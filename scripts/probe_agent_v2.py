"""One-off probe: round-1 agent decision through the formal v2 code path.

Uses real_only smoke val metrics as the "previous" feedback, calls the LLM via
run_conditions.agent_proposal (same prompt, retry loop and validator as the
formal run), and writes the full record.  No training, no generation, no test.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
sys.path.insert(0, str(PROJECT / "scripts"))
import run_conditions as rc  # noqa: E402
from defectaug.allocation import BudgetProtocol  # noqa: E402

protocol = yaml.safe_load(rc.PROTOCOL_PATH.read_text(encoding="utf-8"))
ctx = rc.Ctx(protocol, "agentprobe", 1, "0")
ctx.agent_endpoint = protocol["agent_policy"]["endpoint"]
ctx.agent_model = protocol["agent_policy"]["model"]

bp = BudgetProtocol.from_protocol(protocol, rounds=3)
prev = PROJECT / "reports/v2/smoke/yolo_eval/v2_smoke_real_only_s2070104565_val/metrics.json"
seed = protocol["experiment"]["final_seeds"][0]
cond = "agent_multi_round"
allocated = {c: 0 for c in rc.CLASSES}
cap = rc.capacity(ctx, cond, seed)
obs = rc.observation(ctx, cond, seed, 1, bp, prev, allocated, cap, [])

t0 = time.time()
alloc, meta = rc.agent_proposal(ctx, obs, seed, 1, bp, allocated, cap)
wall = time.time() - t0

out = PROJECT / "reports/v2/agentprobe/probe_r1.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"observation": obs, "allocation": alloc, "meta": meta,
                           "wall_seconds": round(wall, 2)}, ensure_ascii=False, indent=1),
               encoding="utf-8")

print("capacity", cap)
print("allocation", alloc, "sum", sum(alloc.values()) if alloc else None)
print("repaired", meta.get("repaired"), "attempts", len(meta["attempts"]), "wall", round(wall, 1), "s")
for a in meta["attempts"]:
    r = a["response"]
    print(f"  attempt {a['attempt']}: served={r.get('served_model')} finish={r.get('finish_reason')} "
          f"usage={r.get('usage', {}).get('completion_tokens')} reasoning_chars={r.get('reasoning_chars')} "
          f"t={r.get('elapsed_seconds')}s err={a.get('error')}")
print("rationale:", (meta.get("rationale") or "")[:400])
print("wrote", out.relative_to(PROJECT))
