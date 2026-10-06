"""ONE-TIME v2 final test evaluation (18 models = 6 conditions x 3 seeds).

Step 1 (freeze): hash the frozen protocol and every final model's weights,
data yaml and content lock into reports/v2/final/protocol_freeze.json.  If
the freeze file already exists, the current hashes must match it exactly.
Step 2 (test): each model passes the test-access gate in evaluate_yolo.py
(protocol sha256 + unlock token + audit log).  Existing test metrics are
skipped, never re-run to pick a result.  Multi-round conditions use r3.

Usage:
    python scripts/final_test_v2.py --freeze-only
    python scripts/final_test_v2.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PY = PROJECT / ".venv" / "Scripts" / "python.exe"
PROTOCOL = PROJECT / "configs" / "experiment_v2.yaml"
FINAL = PROJECT / "reports" / "v2" / "final"
EVAL_ROOT = FINAL / "yolo_eval"
FREEZE = FINAL / "protocol_freeze.json"
TAG = "final"
CONDITIONS = ("real_only", "real_only_repeat_matched", "uniform_fixed",
              "formula_one_shot", "formula_multi_round", "agent_multi_round")


def log(msg: str) -> None:
    print(f"[{time.strftime('%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def final_models(p: dict) -> list[dict]:
    """The model that represents each condition/seed on test (r3 for multi-round)."""
    conds = {c["name"]: c for c in p["conditions"]}
    out = []
    for seed in p["experiment"]["final_seeds"]:
        for name in CONDITIONS:
            c = conds[name]
            rounds = int(c.get("rounds", 1))
            run = f"v2_{TAG}_{name}_s{seed}" + (f"_r{rounds}" if rounds > 1 else "")
            if c.get("augmented"):
                key = f"v2_{TAG}_{name}_{seed}_r{rounds}"
                data = PROJECT / "configs" / "generated" / f"dataset_{key}.yaml"
                lock = PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json"
            else:
                data = PROJECT / c.get("dataset_yaml", p["dataset"]["yaml"])
                lock = PROJECT / c.get("content_lock", p["dataset"]["content_lock"])
            weights = PROJECT / "runs" / "yolo" / run / "weights" / p["detector"]["eval_weights"]
            val = EVAL_ROOT / f"{run}_val" / "metrics.json"
            out.append({"condition": name, "seed": seed, "run": run,
                        "weights": weights, "data": data, "lock": lock, "val_metrics": val})
    return out
def rel(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def snapshot(p: dict) -> dict:
    models = []
    for m in final_models(p):
        for k in ("weights", "data", "lock", "val_metrics"):
            if not m[k].is_file():
                raise SystemExit(f"FATAL: missing {k} for {m['run']}: {m[k]}")
        # best.pt must be the same file that produced the val result.
        val_doc = json.loads(m["val_metrics"].read_text(encoding="utf-8"))
        w_hash = sha256(m["weights"])
        if val_doc["inputs"]["weights"]["sha256"] != w_hash:
            raise SystemExit(f"FATAL: {m['run']} best.pt changed since val evaluation")
        models.append({"condition": m["condition"], "seed": m["seed"], "run": m["run"],
                       "weights": rel(m["weights"]), "weights_sha256": w_hash,
                       "data_yaml": rel(m["data"]), "data_yaml_sha256": sha256(m["data"]),
                       "content_lock": rel(m["lock"]), "content_lock_sha256": sha256(m["lock"])})
    return {"protocol": rel(PROTOCOL), "protocol_sha256": sha256(PROTOCOL),
            "experiment_status": p["experiment"]["status"], "models": models}


def freeze(p: dict) -> dict:
    if p["experiment"]["status"] != "frozen_final":
        raise SystemExit("FATAL: protocol status is not frozen_final")
    snap = snapshot(p)
    if FREEZE.is_file():
        old = json.loads(FREEZE.read_text(encoding="utf-8"))
        old_cmp = {k: v for k, v in old.items() if k != "frozen_at"}
        if old_cmp != snap:
            raise SystemExit("FATAL: current artifacts differ from protocol_freeze.json")
        log(f"freeze verified: protocol {snap['protocol_sha256'][:12]}, {len(snap['models'])} models")
        return old
    snap["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    FREEZE.write_text(json.dumps(snap, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"freeze written: protocol {snap['protocol_sha256'][:12]}, {len(snap['models'])} models")
    return snap


def run_test(p: dict, frozen: dict) -> None:
    det = p["detector"]
    token = p["final_test_access"]["unlock_token"]
    for m in frozen["models"]:
        name = f"{m['run']}_test"
        if (EVAL_ROOT / name / "metrics.json").is_file():
            log(f"SKIP {name} (metrics present)")
            continue
        log(f"TEST {name}")
        cmd = [PY, "scripts/evaluate_yolo.py", "--weights", m["weights"], "--data", m["data_yaml"],
               "--split", "test", "--content-lock", m["content_lock"], "--content-lock-mode", "required",
               "--expected-weights-sha256", m["weights_sha256"],
               "--expected-data-sha256", m["data_yaml_sha256"],
               "--protocol", frozen["protocol"], "--expected-protocol-sha256", frozen["protocol_sha256"],
               "--final-test-unlock-token", token,
               "--imgsz", det["imgsz"], "--batch", det["batch"], "--device", "0",
               "--workers", det["workers"], "--seed", m["seed"], "--conf", 0.001, "--iou", 0.7,
               "--max-det", 300, "--output-root", rel(EVAL_ROOT), "--name", name, "--exist-ok"]
        logf = EVAL_ROOT / f"{name}.log"
        with logf.open("w", encoding="utf-8") as s:
            rc = subprocess.call([str(c) for c in cmd], cwd=PROJECT, stdout=s, stderr=subprocess.STDOUT)
        if rc != 0:
            raise SystemExit(f"FATAL: test evaluation failed for {name} (see {logf.name})")
    log("final test evaluation finished")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze-only", action="store_true")
    args = ap.parse_args()
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    frozen = freeze(p)
    if not args.freeze_only:
        run_test(p, frozen)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
