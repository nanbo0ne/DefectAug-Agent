"""v3 tuning selection + protocol freeze (run once, between tuning and final).

1. Reads round-3 val mAP50 of formula_tuned_multi_round for every grid point
   (tags tune_g0..tune_gN, tuning seed only), picks the best (ties -> lower
   index) and writes it into formula_tuned_policy.selected.
2. Sets experiment.status to frozen_final and records the protocol sha256 and
   the hashes of every v3 code file in reports/v3/protocol_freeze_v3.json.
   After this, configs/experiment_v3.yaml must not change.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs" / "experiment_v3.yaml"
CODE = [
    "src/defectaug/gen_params.py", "src/defectaug/allocation.py", "src/defectaug/policy.py",
    "src/defectaug/patch_copy.py", "src/defectaug/data_lock.py", "src/defectaug/test_access.py",
    "scripts/run_conditions_v3.py", "scripts/run_synth_generation_v3.py",
    "scripts/val_miss_analysis_v3.py", "scripts/build_synth_dataset.py",
    "scripts/train_yolo.py", "scripts/evaluate_yolo.py", "scripts/agent_decide.py",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if p["experiment"]["status"] == "frozen_final":
        raise SystemExit("already frozen")
    ft = p["formula_tuned_policy"]
    seed = ft["tuning_seed"]
    scores = []
    for i, w in enumerate(ft["grid"]):
        m = (PROJECT / "reports" / "v3" / f"tune_g{i}" / "yolo_eval"
             / f"v3_tune_g{i}_formula_tuned_multi_round_s{seed}_r3_val" / "metrics.json")
        ap50 = json.loads(m.read_text(encoding="utf-8"))["metrics"]["overall"]["ap50"]
        scores.append({"index": i, "weights": w, "val_map50_r3": ap50})
    best = max(scores, key=lambda s: (s["val_map50_r3"], -s["index"]))
    text = PROTOCOL.read_text(encoding="utf-8")
    sel = "[" + ", ".join(f"{v:.2f}" for v in best["weights"]) + "]"
    text, n = re.subn(r"  selected: null[^\n]*", f"  selected: {sel}   # grid index {best['index']}, chosen on tuning-seed val", text)
    assert n == 1
    stamp = time.strftime("%Y-%m-%d")
    text, n1 = re.subn(r"  status: draft\n", "  status: frozen_final\n", text)
    text, n2 = re.subn(r"  frozen_at: null\n", f"  frozen_at: {stamp}\n", text)
    assert n1 == 1 and n2 == 1
    PROTOCOL.write_text(text, encoding="utf-8")
    record = {
        "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "protocol": "configs/experiment_v3.yaml",
        "protocol_sha256": sha(PROTOCOL),
        "tuning_scores": scores,
        "selected": best,
        "code_sha256": {c: sha(PROJECT / c) for c in CODE},
        "split_sha256": {f.name: sha(f) for f in sorted((PROJECT / "data/splits/v3").glob("*.txt"))},
    }
    out = PROJECT / "reports" / "v3" / "protocol_freeze_v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: record[k] for k in ("protocol_sha256", "tuning_scores", "selected")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
