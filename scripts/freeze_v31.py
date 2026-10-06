"""v3.1 protocol freeze (run once, after dev on the tuning seed, before any final31 run).

Sets experiment.status to frozen_final in configs/experiment_v31.yaml and records
the protocol sha256 plus the hashes of every file the v3.1 runs execute in
reports/v3/protocol_freeze_v31.json.  After this, none of them may change;
scripts/check_freeze_v31.py verifies that (locally and on the execution host).
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path

import yaml

PROJECT = Path(__file__).resolve().parents[1]
PROTOCOL = PROJECT / "configs" / "experiment_v31.yaml"
RECORD = PROJECT / "reports" / "v3" / "protocol_freeze_v31.json"
CODE = [
    "configs/experiment_v3.yaml",
    "src/defectaug/gen_params.py", "src/defectaug/allocation.py", "src/defectaug/policy.py",
    "src/defectaug/patch_copy.py", "src/defectaug/data_lock.py", "src/defectaug/test_access.py",
    "src/defectaug/vision_agent.py", "src/defectaug/vision_tools.py",
    "scripts/run_conditions_v3.py", "scripts/run_conditions_v31.py", "scripts/run_conditions_v31e.py",
    "scripts/run_synth_generation_v3.py", "scripts/run_synth_generation_v31.py",
    "scripts/val_miss_analysis_v3.py", "scripts/val_miss_analysis_v31.py", "scripts/build_synth_dataset.py",
    "scripts/train_yolo.py", "scripts/evaluate_yolo.py", "scripts/agent_decide.py",
    "data/splits/v3/train_fewshot.txt", "data/splits/v3/val.txt", "data/splits/v3/test.txt",
    "data/manifests/dataset_content_lock_v3.json", "data/manifests/dataset_content_lock_realmatch_v3.json",
    "requirements-lock-v31.txt", "yolo11n.pt",
]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    p = yaml.safe_load(PROTOCOL.read_text(encoding="utf-8"))
    if p["experiment"]["status"] == "frozen_final":
        raise SystemExit("already frozen")
    text = PROTOCOL.read_text(encoding="utf-8")
    text, n1 = re.subn(r"  status: draft\n", "  status: frozen_final\n", text)
    text, n2 = re.subn(r"  frozen_at: null\n", f"  frozen_at: {time.strftime('%Y-%m-%d')}\n", text)
    assert n1 == 1 and n2 == 1
    PROTOCOL.write_text(text, encoding="utf-8")
    record = {"frozen_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "protocol": "configs/experiment_v31.yaml",
              "protocol_sha256": sha(PROTOCOL), "code_sha256": {c: sha(PROJECT / c) for c in CODE}}
    RECORD.write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps({"protocol_sha256": record["protocol_sha256"], "files": len(CODE)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
