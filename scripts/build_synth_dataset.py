"""Build a dataset YAML + byte-level content lock for a mixed manifest.

For each (condition, seed, round) the train manifest lists the frozen 100
real few-shot images plus every synthetic image produced in rounds 1..k.
Validation and test manifests stay the frozen ones.  Labels and masks for
both real and synthetic stems resolve through the shared companion root
data/synthetic/lock_support/{labels,masks} populated by
run_synth_generation.py (canonical files are copied there once).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from defectaug.data_lock import build_content_lock, write_content_lock

PROJECT = Path(__file__).resolve().parents[1]
POOL_ROOT = PROJECT / "data" / "synthetic"
LABELS_ROOT = POOL_ROOT / "lock_support" / "labels"
MASKS_ROOT = POOL_ROOT / "lock_support" / "masks"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--manifest", help="Override manifest path.")
    # v3 points val/test at data/splits/v3; the default keeps v2 builds unchanged.
    parser.add_argument("--splits-dir", default="data/splits/v2")
    args = parser.parse_args()

    key = f"{args.condition}_{args.seed}_r{args.round}"
    pool = POOL_ROOT / f"{args.condition}_{args.seed}"
    manifest = Path(args.manifest) if args.manifest else pool / f"train_manifest_r{args.round}.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)

    yaml_dir = PROJECT / "configs" / "generated"
    yaml_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = yaml_dir / f"dataset_{key}.yaml"
    document = {
        "path": str(PROJECT).replace("\\", "/"),
        "train": str(manifest).replace("\\", "/"),
        "val": str(PROJECT / args.splits_dir / "val.txt").replace("\\", "/"),
        "test": str(PROJECT / args.splits_dir / "test.txt").replace("\\", "/"),
        "names": {0: "inclusion", 1: "patches", 2: "scratches"},
    }
    yaml_path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")

    lock_path = PROJECT / "data" / "manifests" / f"dataset_content_lock_{key}.json"
    lock = build_content_lock(
        project_root=PROJECT,
        data_yaml=yaml_path,
        labels_root=LABELS_ROOT,
        masks_root=MASKS_ROOT,
    )
    write_content_lock(lock, lock_path, force=lock_path.exists())
    print(json.dumps({"yaml": str(yaml_path), "lock": str(lock_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
