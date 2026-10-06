"""Relocate and lock the train/validation-only Real-only transfer bundle.

The ID templates contain no absolute paths.  This script binds them to the
directory in which the bundle was extracted, writes the Ultralytics dataset
YAML, and creates a byte-level content lock for train_fewshot and val only.
Image and mask files are hashed as opaque bytes and are never decoded here.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from defectaug.data_lock import build_content_lock, verify_content_lock, write_content_lock


PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
EXPECTED_COUNTS = {
    "train_fewshot": {"inclusion": 10, "patches": 30, "scratches": 60},
    "val": {"inclusion": 50, "patches": 50, "scratches": 50},
}
EXPECTED_MODEL_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(split: str) -> list[str]:
    source = PROJECT / "data" / "splits" / "portable" / f"{split}_ids.txt"
    values = [line.strip() for line in source.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate ID in {source}")
    counts = {name: sum(value.startswith(f"{name}_") for value in values) for name in CLASSES}
    if counts != EXPECTED_COUNTS[split]:
        raise RuntimeError(f"unexpected {split} class counts: {counts}")
    return values


def bind_manifest(split: str, ids: list[str]) -> Path:
    image_root = PROJECT / "data" / "canonical" / "images" / "all"
    label_root = PROJECT / "data" / "canonical" / "labels" / "all"
    mask_root = PROJECT / "data" / "canonical" / "masks" / "all"
    paths: list[Path] = []
    for item_id in ids:
        image = image_root / f"{item_id}.jpg"
        label = label_root / f"{item_id}.txt"
        mask = mask_root / f"{item_id}.png"
        for candidate in (image, label, mask):
            if not candidate.is_file():
                raise FileNotFoundError(candidate)
        paths.append(image.resolve())
    destination = PROJECT / "data" / "splits" / "v1" / f"{split}.txt"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(str(path) for path in paths) + "\n", encoding="utf-8", newline="")
    return destination


def main() -> int:
    train_ids = read_ids("train_fewshot")
    val_ids = read_ids("val")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"train/validation ID overlap: {sorted(overlap)[:5]}")

    train_manifest = bind_manifest("train_fewshot", train_ids)
    val_manifest = bind_manifest("val", val_ids)
    data_yaml = PROJECT / "configs" / "dataset_real_only_v1.yaml"
    data_yaml.parent.mkdir(parents=True, exist_ok=True)
    dataset = {
        "path": str(PROJECT.resolve()),
        "train": str(train_manifest.resolve()),
        "val": str(val_manifest.resolve()),
        "names": {0: "inclusion", 1: "patches", 2: "scratches"},
    }
    data_yaml.write_text(
        yaml.safe_dump(dataset, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="",
    )

    model = PROJECT / "yolo11n.pt"
    if sha256_file(model) != EXPECTED_MODEL_SHA256:
        raise RuntimeError("pinned yolo11n.pt SHA-256 mismatch")

    lock_path = PROJECT / "data" / "manifests" / "dataset_real_only_trainval_lock_v1.json"
    document = build_content_lock(
        project_root=PROJECT,
        data_yaml=data_yaml,
        labels_root=PROJECT / "data" / "canonical" / "labels" / "all",
        masks_root=PROJECT / "data" / "canonical" / "masks" / "all",
        active_splits={"train_fewshot": "train", "val": "val"},
    )
    write_content_lock(document, lock_path, force=True)
    verification = verify_content_lock(
        lock_path,
        project_root=PROJECT,
        splits=("train_fewshot", "val"),
        expected_data_yaml=data_yaml,
    )
    report = {
        "prepared_at": datetime.now().astimezone().isoformat(),
        "project_root": str(PROJECT.resolve()),
        "test_in_bundle": False,
        "test_accessed": False,
        "train_manifest": str(train_manifest),
        "val_manifest": str(val_manifest),
        "dataset_yaml": str(data_yaml),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "content_lock": verification.as_dict(),
        "initial_model_sha256": EXPECTED_MODEL_SHA256,
    }
    output = PROJECT / "reports" / "real_only_machine_setup.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

