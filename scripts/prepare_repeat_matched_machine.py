"""Relocate and lock the train/validation-only repeat-matched control bundle."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from defectaug.data_lock import build_content_lock, verify_content_lock, write_content_lock
from prepare_real_only_machine import EXPECTED_MODEL_SHA256, main as prepare_real_only
from prepare_real_repeat_control import build_repeated_rows


PROJECT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    prepare_real_only()
    source_manifest = PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt"
    source_paths = [
        Path(line.strip()).resolve(strict=True)
        for line in source_manifest.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]
    repeated, targets = build_repeated_rows(source_paths, 250)
    repeat_manifest = PROJECT / "data" / "splits" / "v1" / "train_fewshot_repeat_matched.txt"
    repeat_manifest.write_text(
        "\n".join(str(path) for path in repeated) + "\n", encoding="utf-8", newline=""
    )
    val_manifest = PROJECT / "data" / "splits" / "v1" / "val.txt"
    data_yaml = PROJECT / "configs" / "dataset_repeat_matched_v1.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(PROJECT.resolve()),
                "train": str(repeat_manifest.resolve()),
                "val": str(val_manifest.resolve()),
                "names": {0: "inclusion", 1: "patches", 2: "scratches"},
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="",
    )
    lock_path = PROJECT / "data" / "manifests" / "dataset_repeat_matched_trainval_lock_v1.json"
    document = build_content_lock(
        project_root=PROJECT,
        data_yaml=data_yaml,
        labels_root=PROJECT / "data" / "canonical" / "labels" / "all",
        masks_root=PROJECT / "data" / "canonical" / "masks" / "all",
        allow_duplicate_train_entries=True,
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
        "condition": "real_only_repeat_matched",
        "test_in_bundle": False,
        "test_accessed": False,
        "source_real_images": 100,
        "train_rows": len(repeated),
        "train_rows_by_class": targets,
        "train_manifest": str(repeat_manifest),
        "train_manifest_sha256": sha256_file(repeat_manifest),
        "val_manifest": str(val_manifest),
        "dataset_yaml": str(data_yaml),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "content_lock": verification.as_dict(),
        "initial_model_sha256": EXPECTED_MODEL_SHA256,
    }
    output = PROJECT / "reports" / "repeat_matched_machine_setup.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
