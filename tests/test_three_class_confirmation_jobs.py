from __future__ import annotations

import csv
import importlib.util
from collections import Counter
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts/prepare_three_class_confirmation_jobs.py"
SPEC = importlib.util.spec_from_file_location("prepare_three_class_confirmation_jobs", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_three_class_confirmation_table_is_balanced_and_hashed() -> None:
    path = PROJECT / "data/manifests/generation_jobs_three_class_confirmation_v1.csv"
    protocol = yaml.safe_load(
        (PROJECT / "configs/experiment_v1.yaml").read_text(encoding="utf-8")
    )
    confirmation = protocol["generation"]["three_class_confirmation"]
    assert confirmation["frozen_job_manifest"] == str(path.relative_to(PROJECT)).replace(
        "\\", "/"
    )
    assert MODULE.sha256_file(path) == confirmation["frozen_job_manifest_sha256"]
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 30
    assert Counter(row["primary_class"] for row in rows) == {
        "inclusion": 10,
        "patches": 10,
        "scratches": 10,
    }
    for class_name in MODULE.CLASSES:
        class_rows = [row for row in rows if row["primary_class"] == class_name]
        assert len({row["source_id"] for row in class_rows}) == 5
        assert {int(row["seed"]) for row in class_rows} == {20260820, 20260821}
    assert len({row["job_id"] for row in rows}) == 30
    for row in rows:
        for field, digest_field in (
            ("image", "image_sha256"),
            ("mask", "mask_sha256"),
            ("label", "label_sha256"),
        ):
            assert MODULE.sha256_file(Path(row[field])) == row[digest_field]
