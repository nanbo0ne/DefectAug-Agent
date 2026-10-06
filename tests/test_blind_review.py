"""Blind-review packaging tests that never decode candidate image content."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "scripts" / "prepare_blind_review.py"
SPEC = importlib.util.spec_from_file_location("prepare_blind_review", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def make_manifest(tmp_path: Path, name: str, payload: bytes) -> Path:
    image = tmp_path / f"{name}.png"
    image.write_bytes(payload)
    digest = MODULE.sha256_file(image)
    manifest = tmp_path / f"{name}.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "record_type": "result",
                "run_id": f"run-{name}",
                "status": "ok",
                "job": {
                    "job_id": "scratches_182__s7",
                    "source_id": "scratches_182",
                    "output": str(image),
                },
                "result": {
                    "composite_output": str(image),
                    "composite_sha256": digest,
                    "review_status": MODULE.PENDING_REVIEW,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return manifest


def test_blind_package_hides_origin_and_copies_bytes(tmp_path: Path) -> None:
    first = make_manifest(tmp_path, "qwen_method", b"candidate-one")
    second = make_manifest(tmp_path, "sdxl_method", b"candidate-two")
    package = tmp_path / "reviewer_package"
    mapping = tmp_path / "private" / "mapping.jsonl"

    plan = MODULE.prepare_package(
        manifests=[first, second],
        package_dir=package,
        mapping=mapping,
        seed=17,
    )

    assert plan["candidate_count"] == 2
    sheet = (package / "review_sheet.csv").read_text(encoding="utf-8-sig")
    public = (package / "package_manifest.json").read_text(encoding="utf-8")
    assert "qwen_method" not in sheet + public
    assert "sdxl_method" not in sheet + public
    assert "scratches_182" not in sheet + public
    assert "R0001" in sheet and "R0002" in sheet
    private = mapping.read_text(encoding="utf-8")
    assert "qwen_method" in private and "sdxl_method" in private
    for item in plan["items"]:
        assert MODULE.sha256_file(Path(item["review_image"])) == item["candidate_sha256"]


def test_blind_package_rejects_mapping_inside_reviewer_directory(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, "generator", b"candidate")
    package = tmp_path / "package"
    try:
        MODULE.prepare_package(
            manifests=[manifest],
            package_dir=package,
            mapping=package / "mapping.jsonl",
            seed=1,
            dry_run=True,
        )
    except ValueError as exc:
        assert "outside" in str(exc)
    else:
        raise AssertionError("origin mapping inside reviewer package was accepted")


def test_blind_package_requires_explicit_pending_status(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, "generator", b"candidate")
    text = manifest.read_text(encoding="utf-8").replace(
        MODULE.PENDING_REVIEW, "accepted"
    )
    manifest.write_text(text, encoding="utf-8")
    try:
        MODULE.load_candidates([manifest])
    except ValueError as exc:
        assert "pending blind review" in str(exc)
    else:
        raise AssertionError("non-pending candidate entered blind-review package")


def test_append_only_assignment_repairs_an_old_missing_status(tmp_path: Path) -> None:
    manifest = make_manifest(tmp_path, "old_runner", b"candidate")
    record = json.loads(manifest.read_text(encoding="utf-8"))
    del record["result"]["review_status"]
    assignment = {
        "record_type": "review_status_assignment",
        "run_id": record["run_id"],
        "job_id": record["job"]["job_id"],
        "review_status": MODULE.PENDING_REVIEW,
        "reason": "old runner omitted a protocol-required field",
    }
    manifest.write_text(
        json.dumps(record) + "\n" + json.dumps(assignment) + "\n", encoding="utf-8"
    )

    candidates = MODULE.load_candidates([manifest])

    assert candidates[0]["review_status_source"] == "append_only_assignment"
