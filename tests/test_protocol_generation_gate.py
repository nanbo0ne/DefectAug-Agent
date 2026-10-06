from __future__ import annotations

import csv
from pathlib import Path

import yaml


PROJECT = Path(__file__).resolve().parents[1]


def load_protocol() -> dict:
    return yaml.safe_load(
        (PROJECT / "configs/experiment_v1.yaml").read_text(encoding="utf-8")
    )


def test_scratches_is_engineering_gate_and_three_class_confirmation_is_required() -> None:
    generation = load_protocol()["generation"]
    pilot = generation["pilot"]
    confirmation = generation["three_class_confirmation"]

    assert pilot["engineering_class"] == "scratches"
    assert pilot["purpose"] == "engineering_gate_only"
    assert confirmation["required_before_generator_freeze"] is True
    assert confirmation["classes"] == ["inclusion", "patches", "scratches"]
    assert confirmation["sources_per_class"] == 5
    assert confirmation["seeds_per_source"] == 2
    assert confirmation["candidates_per_class_per_generator"] == 10
    assert confirmation["candidates_per_generator"] == 30
    assert confirmation["class_tool_mapping_shared_across_methods"] is True
    assert confirmation["selection_metrics"] == [
        "macro_mean_valid_candidate_rate",
        "worst_class_valid_candidate_rate",
    ]


def test_pilot_thresholds_are_diagnostic_and_formal_qc_is_fully_automatic() -> None:
    # Amended 2026-09-19 (docs/14): the loop is fully automatic; quality is
    # judged by hard code invariants plus downstream detector utility.
    quality = load_protocol()["quality_control"]

    assert quality["thresholds_enforced"] is False
    assert quality["inside_mask_changed_pixel_fraction_min"] is None
    assert quality["changed_region_mask_iou_min"] is None
    assert quality["semantic_auto_acceptance_threshold"] is None
    assert quality["class_semantic_score_min"] is None
    assert quality["outside_mask_changed_pixel_fraction_max"] == 0.0
    assert quality["source_mask_xml_iou_min"] == 0.50
    assert quality["pilot_human_review_fraction"] == 0.0
    assert quality["formal_accepted_human_review_fraction"] == 0.0
    assert quality["formal_blinded_human_review"] is False
    assert quality["formal_second_reviewer_fraction"] == 0.0
    assert quality["formal_second_reviewer_sampling"] == "none"
    assert quality["reviewer_disagreement_requires_adjudication"] is False


def test_one_row_smoke_manifest_is_exact_first_frozen_job() -> None:
    frozen = PROJECT / "data/manifests/generation_jobs_scratch_gate_v1.csv"
    smoke = PROJECT / "data/manifests/generation_jobs_scratch_gate_smoke_first_v1.csv"
    with frozen.open("r", encoding="utf-8-sig", newline="") as stream:
        frozen_rows = list(csv.DictReader(stream))
    with smoke.open("r", encoding="utf-8-sig", newline="") as stream:
        smoke_rows = list(csv.DictReader(stream))

    assert len(frozen_rows) == 30
    assert smoke_rows == frozen_rows[:1]
