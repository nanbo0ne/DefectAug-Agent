import numpy as np
from pathlib import Path

from defectaug.quality import (
    candidate_metrics,
    hard_composite,
    load_yolo_box_mask,
    quality_report,
    save_and_verify_candidate,
)


def test_hard_composite_preserves_outside_mask() -> None:
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    generated = np.full((20, 20, 3), 255, dtype=np.uint8)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    result = hard_composite(original, generated, mask)
    metrics = candidate_metrics(original, result, mask)
    assert metrics.outside_changed_fraction == 0.0
    assert metrics.inside_changed_fraction == 1.0
    assert metrics.changed_region_mask_iou == 1.0


def test_shared_report_separates_raw_drift_composite_invariant_and_box_constraint(
    tmp_path: Path,
) -> None:
    original = np.zeros((20, 20, 3), dtype=np.uint8)
    raw = original.copy()
    raw[5:15, 5:15] = 120
    raw[0, 0] = 120  # raw generator drift outside the permitted mask
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 255
    composite = hard_composite(original, raw, mask)
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    boxes = load_yolo_box_mask(label, 20, 20)

    report = quality_report(
        original,
        raw,
        composite,
        mask,
        yolo_box_mask=boxes,
        pixel_threshold=8,
        boundary_ring_width=1,
    )

    assert report.raw_outside_changed_pixels == 1
    assert report.composite_outside_changed_pixels == 0
    assert report.composite_hard_invariant is True
    assert report.inside_changed_fraction == 1.0
    assert report.effective_change_within_yolo_box_fraction == 1.0
    assert report.boundary_ring_seam_score > 0


def test_save_and_verify_persists_binary_mask_and_hashes(tmp_path: Path) -> None:
    original = np.zeros((8, 8, 3), dtype=np.uint8)
    raw = np.full((8, 8, 3), 80, dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 200
    label = tmp_path / "sample.txt"
    label.write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    composite = tmp_path / "candidate.png"
    final_mask = tmp_path / "candidate.mask.png"

    result = save_and_verify_candidate(
        original=original,
        raw_candidate=raw,
        mask=mask,
        composite_path=composite,
        final_mask_path=final_mask,
        label_path=label,
    )

    assert composite.is_file() and final_mask.is_file()
    assert len(result["composite_sha256"]) == 64
    assert len(result["final_binary_mask_sha256"]) == 64
    assert len(result["label_sha256"]) == 64
    assert result["quality"]["composite_hard_invariant"] is True
