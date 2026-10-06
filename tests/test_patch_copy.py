from __future__ import annotations

import numpy as np

from defectaug.patch_copy import (
    YoloBox,
    copy_real_defect,
    gaussian_feather_boundary_ring,
    poisson_blend_boundary_ring,
)


def test_patch_copy_preserves_every_pixel_outside_inserted_mask() -> None:
    source = np.full((64, 64, 3), 100, dtype=np.uint8)
    source[20:32, 18:36] = 210
    source_mask = np.zeros((64, 64), dtype=bool)
    source_mask[20:32, 18:36] = True
    target = np.full((64, 64, 3), 90, dtype=np.uint8)
    target[4:12, 4:12] = 140
    boxes = [YoloBox(0, 0.125, 0.125, 0.125, 0.125)]

    result = copy_real_defect(
        source_image=source,
        source_mask=source_mask,
        target_image=target,
        target_boxes=boxes,
        class_id=0,
        rng=np.random.default_rng(7),
        scale=0.8,
        max_overlap_fraction=0.0,
    )

    assert result.metrics.outside_mask_changed_pixels == 0
    assert np.array_equal(result.image[~result.inserted_mask], target[~result.inserted_mask])
    assert result.metrics.inside_mask_changed_fraction > 0.95
    assert 0.0 <= result.metrics.existing_box_overlap_fraction <= 0.0


def test_inserted_yolo_box_tightly_contains_mask() -> None:
    source = np.full((80, 80, 3), 80, dtype=np.uint8)
    source[30:44, 25:39] = 180
    source_mask = np.zeros((80, 80), dtype=bool)
    source_mask[30:44, 25:39] = True
    target = np.full((80, 80, 3), 70, dtype=np.uint8)

    result = copy_real_defect(
        source_image=source,
        source_mask=source_mask,
        target_image=target,
        target_boxes=[YoloBox(1, 0.1, 0.1, 0.1, 0.1)],
        class_id=1,
        rng=np.random.default_rng(11),
        scale=1.0,
        max_overlap_fraction=0.0,
    )

    ys, xs = np.where(result.inserted_mask)
    box = result.inserted_box
    x0 = (box.center_x - box.width / 2) * 80
    x1 = (box.center_x + box.width / 2) * 80
    y0 = (box.center_y - box.height / 2) * 80
    y1 = (box.center_y + box.height / 2) * 80
    assert np.isclose(x0, xs.min())
    assert np.isclose(x1, xs.max() + 1)
    assert np.isclose(y0, ys.min())
    assert np.isclose(y1, ys.max() + 1)
    assert box.class_id == 1


def test_poisson_boundary_blend_changes_only_exterior_ring() -> None:
    target = np.full((72, 72, 3), 80, dtype=np.uint8)
    target[:, :, 1] = np.arange(72, dtype=np.uint8)[:, None] + 60
    baseline = target.copy()
    inserted_mask = np.zeros((72, 72), dtype=bool)
    inserted_mask[27:45, 25:47] = True
    baseline[inserted_mask] = np.array([195, 110, 45], dtype=np.uint8)

    for clone_mode in ("normal", "mixed"):
        result = poisson_blend_boundary_ring(
            baseline_image=baseline,
            target_image=target,
            inserted_mask=inserted_mask,
            ring_radius_pixels=4,
            clone_mode=clone_mode,
        )

        assert np.array_equal(result.image[inserted_mask], baseline[inserted_mask])
        assert np.array_equal(result.image[~result.dilated_mask], target[~result.dilated_mask])
        assert not np.any(result.boundary_ring & inserted_mask)
        assert np.array_equal(result.inserted_mask, inserted_mask)
        assert result.metrics.core_changed_vs_baseline_pixels == 0
        assert result.metrics.outside_dilated_mask_changed_vs_target_pixels == 0
        assert result.metrics.boundary_ring_pixels == int(result.boundary_ring.sum())


def test_poisson_boundary_blend_is_deterministic() -> None:
    target = np.full((64, 64, 3), 90, dtype=np.uint8)
    baseline = target.copy()
    inserted_mask = np.zeros((64, 64), dtype=bool)
    inserted_mask[20:42, 28:36] = True
    baseline[inserted_mask] = 175

    first = poisson_blend_boundary_ring(
        baseline_image=baseline,
        target_image=target,
        inserted_mask=inserted_mask,
        ring_radius_pixels=3,
    )
    second = poisson_blend_boundary_ring(
        baseline_image=baseline,
        target_image=target,
        inserted_mask=inserted_mask,
        ring_radius_pixels=3,
    )

    assert np.array_equal(first.image, second.image)
    assert first.metrics == second.metrics


def test_gaussian_feather_changes_only_exterior_ring_and_reduces_step_edge() -> None:
    target = np.full((80, 80, 3), 75, dtype=np.uint8)
    baseline = target.copy()
    inserted_mask = np.zeros((80, 80), dtype=bool)
    inserted_mask[28:52, 25:55] = True
    baseline[inserted_mask] = 195

    result = gaussian_feather_boundary_ring(
        baseline_image=baseline,
        target_image=target,
        inserted_mask=inserted_mask,
        ring_radius_pixels=4,
    )

    assert np.array_equal(result.image[inserted_mask], baseline[inserted_mask])
    assert np.array_equal(result.image[~result.dilated_mask], target[~result.dilated_mask])
    assert result.metrics.core_changed_vs_baseline_pixels == 0
    assert result.metrics.outside_dilated_mask_changed_vs_target_pixels == 0
    assert result.metrics.ring_changed_vs_target_pixels > 0
    assert result.metrics.inner_boundary_jump_after < result.metrics.inner_boundary_jump_before
    assert result.metrics.seam_score_reduction_fraction > 0
