"""Deterministic, mask-guided real-defect copy-paste augmentation.

This module deliberately operates only on an explicit source manifest.  The
research runner supplies the frozen ``train_fewshot`` manifest, so validation
and locked-test images cannot silently become generation sources or targets.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

    def line(self) -> str:
        return (
            f"{self.class_id} {self.center_x:.6f} {self.center_y:.6f} "
            f"{self.width:.6f} {self.height:.6f}"
        )


@dataclass(frozen=True)
class PatchCopyMetrics:
    inserted_mask_pixels: int
    inserted_mask_fraction: float
    outside_mask_changed_pixels: int
    inside_mask_changed_fraction: float
    mean_absolute_change_inside: float
    existing_box_overlap_fraction: float
    boundary_mean_absolute_change: float
    raw_background_shift: tuple[float, float, float]
    source_background_shift: tuple[float, float, float]
    background_shift_clipped: bool
    scale: float
    rotation_degrees: int
    horizontal_flip: bool
    vertical_flip: bool

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class PatchCopyResult:
    image: np.ndarray
    inserted_mask: np.ndarray
    inserted_box: YoloBox
    metrics: PatchCopyMetrics


@dataclass(frozen=True)
class BoundaryBlendMetrics:
    """Objective diagnostics for a boundary-ring-only Poisson blend.

    The detector label remains tied to ``inserted_mask``.  The defect core is
    copied byte-for-byte from the frozen Patch/Copy-Paste baseline; only the
    exterior ring may differ from the real target image.
    """

    ring_radius_pixels: int
    clone_mode: str
    inserted_mask_pixels: int
    boundary_ring_pixels: int
    core_changed_vs_baseline_pixels: int
    outside_dilated_mask_changed_vs_target_pixels: int
    ring_changed_vs_target_pixels: int
    ring_changed_vs_target_fraction: float
    ring_mean_absolute_change_vs_target: float
    inner_boundary_jump_before: float
    inner_boundary_jump_after: float
    outer_boundary_jump_target: float
    outer_boundary_jump_after: float
    seam_score_before: float
    seam_score_after: float
    seam_score_reduction_fraction: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class BoundaryBlendResult:
    image: np.ndarray
    inserted_mask: np.ndarray
    boundary_ring: np.ndarray
    dilated_mask: np.ndarray
    metrics: BoundaryBlendMetrics


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as stream:
        return np.asarray(stream.convert("RGB"), dtype=np.uint8)


def read_binary_mask(path: Path, threshold: int = 127) -> np.ndarray:
    with Image.open(path) as stream:
        mask = np.asarray(stream.convert("L"), dtype=np.uint8)
    return mask >= threshold


def read_yolo_boxes(path: Path) -> list[YoloBox]:
    boxes: list[YoloBox] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_number} is not a five-field YOLO label")
        try:
            class_id = int(fields[0])
            center_x, center_y, width, height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number} contains invalid numeric fields") from exc
        if class_id < 0 or not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < width <= 1.0
            and 0.0 < height <= 1.0
        ):
            raise ValueError(f"{path}:{line_number} contains out-of-range values")
        boxes.append(YoloBox(class_id, center_x, center_y, width, height))
    if not boxes:
        raise ValueError(f"YOLO label is empty: {path}")
    return boxes


def _largest_component(mask: np.ndarray, min_pixels: int) -> np.ndarray:
    binary = mask.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        raise ValueError("source mask contains no foreground component")
    component_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component_index, cv2.CC_STAT_AREA])
    if area < min_pixels:
        raise ValueError(f"largest source-mask component has only {area} pixels")
    return labels == component_index


def _crop_component(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    margin: int,
    min_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    component = _largest_component(mask, min_pixels)
    ys, xs = np.where(component)
    y0 = max(0, int(ys.min()) - margin)
    y1 = min(image.shape[0], int(ys.max()) + margin + 1)
    x0 = max(0, int(xs.min()) - margin)
    x1 = min(image.shape[1], int(xs.max()) + margin + 1)
    return image[y0:y1, x0:x1].copy(), component[y0:y1, x0:x1].copy()


def _transform_patch(
    patch: np.ndarray,
    mask: np.ndarray,
    *,
    scale: float,
    rotation_degrees: int,
    horizontal_flip: bool,
    vertical_flip: bool,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.1 <= scale <= 2.0:
        raise ValueError("scale must be in [0.1, 2.0]")
    if rotation_degrees not in (0, 90, 180, 270):
        raise ValueError("rotation must be a right angle")
    height = max(2, int(round(patch.shape[0] * scale)))
    width = max(2, int(round(patch.shape[1] * scale)))
    transformed = cv2.resize(patch, (width, height), interpolation=cv2.INTER_LINEAR)
    transformed_mask = cv2.resize(
        mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    if rotation_degrees:
        rotate_code = {
            90: cv2.ROTATE_90_CLOCKWISE,
            180: cv2.ROTATE_180,
            270: cv2.ROTATE_90_COUNTERCLOCKWISE,
        }[rotation_degrees]
        transformed = cv2.rotate(transformed, rotate_code)
        transformed_mask = cv2.rotate(transformed_mask.astype(np.uint8), rotate_code).astype(bool)
    if horizontal_flip:
        transformed = cv2.flip(transformed, 1)
        transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 1).astype(bool)
    if vertical_flip:
        transformed = cv2.flip(transformed, 0)
        transformed_mask = cv2.flip(transformed_mask.astype(np.uint8), 0).astype(bool)
    if int(transformed_mask.sum()) < 16:
        raise ValueError("transformed patch mask became too small")
    return transformed, transformed_mask


def _box_union_mask(boxes: Sequence[YoloBox], height: int, width: int) -> np.ndarray:
    union = np.zeros((height, width), dtype=bool)
    for box in boxes:
        x0 = max(0, int(np.floor((box.center_x - box.width / 2) * width)))
        x1 = min(width, int(np.ceil((box.center_x + box.width / 2) * width)))
        y0 = max(0, int(np.floor((box.center_y - box.height / 2) * height)))
        y1 = min(height, int(np.ceil((box.center_y + box.height / 2) * height)))
        union[y0:y1, x0:x1] = True
    return union


def _find_placement(
    patch_mask: np.ndarray,
    existing_boxes: Sequence[YoloBox],
    image_shape: tuple[int, int],
    rng: np.random.Generator,
    *,
    max_overlap_fraction: float,
    attempts: int,
) -> tuple[int, int, float]:
    height, width = image_shape
    patch_height, patch_width = patch_mask.shape
    if patch_height > height or patch_width > width:
        raise ValueError("transformed patch does not fit target image")
    occupied = _box_union_mask(existing_boxes, height, width)
    candidates: list[tuple[float, int, int]] = []
    for _ in range(attempts):
        y = int(rng.integers(0, height - patch_height + 1))
        x = int(rng.integers(0, width - patch_width + 1))
        overlap = float(occupied[y : y + patch_height, x : x + patch_width][patch_mask].mean())
        candidates.append((overlap, y, x))
        if overlap <= max_overlap_fraction:
            return y, x, overlap
    overlap, y, x = min(candidates)
    if overlap > max_overlap_fraction:
        raise ValueError(
            f"no placement met overlap threshold {max_overlap_fraction:.3f}; best={overlap:.3f}"
        )
    return y, x, overlap


def _mask_to_yolo_box(mask: np.ndarray, class_id: int) -> YoloBox:
    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise ValueError("cannot label an empty inserted mask")
    height, width = mask.shape
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    return YoloBox(
        class_id=class_id,
        center_x=((x0 + x1) / 2) / width,
        center_y=((y0 + y1) / 2) / height,
        width=(x1 - x0) / width,
        height=(y1 - y0) / height,
    )


def _cross_boundary_jump(image: np.ndarray, region: np.ndarray) -> float:
    """Return mean absolute RGB jump across the 4-neighbour region boundary."""

    horizontal = region[:, :-1] != region[:, 1:]
    vertical = region[:-1, :] != region[1:, :]
    values: list[np.ndarray] = []
    if np.any(horizontal):
        difference = np.abs(
            image[:, :-1].astype(np.int16) - image[:, 1:].astype(np.int16)
        ).mean(axis=2)
        values.append(difference[horizontal])
    if np.any(vertical):
        difference = np.abs(
            image[:-1, :].astype(np.int16) - image[1:, :].astype(np.int16)
        ).mean(axis=2)
        values.append(difference[vertical])
    if not values:
        return 0.0
    return float(np.concatenate(values).mean())


def poisson_blend_boundary_ring(
    *,
    baseline_image: np.ndarray,
    target_image: np.ndarray,
    inserted_mask: np.ndarray,
    ring_radius_pixels: int = 4,
    clone_mode: str = "normal",
    changed_threshold: int = 3,
) -> BoundaryBlendResult:
    """Blend only a narrow ring *outside* a copied real defect.

    OpenCV's Poisson solver needs the whole defect-plus-ring region to infer a
    transition.  Its unrestricted result is therefore used only as an
    intermediate proposal.  The final composite restores the complete defect
    core from ``baseline_image`` and copies proposal pixels only in the
    exterior ring.  Pixels beyond the dilated mask are restored from the real
    target.  These two explicit projections make the method suitable for a
    label-preserving A/B gate even though ``seamlessClone`` itself can modify a
    wider numerical support than the supplied mask.
    """

    if baseline_image.shape != target_image.shape or baseline_image.ndim != 3:
        raise ValueError("baseline and target must be same-shaped RGB images")
    if baseline_image.shape[2] != 3:
        raise ValueError("baseline and target must have three channels")
    if inserted_mask.shape != target_image.shape[:2]:
        raise ValueError("inserted mask dimensions do not match images")
    if inserted_mask.dtype != np.bool_:
        inserted_mask = inserted_mask.astype(bool)
    if not np.any(inserted_mask):
        raise ValueError("inserted mask is empty")
    if not 1 <= ring_radius_pixels <= 32:
        raise ValueError("ring radius must be in [1, 32]")
    clone_flags = {
        "normal": cv2.NORMAL_CLONE,
        "mixed": cv2.MIXED_CLONE,
    }
    if clone_mode not in clone_flags:
        raise ValueError("clone mode must be 'normal' or 'mixed'")
    if changed_threshold < 0:
        raise ValueError("changed threshold must be non-negative")

    kernel_size = ring_radius_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(inserted_mask.astype(np.uint8), kernel).astype(bool)
    boundary_ring = dilated_mask & ~inserted_mask
    if not np.any(boundary_ring):
        raise ValueError("boundary ring is empty")

    # Channel order does not affect the independent Poisson solves, but using
    # BGR here follows OpenCV's documented image convention.
    source_bgr = cv2.cvtColor(baseline_image, cv2.COLOR_RGB2BGR)
    target_bgr = cv2.cvtColor(target_image, cv2.COLOR_RGB2BGR)
    center = (target_image.shape[1] // 2, target_image.shape[0] // 2)
    proposal_bgr = cv2.seamlessClone(
        source_bgr,
        target_bgr,
        dilated_mask.astype(np.uint8) * 255,
        center,
        clone_flags[clone_mode],
    )
    proposal = cv2.cvtColor(proposal_bgr, cv2.COLOR_BGR2RGB)

    output = target_image.copy()
    output[boundary_ring] = proposal[boundary_ring]
    output[inserted_mask] = baseline_image[inserted_mask]

    core_difference = np.abs(
        output.astype(np.int16) - baseline_image.astype(np.int16)
    ).max(axis=2)
    target_difference = np.abs(
        output.astype(np.int16) - target_image.astype(np.int16)
    ).max(axis=2)
    target_mae = np.abs(
        output.astype(np.int16) - target_image.astype(np.int16)
    ).mean(axis=2)
    core_changed = int((core_difference[inserted_mask] > 0).sum())
    outside_changed = int((target_difference[~dilated_mask] > 0).sum())
    if core_changed:
        raise RuntimeError("boundary blend changed frozen defect-core pixels")
    if outside_changed:
        raise RuntimeError("boundary blend changed pixels beyond the allowed ring")

    ring_changed = int((target_difference[boundary_ring] >= changed_threshold).sum())
    inner_before = _cross_boundary_jump(baseline_image, inserted_mask)
    inner_after = _cross_boundary_jump(output, inserted_mask)
    outer_target = _cross_boundary_jump(target_image, dilated_mask)
    outer_after = _cross_boundary_jump(output, dilated_mask)
    score_before = inner_before
    score_after = inner_after + max(0.0, outer_after - outer_target)
    reduction = (score_before - score_after) / score_before if score_before > 0 else 0.0

    return BoundaryBlendResult(
        image=output,
        inserted_mask=inserted_mask.copy(),
        boundary_ring=boundary_ring,
        dilated_mask=dilated_mask,
        metrics=BoundaryBlendMetrics(
            ring_radius_pixels=ring_radius_pixels,
            clone_mode=clone_mode,
            inserted_mask_pixels=int(inserted_mask.sum()),
            boundary_ring_pixels=int(boundary_ring.sum()),
            core_changed_vs_baseline_pixels=core_changed,
            outside_dilated_mask_changed_vs_target_pixels=outside_changed,
            ring_changed_vs_target_pixels=ring_changed,
            ring_changed_vs_target_fraction=float(ring_changed / boundary_ring.sum()),
            ring_mean_absolute_change_vs_target=float(target_mae[boundary_ring].mean()),
            inner_boundary_jump_before=inner_before,
            inner_boundary_jump_after=inner_after,
            outer_boundary_jump_target=outer_target,
            outer_boundary_jump_after=outer_after,
            seam_score_before=score_before,
            seam_score_after=score_after,
            seam_score_reduction_fraction=float(reduction),
        ),
    )


def gaussian_feather_boundary_ring(
    *,
    baseline_image: np.ndarray,
    target_image: np.ndarray,
    inserted_mask: np.ndarray,
    ring_radius_pixels: int = 4,
    sigma: float | None = None,
    changed_threshold: int = 3,
) -> BoundaryBlendResult:
    """Feather an abrupt copy boundary outward without altering its core.

    This is the deterministic fallback for the case where ring-only Poisson is
    a no-op.  A Gaussian proposal carries a small amount of the real defect's
    edge appearance into the exterior ring; distance weighting brings the
    proposal smoothly back to the untouched target at the ring's outer edge.
    It can reduce an abrupt cut edge, but may create a halo, so human review is
    still mandatory.
    """

    if baseline_image.shape != target_image.shape or baseline_image.ndim != 3:
        raise ValueError("baseline and target must be same-shaped RGB images")
    if baseline_image.shape[2] != 3:
        raise ValueError("baseline and target must have three channels")
    if inserted_mask.shape != target_image.shape[:2]:
        raise ValueError("inserted mask dimensions do not match images")
    if inserted_mask.dtype != np.bool_:
        inserted_mask = inserted_mask.astype(bool)
    if not np.any(inserted_mask):
        raise ValueError("inserted mask is empty")
    if not 1 <= ring_radius_pixels <= 32:
        raise ValueError("ring radius must be in [1, 32]")
    if changed_threshold < 0:
        raise ValueError("changed threshold must be non-negative")
    if sigma is None:
        sigma = max(0.8, ring_radius_pixels / 2)
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    kernel_size = ring_radius_pixels * 2 + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    dilated_mask = cv2.dilate(inserted_mask.astype(np.uint8), kernel).astype(bool)
    boundary_ring = dilated_mask & ~inserted_mask
    if not np.any(boundary_ring):
        raise ValueError("boundary ring is empty")

    blurred = cv2.GaussianBlur(
        baseline_image,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_REFLECT101,
    ).astype(np.float32)
    distance = cv2.distanceTransform((~inserted_mask).astype(np.uint8), cv2.DIST_L2, 5)
    alpha = np.clip(
        (ring_radius_pixels + 1.0 - distance) / (ring_radius_pixels + 1.0),
        0.0,
        1.0,
    )
    alpha *= boundary_ring.astype(np.float32)
    proposal = np.rint(
        target_image.astype(np.float32) * (1.0 - alpha[..., None])
        + blurred * alpha[..., None]
    ).astype(np.uint8)

    output = target_image.copy()
    output[boundary_ring] = proposal[boundary_ring]
    output[inserted_mask] = baseline_image[inserted_mask]

    core_difference = np.abs(
        output.astype(np.int16) - baseline_image.astype(np.int16)
    ).max(axis=2)
    target_difference = np.abs(
        output.astype(np.int16) - target_image.astype(np.int16)
    ).max(axis=2)
    target_mae = np.abs(
        output.astype(np.int16) - target_image.astype(np.int16)
    ).mean(axis=2)
    core_changed = int((core_difference[inserted_mask] > 0).sum())
    outside_changed = int((target_difference[~dilated_mask] > 0).sum())
    if core_changed:
        raise RuntimeError("boundary feather changed frozen defect-core pixels")
    if outside_changed:
        raise RuntimeError("boundary feather changed pixels beyond the allowed ring")

    ring_changed = int((target_difference[boundary_ring] >= changed_threshold).sum())
    inner_before = _cross_boundary_jump(baseline_image, inserted_mask)
    inner_after = _cross_boundary_jump(output, inserted_mask)
    outer_target = _cross_boundary_jump(target_image, dilated_mask)
    outer_after = _cross_boundary_jump(output, dilated_mask)
    score_before = inner_before
    score_after = inner_after + max(0.0, outer_after - outer_target)
    reduction = (score_before - score_after) / score_before if score_before > 0 else 0.0

    return BoundaryBlendResult(
        image=output,
        inserted_mask=inserted_mask.copy(),
        boundary_ring=boundary_ring,
        dilated_mask=dilated_mask,
        metrics=BoundaryBlendMetrics(
            ring_radius_pixels=ring_radius_pixels,
            clone_mode=f"gaussian_sigma_{sigma:g}",
            inserted_mask_pixels=int(inserted_mask.sum()),
            boundary_ring_pixels=int(boundary_ring.sum()),
            core_changed_vs_baseline_pixels=core_changed,
            outside_dilated_mask_changed_vs_target_pixels=outside_changed,
            ring_changed_vs_target_pixels=ring_changed,
            ring_changed_vs_target_fraction=float(ring_changed / boundary_ring.sum()),
            ring_mean_absolute_change_vs_target=float(target_mae[boundary_ring].mean()),
            inner_boundary_jump_before=inner_before,
            inner_boundary_jump_after=inner_after,
            outer_boundary_jump_target=outer_target,
            outer_boundary_jump_after=outer_after,
            seam_score_before=score_before,
            seam_score_after=score_after,
            seam_score_reduction_fraction=float(reduction),
        ),
    )


def copy_real_defect(
    *,
    source_image: np.ndarray,
    source_mask: np.ndarray,
    target_image: np.ndarray,
    target_boxes: Sequence[YoloBox],
    class_id: int,
    rng: np.random.Generator,
    scale: float,
    rotation_degrees: int = 0,
    horizontal_flip: bool = False,
    vertical_flip: bool = False,
    crop_margin: int = 5,
    min_component_pixels: int = 32,
    max_overlap_fraction: float = 0.05,
    placement_attempts: int = 512,
    feather_sigma: float = 0.8,
) -> PatchCopyResult:
    """Copy one real masked component into a same-class training image.

    Pixels outside the inserted mask remain byte-identical.  The source
    background ring is used only to estimate a per-channel brightness shift;
    it is never pasted into the target.
    """

    if source_image.ndim != 3 or source_image.shape[2] != 3:
        raise ValueError("source image must be RGB")
    if target_image.ndim != 3 or target_image.shape[2] != 3:
        raise ValueError("target image must be RGB")
    if source_mask.shape != source_image.shape[:2]:
        raise ValueError("source mask dimensions do not match source image")
    patch, patch_mask = _crop_component(
        source_image, source_mask, margin=crop_margin, min_pixels=min_component_pixels
    )
    patch, patch_mask = _transform_patch(
        patch,
        patch_mask,
        scale=scale,
        rotation_degrees=rotation_degrees,
        horizontal_flip=horizontal_flip,
        vertical_flip=vertical_flip,
    )
    y, x, overlap = _find_placement(
        patch_mask,
        target_boxes,
        target_image.shape[:2],
        rng,
        max_overlap_fraction=max_overlap_fraction,
        attempts=placement_attempts,
    )

    patch_height, patch_width = patch_mask.shape
    target_region = target_image[y : y + patch_height, x : x + patch_width]
    ring_kernel = np.ones((5, 5), dtype=np.uint8)
    patch_ring = cv2.dilate(patch_mask.astype(np.uint8), ring_kernel).astype(bool) & ~patch_mask
    target_ring = patch_ring
    if np.any(patch_ring):
        source_mean = patch[patch_ring].mean(axis=0)
        target_mean = target_region[target_ring].mean(axis=0)
        raw_shift = target_mean - source_mean
        shift = np.clip(raw_shift, -48.0, 48.0)
    else:
        raw_shift = np.zeros(3, dtype=np.float64)
        shift = np.zeros(3, dtype=np.float64)
    corrected = np.clip(patch.astype(np.float32) + shift, 0, 255)

    alpha = cv2.GaussianBlur(
        patch_mask.astype(np.float32), (0, 0), sigmaX=feather_sigma, sigmaY=feather_sigma
    )
    alpha = np.clip(alpha, 0.0, 1.0) * patch_mask.astype(np.float32)
    alpha3 = alpha[..., None]
    blended = np.rint(target_region * (1.0 - alpha3) + corrected * alpha3).astype(np.uint8)

    output = target_image.copy()
    output_region = output[y : y + patch_height, x : x + patch_width]
    output_region[patch_mask] = blended[patch_mask]
    inserted_mask = np.zeros(target_image.shape[:2], dtype=bool)
    inserted_mask[y : y + patch_height, x : x + patch_width] = patch_mask

    delta = np.abs(output.astype(np.int16) - target_image.astype(np.int16)).max(axis=2)
    changed = delta >= 3
    outside_changed = int(np.logical_and(changed, ~inserted_mask).sum())
    if outside_changed:
        raise RuntimeError("copy-paste changed pixels outside the inserted mask")
    inserted_pixels = int(inserted_mask.sum())
    eroded = cv2.erode(patch_mask.astype(np.uint8), np.ones((3, 3), dtype=np.uint8)).astype(bool)
    inner_boundary = patch_mask & ~eroded
    boundary_delta = np.abs(
        output[y : y + patch_height, x : x + patch_width].astype(np.int16)
        - target_region.astype(np.int16)
    ).max(axis=2)
    new_box = _mask_to_yolo_box(inserted_mask, class_id)
    return PatchCopyResult(
        image=output,
        inserted_mask=inserted_mask,
        inserted_box=new_box,
        metrics=PatchCopyMetrics(
            inserted_mask_pixels=inserted_pixels,
            inserted_mask_fraction=float(inserted_pixels / inserted_mask.size),
            outside_mask_changed_pixels=outside_changed,
            inside_mask_changed_fraction=float(changed[inserted_mask].mean()),
            mean_absolute_change_inside=float(delta[inserted_mask].mean()),
            existing_box_overlap_fraction=overlap,
            boundary_mean_absolute_change=(
                float(boundary_delta[inner_boundary].mean()) if np.any(inner_boundary) else 0.0
            ),
            raw_background_shift=tuple(float(value) for value in raw_shift),
            source_background_shift=tuple(float(value) for value in shift),
            background_shift_clipped=bool(np.any(np.abs(raw_shift) > 48.0)),
            scale=scale,
            rotation_degrees=rotation_degrees,
            horizontal_flip=horizontal_flip,
            vertical_flip=vertical_flip,
        ),
    )


__all__ = [
    "BoundaryBlendMetrics",
    "BoundaryBlendResult",
    "PatchCopyMetrics",
    "PatchCopyResult",
    "YoloBox",
    "copy_real_defect",
    "gaussian_feather_boundary_ring",
    "poisson_blend_boundary_ring",
    "read_binary_mask",
    "read_rgb",
    "read_yolo_boxes",
]
