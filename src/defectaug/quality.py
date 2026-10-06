"""Generator-independent quality checks for synthetic candidates."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class QualityMetrics:
    """Legacy metrics retained for the early AIO pilot."""

    outside_changed_fraction: float
    inside_changed_fraction: float
    changed_region_mask_iou: float
    mean_absolute_change_inside: float

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class CandidateQualityReport:
    """Non-overlapping measurements shared by the formal generator runners."""

    pixel_threshold: int
    boundary_ring_width: int
    mask_pixels: int
    mask_fraction: float
    raw_outside_changed_pixels: int
    raw_outside_changed_fraction: float
    composite_outside_changed_pixels: int
    composite_outside_changed_fraction: float
    composite_hard_invariant: bool
    inside_changed_pixels: int
    inside_changed_fraction: float
    mean_absolute_change_inside: float
    boundary_inner_pixels: int
    boundary_outer_pixels: int
    boundary_inner_mean_absolute_change: float
    boundary_outer_mean_absolute_change: float
    boundary_ring_seam_score: float
    yolo_box_pixels: int | None
    mask_within_yolo_box_fraction: float | None
    effective_changed_pixels: int
    effective_change_within_yolo_box_fraction: float | None
    effective_change_outside_yolo_box_pixels: int | None

    def as_dict(self) -> dict[str, int | float | bool | None]:
        return asdict(self)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def normalize_mask(mask: np.ndarray, threshold: int = 127) -> np.ndarray:
    if not 0 <= threshold <= 255:
        raise ValueError("mask threshold must be in [0, 255]")
    if mask.ndim == 3:
        mask = mask.max(axis=2)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2-D or 3-channel; got shape {mask.shape}")
    return mask >= threshold


def hard_composite(
    original: np.ndarray,
    generated: np.ndarray,
    mask: np.ndarray,
    mask_threshold: int = 127,
) -> np.ndarray:
    if original.shape != generated.shape:
        raise ValueError(f"shape mismatch: {original.shape} != {generated.shape}")
    binary = normalize_mask(mask, mask_threshold)
    if binary.shape != original.shape[:2]:
        raise ValueError("mask dimensions do not match image")
    output = original.copy()
    output[binary] = generated[binary]
    return output


def _delta(original: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if original.shape != candidate.shape:
        raise ValueError(f"shape mismatch: {original.shape} != {candidate.shape}")
    delta = np.abs(original.astype(np.int16) - candidate.astype(np.int16))
    return delta.max(axis=2) if delta.ndim == 3 else delta


def _binary_morphology(mask: np.ndarray, radius: int, *, dilate: bool) -> np.ndarray:
    """Square-kernel boolean morphology without an OpenCV runtime dependency."""

    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.zeros_like(mask, dtype=bool) if dilate else np.ones_like(mask, dtype=bool)
    height, width = mask.shape
    for y_offset in range(radius * 2 + 1):
        for x_offset in range(radius * 2 + 1):
            view = padded[y_offset : y_offset + height, x_offset : x_offset + width]
            if dilate:
                result |= view
            else:
                result &= view
    return result


def candidate_metrics(
    original: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
    pixel_threshold: int = 8,
) -> QualityMetrics:
    """Legacy metric set; formal runners use :func:`quality_report`."""

    if pixel_threshold <= 0:
        raise ValueError("pixel threshold must be positive")
    binary = normalize_mask(mask)
    if binary.shape != original.shape[:2]:
        raise ValueError("mask dimensions do not match image")
    if not np.any(binary):
        raise ValueError("mask is empty")
    delta = _delta(original, candidate)
    changed = delta >= pixel_threshold
    outside = ~binary
    intersection = int(np.logical_and(changed, binary).sum())
    union = int(np.logical_or(changed, binary).sum())
    return QualityMetrics(
        outside_changed_fraction=float(changed[outside].mean()) if np.any(outside) else 0.0,
        inside_changed_fraction=float(changed[binary].mean()),
        changed_region_mask_iou=intersection / union if union else 0.0,
        mean_absolute_change_inside=float(delta[binary].mean()),
    )


def load_yolo_box_mask(label_path: Path, height: int, width: int) -> np.ndarray:
    """Rasterize normalized YOLO boxes into a boolean constraint mask."""

    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    if not label_path.is_file():
        raise FileNotFoundError(f"YOLO label does not exist: {label_path}")
    result = np.zeros((height, width), dtype=bool)
    parsed = 0
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"{label_path}:{line_number} is not a 5-field YOLO box")
        try:
            class_id = int(fields[0])
            center_x, center_y, box_width, box_height = map(float, fields[1:])
        except ValueError as exc:
            raise ValueError(f"{label_path}:{line_number} has invalid numeric fields") from exc
        if class_id < 0:
            raise ValueError(f"{label_path}:{line_number} has a negative class id")
        if not (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        ):
            raise ValueError(f"{label_path}:{line_number} has out-of-range coordinates")
        x0 = max(0, int(np.floor((center_x - box_width / 2) * width)))
        y0 = max(0, int(np.floor((center_y - box_height / 2) * height)))
        x1 = min(width, int(np.ceil((center_x + box_width / 2) * width)))
        y1 = min(height, int(np.ceil((center_y + box_height / 2) * height)))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"{label_path}:{line_number} rasterizes to an empty box")
        result[y0:y1, x0:x1] = True
        parsed += 1
    if parsed == 0:
        raise ValueError(f"YOLO label is empty: {label_path}")
    return result


def quality_report(
    original: np.ndarray,
    raw_candidate: np.ndarray,
    composite: np.ndarray,
    mask: np.ndarray,
    *,
    yolo_box_mask: np.ndarray | None,
    pixel_threshold: int = 8,
    boundary_ring_width: int = 2,
    mask_threshold: int = 127,
) -> CandidateQualityReport:
    """Measure raw drift, hard-composite integrity, coverage, seam risk and labels."""

    if pixel_threshold <= 0:
        raise ValueError("pixel threshold must be positive")
    if boundary_ring_width <= 0:
        raise ValueError("boundary ring width must be positive")
    if original.shape != raw_candidate.shape or original.shape != composite.shape:
        raise ValueError("original, raw candidate and composite must have identical shapes")
    binary = normalize_mask(mask, mask_threshold)
    if binary.shape != original.shape[:2] or not np.any(binary):
        raise ValueError("mask must be non-empty and match the image")
    if yolo_box_mask is not None and yolo_box_mask.shape != binary.shape:
        raise ValueError("YOLO box constraint dimensions do not match image")

    raw_delta = _delta(original, raw_candidate)
    composite_delta = _delta(original, composite)
    raw_changed = raw_delta >= pixel_threshold
    composite_changed = composite_delta >= pixel_threshold
    outside = ~binary
    mask_pixels = int(binary.sum())
    outside_pixels = int(outside.sum())
    raw_outside_changed = int(np.logical_and(raw_changed, outside).sum())
    composite_outside_changed = int(np.logical_and(composite_changed, outside).sum())
    effective_changed = np.logical_and(composite_changed, binary)
    effective_changed_pixels = int(effective_changed.sum())

    eroded = _binary_morphology(binary, boundary_ring_width, dilate=False)
    dilated = _binary_morphology(binary, boundary_ring_width, dilate=True)
    inner_ring = np.logical_and(binary, ~eroded)
    outer_ring = np.logical_and(dilated, ~binary)
    inner_mean = float(composite_delta[inner_ring].mean()) if np.any(inner_ring) else 0.0
    outer_mean = float(composite_delta[outer_ring].mean()) if np.any(outer_ring) else 0.0

    yolo_pixels: int | None = None
    mask_within_yolo: float | None = None
    changed_within_yolo: float | None = None
    changed_outside_yolo: int | None = None
    if yolo_box_mask is not None:
        boxes = yolo_box_mask.astype(bool)
        yolo_pixels = int(boxes.sum())
        mask_within_yolo = float(np.logical_and(binary, boxes).sum() / mask_pixels)
        changed_outside_yolo = int(np.logical_and(effective_changed, ~boxes).sum())
        changed_within_yolo = (
            float(np.logical_and(effective_changed, boxes).sum() / effective_changed_pixels)
            if effective_changed_pixels
            else 0.0
        )

    return CandidateQualityReport(
        pixel_threshold=pixel_threshold,
        boundary_ring_width=boundary_ring_width,
        mask_pixels=mask_pixels,
        mask_fraction=float(mask_pixels / binary.size),
        raw_outside_changed_pixels=raw_outside_changed,
        raw_outside_changed_fraction=float(raw_outside_changed / outside_pixels) if outside_pixels else 0.0,
        composite_outside_changed_pixels=composite_outside_changed,
        composite_outside_changed_fraction=float(composite_outside_changed / outside_pixels) if outside_pixels else 0.0,
        composite_hard_invariant=composite_outside_changed == 0,
        inside_changed_pixels=effective_changed_pixels,
        inside_changed_fraction=float(effective_changed_pixels / mask_pixels),
        mean_absolute_change_inside=float(composite_delta[binary].mean()),
        boundary_inner_pixels=int(inner_ring.sum()),
        boundary_outer_pixels=int(outer_ring.sum()),
        boundary_inner_mean_absolute_change=inner_mean,
        boundary_outer_mean_absolute_change=outer_mean,
        boundary_ring_seam_score=float(abs(inner_mean - outer_mean) / 255.0),
        yolo_box_pixels=yolo_pixels,
        mask_within_yolo_box_fraction=mask_within_yolo,
        effective_changed_pixels=effective_changed_pixels,
        effective_change_within_yolo_box_fraction=changed_within_yolo,
        effective_change_outside_yolo_box_pixels=changed_outside_yolo,
    )


def write_png(path: Path, image: np.ndarray, *, overwrite: bool = False) -> str:
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing PNG: {path}")
    if path.suffix.lower() != ".png":
        raise ValueError(f"exact pixel verification requires PNG output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if image.dtype != np.uint8 or image.ndim not in (2, 3):
        raise ValueError("PNG image must be a 2-D/3-D uint8 numpy array")
    Image.fromarray(image).save(path, format="PNG", optimize=False)
    return sha256_file(path)


def write_binary_mask(
    path: Path,
    mask: np.ndarray,
    *,
    threshold: int = 127,
    overwrite: bool = False,
) -> tuple[np.ndarray, str]:
    binary_u8 = normalize_mask(mask, threshold).astype(np.uint8) * 255
    digest = write_png(path, binary_u8, overwrite=overwrite)
    with Image.open(path) as stream:
        reloaded = np.asarray(stream.convert("L"), dtype=np.uint8)
    if reloaded is None or not np.array_equal(binary_u8, reloaded):
        raise RuntimeError(f"binary mask changed after PNG round-trip: {path}")
    return binary_u8, digest


def save_and_verify_candidate(
    *,
    original: np.ndarray,
    raw_candidate: np.ndarray,
    mask: np.ndarray,
    composite_path: Path,
    final_mask_path: Path,
    label_path: Path,
    pixel_threshold: int = 8,
    boundary_ring_width: int = 2,
    mask_threshold: int = 127,
    overwrite: bool = False,
) -> dict[str, object]:
    """Hard-composite, persist, reload and produce the shared formal QC record."""

    if original.shape != raw_candidate.shape:
        raise ValueError("raw candidate must be resized to the source dimensions")
    if composite_path.resolve() == final_mask_path.resolve():
        raise ValueError("composite and final mask outputs must be distinct")
    if not overwrite:
        existing = [str(path) for path in (composite_path, final_mask_path) if path.exists()]
        if existing:
            raise FileExistsError(f"refusing to overwrite candidate artifacts: {existing}")
    binary_u8, mask_sha256 = write_binary_mask(
        final_mask_path, mask, threshold=mask_threshold, overwrite=overwrite
    )
    composite = hard_composite(original, raw_candidate, binary_u8, mask_threshold)
    composite_sha256 = write_png(composite_path, composite, overwrite=overwrite)
    with Image.open(composite_path) as stream:
        reloaded = np.asarray(stream.convert("RGB"), dtype=np.uint8)
    if reloaded is None or reloaded.shape != original.shape:
        raise RuntimeError(f"composite is unreadable after PNG round-trip: {composite_path}")
    boxes = load_yolo_box_mask(label_path, original.shape[0], original.shape[1])
    report = quality_report(
        original,
        raw_candidate,
        reloaded,
        binary_u8,
        yolo_box_mask=boxes,
        pixel_threshold=pixel_threshold,
        boundary_ring_width=boundary_ring_width,
        mask_threshold=mask_threshold,
    )
    if not report.composite_hard_invariant:
        raise RuntimeError(
            f"hard-composite invariant failed: {report.composite_outside_changed_pixels} "
            "outside-mask pixels changed"
        )
    return {
        "composite_sha256": composite_sha256,
        "final_binary_mask_sha256": mask_sha256,
        "label_sha256": sha256_file(label_path),
        "quality": report.as_dict(),
    }
