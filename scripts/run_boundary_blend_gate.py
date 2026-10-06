"""Run a train-only A/B gate for boundary-ring Poisson blending.

A is the frozen nine-image Patch/Copy-Paste pilot.  B keeps every inserted
defect-core pixel and the inherited YOLO label unchanged, while allowing only a
3--5 px exterior ring to use OpenCV Poisson blending.  The script never reads
validation or locked-test image content.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import median

import numpy as np
from PIL import Image, ImageDraw

from defectaug.patch_copy import (
    gaussian_feather_boundary_ring,
    poisson_blend_boundary_ring,
    read_binary_mask,
    read_rgb,
)


PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = PROJECT / "data" / "synthetic" / "patch_copy_pilot_v2"
DEFAULT_OUTPUT = PROJECT / "data" / "synthetic" / "boundary_blend_gate_v2"
DEFAULT_PREVIEW = PROJECT / "reports" / "boundary_blend_gate_v2.png"
DEFAULT_VARIANTS_PREVIEW = PROJECT / "reports" / "boundary_blend_variants_v2.png"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def _frozen_train_paths() -> set[Path]:
    manifest = PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt"
    paths = {
        _resolve_project_path(raw.strip())
        for raw in manifest.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    }
    if len(paths) != 100:
        raise RuntimeError(f"expected 100 frozen training paths, got {len(paths)}")
    for path in paths:
        lowered = str(path).lower()
        if "test" in lowered or "valid" in lowered:
            raise RuntimeError("frozen training manifest references validation/test content")
    return paths


def _load_and_verify_baseline(root: Path) -> list[dict[str, object]]:
    records_path = root / "records.jsonl"
    records = [
        json.loads(line)
        for line in records_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(records) != 9 or len({str(row["id"]) for row in records}) != 9:
        raise RuntimeError("A/B gate requires the frozen nine unique baseline tasks")
    allowed = _frozen_train_paths()
    for row in records:
        source = _resolve_project_path(str(row["source"]))
        target = _resolve_project_path(str(row["target"]))
        if source not in allowed or target not in allowed:
            raise RuntimeError(f"{row['id']} source/target is outside frozen train_fewshot")
        for path_key, hash_key in (
            ("image", "image_sha256"),
            ("mask", "mask_sha256"),
            ("label", "label_sha256"),
        ):
            path = _resolve_project_path(str(row[path_key]))
            if sha256_file(path) != row[hash_key]:
                raise RuntimeError(f"{row['id']} frozen {path_key} hash mismatch")
    return records


def _write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", optimize=False)
    return sha256_file(path)


def _variant_stats(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["clone_mode"]), int(row["ring_radius_pixels"]))].append(row)
    stats: list[dict[str, object]] = []
    for (mode, radius), rows in sorted(grouped.items()):
        reductions = [float(row["quality"]["seam_score_reduction_fraction"]) for row in rows]
        ring_mae = [float(row["quality"]["ring_mean_absolute_change_vs_target"]) for row in rows]
        inner_before = [float(row["quality"]["inner_boundary_jump_before"]) for row in rows]
        inner_after = [float(row["quality"]["inner_boundary_jump_after"]) for row in rows]
        stats.append(
            {
                "clone_mode": mode,
                "ring_radius_pixels": radius,
                "count": len(rows),
                "positive_seam_reduction_count": sum(value > 0 for value in reductions),
                "median_seam_score_reduction_fraction": median(reductions),
                "mean_seam_score_reduction_fraction": sum(reductions) / len(reductions),
                "median_ring_mae_vs_target": median(ring_mae),
                "median_inner_boundary_jump_before": median(inner_before),
                "median_inner_boundary_jump_after": median(inner_after),
                "hard_invariants_pass": all(
                    int(row["quality"]["core_changed_vs_baseline_pixels"]) == 0
                    and int(row["quality"]["outside_dilated_mask_changed_vs_target_pixels"]) == 0
                    for row in rows
                ),
            }
        )
    return stats


def _select_engineering_variant(stats: list[dict[str, object]]) -> dict[str, object]:
    """Select a preview variant; this is calibration, not a paper result."""

    return max(
        stats,
        key=lambda row: (
            int(row["positive_seam_reduction_count"]),
            float(row["median_seam_score_reduction_fraction"]),
            -float(row["median_ring_mae_vs_target"]),
            -int(row["ring_radius_pixels"]),
            str(row["clone_mode"]) == "normal",
        ),
    )


def _captioned(image: np.ndarray, title: str) -> Image.Image:
    tile = Image.fromarray(image).convert("RGB")
    canvas = Image.new("RGB", (tile.width, tile.height + 22), "white")
    canvas.paste(tile, (0, 22))
    ImageDraw.Draw(canvas).text((4, 4), title, fill="black")
    return canvas


def _save_selected_preview(
    baseline_rows: list[dict[str, object]],
    variant_rows: list[dict[str, object]],
    destination: Path,
) -> None:
    selected = {str(row["id"]): row for row in variant_rows}
    rows: list[Image.Image] = []
    for baseline in baseline_rows:
        row = selected[str(baseline["id"])]
        target = read_rgb(_resolve_project_path(str(baseline["target"])))
        a_image = read_rgb(_resolve_project_path(str(baseline["image"])))
        b_image = read_rgb(_resolve_project_path(str(row["image"])))
        difference = np.clip(
            np.abs(b_image.astype(np.int16) - a_image.astype(np.int16)) * 6,
            0,
            255,
        ).astype(np.uint8)
        ring = read_binary_mask(_resolve_project_path(str(row["ring_mask"])))
        mask = read_binary_mask(_resolve_project_path(str(baseline["mask"])))
        overlay = target.copy()
        overlay[ring] = np.array([255, 210, 0], dtype=np.uint8)
        overlay[mask] = np.array([230, 40, 40], dtype=np.uint8)
        titles = (
            f"{row['id']} target",
            "A fixed copy",
            f"B {row['clone_mode']} r={row['ring_radius_pixels']}",
            "|B-A| x6",
            "red=core yellow=ring",
        )
        tiles = [_captioned(image, title) for image, title in zip((target, a_image, b_image, difference, overlay), titles)]
        strip = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, 0))
            x += tile.width
        rows.append(strip)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False)


def _save_variants_preview(
    baseline_rows: list[dict[str, object]],
    all_rows: list[dict[str, object]],
    destination: Path,
) -> None:
    variants = sorted(
        {(str(row["clone_mode"]), int(row["ring_radius_pixels"])) for row in all_rows},
        key=lambda item: (item[0], item[1]),
    )
    lookup = {
        (str(row["id"]), str(row["clone_mode"]), int(row["ring_radius_pixels"])): row
        for row in all_rows
    }
    rows: list[Image.Image] = []
    for baseline in baseline_rows:
        images = [read_rgb(_resolve_project_path(str(baseline["target"]))), read_rgb(_resolve_project_path(str(baseline["image"])))]
        titles = [f"{baseline['id']} target", "A fixed copy"]
        for mode, radius in variants:
            row = lookup[(str(baseline["id"]), mode, radius)]
            images.append(read_rgb(_resolve_project_path(str(row["image"]))))
            titles.append(f"{mode} r={radius}")
        tiles = [_captioned(image, title) for image, title in zip(images, titles)]
        strip = Image.new("RGB", (sum(tile.width for tile in tiles), tiles[0].height), "white")
        x = 0
        for tile in tiles:
            strip.paste(tile, (x, 0))
            x += tile.width
        rows.append(strip)
    canvas = Image.new("RGB", (rows[0].width, sum(row.height for row in rows)), "white")
    y = 0
    for row in rows:
        canvas.paste(row, (0, y))
        y += row.height
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination, format="PNG", optimize=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--radii", nargs="+", type=int, default=[3, 4, 5])
    parser.add_argument("--clone-modes", nargs="+", default=["normal", "mixed", "gaussian"])
    parser.add_argument("--preview", default=str(DEFAULT_PREVIEW))
    parser.add_argument("--variants-preview", default=str(DEFAULT_VARIANTS_PREVIEW))
    args = parser.parse_args()

    if any(radius not in (3, 4, 5) for radius in args.radii):
        raise ValueError("this frozen gate allows only 3, 4, or 5 px rings")
    if any(mode not in ("normal", "mixed", "gaussian") for mode in args.clone_modes):
        raise ValueError("clone modes must be normal, mixed, or gaussian")
    baseline_root = _resolve_project_path(args.baseline)
    output_root = _resolve_project_path(args.output)
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty gate directory: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    baseline_rows = _load_and_verify_baseline(baseline_root)
    records: list[dict[str, object]] = []

    for baseline in baseline_rows:
        a_image = read_rgb(_resolve_project_path(str(baseline["image"])))
        target = read_rgb(_resolve_project_path(str(baseline["target"])))
        mask = read_binary_mask(_resolve_project_path(str(baseline["mask"])))
        for mode in args.clone_modes:
            for radius in args.radii:
                if mode == "gaussian":
                    result = gaussian_feather_boundary_ring(
                        baseline_image=a_image,
                        target_image=target,
                        inserted_mask=mask,
                        ring_radius_pixels=radius,
                    )
                else:
                    result = poisson_blend_boundary_ring(
                        baseline_image=a_image,
                        target_image=target,
                        inserted_mask=mask,
                        ring_radius_pixels=radius,
                        clone_mode=mode,
                    )
                variant = f"{mode}_r{radius}"
                image_out = output_root / variant / "images" / f"{baseline['id']}.png"
                ring_out = output_root / variant / "rings" / f"{baseline['id']}.png"
                image_hash = _write_png(image_out, result.image)
                ring_hash = _write_png(ring_out, result.boundary_ring.astype(np.uint8) * 255)
                records.append(
                    {
                        "id": baseline["id"],
                        "class": baseline["class"],
                        "method": (
                            "gaussian_boundary_ring_only"
                            if mode == "gaussian"
                            else "poisson_boundary_ring_only"
                        ),
                        "clone_mode": mode,
                        "ring_radius_pixels": radius,
                        "source": baseline["source"],
                        "target": baseline["target"],
                        "baseline_image": baseline["image"],
                        "image": str(image_out),
                        "ring_mask": str(ring_out),
                        "inserted_mask": baseline["mask"],
                        "label": baseline["label"],
                        "seed": baseline["seed"],
                        "scale": baseline["quality"]["scale"],
                        "rotation_degrees": baseline["quality"]["rotation_degrees"],
                        "horizontal_flip": baseline["quality"]["horizontal_flip"],
                        "vertical_flip": baseline["quality"]["vertical_flip"],
                        "baseline_image_sha256": baseline["image_sha256"],
                        "image_sha256": image_hash,
                        "ring_mask_sha256": ring_hash,
                        "inserted_mask_sha256": baseline["mask_sha256"],
                        "label_sha256": baseline["label_sha256"],
                        "quality": result.metrics.as_dict(),
                    }
                )

    records_path = output_root / "records.jsonl"
    records_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n",
        encoding="utf-8",
    )
    stats = _variant_stats(records)
    selected = _select_engineering_variant(stats)
    selected_rows = [
        row
        for row in records
        if row["clone_mode"] == selected["clone_mode"]
        and row["ring_radius_pixels"] == selected["ring_radius_pixels"]
    ]

    csv_path = output_root / "variant_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0].keys()))
        writer.writeheader()
        writer.writerows(stats)

    preview = _resolve_project_path(args.preview)
    variants_preview = _resolve_project_path(args.variants_preview)
    _save_selected_preview(baseline_rows, selected_rows, preview)
    _save_variants_preview(baseline_rows, records, variants_preview)

    objective_gate = bool(
        selected["hard_invariants_pass"]
        and int(selected["positive_seam_reduction_count"]) >= 6
        and float(selected["median_seam_score_reduction_fraction"]) >= 0.05
    )
    summary = {
        "gate": "patch_copy_vs_boundary_ring_blending_v2",
        "purpose": "engineering_quality_calibration_not_paper_effect_result",
        "baseline": str(baseline_root),
        "baseline_records_sha256": sha256_file(baseline_root / "records.jsonl"),
        "task_count": len(baseline_rows),
        "candidate_count": len(records),
        "radii_pixels": sorted(set(args.radii)),
        "clone_modes": list(args.clone_modes),
        "same_source_target_transform_seed": True,
        "defect_core_byte_identical_to_A": True,
        "labels_byte_identical_to_A": True,
        "test_accessed": False,
        "validation_accessed": False,
        "variant_stats": stats,
        "selected_for_preview_only": selected,
        "selection_warning": "selected on the same 9-task engineering calibration; not an unbiased paper result",
        "objective_gate_thresholds": {
            "hard_invariants_all_9": True,
            "positive_seam_reduction_count_min": 6,
            "median_seam_score_reduction_fraction_min": 0.05,
            "human_blind_review_required": True,
        },
        "objective_gate_pass": objective_gate,
        "formal_generation_allowed": False,
        "records_sha256": sha256_file(records_path),
        "preview": str(preview),
        "preview_sha256": sha256_file(preview),
        "variants_preview": str(variants_preview),
        "variants_preview_sha256": sha256_file(variants_preview),
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
