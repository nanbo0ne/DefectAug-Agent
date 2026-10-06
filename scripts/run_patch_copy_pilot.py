"""Run a small train-only real-defect Patch/Copy-Paste pilot.

The pilot is deliberately independent of Qwen/SDXL/GAN.  It tests whether
real masks can provide immediately reliable labels and a reproducible
generator-independent baseline before investing in a learned generator.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from defectaug.patch_copy import copy_real_defect, read_binary_mask, read_rgb, read_yolo_boxes


PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_train_paths() -> list[Path]:
    manifest = PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt"
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    paths = [Path(raw.strip()) for raw in manifest.read_text(encoding="utf-8").splitlines() if raw.strip()]
    if len(paths) != 100:
        raise RuntimeError(f"expected the frozen 100-image train manifest, got {len(paths)}")
    if any("test" in str(path).lower() or "valid" in str(path).lower() for path in paths):
        raise RuntimeError("train manifest unexpectedly references validation/test paths")
    return paths


def label_path(image_path: Path) -> Path:
    return PROJECT / "data" / "canonical" / "labels" / "all" / f"{image_path.stem}.txt"


def mask_path(image_path: Path) -> Path:
    return PROJECT / "data" / "canonical" / "masks" / "all" / f"{image_path.stem}.png"


def write_png(path: Path, image: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(path, format="PNG", optimize=False)
    return sha256_file(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--count-per-class", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260823)
    parser.add_argument("--scale", type=float, default=0.75)
    parser.add_argument("--output", default="data/synthetic/patch_copy_pilot_v1")
    args = parser.parse_args()
    if args.count_per_class <= 0:
        raise ValueError("count-per-class must be positive")

    train_paths = resolve_train_paths()
    by_class: dict[str, list[Path]] = {name: [] for name in CLASSES}
    for path in train_paths:
        class_name = path.stem.rsplit("_", 1)[0]
        if class_name in by_class:
            by_class[class_name].append(path)
    if any(len(value) < 2 for value in by_class.values()):
        raise RuntimeError("each class needs at least two train sources")

    output_root = (PROJECT / args.output).resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to reuse non-empty pilot directory: {output_root}")
    image_dir = output_root / "images"
    mask_dir = output_root / "masks"
    label_dir = output_root / "labels"
    records: list[dict[str, object]] = []
    root_rng = np.random.default_rng(args.seed)

    for class_name in CLASSES:
        sources = by_class[class_name]
        for index in range(args.count_per_class):
            source = sources[index % len(sources)]
            target = sources[(index + 1) % len(sources)]
            source_image = read_rgb(source)
            source_mask = read_binary_mask(mask_path(source))
            target_image = read_rgb(target)
            target_boxes = read_yolo_boxes(label_path(target))
            rng = np.random.default_rng(int(root_rng.integers(0, 2**32 - 1)))
            result = copy_real_defect(
                source_image=source_image,
                source_mask=source_mask,
                target_image=target_image,
                target_boxes=target_boxes,
                class_id=CLASS_ID[class_name],
                rng=rng,
                scale=args.scale,
                rotation_degrees=(index % 4) * 90,
                horizontal_flip=(index % 2 == 1),
                vertical_flip=False,
            )
            stem = f"{class_name}_copy_{index:03d}"
            image_out = image_dir / f"{stem}.png"
            mask_out = mask_dir / f"{stem}.png"
            label_out = label_dir / f"{stem}.txt"
            image_sha = write_png(image_out, result.image)
            mask_sha = write_png(mask_out, result.inserted_mask.astype(np.uint8) * 255)
            label_lines = [box.line() for box in target_boxes] + [result.inserted_box.line()]
            label_out.parent.mkdir(parents=True, exist_ok=True)
            label_out.write_text("\n".join(label_lines) + "\n", encoding="utf-8")
            records.append(
                {
                    "id": stem,
                    "class": class_name,
                    "source": str(source),
                    "target": str(target),
                    "image": str(image_out),
                    "mask": str(mask_out),
                    "label": str(label_out),
                    "image_sha256": image_sha,
                    "mask_sha256": mask_sha,
                    "label_sha256": sha256_file(label_out),
                    "seed": args.seed,
                    "automatic_qc": {
                        "hard_invariants_pass": bool(
                            result.metrics.outside_mask_changed_pixels == 0
                            and result.metrics.existing_box_overlap_fraction <= 0.05
                            and result.metrics.inside_mask_changed_fraction >= 0.50
                        ),
                        "background_shift_clipped": result.metrics.background_shift_clipped,
                        "semantic_and_seam_review": "pending_human_review",
                    },
                    "quality": result.metrics.as_dict(),
                }
            )

    jsonl = output_root / "records.jsonl"
    jsonl.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + "\n", encoding="utf-8")
    csv_path = output_root / "manifest.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["id", "class", "source", "target", "image", "mask", "label", "image_sha256", "mask_sha256", "label_sha256"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in records)
    combined_manifest = output_root / "train_real_plus_patch_copy.txt"
    combined_manifest.write_text(
        "\n".join(str(path) for path in train_paths)
        + "\n"
        + "\n".join(str(row["image"]) for row in records)
        + "\n",
        encoding="utf-8",
    )
    dataset_yaml = output_root / "dataset_real_plus_patch_copy.yaml"
    dataset_yaml.write_text(
        "\n".join(
            (
                f"path: {PROJECT.as_posix()}",
                f"train: {combined_manifest.as_posix()}",
                f"val: {(PROJECT / 'data/splits/v1/val.txt').as_posix()}",
                "names:",
                "  0: inclusion",
                "  1: patches",
                "  2: scratches",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    summary = {
        "pilot": "real_train_patch_copy_v1",
        "source_manifest": str(PROJECT / "data/splits/v1/train_fewshot.txt"),
        "count": len(records),
        "count_by_class": {name: sum(row["class"] == name for row in records) for name in CLASSES},
        "scale": args.scale,
        "seed": args.seed,
        "test_accessed": False,
        "validation_accessed": False,
        "combined_train_rows": len(train_paths) + len(records),
        "combined_train_manifest": str(combined_manifest),
        "combined_train_manifest_sha256": sha256_file(combined_manifest),
        "dataset_yaml": str(dataset_yaml),
        "dataset_yaml_sha256": sha256_file(dataset_yaml),
        "automatic_qc": {
            "outside_mask_changed_pixels_max": 0,
            "existing_box_overlap_fraction_max": 0.05,
            "inside_mask_changed_fraction_min": 0.50,
            "brightness_shift_clip": 48.0,
            "semantic_and_seam_decision": "human_review_required",
        },
        "records_sha256": sha256_file(jsonl),
    }
    (output_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
