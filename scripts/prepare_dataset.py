"""Build the canonical three-class NEU/SD-saliency dataset and fixed v1 split.

The source archives remain untouched.  Canonical images and masks are hard-linked
when possible, labels are regenerated from the original NEU XML files, and every
split decision is recorded in CSV/JSON manifests.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
NEU = PROJECT / "data" / "raw" / "neu_det" / "NEU-DET"
SD900 = PROJECT / "data" / "external" / "MCITF" / "SD-saliency-900"
CANONICAL = PROJECT / "data" / "canonical"
SPLITS = PROJECT / "data" / "splits" / "v1"
MANIFESTS = PROJECT / "data" / "manifests"
CONFIGS = PROJECT / "configs"

CLASS_NAMES = ["inclusion", "patches", "scratches"]
CLASS_TO_ID = {name: index for index, name in enumerate(CLASS_NAMES)}
PREFIX_TO_CLASS = {"In": "inclusion", "Pa": "patches", "Sc": "scratches"}
CLASS_TO_PREFIX = {value: key for key, value in PREFIX_TO_CLASS.items()}
DEFAULT_FEWSHOT = {"inclusion": 10, "patches": 30, "scratches": 60}
QUARANTINE = {"patches_101", "patches_105"}


@dataclass
class Record:
    stem: str
    primary_class: str
    source_image: str
    source_xml: str
    source_mask: str
    canonical_image: str
    canonical_label: str
    canonical_mask: str
    sha256: str
    object_count: int
    class_set: str
    mask_xml_iou: float | None
    fewshot_eligible: bool
    split: str = "unused"
    excluded_reason: str = ""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_xml(path: Path) -> tuple[int, int, list[tuple[str, list[int]]]]:
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    objects: list[tuple[str, list[int]]] = []
    for node in root.findall("object"):
        box = node.find("bndbox")
        if box is None:
            continue
        objects.append(
            (
                node.findtext("name", ""),
                [
                    int(float(box.findtext("xmin", "0"))),
                    int(float(box.findtext("ymin", "0"))),
                    int(float(box.findtext("xmax", "0"))),
                    int(float(box.findtext("ymax", "0"))),
                ],
            )
        )
    return width, height, objects


def mask_bbox(mask_path: Path) -> list[int] | None:
    pixels = np.asarray(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.nonzero(pixels)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def find_neu_files() -> tuple[dict[str, Path], dict[str, Path]]:
    images = {
        path.stem: path
        for directory in (NEU / "train" / "images", NEU / "valid" / "images")
        for path in directory.glob("*.jpg")
    }
    xmls = {path.stem: path for path in (NEU / "ANNOTATIONS").glob("*.xml")}
    return images, xmls


def link_or_copy(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return "existing"
    try:
        os.link(source, target)
        return "hardlink"
    except OSError:
        shutil.copy2(source, target)
        return "copy"


def yolo_rows(width: int, height: int, objects: list[tuple[str, list[int]]]) -> list[str]:
    rows: list[str] = []
    for class_name, (xmin, ymin, xmax, ymax) in objects:
        if class_name not in CLASS_TO_ID:
            raise ValueError(f"Unexpected class {class_name!r}")
        xmin = max(0, min(width, xmin))
        xmax = max(0, min(width, xmax))
        ymin = max(0, min(height, ymin))
        ymax = max(0, min(height, ymax))
        if xmax <= xmin or ymax <= ymin:
            raise ValueError(f"Invalid box for {class_name}: {(xmin, ymin, xmax, ymax)}")
        x_center = ((xmin + xmax) / 2) / width
        y_center = ((ymin + ymax) / 2) / height
        box_width = (xmax - xmin) / width
        box_height = (ymax - ymin) / height
        rows.append(
            f"{CLASS_TO_ID[class_name]} {x_center:.6f} {y_center:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )
    return rows


def build_records() -> list[Record]:
    neu_images, neu_xmls = find_neu_files()
    source_dir = SD900 / "Source Images"
    mask_dir = SD900 / "Ground truth"
    records: list[Record] = []

    for source in sorted(source_dir.glob("*.bmp")):
        prefix, index = source.stem.split("_", 1)
        primary_class = PREFIX_TO_CLASS[prefix]
        stem = f"{primary_class}_{index}"
        image_path = neu_images[stem]
        xml_path = neu_xmls[stem]
        mask_path = mask_dir / f"{prefix}_{index}.png"
        width, height, objects = parse_xml(xml_path)
        classes = sorted({class_name for class_name, _ in objects})
        outside = [class_name for class_name in classes if class_name not in CLASS_TO_ID]
        excluded_reason = ""
        if stem in QUARANTINE:
            excluded_reason = "known_duplicate_pair"
        elif outside:
            excluded_reason = "outside_three_class_subset:" + ",".join(outside)

        current_iou: float | None = None
        if len(objects) == 1 and objects[0][0] == primary_class:
            current_mask_box = mask_bbox(mask_path)
            current_iou = bbox_iou(current_mask_box, objects[0][1]) if current_mask_box else 0.0
        fewshot_eligible = (
            not excluded_reason
            and len(objects) == 1
            and objects[0][0] == primary_class
            and current_iou is not None
            and current_iou >= 0.5
        )

        records.append(
            Record(
                stem=stem,
                primary_class=primary_class,
                source_image=str(image_path.resolve()),
                source_xml=str(xml_path.resolve()),
                source_mask=str(mask_path.resolve()),
                canonical_image=str((CANONICAL / "images" / "all" / f"{stem}.jpg").resolve()),
                canonical_label=str((CANONICAL / "labels" / "all" / f"{stem}.txt").resolve()),
                canonical_mask=str((CANONICAL / "masks" / "all" / f"{stem}.png").resolve()),
                sha256=sha256(image_path),
                object_count=len(objects),
                class_set="|".join(classes),
                mask_xml_iou=current_iou,
                fewshot_eligible=fewshot_eligible,
                excluded_reason=excluded_reason,
            )
        )
    return records


def assign_splits(
    records: list[Record], seed: int, fewshot: dict[str, int], val_per_class: int, test_per_class: int
) -> None:
    rng = random.Random(seed)
    for class_name in CLASS_NAMES:
        candidates = [
            record
            for record in records
            if record.primary_class == class_name and record.fewshot_eligible
        ]
        rng.shuffle(candidates)
        needed = fewshot[class_name]
        if len(candidates) < needed:
            raise RuntimeError(
                f"Not enough few-shot candidates for {class_name}: {len(candidates)} < {needed}"
            )
        for record in candidates[:needed]:
            record.split = "train_fewshot"

        remaining = [
            record
            for record in records
            if record.primary_class == class_name
            and not record.excluded_reason
            and record.split == "unused"
        ]
        rng.shuffle(remaining)
        if len(remaining) < val_per_class + test_per_class:
            raise RuntimeError(f"Not enough remaining records for {class_name}")
        for record in remaining[:val_per_class]:
            record.split = "val"
        for record in remaining[val_per_class : val_per_class + test_per_class]:
            record.split = "test"

    for record in records:
        if record.excluded_reason:
            record.split = "quarantine"


def materialize(records: list[Record]) -> Counter:
    operations = Counter()
    for record in records:
        if record.split == "quarantine":
            continue
        image_source = Path(record.source_image)
        mask_source = Path(record.source_mask)
        image_target = Path(record.canonical_image)
        mask_target = Path(record.canonical_mask)
        label_target = Path(record.canonical_label)
        operations[f"image_{link_or_copy(image_source, image_target)}"] += 1
        operations[f"mask_{link_or_copy(mask_source, mask_target)}"] += 1
        width, height, objects = parse_xml(Path(record.source_xml))
        label_target.parent.mkdir(parents=True, exist_ok=True)
        label_target.write_text("\n".join(yolo_rows(width, height, objects)) + "\n", encoding="utf-8")
        operations["label_written"] += 1
    return operations


def write_outputs(records: list[Record], seed: int, operations: Counter) -> None:
    SPLITS.mkdir(parents=True, exist_ok=True)
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)

    for split in ("train_fewshot", "val", "test", "unused", "quarantine"):
        paths = sorted(
            record.canonical_image
            for record in records
            if record.split == split and record.canonical_image
        )
        (SPLITS / f"{split}.txt").write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")

    rows = [asdict(record) for record in sorted(records, key=lambda item: item.stem)]
    with (MANIFESTS / "sd900_three_class_v1.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    split_counts = Counter(record.split for record in records)
    class_split_counts = Counter((record.split, record.primary_class) for record in records)
    instance_counts = Counter()
    for record in records:
        if record.split in {"quarantine", "unused"}:
            continue
        _, _, objects = parse_xml(Path(record.source_xml))
        for class_name, _ in objects:
            instance_counts[(record.split, class_name)] += 1

    summary = {
        "version": "v1",
        "seed": seed,
        "fewshot_requested": DEFAULT_FEWSHOT,
        "split_counts": dict(split_counts),
        "class_split_image_counts": {
            f"{split}:{class_name}": count
            for (split, class_name), count in sorted(class_split_counts.items())
        },
        "class_split_instance_counts": {
            f"{split}:{class_name}": count
            for (split, class_name), count in sorted(instance_counts.items())
        },
        "operations": dict(operations),
        "quarantine": [
            {"stem": record.stem, "reason": record.excluded_reason}
            for record in records
            if record.split == "quarantine"
        ],
    }
    (MANIFESTS / "sd900_three_class_v1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    dataset_yaml = "\n".join(
        [
            f"path: {PROJECT.as_posix()}",
            f"train: {(SPLITS / 'train_fewshot.txt').as_posix()}",
            f"val: {(SPLITS / 'val.txt').as_posix()}",
            f"test: {(SPLITS / 'test.txt').as_posix()}",
            "names:",
            "  0: inclusion",
            "  1: patches",
            "  2: scratches",
            "",
        ]
    )
    (CONFIGS / "dataset_sd900_3class_v1.yaml").write_text(dataset_yaml, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--val-per-class", type=int, default=50)
    parser.add_argument("--test-per-class", type=int, default=50)
    args = parser.parse_args()

    records = build_records()
    assign_splits(records, args.seed, DEFAULT_FEWSHOT, args.val_per_class, args.test_per_class)
    operations = materialize(records)
    write_outputs(records, args.seed, operations)


if __name__ == "__main__":
    main()

