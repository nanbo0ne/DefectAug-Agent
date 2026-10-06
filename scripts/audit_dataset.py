"""Audit the purchased GG38 archives and the official SD-saliency-900 mirror.

The script is intentionally read-only with respect to source datasets.  It writes
only two reports under ``reports/`` so that every later split decision has an
auditable factual basis.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT = Path(__file__).resolve().parents[1]
RAW = PROJECT / "data" / "raw"
REPORTS = PROJECT / "reports"

NEU = RAW / "neu_det" / "NEU-DET"
TARGET = RAW / "target_detection" / "steel_bug_detect"
SEM_CONTAINER = RAW / "semantic_segmentation"
OFFICIAL_SD900 = PROJECT / "data" / "external" / "MCITF" / "SD-saliency-900"

NEU_CLASSES = [
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
]
SD900_PREFIX = {"In": "inclusion", "Pa": "patches", "Sc": "scratches"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_xml(path: Path) -> tuple[tuple[int, int], list[dict]]:
    root = ET.parse(path).getroot()
    width = int(root.findtext("size/width", "0"))
    height = int(root.findtext("size/height", "0"))
    objects: list[dict] = []
    for node in root.findall("object"):
        box = node.find("bndbox")
        if box is None:
            continue
        objects.append(
            {
                "class": node.findtext("name", ""),
                "box": [
                    int(float(box.findtext("xmin", "0"))),
                    int(float(box.findtext("ymin", "0"))),
                    int(float(box.findtext("xmax", "0"))),
                    int(float(box.findtext("ymax", "0"))),
                ],
            }
        )
    return (width, height), objects


def parse_yolo(path: Path) -> list[list[float]]:
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(value) for value in line.split()])
    return rows


def image_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}
    )


def find_semantic_root() -> Path:
    candidates = [path for path in SEM_CONTAINER.iterdir() if path.is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one semantic root, found {len(candidates)}")
    return candidates[0]


def audit_neu() -> tuple[dict, dict[str, Path], dict[str, tuple[tuple[int, int], list[dict]]]]:
    images = sorted((NEU / "train" / "images").glob("*.jpg")) + sorted(
        (NEU / "valid" / "images").glob("*.jpg")
    )
    labels = sorted((NEU / "train" / "labels").glob("*.txt")) + sorted(
        (NEU / "valid" / "labels").glob("*.txt")
    )
    image_by_stem = {path.stem: path for path in images}
    xml_by_stem = {path.stem: path for path in (NEU / "ANNOTATIONS").glob("*.xml")}
    label_by_stem = {path.stem: path for path in labels}

    corrupt: list[str] = []
    dimensions = Counter()
    hashes: dict[str, list[str]] = defaultdict(list)
    for path in images:
        try:
            with Image.open(path) as image:
                image.load()
                dimensions[f"{image.width}x{image.height}:{image.mode}"] += 1
        except Exception as exc:  # pragma: no cover - audit output is the evidence
            corrupt.append(f"{path}: {exc}")
            continue
        hashes[sha256(path)].append(path.name)

    duplicate_groups = sorted(
        (names for names in hashes.values() if len(names) > 1), key=lambda names: names[0]
    )
    instance_counts = Counter()
    image_counts = Counter()
    invalid_labels: list[dict] = []
    empty_labels: list[str] = []
    xml_cache: dict[str, tuple[tuple[int, int], list[dict]]] = {}

    for stem, image_path in image_by_stem.items():
        xml_path = xml_by_stem.get(stem)
        label_path = label_by_stem.get(stem)
        if xml_path is None or label_path is None:
            invalid_labels.append(
                {"stem": stem, "reason": "missing_xml_or_yolo", "image": str(image_path)}
            )
            continue
        size, objects = parse_xml(xml_path)
        xml_cache[stem] = (size, objects)
        rows = parse_yolo(label_path)
        if not rows:
            empty_labels.append(stem)
        seen = set()
        for row in rows:
            if len(row) != 5:
                invalid_labels.append({"stem": stem, "reason": "wrong_column_count", "row": row})
                continue
            class_id = int(row[0])
            coords = row[1:]
            if class_id < 0 or class_id >= len(NEU_CLASSES):
                invalid_labels.append({"stem": stem, "reason": "class_out_of_range", "row": row})
                continue
            if any(value < 0 or value > 1 for value in coords) or coords[2] <= 0 or coords[3] <= 0:
                invalid_labels.append({"stem": stem, "reason": "invalid_normalized_box", "row": row})
            class_name = NEU_CLASSES[class_id]
            instance_counts[class_name] += 1
            seen.add(class_name)
        for class_name in seen:
            image_counts[class_name] += 1

    summary = {
        "images": len(images),
        "unique_image_hashes": len(hashes),
        "xml_files": len(xml_by_stem),
        "yolo_files": len(label_by_stem),
        "dimensions": dict(dimensions),
        "corrupt_images": corrupt,
        "duplicate_groups": duplicate_groups,
        "empty_labels": empty_labels,
        "invalid_labels": invalid_labels,
        "instance_counts": dict(instance_counts),
        "images_containing_class": dict(image_counts),
    }
    return summary, image_by_stem, xml_cache


def audit_target_overlap(neu_images: dict[str, Path]) -> dict:
    target_images = sorted((TARGET / "train" / "IMAGES").glob("*.jpg")) + sorted(
        (TARGET / "test" / "IMAGES").glob("*.jpg")
    )
    neu_hashes = Counter(sha256(path) for path in neu_images.values())
    target_hashes = Counter(sha256(path) for path in target_images)
    overlap_files = sum(min(count, target_hashes[digest]) for digest, count in neu_hashes.items())
    return {
        "images": len(target_images),
        "unique_image_hashes": len(target_hashes),
        "exact_hash_overlap_files_with_neu": overlap_files,
        "same_multiset_of_hashes": neu_hashes == target_hashes,
    }


def audit_semantic(neu_images: dict[str, Path]) -> dict:
    root = find_semantic_root()
    images = sorted((root / "images").glob("*.jpg"))
    masks = {path.stem: path for path in (root / "SegmentationClass").glob("*.png")}
    annotations = {path.stem: path for path in (root / "Annotations").glob("*.json")}
    label_shapes = Counter()
    label_images = Counter()
    empty_masks: list[str] = []
    invalid_mask_values: dict[str, list[int]] = {}
    corrupt: list[str] = []

    for image_path in images:
        stem = image_path.stem
        try:
            with Image.open(image_path) as image:
                image.load()
            mask_path = masks[stem]
            mask = np.asarray(Image.open(mask_path))
            values = sorted(int(value) for value in np.unique(mask))
            if not any(value != 0 for value in values):
                empty_masks.append(stem)
            if any(value not in {0, 1, 2, 3} for value in values):
                invalid_mask_values[stem] = values
            payload = json.loads(annotations[stem].read_text(encoding="utf-8"))
            present = set()
            for shape in payload.get("shapes", []):
                label = str(shape.get("label", ""))
                label_shapes[label] += 1
                present.add(label)
            for label in present:
                label_images[label] += 1
        except Exception as exc:  # pragma: no cover
            corrupt.append(f"{stem}: {exc}")

    neu_hashes = {sha256(path) for path in neu_images.values()}
    exact_overlap = sum(sha256(path) in neu_hashes for path in images)
    return {
        "images": len(images),
        "masks": len(masks),
        "json_annotations": len(annotations),
        "shape_counts": dict(label_shapes),
        "images_containing_label": dict(label_images),
        "empty_masks": empty_masks,
        "invalid_mask_values": invalid_mask_values,
        "corrupt_or_missing": corrupt,
        "exact_image_overlap_with_neu": exact_overlap,
        "unverified_or_derived_images": len(images) - exact_overlap,
    }


def mask_bbox(mask_path: Path) -> list[int] | None:
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.nonzero(mask)
    if not len(xs):
        return None
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def bbox_iou(a: list[int], b: list[int]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def audit_official_sd900(
    neu_images: dict[str, Path], xml_cache: dict[str, tuple[tuple[int, int], list[dict]]]
) -> dict:
    sources = sorted((OFFICIAL_SD900 / "Source Images").glob("*.bmp"))
    masks = {path.stem: path for path in (OFFICIAL_SD900 / "Ground truth").glob("*.png")}
    counts = Counter()
    empty_masks: list[str] = []
    missing_neu: list[str] = []
    pixel_mae: list[float] = []
    single_object_ious: list[float] = []
    single_object_ious_by_class: dict[str, list[float]] = defaultdict(list)

    for source in sources:
        prefix, index = source.stem.split("_", 1)
        class_name = SD900_PREFIX[prefix]
        counts[class_name] += 1
        neu_stem = f"{class_name}_{index}"
        neu_image = neu_images.get(neu_stem)
        if neu_image is None:
            missing_neu.append(neu_stem)
            continue
        source_pixels = np.asarray(Image.open(source).convert("L"), dtype=np.int16)
        neu_pixels = np.asarray(Image.open(neu_image).convert("L"), dtype=np.int16)
        pixel_mae.append(float(np.abs(source_pixels - neu_pixels).mean()))

        current_mask = masks.get(source.stem)
        box = mask_bbox(current_mask) if current_mask else None
        if box is None:
            empty_masks.append(source.stem)
            continue
        _, objects = xml_cache[neu_stem]
        matching = [item["box"] for item in objects if item["class"] == class_name]
        if len(objects) == 1 and len(matching) == 1:
            current_iou = bbox_iou(box, matching[0])
            single_object_ious.append(current_iou)
            single_object_ious_by_class[class_name].append(current_iou)

    iou_sorted = sorted(single_object_ious)
    return {
        "git_commit": _git_commit(OFFICIAL_SD900.parents[0]),
        "source_images": len(sources),
        "masks": len(masks),
        "class_counts": dict(counts),
        "missing_neu_filename_matches": missing_neu,
        "empty_masks": empty_masks,
        "source_to_neu_pixel_mae": {
            "mean": statistics.fmean(pixel_mae) if pixel_mae else None,
            "max": max(pixel_mae) if pixel_mae else None,
        },
        "single_object_mask_bbox_vs_xml_iou": {
            "count": len(iou_sorted),
            "mean": statistics.fmean(iou_sorted) if iou_sorted else None,
            "median": statistics.median(iou_sorted) if iou_sorted else None,
            "p10": iou_sorted[max(0, int(len(iou_sorted) * 0.10) - 1)] if iou_sorted else None,
            "min": min(iou_sorted) if iou_sorted else None,
            "count_ge_0_5": sum(value >= 0.5 for value in iou_sorted),
            "by_class": {
                class_name: {
                    "count": len(values),
                    "count_ge_0_5": sum(value >= 0.5 for value in values),
                    "median": statistics.median(values) if values else None,
                }
                for class_name, values in sorted(single_object_ious_by_class.items())
            },
        },
    }


def _git_commit(repo: Path) -> str | None:
    head = repo / ".git" / "HEAD"
    if not head.exists():
        return None
    value = head.read_text(encoding="utf-8").strip()
    if value.startswith("ref: "):
        ref = repo / ".git" / value[5:]
        return ref.read_text(encoding="utf-8").strip() if ref.exists() else None
    return value


def render_markdown(report: dict) -> str:
    neu = report["neu_det"]
    semantic = report["purchased_semantic"]
    official = report["official_sd_saliency_900"]
    lines = [
        "# GG38 数据集机器审计报告",
        "",
        "> 本报告由 `scripts/audit_dataset.py` 生成。源数据只读，结果写入 `reports/`。",
        "",
        "## 结论",
        "",
        f"- NEU-DET：{neu['images']} 张图片，{neu['unique_image_hashes']} 个唯一文件哈希。",
        f"- 购买包的目标检测版与 NEU-DET 完全同源：`same_multiset_of_hashes={report['target_detection_overlap']['same_multiset_of_hashes']}`。",
        f"- 购买包的语义分割版：{semantic['images']} 张，其中仅 {semantic['exact_image_overlap_with_neu']} 张与 NEU 原图逐文件相同；其余 {semantic['unverified_or_derived_images']} 张暂不作为真实样本。",
        f"- 官方 SD-saliency-900：{official['source_images']} 张图和 {official['masks']} 张 Mask，Git 提交 `{official['git_commit']}`。",
        "",
        "## NEU-DET 类别实例数",
        "",
        "| 类别 | 实例数 | 含该类图片数 |",
        "|---|---:|---:|",
    ]
    for class_name in NEU_CLASSES:
        lines.append(
            f"| {class_name} | {neu['instance_counts'].get(class_name, 0)} | "
            f"{neu['images_containing_class'].get(class_name, 0)} |"
        )
    lines.extend(
        [
            "",
            "## 质量与使用决定",
            "",
            f"- 损坏 NEU 图片：{len(neu['corrupt_images'])}。",
            f"- 空 NEU 标签：{len(neu['empty_labels'])}。",
            f"- 非法 NEU 标签：{len(neu['invalid_labels'])}。",
            f"- NEU 重复组：`{json.dumps(neu['duplicate_groups'], ensure_ascii=False)}`。",
            f"- 官方 Mask 空图：{len(official['empty_masks'])}。",
            f"- 官方图片与 NEU 同名图平均像素 MAE：{official['source_to_neu_pixel_mae']['mean']:.4f}（格式转换/JPEG压缩差异）。",
            "- 正式检测真实源只使用 NEU-DET；Mask 优先使用官方 SD-saliency-900。",
            "- 购买语义分割包中未能追溯的派生图全部隔离，不参与训练、验证或测试。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    neu, neu_images, xml_cache = audit_neu()
    report = {
        "neu_det": neu,
        "target_detection_overlap": audit_target_overlap(neu_images),
        "purchased_semantic": audit_semantic(neu_images),
        "official_sd_saliency_900": audit_official_sd900(neu_images, xml_cache),
    }
    (REPORTS / "dataset_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (REPORTS / "dataset_audit.md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
