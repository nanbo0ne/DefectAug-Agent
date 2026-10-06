"""v3.1 (= v3 + --missed-output) per-instance miss analysis on val for the agent observation.

Runs a trained detector on the frozen val split, matches predictions to
ground-truth boxes (greedy, IoU >= 0.5, confidence-ordered), and reports for
each class:

- recall split by size tertile and contrast tertile, where the tertile edges
  are computed from the 100 train images only (no val statistics leak into
  the bucket definitions);
- misclassifications (GT matched by a box of another class at IoU >= 0.5);
- false positives per image.

Only aggregate counts leave this script; no val image paths or pixels are
passed to the agent.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
CANON_LABELS = PROJECT / "data" / "canonical" / "labels" / "all"
CANON_MASKS = PROJECT / "data" / "canonical" / "masks" / "all"
SPLITS = PROJECT / "data" / "splits" / "v3"
IOU_MATCH = 0.5
RING = 5


def _read_manifest(path: Path) -> list[Path]:
    return [Path(l.strip()) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _gt_instances(image_path: Path) -> list[dict]:
    """GT boxes of one image with pixel size and local contrast."""
    # PIL, not cv2.imread: ultralytics monkey-patches cv2.imread once imported.
    grey = np.asarray(Image.open(image_path).convert("L"), dtype=np.float32)
    h_img, w_img = grey.shape
    mask_path = CANON_MASKS / f"{image_path.stem}.png"
    mask = np.asarray(Image.open(mask_path).convert("L")) > 0 if mask_path.is_file() else None
    out = []
    for line in (CANON_LABELS / f"{image_path.stem}.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        c, x, y, w, h = map(float, line.split())
        x0, y0 = max(0, int((x - w / 2) * w_img)), max(0, int((y - h / 2) * h_img))
        x1, y1 = min(w_img, int(np.ceil((x + w / 2) * w_img))), min(h_img, int(np.ceil((y + h / 2) * h_img)))
        box = np.zeros((h_img, w_img), bool)
        box[y0:y1, x0:x1] = True
        inside = box & mask if mask is not None else box
        if not inside.any():
            inside = box
        ring = cv2.dilate(box.astype(np.uint8), np.ones((2 * RING + 1,) * 2, np.uint8)).astype(bool) & ~box
        contrast = float(abs(grey[inside].mean() - grey[ring].mean())) if ring.any() else 0.0
        out.append({
            "cls": int(c),
            "xyxy": (x0, y0, x1, y1),
            "size": float(np.sqrt(max(1, x1 - x0) * max(1, y1 - y0))),
            "contrast": contrast,
        })
    return out


def tertile_edges() -> dict:
    """Bucket edges and per-class median GT size, from the 100 train images only."""
    sizes, contrasts = [], []
    per_class: dict[str, list[float]] = {n: [] for n in CLASSES}
    for p in _read_manifest(SPLITS / "train_fewshot.txt"):
        for g in _gt_instances(p):
            sizes.append(g["size"])
            contrasts.append(g["contrast"])
            per_class[CLASSES[g["cls"]]].append(g["size"])
    return {
        "train_median_size_px": {n: float(np.median(v)) for n, v in per_class.items()},
        "size": [float(v) for v in np.percentile(sizes, [100 / 3, 200 / 3])],
        "contrast": [float(v) for v in np.percentile(contrasts, [100 / 3, 200 / 3])],
    }


def _bucket(value: float, edges: list[float]) -> str:
    return "low" if value < edges[0] else ("mid" if value < edges[1] else "high")


def _iou(a, b) -> float:
    ix0, iy0, ix1, iy1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix1 - ix0) * max(0.0, iy1 - iy0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def analyse(weights: Path, conf: float, imgsz: int, device: str, missed_out: list | None = None) -> dict:
    from ultralytics import YOLO

    edges = tertile_edges()
    model = YOLO(str(weights))
    val = _read_manifest(SPLITS / "val.txt")
    stats = {
        name: {
            "gt": 0, "hit": 0, "misclassified_as": {n: 0 for n in CLASSES if n != name},
            "by_size": {b: [0, 0] for b in ("low", "mid", "high")},
            "by_contrast": {b: [0, 0] for b in ("low", "mid", "high")},
            "false_positives": 0, "missed_sizes": [],
        }
        for name in CLASSES
    }
    for start in range(0, len(val), 16):
        batch = val[start:start + 16]
        results = model.predict([str(p) for p in batch], conf=conf, imgsz=imgsz, device=device,
                                verbose=False)
        for path, res in zip(batch, results):
            gts = _gt_instances(path)
            preds = sorted(
                zip(res.boxes.xyxy.cpu().numpy().tolist(), res.boxes.cls.cpu().numpy().astype(int).tolist(),
                    res.boxes.conf.cpu().numpy().tolist()),
                key=lambda t: -t[2],
            )
            matched_gt: set[int] = set()
            for box, cls, _ in preds:
                best, best_iou = None, IOU_MATCH
                for i, g in enumerate(gts):
                    if i in matched_gt or g["cls"] != cls:
                        continue
                    v = _iou(box, g["xyxy"])
                    if v >= best_iou:
                        best, best_iou = i, v
                if best is None:
                    stats[CLASSES[cls]]["false_positives"] += 1
                else:
                    matched_gt.add(best)
            for i, g in enumerate(gts):
                s = stats[CLASSES[g["cls"]]]
                hit = i in matched_gt
                s["gt"] += 1
                s["hit"] += int(hit)
                for key, value in (("by_size", g["size"]), ("by_contrast", g["contrast"])):
                    cell = s[key][_bucket(value, edges[key.split("_")[1]])]  # size|contrast
                    cell[0] += int(hit)
                    cell[1] += 1
                if not hit:
                    s["missed_sizes"].append(g["size"])
                    other = [
                        CLASSES[cls] for box, cls, _ in preds
                        if cls != g["cls"] and _iou(box, g["xyxy"]) >= IOU_MATCH
                    ]
                    if other:
                        s["misclassified_as"][other[0]] += 1
                    if missed_out is not None:
                        missed_out.append({"image": str(path), "cls": CLASSES[g["cls"]], "xyxy": g["xyxy"],
                                           "size": round(g["size"], 1), "contrast": round(g["contrast"], 1),
                                           "misclassified_as": other[0] if other else None})
    out = {"conf": conf, "iou_match": IOU_MATCH,
           "tertile_edges_from_train": {k: edges[k] for k in ("size", "contrast")},
           "val_images": len(val), "classes": {}}
    for name, s in stats.items():
        def rate(cell):
            return {"recall": round(cell[0] / cell[1], 3) if cell[1] else None, "n": cell[1]}
        out["classes"][name] = {
            "gt_instances": s["gt"],
            "recall": round(s["hit"] / s["gt"], 3) if s["gt"] else None,
            "missed": s["gt"] - s["hit"],
            "misclassified_as": s["misclassified_as"],
            "false_positives_per_image": round(s["false_positives"] / len(val), 3),
            "train_median_size_px": round(edges["train_median_size_px"][name], 1),
            "median_missed_size_px": (round(float(np.median(s["missed_sizes"])), 1)
                                      if s["missed_sizes"] else None),
            "recall_by_size_tertile": {b: rate(c) for b, c in s["by_size"].items()},
            "recall_by_contrast_tertile": {b: rate(c) for b, c in s["by_contrast"].items()},
        }
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--missed-output", default=None,
                        help="optional JSON list of missed val instances (for v3.1 vision tools)")
    args = parser.parse_args()
    missed: list | None = [] if args.missed_output else None
    report = analyse(Path(args.weights), args.conf, args.imgsz, args.device, missed)
    if args.missed_output:
        Path(args.missed_output).write_text(json.dumps(missed, indent=1), encoding="utf-8")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
