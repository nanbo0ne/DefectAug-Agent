"""v3.1 vision tools for the agent: labelled montages of defect crops.

Every montage is a single RGB image (PNG, base64) so the agent sees many
instances in one message.  Tiles are numbered; the caller passes the number ->
identity mapping in text.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw

TILE = 112
PAD = 0.6          # context around a box, as a fraction of its size
MIN_CROP = 24      # px in the source image


def _crop(img: Image.Image, xyxy: Sequence[float], label: str, color=(255, 64, 64)) -> Image.Image:
    x0, y0, x1, y1 = (float(v) for v in xyxy)
    w, h = max(x1 - x0, 1.0), max(y1 - y0, 1.0)
    side = max(w, h) * (1 + 2 * PAD)
    side = max(side, MIN_CROP)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    box = (int(cx - side / 2), int(cy - side / 2), int(cx + side / 2), int(cy + side / 2))
    crop = img.crop(box)                     # PIL pads outside the image with black
    scale = TILE / crop.width
    crop = crop.resize((TILE, TILE), Image.BILINEAR).convert("RGB")
    d = ImageDraw.Draw(crop)
    bx = [(x0 - box[0]) * scale, (y0 - box[1]) * scale, (x1 - box[0]) * scale, (y1 - box[1]) * scale]
    d.rectangle(bx, outline=color, width=1)
    d.rectangle([0, 0, 7 * len(label) + 4, 12], fill=(0, 0, 0))
    d.text((2, 0), label, fill=(255, 255, 0))
    return crop


def montage(tiles: list[Image.Image], cols: int = 8) -> Image.Image:
    rows = max(1, (len(tiles) + cols - 1) // cols)
    sheet = Image.new("RGB", (cols * (TILE + 2), rows * (TILE + 2)), (40, 40, 40))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * (TILE + 2), (i // cols) * (TILE + 2)))
    return sheet


def to_data_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def evenly(items: list, k: int) -> list:
    if len(items) <= k:
        return list(items)
    idx = np.linspace(0, len(items) - 1, k).round().astype(int)
    return [items[i] for i in idx]


def missed_montage(missed: list[dict], cls: str, k: int = 24) -> tuple[Image.Image | None, int]:
    """Missed val instances of one class, smallest to largest (red box = the GT)."""
    items = sorted((m for m in missed if m["cls"] == cls), key=lambda m: m["size"])
    pick = evenly(items, k)
    tiles = []
    for i, m in enumerate(pick, 1):
        with Image.open(m["image"]) as im:
            tag = f"{i}" + ("x" if m.get("misclassified_as") else "")
            tiles.append(_crop(im.convert("RGB"), m["xyxy"], tag))
    return (montage(tiles) if tiles else None), len(items)


def _mask_bbox(mask_path: Path) -> tuple[int, int, int, int] | None:
    m = np.asarray(Image.open(mask_path).convert("L")) > 0
    ys, xs = np.where(m)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def source_montage(sources: list[Path], masks_dir: Path) -> Image.Image:
    """Each train source defect (mask bbox), tile label = numeric stem suffix."""
    # Whole source image (defects are often image-sized strips), mask outline in blue.
    tiles = []
    for p in sources:
        mask = np.asarray(Image.open(masks_dir / f"{p.stem}.png").convert("L").resize((TILE, TILE))) > 0
        with Image.open(p) as im:
            t = im.convert("RGB").resize((TILE, TILE), Image.BILINEAR)
        arr = np.asarray(t).copy()
        edge = mask & ~(np.roll(mask, 1, 0) & np.roll(mask, -1, 0) & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
        arr[edge] = (64, 200, 255)
        t = Image.fromarray(arr)
        d = ImageDraw.Draw(t)
        label = p.stem.rsplit("_", 1)[1]
        d.rectangle([0, 0, 7 * len(label) + 4, 12], fill=(0, 0, 0))
        d.text((2, 0), label, fill=(255, 255, 0))
        tiles.append(t)
    return montage(tiles, cols=10)


def synthetic_montage(image_paths: list[Path], k: int = 16) -> Image.Image | None:
    """Synthetic samples: the inserted box is the last label line."""
    tiles = []
    for i, p in enumerate(evenly(sorted(image_paths), k), 1):
        lab = p.parent.parent / "labels" / f"{p.stem}.txt"
        lines = [l for l in lab.read_text(encoding="utf-8").splitlines() if l.strip()]
        _, x, y, w, h = map(float, lines[-1].split())
        with Image.open(p) as im:
            W, H = im.size
            tiles.append(_crop(im.convert("RGB"), ((x - w / 2) * W, (y - h / 2) * H,
                                                   (x + w / 2) * W, (y + h / 2) * H), str(i),
                               color=(64, 255, 64)))
    return montage(tiles) if tiles else None
