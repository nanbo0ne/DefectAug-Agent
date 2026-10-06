"""Render the three first AIO candidates beside their source, mask and composite."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
PILOT = PROJECT / "data/synthetic/pilot/aio"
OUTPUT = PROJECT / "reports/aio_pilot_first3.png"
SOURCE = PROJECT / "data/canonical/images/all/scratches_100.jpg"
MASK = PROJECT / "data/canonical/masks/all/scratches_100.png"
SEEDS = [20260820, 20260821, 20260822]


def cell(path: Path, label: str, size: int = 260) -> Image.Image:
    image = Image.open(path).convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size + 30), "white")
    canvas.paste(image, (0, 0))
    ImageDraw.Draw(canvas).text((6, size + 7), label, fill="black", font=ImageFont.load_default())
    return canvas


def main() -> None:
    width, height = 260, 290
    header = 34
    sheet = Image.new("RGB", (4 * width, header + 3 * height), (225, 225, 225))
    draw = ImageDraw.Draw(sheet)
    for column, title in enumerate(("source", "mask", "raw AIO edit", "mask composite")):
        draw.text((column * width + 6, 10), title, fill="black", font=ImageFont.load_default())
    for row, seed in enumerate(SEEDS):
        record = json.loads((PILOT / f"scratches_100_seed{seed}.json").read_text(encoding="utf-8"))
        denoise = record["denoise"]
        paths = [
            SOURCE,
            MASK,
            Path(record["raw_output"]),
            Path(record["composite_output"]),
        ]
        labels = [
            f"seed {seed}; denoise {denoise}",
            "official binary defect mask",
            f"raw; {record['elapsed_seconds']:.1f}s",
            f"inside changed {record['metrics']['inside_changed_fraction']:.1%}",
        ]
        for column, (path, label) in enumerate(zip(paths, labels, strict=True)):
            sheet.paste(cell(path, label, width), (column * width, header + row * height))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()
