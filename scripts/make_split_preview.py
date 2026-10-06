"""Render a visual sanity sheet for the fixed few-shot split."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_LIST = PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt"
OUTPUT = PROJECT / "reports" / "fewshot_preview.png"
CLASS_NAMES = ["inclusion", "patches", "scratches"]
COLORS = [(239, 68, 68), (34, 197, 94), (59, 130, 246)]


def label_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\labels\\")).with_suffix(".txt")


def mask_path(image_path: Path) -> Path:
    return Path(str(image_path).replace("\\images\\", "\\masks\\")).with_suffix(".png")


def render_cell(image_path: Path, cell_size: int = 250) -> Image.Image:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path(image_path)).convert("L")
    overlay = Image.new("RGB", image.size, (255, 40, 40))
    blended = Image.composite(Image.blend(image, overlay, 0.35), image, mask)
    draw = ImageDraw.Draw(blended)
    width, height = blended.size
    for line in label_path(image_path).read_text(encoding="utf-8").splitlines():
        class_id, xc, yc, bw, bh = [float(value) for value in line.split()]
        x1 = (xc - bw / 2) * width
        y1 = (yc - bh / 2) * height
        x2 = (xc + bw / 2) * width
        y2 = (yc + bh / 2) * height
        draw.rectangle((x1, y1, x2, y2), outline=COLORS[int(class_id)], width=3)
    canvas = Image.new("RGB", (cell_size, cell_size + 28), "white")
    canvas.paste(blended.resize((cell_size, cell_size), Image.Resampling.NEAREST), (0, 0))
    ImageDraw.Draw(canvas).text((6, cell_size + 5), image_path.stem, fill="black", font=ImageFont.load_default())
    return canvas


def main() -> None:
    images = [Path(line.strip()) for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected: list[Path] = []
    for class_name in CLASS_NAMES:
        selected.extend(sorted(path for path in images if path.stem.startswith(class_name + "_"))[:6])
    columns, rows = 6, len(CLASS_NAMES)
    cell_width, cell_height = 250, 278
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), (235, 235, 235))
    for index, path in enumerate(selected):
        sheet.paste(render_cell(path, cell_width), ((index % columns) * cell_width, (index // columns) * cell_height))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT)
    print(OUTPUT)


if __name__ == "__main__":
    main()

