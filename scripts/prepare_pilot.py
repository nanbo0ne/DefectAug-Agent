"""Create a deterministic manifest for the first Scratches generation pilot."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
TRAIN_LIST = PROJECT / "data" / "splits" / "v1" / "train_fewshot.txt"
OUTPUT = PROJECT / "data" / "manifests" / "pilot_scratches_v1.csv"


def sibling(path: Path, old: str, new: str, suffix: str) -> Path:
    return Path(str(path).replace(f"\\{old}\\", f"\\{new}\\")).with_suffix(suffix)


def main() -> None:
    images = sorted(
        Path(line.strip())
        for line in TRAIN_LIST.read_text(encoding="utf-8").splitlines()
        if line.strip() and Path(line.strip()).stem.startswith("scratches_")
    )[:10]
    rows = [
        {
            "source_id": path.stem,
            "image": str(path),
            "mask": str(sibling(path, "images", "masks", ".png")),
            "label": str(sibling(path, "images", "labels", ".txt")),
            "class": "scratches",
            "status": "not_generated",
        }
        for path in images
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUTPUT)


if __name__ == "__main__":
    main()

