"""Build the deterministic real-only exposure-matched training control.

The augmented conditions contain 100 real + 150 accepted synthetic images.
This control contains no synthetic pixels: it repeats only the frozen 100 real
few-shot rows to 250 rows while preserving the 1:3:6 class ratio exactly.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")


def class_name(path: Path) -> str:
    matches = [name for name in CLASSES if path.stem.startswith(f"{name}_")]
    if len(matches) != 1:
        raise ValueError(f"cannot infer one class from {path}")
    return matches[0]


def proportional_targets(counts: dict[str, int], total: int) -> dict[str, int]:
    if total <= 0 or not counts or any(value <= 0 for value in counts.values()):
        raise ValueError("counts and total must be positive")
    count_total = sum(counts.values())
    raw = {name: total * count / count_total for name, count in counts.items()}
    targets = {name: int(value) for name, value in raw.items()}
    remaining = total - sum(targets.values())
    for name in sorted(counts, key=lambda item: (-(raw[item] - targets[item]), item)):
        if remaining == 0:
            break
        targets[name] += 1
        remaining -= 1
    return targets


def build_repeated_rows(paths: list[Path], total: int) -> tuple[list[Path], dict[str, int]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in paths:
        grouped[class_name(path)].append(path)
    if set(grouped) != set(CLASSES):
        raise ValueError(f"expected exactly classes {CLASSES}, got {sorted(grouped)}")
    for values in grouped.values():
        values.sort(key=lambda path: path.stem)
    targets = proportional_targets({name: len(grouped[name]) for name in CLASSES}, total)
    expanded = {
        name: [grouped[name][index % len(grouped[name])] for index in range(targets[name])]
        for name in CLASSES
    }
    # Deterministically interleave classes instead of emitting three large blocks.
    rows: list[Path] = []
    while any(expanded.values()):
        for name in CLASSES:
            if expanded[name]:
                rows.append(expanded[name].pop(0))
    return rows, targets


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=PROJECT / "data/splits/v1/train_fewshot.txt")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT / "data/splits/v1/train_fewshot_repeat_matched.txt",
    )
    parser.add_argument("--total", type=int, default=250)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    input_path = args.input.resolve(strict=True)
    paths = [Path(line.strip()).resolve(strict=True) for line in input_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    rows, targets = build_repeated_rows(paths, args.total)
    output = args.output.resolve()
    content = "\n".join(str(path) for path in rows) + "\n"
    if output.exists() and not args.overwrite:
        if output.read_text(encoding="utf-8-sig") != content:
            raise FileExistsError(f"refusing to replace different frozen control: {output}")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content, encoding="utf-8")
    print(json.dumps({"output": str(output), "rows": len(rows), "by_class": targets}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
