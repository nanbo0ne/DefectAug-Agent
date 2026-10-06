"""Compute the transparent formula allocation for one generation round.

Wraps defectaug.policy.formula_allocation and adds two protocol constraints
the raw allocator does not know about:

- remaining per-class source capacity (accepted <= 8 per source image);
- optional cumulative cap 0.70 * total_budget shared by all rounds.

Excess over capacity is redistributed to uncapped classes by descending
weakness score (stable class-name tie-break), preserving the round total.
Prints the allocation as `inclusion:N,patches:N,scratches:N`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from defectaug.policy import ClassMetrics, formula_allocation

CLASSES = ("inclusion", "patches", "scratches")
SOURCES_PER_CLASS = {"inclusion": 10, "patches": 30, "scratches": 60}
MAX_ACCEPTED_PER_SOURCE = 8
PROJECT = Path(__file__).resolve().parents[1]


def pool_capacity(pool_dir: Path, seed: int, tag: str) -> dict[str, int]:
    capacity: dict[str, int] = {}
    for name in CLASSES:
        accepted = len(list(pool_dir.glob(f"r*/images/{name}_{tag}{seed}_r*_*.png")))
        capacity[name] = max(0, MAX_ACCEPTED_PER_SOURCE * SOURCES_PER_CLASS[name] - accepted)
    return capacity


def parse_map(text: str, cast=int) -> dict:
    out = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(":")
        out[name.strip()] = cast(value.strip())
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="metrics.json from evaluate_yolo.")
    parser.add_argument("--total", type=int, required=True)
    parser.add_argument("--minimum", type=int, default=0)
    parser.add_argument("--maximum-fraction", type=float, default=0.70)
    parser.add_argument("--real", default="inclusion:10,patches:30,scratches:60")
    parser.add_argument(
        "--capacity",
        default="",
        help="Remaining per-class capacity, e.g. inclusion:80,patches:105,scratches:105",
    )
    parser.add_argument(
        "--pool-dir",
        default="",
        help="Synthetic pool dir; capacity auto-computed from accepted files when set.",
    )
    parser.add_argument("--tag", default="fm", help="Pool filename tag (u/f1/fm/ag).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    doc = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    per_class = {c["class_name"]: c for c in doc["metrics"]["per_class"]}
    real = parse_map(args.real)
    metrics = {
        name: ClassMetrics(
            recall=float(per_class[name]["recall"]),
            ap50=float(per_class[name]["ap50"]),
            real_images=real[name],
        )
        for name in CLASSES
    }
    scores, allocation = formula_allocation(
        metrics,
        total=args.total,
        minimum=args.minimum,
        maximum_fraction=args.maximum_fraction,
    )

    capacity = parse_map(args.capacity) if args.capacity else {}
    if args.pool_dir:
        capacity = pool_capacity(Path(args.pool_dir), args.seed, args.tag)
        cap_total = int(0.70 * 150)
        for name in CLASSES:
            capacity[name] = min(capacity[name], cap_total)
    if capacity:
        order = sorted(CLASSES, key=lambda n: (-scores[n], n))
        overflow = 0
        for name in CLASSES:
            cap = capacity.get(name)
            if cap is not None and allocation[name] > cap:
                overflow += allocation[name] - cap
                allocation[name] = cap
        for name in order:
            if overflow <= 0:
                break
            cap = capacity.get(name, args.total)
            room = max(0, cap - allocation[name])
            take = min(room, overflow)
            allocation[name] += take
            overflow -= take
        if overflow > 0:
            raise RuntimeError(f"capacity leaves {overflow} units unallocatable")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {"scores": scores, "allocation": allocation, "capacity": capacity},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    print(",".join(f"{name}:{allocation[name]}" for name in CLASSES))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
