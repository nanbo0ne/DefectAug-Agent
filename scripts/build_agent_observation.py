"""Build the observation JSON consumed by agent_decide.py.

Combines the previous round's validation metrics with the cumulative
synthetic-generation state (accepted counts, remaining capacity) and the
round history so far.  Round 1 observations use the seed's real_only
validation metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
REAL_IMAGES = {"inclusion": 10, "patches": 30, "scratches": 60}
MAX_ACCEPTED_PER_SOURCE = 8
TOTAL_BUDGET = 150
ROUND_BUDGET = 50
TOTAL_ROUNDS = 3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="metrics.json of the previous round.")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--state", required=True, help="synth pool state.json (may not exist for r1).")
    parser.add_argument("--history", default="", help="Optional JSON list of previous allocations+metrics.")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    doc = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    per_class = {c["class_name"]: c for c in doc["metrics"]["per_class"]}
    state_path = Path(args.state)
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"per_source": {}}
    accepted_so_far: dict[str, int] = {name: 0 for name in CLASSES}
    pool = PROJECT / "data" / "synthetic" / f"agent_multi_round_{args.seed}"
    for name in CLASSES:
        accepted_so_far[name] = len(list(pool.glob(f"r*/images/{name}_ag{args.seed}_r*_*.png")))

    history = []
    if args.history and Path(args.history).is_file():
        history = json.loads(Path(args.history).read_text(encoding="utf-8"))

    observation = {
        "condition": "agent_multi_round",
        "seed": args.seed,
        "round_index": args.round,
        "total_rounds": TOTAL_ROUNDS,
        "round_budget": ROUND_BUDGET,
        "total_budget": TOTAL_BUDGET,
        "spent_budget": sum(accepted_so_far.values()),
        "real_fewshot_images": REAL_IMAGES,
        "previous_validation": {
            "overall": doc["metrics"]["overall"],
            "per_class": {
                name: {
                    "recall": per_class[name]["recall"],
                    "ap50": per_class[name]["ap50"],
                    "ap50_95": per_class[name]["ap50_95"],
                    "precision": per_class[name]["precision"],
                }
                for name in CLASSES
            },
            "worst_class": doc["metrics"]["worst_class"]["class_name"],
        },
        "per_class": {
            name: {
                "real_images": REAL_IMAGES[name],
                "accepted_so_far": accepted_so_far[name],
                "remaining_capacity": MAX_ACCEPTED_PER_SOURCE * REAL_IMAGES[name] - accepted_so_far[name],
            }
            for name in CLASSES
        },
        "history": history,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(observation, ensure_ascii=False, indent=2), encoding="utf-8")
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
