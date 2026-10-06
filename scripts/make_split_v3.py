"""Build data/splits/v3 from v2.

train_fewshot, val, quarantine and train_fewshot_repeat_matched are carried
over unchanged.  The v2 test split was evaluated once (frozen 2026-10-01) and
is retired alongside the v1 test; neither is used again in any v3 role.  A new
locked test split is drawn from v2 `unused` with the same rule as v2: per
class, sorted -> seeded shuffle -> first N.

The draw seed is derived from a fixed label, so anyone rerunning this script
gets the same split.  No image content is read; only file lists.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
V2 = PROJECT / "data" / "splits" / "v2"
V3 = PROJECT / "data" / "splits" / "v3"
CLASSES = ("inclusion", "patches", "scratches")
TEST_PER_CLASS = 50
SEED_LABEL = "defectaug-v3|test-split"


def derive_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little") & 0x7FFFFFFF


def read(name: str) -> list[str]:
    return [l.strip() for l in (V2 / f"{name}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]


def cls(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[0]


def write(name: str, lines: list[str]) -> None:
    (V3 / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if V3.exists() and any(V3.iterdir()):
        raise SystemExit(f"{V3} already populated; refusing to overwrite a frozen split")
    V3.mkdir(parents=True, exist_ok=True)

    train, val, v2_test = read("train_fewshot"), read("val"), read("test")
    unused, quarantine = read("unused"), read("quarantine")
    retired_v1, repeat = read("retired_test_v1"), read("train_fewshot_repeat_matched")

    seed = derive_seed(SEED_LABEL)
    rng = random.Random(seed)
    new_test: list[str] = []
    new_unused: list[str] = []
    for c in CLASSES:
        pool = sorted(p for p in unused if cls(p) == c)
        rng.shuffle(pool)
        if len(pool) < TEST_PER_CLASS:
            raise SystemExit(f"not enough unused {c}: {len(pool)}")
        new_test += sorted(pool[:TEST_PER_CLASS])
        new_unused += sorted(pool[TEST_PER_CLASS:])

    sets = {
        "train_fewshot": train, "val": val, "test": new_test, "unused": new_unused,
        "quarantine": quarantine, "retired_test_v1": retired_v1, "retired_test_v2": v2_test,
    }
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            inter = set(sets[a]) & set(sets[b])
            if inter:
                raise SystemExit(f"overlap {a}/{b}: {len(inter)}")
    total = sum(len(v) for v in sets.values())
    if total != 900:
        raise SystemExit(f"expected 900 images, got {total}")

    for name, lines in sets.items():
        write(name, lines)
    write("train_fewshot_repeat_matched", repeat)

    summary = {
        "derived_from": "data/splits/v2",
        "test_draw_seed_label": SEED_LABEL,
        "test_draw_seed": seed,
        "rule": "per class: sorted(v2 unused) -> random.Random(seed).shuffle -> first 50",
        "carried_over_unchanged": ["train_fewshot", "val", "quarantine", "train_fewshot_repeat_matched",
                                   "retired_test_v1"],
        "retired": "v2 test (150, evaluated once 2026-10-01) -> retired_test_v2.txt, excluded from every v3 role",
        "counts": {k: dict(Counter(cls(p) for p in v)) | {"total": len(v)} for k, v in sets.items()},
        "file_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(V3.glob("*.txt"))
        },
    }
    (V3 / "split_v3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("test_draw_seed", "counts")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
