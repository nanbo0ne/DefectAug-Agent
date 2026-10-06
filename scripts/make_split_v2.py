"""Build data/splits/v2 from v1.

train_fewshot and val are carried over unchanged.  The v1 test split is
retired (never used again in any role) and a new locked test split is drawn
from v1 `unused` with the same rule prepare_dataset.py used in v1: per class,
seeded shuffle, take the first N.  Quarantine is carried over.

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
V1 = PROJECT / "data" / "splits" / "v1"
V2 = PROJECT / "data" / "splits" / "v2"
CLASSES = ("inclusion", "patches", "scratches")
TEST_PER_CLASS = 50
SEED_LABEL = "defectaug-v2|test-split"


def derive_seed(label: str) -> int:
    return int.from_bytes(hashlib.sha256(label.encode("utf-8")).digest()[:4], "little") & 0x7FFFFFFF


def read(name: str) -> list[str]:
    return [l.strip() for l in (V1 / f"{name}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]


def cls(path: str) -> str:
    return Path(path).stem.rsplit("_", 1)[0]


def write(name: str, lines: list[str]) -> None:
    (V2 / f"{name}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    if V2.exists() and any(V2.iterdir()):
        raise SystemExit(f"{V2} already populated; refusing to overwrite a frozen split")
    V2.mkdir(parents=True, exist_ok=True)

    train, val, old_test = read("train_fewshot"), read("val"), read("test")
    unused, quarantine = read("unused"), read("quarantine")
    repeat = read("train_fewshot_repeat_matched")

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
        "train_fewshot": train, "val": val, "test": new_test,
        "unused": new_unused, "quarantine": quarantine, "retired_test_v1": old_test,
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
        "derived_from": "data/splits/v1",
        "test_draw_seed_label": SEED_LABEL,
        "test_draw_seed": seed,
        "rule": "per class: sorted(v1 unused) -> random.Random(seed).shuffle -> first 50",
        "carried_over_unchanged": ["train_fewshot", "val", "quarantine", "train_fewshot_repeat_matched"],
        "retired": "v1 test (150) -> retired_test_v1.txt, excluded from every v2 role",
        "counts": {k: dict(Counter(cls(p) for p in v)) | {"total": len(v)} for k, v in sets.items()},
        "file_sha256": {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(V2.glob("*.txt"))
        },
    }
    (V2 / "split_v2_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("test_draw_seed", "counts")}, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
