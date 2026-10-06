"""Deterministic per-class synthetic-data generation for formal conditions.

Tool: patch_copy_gauss5 = real-defect mask-guided Copy-Paste
(defectaug.patch_copy.copy_real_defect) + ring-5 Gaussian boundary feather
(gaussian_feather_boundary_ring).  Labels are derived automatically from the
transformed real mask, so the loop needs no human annotation or review.

The script is cumulative across rounds: for a multi-round condition it keeps
a persistent per-source state (accepted/candidate counters, round-robin
cursor) inside the pool directory so that the global caps
(max_candidates_per_source=12, max_accepted_derivatives_per_source=8) hold
across all rounds of one (condition, seed) pair.

Outputs per (condition, seed, round):
    data/synthetic/{condition}_{seed}/r{round}/images|labels|masks/
    data/synthetic/{condition}_{seed}/state.json
    data/synthetic/{condition}_{seed}/ledger.jsonl
    data/synthetic/{condition}_{seed}/train_manifest_r{round}.txt

A shared companion root (data/synthetic/lock_support/{labels,masks}) holds
copies of every canonical and synthetic label/mask so that a single
labels_root/masks_root pair resolves every stem when a dataset content lock
is built for the mixed real+synthetic manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

from defectaug.gen_params import DEFAULT, ClassGenParams, params_from_dict
from defectaug.patch_copy import (
    copy_real_defect,
    gaussian_feather_boundary_ring,
    read_binary_mask,
    read_rgb,
    read_yolo_boxes,
)

PROJECT = Path(__file__).resolve().parents[1]
CLASSES = ("inclusion", "patches", "scratches")
CLASS_ID = {name: index for index, name in enumerate(CLASSES)}
COND_SHORT = {
    "uniform_fixed": "u",
    "formula_one_shot": "f1",
    "formula_multi_round": "fm",
    "agent_multi_round": "ag",
    "formula_tuned_multi_round": "ft",
}
CANON_IMAGES = PROJECT / "data" / "canonical" / "images" / "all"
CANON_LABELS = PROJECT / "data" / "canonical" / "labels" / "all"
CANON_MASKS = PROJECT / "data" / "canonical" / "masks" / "all"
REAL_MANIFEST = PROJECT / "data" / "splits" / "v3" / "train_fewshot.txt"

# v3: raised from 12/8 so inclusion (10 sources) can reach the 105 cumulative cap.
MAX_CANDIDATES_PER_SOURCE = 36
MAX_ACCEPTED_PER_SOURCE = 24
# Hard gate: at least half of the inserted-mask pixels must differ from the
# target by >= 3 grey levels.  Train-only calibration (reports/calibration/
# inside_change_v2.json, 522 valid candidates) gave a minimum of 0.778, so this
# only rejects degenerate "generated but unchanged" candidates.
INSIDE_CHANGE_MIN = 0.50
RING_RADIUS = 5
CROP_MARGIN = 5
MIN_SCALE = 0.6
MAX_SCALE = 1.4


def feasible_max_scale(source_mask: np.ndarray, target_hw: tuple[int, int]) -> float:
    """Largest scale whose transformed largest-component patch still fits.

    Worst case across the {0,90,180,270} rotations is max(h, w) along the
    shorter target side, so the bound uses the longer crop side.
    """

    binary = source_mask.astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return 0.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    component = stats[1 + int(np.argmax(areas))]
    comp_h = int(component[cv2.CC_STAT_HEIGHT]) + 2 * CROP_MARGIN
    comp_w = int(component[cv2.CC_STAT_WIDTH]) + 2 * CROP_MARGIN
    limit = min(target_hw) - 2
    long_side = max(comp_h, comp_w)
    if long_side <= 0:
        return 0.0
    return max(0.0, min(MAX_SCALE, limit / long_side * 0.95))


def _largest_component_pixels(mask: np.ndarray) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return int(stats[1:, cv2.CC_STAT_AREA].max()) if count > 1 else 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seed_for(condition: str, seed: int, round_index: int, attempt: int) -> int:
    raw = f"{condition}|{seed}|r{round_index}|a{attempt}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little")


def _load_json(path: Path, default):
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def _write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_train_sources(manifest: Path | None = None) -> dict[str, list[Path]]:
    manifest = manifest or REAL_MANIFEST
    paths = [
        Path(raw.strip())
        for raw in manifest.read_text(encoding="utf-8").splitlines()
        if raw.strip()
    ]
    if len(paths) != 100:
        raise RuntimeError(f"expected 100 frozen train images, got {len(paths)}")
    by_class: dict[str, list[Path]] = {name: [] for name in CLASSES}
    for path in paths:
        class_name = path.stem.rsplit("_", 1)[0]
        if class_name not in by_class:
            raise RuntimeError(f"unexpected class in train manifest: {path}")
        by_class[class_name].append(path)
    for name in CLASSES:
        by_class[name].sort(key=lambda p: p.stem)
    return by_class


def populate_lock_support() -> tuple[Path, Path]:
    labels_root = PROJECT / "data" / "synthetic" / "lock_support" / "labels"
    masks_root = PROJECT / "data" / "synthetic" / "lock_support" / "masks"
    labels_root.mkdir(parents=True, exist_ok=True)
    masks_root.mkdir(parents=True, exist_ok=True)
    for src in CANON_LABELS.glob("*.txt"):
        dst = labels_root / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
    for src in CANON_MASKS.glob("*.*"):
        dst = masks_root / src.name
        if not dst.exists():
            shutil.copyfile(src, dst)
    return labels_root, masks_root


def _reject_key(message: str) -> str:
    """Coarse reject category for the per-class quality receipt."""
    m = message.lower()
    if "overlap" in m or "placement" in m:
        return "no_placement"
    if "inside-mask change" in m:
        return "inside_change_below_gate"
    if "outside" in m or "beyond the allowed ring" in m or "core pixels" in m:
        return "invariant_violation"
    if "too small" in m or "component" in m or "pixels" in m:
        return "mask_too_small"
    return "other"


def parse_alloc(text: str) -> dict[str, int]:
    alloc: dict[str, int] = {name: 0 for name in CLASSES}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        name, _, value = part.partition(":")
        name = name.strip()
        if name not in alloc:
            raise ValueError(f"unknown class in --alloc: {name!r}")
        alloc[name] = int(value)
    return alloc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=sorted(COND_SHORT))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--alloc", required=True, help="e.g. inclusion:20,patches:15,scratches:15")
    parser.add_argument(
        "--candidate-budget",
        type=int,
        default=150,
        help="Max generation attempts this round (150/round for multi-round; 450 for one-shot conditions).",
    )
    parser.add_argument(
        "--max-overlap",
        type=float,
        default=0.15,
        help="Max fraction of inserted-mask pixels allowed over existing boxes.",
    )
    parser.add_argument(
        "--params",
        default=None,
        help="JSON file mapping class name -> generation params (v3 action space). "
        "Classes not listed use DEFAULT.",
    )
    parser.add_argument("--max-accepted-per-source", type=int, default=MAX_ACCEPTED_PER_SOURCE)
    parser.add_argument("--max-candidates-per-source", type=int, default=MAX_CANDIDATES_PER_SOURCE)
    parser.add_argument("--emit-manifest", action="store_true")
    parser.add_argument("--train-manifest", default=str(REAL_MANIFEST))
    parser.add_argument("--pool-root", default=str(PROJECT / "data" / "synthetic"))
    args = parser.parse_args()
    if args.round < 1:
        raise ValueError("--round must be >= 1")

    alloc = parse_alloc(args.alloc)
    if sum(alloc.values()) <= 0:
        raise ValueError("allocation must be positive")

    # Per-class generation params; validated so out-of-range actions fail loudly.
    class_params: dict[str, ClassGenParams] = {}
    if args.params:
        raw_params = json.loads(Path(args.params).read_text(encoding="utf-8"))
        unknown = set(raw_params) - set(CLASSES)
        if unknown:
            raise ValueError(f"unknown classes in --params: {sorted(unknown)}")
        class_params = {name: params_from_dict(raw) for name, raw in raw_params.items()}

    real_manifest = Path(args.train_manifest)
    by_class = resolve_train_sources(real_manifest)
    all_train = sorted((p for v in by_class.values() for p in v), key=lambda p: p.stem)
    # Largest-component pixel count per source, used by source_weighting.
    source_size = {p.stem: _largest_component_pixels(read_binary_mask(CANON_MASKS / f"{p.stem}.png"))
                   for p in all_train}
    labels_root, masks_root = populate_lock_support()

    pool = Path(args.pool_root) / f"{args.condition}_{args.seed}"
    round_dir = pool / f"r{args.round}"
    for sub in ("images", "labels", "masks"):
        (round_dir / sub).mkdir(parents=True, exist_ok=True)
    # A round may be generated in several calls (top-up after a shortfall).
    # Continue numbering after what is already on disk so nothing is overwritten.
    short = COND_SHORT[args.condition]
    next_index = {
        name: len(list((round_dir / "images").glob(f"{name}_{short}{args.seed}_r{args.round}_*.png")))
        for name in CLASSES
    }
    receipts: dict[str, dict] = {
        name: {"requested": alloc[name], "candidates": 0, "accepted": 0, "rejects": {},
               "params": class_params.get(name, DEFAULT).as_dict()}
        for name in CLASSES
    }
    state_path = pool / "state.json"
    ledger_path = pool / "ledger.jsonl"
    state = _load_json(
        state_path,
        {
            "condition": args.condition,
            "seed": args.seed,
            "per_source": {},
            "cursors": {name: 0 for name in CLASSES},
            "total_candidates": 0,
            "total_accepted": 0,
        },
    )
    per_source: dict[str, dict[str, int]] = state["per_source"]
    cursors: dict[str, int] = state["cursors"]

    round_candidates = 0
    round_accepted = 0
    shortfalls: dict[str, int] = {}
    ledger = ledger_path.open("a", encoding="utf-8")

    for class_name in CLASSES:
        needed = alloc[class_name]
        if needed <= 0:
            continue
        sources = by_class[class_name]
        cp = class_params.get(class_name, DEFAULT)
        # v3 target_pool: background from the class's own train images or from all 100.
        targets = sources if cp.target_pool == "same_class" else all_train
        produced = 0

        def _exhausted(s: Path) -> bool:
            st = per_source.get(s.stem, {})
            return (st.get("accepted", 0) >= args.max_accepted_per_source
                    or st.get("candidates", 0) >= args.max_candidates_per_source)

        while produced < needed and round_candidates < args.candidate_budget:
            attempt = state["total_candidates"]
            rng = np.random.default_rng(_seed_for(args.condition, args.seed, args.round, attempt))
            if cp.source_weighting == "uniform":
                # v2 behaviour: deterministic round-robin over sources.
                idx = cursors[class_name] % len(sources)
                source = sources[idx]
                cursors[class_name] += 1
                if _exhausted(source):
                    if all(_exhausted(s) for s in sources):
                        break
                    continue
            else:
                # v3 source_weighting: sample live sources by largest-component size.
                live = [s for s in sources if not _exhausted(s)]
                if not live:
                    break
                size = np.array([source_size[s.stem] for s in live], dtype=np.float64)
                size = size / np.median(size)
                w = size ** (-1.0 if cp.source_weighting == "prefer_small" else 1.0)
                source = live[int(rng.choice(len(live), p=w / w.sum()))]
            stat = per_source.setdefault(source.stem, {"accepted": 0, "candidates": 0})
            source_mask = read_binary_mask(CANON_MASKS / f"{source.stem}.png")
            # v3: the class's agent/formula-chosen scale range, still capped by
            # the geometric feasibility limit of this source mask.
            max_scale = min(
                cp.scale_max, feasible_max_scale(source_mask, read_rgb(sources[0]).shape[:2])
            )
            if max_scale < cp.scale_min:
                record_skip = {
                    "condition": args.condition, "seed": args.seed, "round": args.round,
                    "class": class_name, "candidate_index": attempt, "source": source.stem,
                    "accepted": False, "reject_reason": f"no feasible scale (max={max_scale:.3f})",
                }
                stat["candidates"] += 1
                state["total_candidates"] += 1
                round_candidates += 1
                receipts[class_name]["candidates"] += 1
                receipts[class_name]["rejects"]["no_feasible_scale"] = (
                    receipts[class_name]["rejects"].get("no_feasible_scale", 0) + 1
                )
                ledger.write(json.dumps(record_skip, ensure_ascii=False) + "\n")
                continue
            scale = float(rng.uniform(cp.scale_min, max_scale))
            rotation = int(rng.choice([0, 90, 180, 270]))
            hflip = bool(rng.integers(0, 2))
            vflip = bool(rng.integers(0, 2))
            target = targets[int(rng.integers(0, len(targets)))]

            record: dict[str, object] = {
                "condition": args.condition,
                "seed": args.seed,
                "round": args.round,
                "class": class_name,
                "candidate_index": attempt,
                "source": source.stem,
                "target": target.stem,
                "scale": round(scale, 4),
                "rotation": rotation,
                "hflip": hflip,
                "vflip": vflip,
            }
            stat["candidates"] += 1
            state["total_candidates"] += 1
            round_candidates += 1
            receipts[class_name]["candidates"] += 1
            try:
                result = copy_real_defect(
                    source_image=read_rgb(source),
                    source_mask=source_mask,
                    target_image=read_rgb(target),
                    target_boxes=read_yolo_boxes(CANON_LABELS / f"{target.stem}.txt"),
                    class_id=CLASS_ID[class_name],
                    rng=rng,
                    scale=scale,
                    rotation_degrees=rotation,
                    horizontal_flip=hflip,
                    vertical_flip=vflip,
                    max_overlap_fraction=cp.max_overlap,
                    placement_attempts=1024,
                )
                blended = gaussian_feather_boundary_ring(
                    baseline_image=result.image,
                    target_image=read_rgb(target),
                    inserted_mask=result.inserted_mask,
                    ring_radius_pixels=RING_RADIUS,
                )
                if result.metrics.inside_mask_changed_fraction < INSIDE_CHANGE_MIN:
                    raise ValueError(
                        f"inside-mask change {result.metrics.inside_mask_changed_fraction:.3f} "
                        f"< {INSIDE_CHANGE_MIN}"
                    )
            except (ValueError, RuntimeError) as exc:
                record["accepted"] = False
                record["reject_reason"] = str(exc)[:300]
                key = _reject_key(str(exc))
                receipts[class_name]["rejects"][key] = receipts[class_name]["rejects"].get(key, 0) + 1
                ledger.write(json.dumps(record, ensure_ascii=False) + "\n")
                continue

            stem = f"{class_name}_{short}{args.seed}_r{args.round}_{next_index[class_name]:03d}"
            next_index[class_name] += 1
            image_path = round_dir / "images" / f"{stem}.png"
            label_path = round_dir / "labels" / f"{stem}.txt"
            mask_path = round_dir / "masks" / f"{stem}.png"
            Image.fromarray(blended.image).save(image_path, format="PNG")
            target_mask = read_binary_mask(CANON_MASKS / f"{target.stem}.png")
            union_mask = target_mask | result.inserted_mask
            Image.fromarray(union_mask.astype(np.uint8) * 255).save(mask_path, format="PNG")
            boxes = read_yolo_boxes(CANON_LABELS / f"{target.stem}.txt")
            label_path.write_text(
                "\n".join(box.line() for box in boxes + [result.inserted_box]) + "\n",
                encoding="utf-8",
            )
            # Shared companion root for content-lock stem resolution.
            shutil.copyfile(label_path, labels_root / label_path.name)
            shutil.copyfile(mask_path, masks_root / mask_path.name)

            stat["accepted"] += 1
            state["total_accepted"] += 1
            produced += 1
            round_accepted += 1
            receipts[class_name]["accepted"] += 1
            record.update(
                {
                    "accepted": True,
                    "stem": stem,
                    "image_sha256": sha256_file(image_path),
                    "inserted_mask_pixels": result.metrics.inserted_mask_pixels,
                    "outside_mask_changed_pixels": result.metrics.outside_mask_changed_pixels,
                    "seam_score_before": round(blended.metrics.seam_score_before, 4),
                    "seam_score_after": round(blended.metrics.seam_score_after, 4),
                    "seam_score_reduction": round(
                        blended.metrics.seam_score_reduction_fraction, 4
                    ),
                }
            )
            ledger.write(json.dumps(record, ensure_ascii=False) + "\n")

        if produced < needed:
            shortfalls[class_name] = needed - produced

    ledger.close()
    state["cursors"] = cursors
    _write_json(state_path, state)

    if args.emit_manifest:
        synth_paths = sorted(pool.glob("r*/images/*.png"))
        manifest_path = pool / f"train_manifest_r{args.round}.txt"
        real_lines = [
            raw.strip()
            for raw in real_manifest.read_text(encoding="utf-8").splitlines()
            if raw.strip()
        ]
        manifest_path.write_text(
            "\n".join(real_lines + [str(p) for p in synth_paths]) + "\n",
            encoding="utf-8",
        )
        print(f"manifest: {manifest_path} ({len(real_lines) + len(synth_paths)} rows)")

    summary = {
        "condition": args.condition,
        "seed": args.seed,
        "round": args.round,
        "requested": alloc,
        "round_candidates": round_candidates,
        "round_accepted": round_accepted,
        "shortfalls": shortfalls,
        "receipts": receipts,
        "total_candidates": state["total_candidates"],
        "total_accepted": state["total_accepted"],
    }
    # Per-call quality receipt; the agent observation aggregates these.
    with (pool / "receipts.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(
            {"round": args.round, "receipts": receipts, "shortfalls": shortfalls},
            ensure_ascii=False,
        ) + "\n")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # rc 2 = shortfall.  The caller must top up or fail; it may not ignore it.
    return 0 if not shortfalls else 2


if __name__ == "__main__":
    raise SystemExit(main())
