"""Calibrate the inside-mask change gate on train-only candidates (CPU).

Replays the frozen patch_copy_gauss5 candidate recipe with a calibration-only
RNG stream (independent of every formal seed) and records, per class, the
distribution of inside_mask_changed_fraction and mean_absolute_change_inside.
Sources and targets are both drawn from train_fewshot; val/test are never read.

Output: reports/calibration/inside_change_v2.{json,md}
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
import run_synth_generation as g  # noqa: E402
from defectaug.patch_copy import (  # noqa: E402
    copy_real_defect,
    gaussian_feather_boundary_ring,
    read_binary_mask,
    read_rgb,
    read_yolo_boxes,
)

CAL_STREAM = "calibration-inside-change-v2"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=200)
    ap.add_argument("--out", default="reports/calibration/inside_change_v2")
    args = ap.parse_args()

    by_class = g.resolve_train_sources()
    rows = []
    for cname in g.CLASSES:
        sources = by_class[cname]
        for k in range(args.per_class):
            seed = int.from_bytes(hashlib.sha256(f"{CAL_STREAM}|{cname}|{k}".encode()).digest()[:8], "little")
            rng = np.random.default_rng(seed)
            source = sources[k % len(sources)]
            smask = read_binary_mask(g.CANON_MASKS / f"{source.stem}.png")
            ms = g.feasible_max_scale(smask, read_rgb(sources[0]).shape[:2])
            if ms < g.MIN_SCALE:
                rows.append({"class": cname, "status": "no_feasible_scale"})
                continue
            scale = float(rng.uniform(g.MIN_SCALE, ms))
            rot = int(rng.choice([0, 90, 180, 270]))
            hf, vf = bool(rng.integers(0, 2)), bool(rng.integers(0, 2))
            target = sources[int(rng.integers(0, len(sources)))]
            try:
                r = copy_real_defect(
                    source_image=read_rgb(source), source_mask=smask,
                    target_image=read_rgb(target),
                    target_boxes=read_yolo_boxes(g.CANON_LABELS / f"{target.stem}.txt"),
                    class_id=g.CLASS_ID[cname], rng=rng, scale=scale, rotation_degrees=rot,
                    horizontal_flip=hf, vertical_flip=vf, max_overlap_fraction=0.15,
                    placement_attempts=1024,
                )
                gaussian_feather_boundary_ring(
                    baseline_image=r.image, target_image=read_rgb(target),
                    inserted_mask=r.inserted_mask, ring_radius_pixels=g.RING_RADIUS,
                )
            except (ValueError, RuntimeError) as exc:
                rows.append({"class": cname, "status": "rejected", "reason": str(exc)[:80]})
                continue
            rows.append({
                "class": cname, "status": "ok",
                "inside_frac": r.metrics.inside_mask_changed_fraction,
                "inside_mae": r.metrics.mean_absolute_change_inside,
                "pixels": r.metrics.inserted_mask_pixels,
            })

    def q(a, p):
        return float(np.percentile(a, p)) if len(a) else float("nan")

    summary = {}
    for cname in g.CLASSES:
        ok = [x for x in rows if x["class"] == cname and x["status"] == "ok"]
        fr = np.array([x["inside_frac"] for x in ok])
        ma = np.array([x["inside_mae"] for x in ok])
        summary[cname] = {
            "n_ok": len(ok),
            "n_rejected": sum(1 for x in rows if x["class"] == cname and x["status"] != "ok"),
            "inside_frac": {p: round(q(fr, p), 4) for p in (0, 1, 5, 10, 25, 50)},
            "inside_mae": {p: round(q(ma, p), 3) for p in (0, 1, 5, 10, 25, 50)},
            "frac_below": {t: int((fr < t).sum()) for t in (0.05, 0.10, 0.20, 0.30)},
        }
    out = PROJECT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
