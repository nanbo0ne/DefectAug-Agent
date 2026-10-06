"""v3 shared generation-parameter action space.

In v2 every augmented condition could only choose a per-class image quota.
v3 widens the action to *how* each class is generated.  The agent and the
formula baselines propose the same structure and both go through
`validate_params`; out-of-range values are rejected (agent) or clamped by the
deterministic rule (formula), so neither side can do anything the other
cannot.

Per-class parameters:

- scale_min / scale_max : uniform paste-scale range, inside [SCALE_LO, SCALE_HI]
                          and scale_max - scale_min >= MIN_SCALE_SPAN;
- target_pool           : "same_class" (background drawn from the class's own
                          train images) or "any_class" (all 100 train images);
- source_weighting      : "uniform" | "prefer_small" | "prefer_large"
                          (sampling weight of each source instance by its
                          largest-component size);
- max_overlap           : max fraction of inserted-mask pixels allowed over
                          existing boxes, inside [OVERLAP_LO, OVERLAP_HI].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

SCALE_LO = 0.30
SCALE_HI = 1.40
MIN_SCALE_SPAN = 0.20
OVERLAP_LO = 0.00
OVERLAP_HI = 0.30
TARGET_POOLS = ("same_class", "any_class")
SOURCE_WEIGHTINGS = ("uniform", "prefer_small", "prefer_large")


class ParamViolation(ValueError):
    """A proposed generation parameter is outside the frozen action space."""


@dataclass(frozen=True)
class ClassGenParams:
    scale_min: float
    scale_max: float
    target_pool: str
    source_weighting: str
    max_overlap: float

    def as_dict(self) -> dict:
        return asdict(self)


# v2-equivalent defaults (scale 0.6-1.4, same-class background, uniform
# round-robin sources, overlap 0.15).  uniform_fixed uses these unchanged.
DEFAULT = ClassGenParams(0.60, 1.40, "same_class", "uniform", 0.15)


def validate_params(name: str, raw: Mapping) -> ClassGenParams:
    """Strict check used for agent proposals: no silent repair."""
    try:
        smin = float(raw["scale_min"])
        smax = float(raw["scale_max"])
        pool = str(raw["target_pool"])
        weighting = str(raw["source_weighting"])
        overlap = float(raw["max_overlap"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ParamViolation(f"{name}: missing/invalid field ({exc})") from exc
    if not SCALE_LO <= smin < smax <= SCALE_HI:
        raise ParamViolation(
            f"{name}: scale range [{smin}, {smax}] must satisfy {SCALE_LO} <= min < max <= {SCALE_HI}"
        )
    if smax - smin < MIN_SCALE_SPAN - 1e-9:
        raise ParamViolation(f"{name}: scale span {smax - smin:.3f} < {MIN_SCALE_SPAN}")
    if pool not in TARGET_POOLS:
        raise ParamViolation(f"{name}: target_pool {pool!r} not in {TARGET_POOLS}")
    if weighting not in SOURCE_WEIGHTINGS:
        raise ParamViolation(f"{name}: source_weighting {weighting!r} not in {SOURCE_WEIGHTINGS}")
    if not OVERLAP_LO <= overlap <= OVERLAP_HI:
        raise ParamViolation(f"{name}: max_overlap {overlap} outside [{OVERLAP_LO}, {OVERLAP_HI}]")
    return ClassGenParams(round(smin, 3), round(smax, 3), pool, weighting, round(overlap, 3))


def clamp_params(raw: Mapping) -> ClassGenParams:
    """Deterministic repair used by the formula rules (always feasible)."""
    smin = min(max(float(raw.get("scale_min", DEFAULT.scale_min)), SCALE_LO), SCALE_HI - MIN_SCALE_SPAN)
    smax = min(max(float(raw.get("scale_max", DEFAULT.scale_max)), smin + MIN_SCALE_SPAN), SCALE_HI)
    pool = raw.get("target_pool", DEFAULT.target_pool)
    weighting = raw.get("source_weighting", DEFAULT.source_weighting)
    overlap = min(max(float(raw.get("max_overlap", DEFAULT.max_overlap)), OVERLAP_LO), OVERLAP_HI)
    return validate_params(
        "clamp",
        {
            "scale_min": smin,
            "scale_max": smax,
            "target_pool": pool if pool in TARGET_POOLS else DEFAULT.target_pool,
            "source_weighting": weighting if weighting in SOURCE_WEIGHTINGS else DEFAULT.source_weighting,
            "max_overlap": overlap,
        },
    )


def formula_params(
    class_name: str,
    *,
    recall: float,
    median_train_px: float,
    median_missed_px: float | None,
) -> ClassGenParams:
    """Fixed, transparent rule mapping val feedback to generation parameters.

    - scale: centre the range on the ratio median_missed_px / median_train_px
      (falls back to 1.0 when nothing was missed), +-0.25, clamped;
    - target_pool: any_class when recall < 0.5 (more background diversity);
    - source_weighting: prefer_small when missed objects are smaller than the
      train median, prefer_large when larger, else uniform;
    - max_overlap: 0.15 (v2 value) — the rule never tunes it.
    """
    ratio = 1.0 if not median_missed_px or median_train_px <= 0 else median_missed_px / median_train_px
    if ratio < 0.85:
        weighting = "prefer_small"
    elif ratio > 1.15:
        weighting = "prefer_large"
    else:
        weighting = "uniform"
    return clamp_params(
        {
            "scale_min": ratio - 0.25,
            "scale_max": ratio + 0.25,
            "target_pool": "any_class" if recall < 0.5 else "same_class",
            "source_weighting": weighting,
            "max_overlap": 0.15,
        }
    )


def params_from_dict(raw: Mapping) -> ClassGenParams:
    return validate_params("load", raw)


__all__ = [
    "ClassGenParams",
    "DEFAULT",
    "MIN_SCALE_SPAN",
    "OVERLAP_HI",
    "OVERLAP_LO",
    "ParamViolation",
    "SCALE_HI",
    "SCALE_LO",
    "SOURCE_WEIGHTINGS",
    "TARGET_POOLS",
    "clamp_params",
    "formula_params",
    "params_from_dict",
    "validate_params",
]
