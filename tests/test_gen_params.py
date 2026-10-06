import pytest

from defectaug.gen_params import (
    DEFAULT, MIN_SCALE_SPAN, SCALE_HI, SCALE_LO, ParamViolation, clamp_params, formula_params,
    validate_params,
)

GOOD = {"scale_min": 0.4, "scale_max": 0.8, "target_pool": "any_class",
        "source_weighting": "prefer_small", "max_overlap": 0.1}


def test_valid_params_pass():
    p = validate_params("x", GOOD)
    assert p.scale_min == 0.4 and p.target_pool == "any_class"


@pytest.mark.parametrize("patch", [
    {"scale_min": 0.1},                     # below SCALE_LO
    {"scale_max": 2.0},                     # above SCALE_HI
    {"scale_min": 0.7, "scale_max": 0.8},   # span < MIN_SCALE_SPAN
    {"target_pool": "val_images"},
    {"source_weighting": "random"},
    {"max_overlap": 0.9},
])
def test_out_of_space_rejected(patch):
    with pytest.raises(ParamViolation):
        validate_params("x", {**GOOD, **patch})


def test_missing_field_rejected():
    raw = dict(GOOD)
    del raw["max_overlap"]
    with pytest.raises(ParamViolation):
        validate_params("x", raw)


def test_clamp_always_feasible():
    p = clamp_params({"scale_min": -5, "scale_max": 9, "target_pool": "?", "max_overlap": 3})
    assert SCALE_LO <= p.scale_min < p.scale_max <= SCALE_HI
    assert p.scale_max - p.scale_min >= MIN_SCALE_SPAN - 1e-9
    assert p.target_pool == DEFAULT.target_pool


def test_formula_rule_small_misses():
    p = formula_params("inclusion", recall=0.3, median_train_px=90.0, median_missed_px=34.0)
    assert p.source_weighting == "prefer_small"
    assert p.target_pool == "any_class"
    assert p.scale_min == SCALE_LO


def test_formula_rule_no_misses_is_neutral():
    p = formula_params("patches", recall=0.9, median_train_px=80.0, median_missed_px=None)
    assert p.source_weighting == "uniform" and p.target_pool == "same_class"
    assert p.scale_min == pytest.approx(0.75) and p.scale_max == pytest.approx(1.25)
