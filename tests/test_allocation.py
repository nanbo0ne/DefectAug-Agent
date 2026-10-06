import pytest

from defectaug.allocation import (
    BudgetProtocol,
    ConstraintViolation,
    headroom,
    project,
    source_capacity_from_state,
    uniform_proposal,
    validate,
)

MULTI = BudgetProtocol(total_budget=150, round_budget=50, rounds=3, max_fraction=0.70, final_min_per_class=10)
ONE = BudgetProtocol(total_budget=150, round_budget=150, rounds=1, max_fraction=0.70, final_min_per_class=10)
ZERO = {"inclusion": 0, "patches": 0, "scratches": 0}
FRESH = {"inclusion": 80, "patches": 240, "scratches": 480}


def test_cumulative_cap_is_105():
    assert MULTI.cumulative_cap == 105


def test_headroom_not_double_counted():
    # 30 already allocated to inclusion; generator can still take 50 more.
    room = headroom(MULTI, {"inclusion": 30, "patches": 10, "scratches": 10},
                    {"inclusion": 50, "patches": 230, "scratches": 470})
    # cap side 105-30=75, capacity side 50 -> 50 (the old code gave 80-2*30=20).
    assert room["inclusion"] == 50


def test_cumulative_cap_binds_across_rounds():
    allocated = {"inclusion": 90, "patches": 5, "scratches": 5}
    room = headroom(MULTI, allocated, {"inclusion": 400, "patches": 400, "scratches": 400})
    assert room["inclusion"] == 15
    with pytest.raises(ConstraintViolation):
        validate(MULTI, {"inclusion": 20, "patches": 15, "scratches": 15}, allocated,
                 {"inclusion": 400, "patches": 400, "scratches": 400}, round_index=2)


def test_valid_plan_passes_unchanged():
    plan = {"inclusion": 30, "patches": 10, "scratches": 10}
    assert validate(MULTI, plan, ZERO, FRESH, 1) == plan


@pytest.mark.parametrize("plan,fragment", [
    ({"inclusion": 30, "patches": 10, "scratches": 9}, "round sum"),
    ({"inclusion": -1, "patches": 26, "scratches": 25}, "negative"),
    ({"inclusion": 1.5, "patches": 24, "scratches": 25}, "not an integer"),
    ({"inclusion": True, "patches": 24, "scratches": 25}, "not an integer"),
    ({"inclusion": 25, "patches": 25}, "not an integer"),
    ("50,0,0", "not an object"),
])
def test_shape_and_sum_violations(plan, fragment):
    with pytest.raises(ConstraintViolation) as exc:
        validate(MULTI, plan, ZERO, FRESH, 1)
    assert fragment in str(exc.value)


def test_final_round_minimum():
    allocated = {"inclusion": 60, "patches": 40, "scratches": 0}
    with pytest.raises(ConstraintViolation) as exc:
        validate(MULTI, {"inclusion": 25, "patches": 25, "scratches": 0}, allocated, FRESH, 3)
    assert "below minimum" in str(exc.value)
    ok = {"inclusion": 20, "patches": 20, "scratches": 10}
    assert validate(MULTI, ok, allocated, FRESH, 3) == ok


def test_lookahead_rejects_dead_end():
    # After round 2 scratches would still be 0 but only 50 remain: fine.
    # A plan that leaves future headroom < future budget must be rejected.
    cap = {"inclusion": 60, "patches": 45, "scratches": 400}
    plan = {"inclusion": 0, "patches": 0, "scratches": 50}
    # remaining after r1: inclusion 60, patches 45, scratches 350 -> room ok
    assert validate(MULTI, plan, ZERO, cap, 1) == plan


def test_project_repairs_to_feasible_and_is_deterministic():
    bad = {"inclusion": 90, "patches": "x", "scratches": -3}
    a = project(MULTI, bad, ZERO, FRESH, 1)
    b = project(MULTI, bad, ZERO, FRESH, 1)
    assert a == b
    assert sum(a.values()) == 50
    validate(MULTI, a, ZERO, FRESH, 1)


def test_project_respects_final_minimum():
    allocated = {"inclusion": 60, "patches": 40, "scratches": 0}
    out = project(MULTI, {"inclusion": 50, "patches": 0, "scratches": 0}, allocated, FRESH, 3)
    assert allocated["scratches"] + out["scratches"] >= 10
    assert sum(out.values()) == 50


def test_project_keeps_valid_plan():
    plan = {"inclusion": 20, "patches": 15, "scratches": 15}
    assert project(MULTI, plan, ZERO, FRESH, 1) == plan


def test_one_shot_uses_same_rules():
    assert validate(ONE, {"inclusion": 50, "patches": 50, "scratches": 50}, ZERO, FRESH, 1)
    with pytest.raises(ConstraintViolation):
        validate(ONE, {"inclusion": 106, "patches": 34, "scratches": 10}, ZERO, FRESH, 1)
    with pytest.raises(ConstraintViolation):
        validate(ONE, {"inclusion": 80, "patches": 70, "scratches": 0}, ZERO, FRESH, 1)


def test_uniform_proposal():
    assert uniform_proposal(ONE) == {"inclusion": 50, "patches": 50, "scratches": 50}
    assert uniform_proposal(MULTI) == {"inclusion": 17, "patches": 17, "scratches": 16}


def test_source_capacity_counts_candidate_exhaustion():
    state = {"per_source": {
        "inclusion_1": {"accepted": 7, "candidates": 12},   # dead: 0
        "inclusion_2": {"accepted": 3, "candidates": 10},   # min(5, 2) = 2
        "inclusion_3": {"accepted": 0, "candidates": 0},    # 8
    }}
    cap = source_capacity_from_state(state, {"inclusion": ["inclusion_1", "inclusion_2", "inclusion_3"]}, 8, 12)
    assert cap == {"inclusion": 10}


def test_protocol_rejects_impossible_config():
    with pytest.raises(ValueError):
        BudgetProtocol(total_budget=150, round_budget=50, rounds=3, max_fraction=0.30, final_min_per_class=10)
