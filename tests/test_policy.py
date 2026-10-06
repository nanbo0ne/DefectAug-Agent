from defectaug.policy import ClassMetrics, formula_allocation


def test_formula_allocation_is_bounded_and_exact() -> None:
    metrics = {
        "inclusion": ClassMetrics(recall=0.30, ap50=0.40, real_images=10),
        "patches": ClassMetrics(recall=0.55, ap50=0.60, real_images=30),
        "scratches": ClassMetrics(recall=0.80, ap50=0.82, real_images=60),
    }
    scores, allocation = formula_allocation(metrics, total=150, minimum=10, maximum_fraction=0.70)
    assert sum(allocation.values()) == 150
    assert all(value >= 10 for value in allocation.values())
    assert allocation["inclusion"] > allocation["scratches"]
    assert scores["inclusion"] > scores["scratches"]

