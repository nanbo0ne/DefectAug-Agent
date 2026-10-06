"""Transparent baselines used before the LLM controller is connected."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ClassMetrics:
    recall: float
    ap50: float
    real_images: int

    def validate(self) -> None:
        if not 0 <= self.recall <= 1:
            raise ValueError(f"recall out of range: {self.recall}")
        if not 0 <= self.ap50 <= 1:
            raise ValueError(f"ap50 out of range: {self.ap50}")
        if self.real_images <= 0:
            raise ValueError("real_images must be positive")


def formula_scores(
    metrics: dict[str, ClassMetrics],
    recall_weight: float = 0.50,
    ap50_weight: float = 0.30,
    rarity_weight: float = 0.20,
) -> dict[str, float]:
    """Calculate the frozen one-shot weakness score from the protocol."""

    if not metrics:
        raise ValueError("metrics cannot be empty")
    if not math.isclose(recall_weight + ap50_weight + rarity_weight, 1.0, abs_tol=1e-9):
        raise ValueError("formula weights must sum to 1")
    max_images = max(item.real_images for item in metrics.values())
    scores: dict[str, float] = {}
    for class_name, item in metrics.items():
        item.validate()
        rarity = 1 - item.real_images / max_images
        scores[class_name] = (
            recall_weight * (1 - item.recall)
            + ap50_weight * (1 - item.ap50)
            + rarity_weight * rarity
        )
    return scores


def largest_remainder_with_bounds(
    scores: dict[str, float], total: int, minimum: int = 0, maximum_fraction: float = 1.0
) -> dict[str, int]:
    """Allocate an integer budget while preserving total, lower bounds and a cap."""

    if total < 0:
        raise ValueError("total must be non-negative")
    if not scores:
        raise ValueError("scores cannot be empty")
    if any(score < 0 for score in scores.values()):
        raise ValueError("scores must be non-negative")
    names = sorted(scores)
    if minimum * len(names) > total:
        raise ValueError("minimum allocation exceeds total")
    cap = max(minimum, math.floor(total * maximum_fraction))
    if cap * len(names) < total:
        raise ValueError("maximum_fraction makes total allocation impossible")

    allocation = {name: minimum for name in names}
    remaining = total - minimum * len(names)
    active = set(names)

    while remaining and active:
        weight_sum = sum(scores[name] for name in active)
        weights = {
            name: (scores[name] / weight_sum if weight_sum > 0 else 1 / len(active))
            for name in active
        }
        raw = {name: remaining * weights[name] for name in active}
        floors = {
            name: min(cap - allocation[name], math.floor(raw[name])) for name in active
        }
        progressed = sum(max(0, value) for value in floors.values())
        for name, value in floors.items():
            if value > 0:
                allocation[name] += value
                remaining -= value
        active = {name for name in active if allocation[name] < cap}
        if not remaining:
            break

        # Largest fractional remainder; ties are stable by class name.
        order = sorted(active, key=lambda name: (-(raw.get(name, 0) % 1), name))
        for name in order:
            if remaining == 0:
                break
            if allocation[name] < cap:
                allocation[name] += 1
                remaining -= 1
                progressed += 1
        if progressed == 0:
            raise RuntimeError("allocation stalled")

    if remaining:
        raise RuntimeError(f"could not allocate {remaining} budget units")
    if sum(allocation.values()) != total:
        raise AssertionError("allocation sum mismatch")
    return allocation


def formula_allocation(
    metrics: dict[str, ClassMetrics],
    total: int,
    minimum: int = 10,
    maximum_fraction: float = 0.70,
    recall_weight: float = 0.50,
    ap50_weight: float = 0.30,
    rarity_weight: float = 0.20,
) -> tuple[dict[str, float], dict[str, int]]:
    scores = formula_scores(
        metrics,
        recall_weight=recall_weight,
        ap50_weight=ap50_weight,
        rarity_weight=rarity_weight,
    )
    return scores, largest_remainder_with_bounds(scores, total, minimum, maximum_fraction)
