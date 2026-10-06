"""Transparent optimizer-step accounting for unequal dataset sizes."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TrainingBudget:
    images: int
    batch_size: int
    steps_per_epoch: int
    epochs: int
    optimizer_steps: int

    def as_dict(self) -> dict[str, int]:
        return asdict(self)


def budget_for_epochs(images: int, batch_size: int, epochs: int) -> TrainingBudget:
    if images <= 0 or batch_size <= 0 or epochs <= 0:
        raise ValueError("images, batch_size and epochs must be positive")
    steps_per_epoch = math.ceil(images / batch_size)
    return TrainingBudget(
        images=images,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        optimizer_steps=steps_per_epoch * epochs,
    )


def budget_for_target_steps(images: int, batch_size: int, target_steps: int) -> TrainingBudget:
    if target_steps <= 0:
        raise ValueError("target_steps must be positive")
    steps_per_epoch = math.ceil(images / batch_size)
    epochs = math.ceil(target_steps / steps_per_epoch)
    return budget_for_epochs(images, batch_size, epochs)
