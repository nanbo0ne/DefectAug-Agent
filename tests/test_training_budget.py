import pytest

from defectaug.training_budget import budget_for_epochs, budget_for_target_steps


def test_equal_step_plan_for_current_protocol() -> None:
    augmented = budget_for_epochs(images=250, batch_size=16, epochs=100)
    real = budget_for_target_steps(images=100, batch_size=16, target_steps=augmented.optimizer_steps)
    assert augmented.steps_per_epoch == 16
    assert augmented.optimizer_steps == 1600
    assert real.steps_per_epoch == 7
    assert real.epochs == 229
    assert real.optimizer_steps == 1603


@pytest.mark.parametrize("value", [0, -1])
def test_training_budget_rejects_non_positive_values(value: int) -> None:
    with pytest.raises(ValueError):
        budget_for_epochs(images=value, batch_size=16, epochs=100)
