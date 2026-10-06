"""Write the frozen equal-step control plan for Real-only versus synthetic runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import yaml

from defectaug.training_budget import budget_for_epochs, budget_for_target_steps


PROJECT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="configs/experiment_v1.yaml")
    parser.add_argument("--output", default="reports/plans/training_budget_v1.json")
    args = parser.parse_args()

    experiment_path = (PROJECT / args.experiment).resolve()
    output_path = (PROJECT / args.output).resolve()
    config = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))
    real_images = sum(int(value) for value in config["dataset"]["real_fewshot_images"].values())
    synthetic_images = int(config["generation"]["accepted_budget_total"])
    augmented_images = real_images + synthetic_images
    batch = int(config["detector"]["batch"])
    final_epochs = int(config["detector"]["final_epochs"])

    augmented = budget_for_epochs(augmented_images, batch, final_epochs)
    real_same_epochs = budget_for_epochs(real_images, batch, final_epochs)
    real_equal_steps = budget_for_target_steps(real_images, batch, augmented.optimizer_steps)
    report = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "planned_control_not_executed",
        "fairness_definition": (
            "V2/Uniform/Formula-adaptive/Agent use the same image count and epochs. "
            "H1 additionally compares synthetic augmentation against Real-only trained "
            "for approximately the same optimizer-step budget."
        ),
        "real_images": real_images,
        "accepted_synthetic_images": synthetic_images,
        "augmented_images": augmented_images,
        "augmented_main_run": augmented.as_dict(),
        "real_same_epochs_reference": real_same_epochs.as_dict(),
        "real_equal_steps_control": real_equal_steps.as_dict(),
        "equal_step_overshoot": real_equal_steps.optimizer_steps - augmented.optimizer_steps,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
