"""Create a transparent one-shot or per-round formula budget plan."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml

from defectaug.policy import ClassMetrics, formula_allocation


PROJECT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True, help="JSON from evaluate_yolo.py")
    parser.add_argument("--experiment", default="configs/experiment_v1.yaml")
    parser.add_argument("--output", default="reports/plans/formula_one_shot_v1.json")
    parser.add_argument(
        "--policy",
        choices=("formula_policy", "formula_multiround_policy"),
        default="formula_policy",
    )
    parser.add_argument(
        "--round",
        type=int,
        default=None,
        help="Required for formula_multiround_policy; validation metrics must be from that round.",
    )
    args = parser.parse_args()

    metrics_path = project_path(args.metrics)
    experiment_path = project_path(args.experiment)
    output_path = project_path(args.output)
    report = json.loads(metrics_path.read_text(encoding="utf-8"))
    experiment = yaml.safe_load(experiment_path.read_text(encoding="utf-8"))

    if report.get("evaluation", {}).get("split") != "val":
        raise ValueError("formula allocation must be derived from validation, never test")
    if report.get("evaluation", {}).get("test_access_authorized") is not False:
        raise ValueError("input report does not prove that test access was disabled")

    real_counts = experiment["dataset"]["real_fewshot_images"]
    policy = experiment[args.policy]
    if args.policy == "formula_multiround_policy":
        if args.round is None or not 1 <= args.round <= int(policy["rounds"]):
            raise ValueError("formula_multiround_policy requires --round within configured rounds")
        total = int(policy["accepted_budget_per_round"])
        minimum = int(policy["minimum_per_active_class"])
    else:
        if args.round is not None:
            raise ValueError("--round is only valid with formula_multiround_policy")
        total = int(experiment["generation"]["accepted_budget_total"])
        minimum = int(policy["minimum_per_class"])
    class_metrics: dict[str, ClassMetrics] = {}
    source_rows: dict[str, dict[str, float | int]] = {}
    for row in report["metrics"]["per_class"]:
        name = row["class_name"]
        if name not in real_counts:
            raise ValueError(f"validation report contains unknown class: {name}")
        if not row.get("evaluated") or row.get("recall") is None or row.get("ap50") is None:
            raise ValueError(f"validation metrics are incomplete for {name}")
        class_metrics[name] = ClassMetrics(
            recall=float(row["recall"]),
            ap50=float(row["ap50"]),
            real_images=int(real_counts[name]),
        )
        source_rows[name] = {
            "recall": float(row["recall"]),
            "ap50": float(row["ap50"]),
            "real_images": int(real_counts[name]),
        }
    if set(class_metrics) != set(real_counts):
        raise ValueError("validation report and configured classes do not match")

    scores, allocation = formula_allocation(
        class_metrics,
        total=total,
        minimum=minimum,
        maximum_fraction=float(policy["maximum_fraction_per_class"]),
        recall_weight=float(policy["recall_weight"]),
        ap50_weight=float(policy["ap50_weight"]),
        rarity_weight=float(policy["rarity_weight"]),
    )
    plan = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "pilot_plan_not_formal_frozen_result",
        "policy": args.policy,
        "round": args.round,
        "input_validation_report": {
            "path": str(metrics_path),
            "sha256": sha256(metrics_path),
            "split": "val",
            "selected_split_sha256": report["inputs"]["selected_split"]["sha256"],
            "weights_sha256": report["inputs"]["weights"]["sha256"],
        },
        "experiment_config": {"path": str(experiment_path), "sha256": sha256(experiment_path)},
        "formula": {
            "recall_weight": float(policy["recall_weight"]),
            "ap50_weight": float(policy["ap50_weight"]),
            "rarity_weight": float(policy["rarity_weight"]),
            "minimum_per_class": minimum,
            "maximum_fraction_per_class": float(policy["maximum_fraction_per_class"]),
            "allocation_method": policy["allocation"],
        },
        "accepted_budget_total": total,
        "source_metrics": source_rows,
        "weakness_scores": scores,
        "accepted_budget_by_class": allocation,
        "budget_sum_check": sum(allocation.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(plan, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
