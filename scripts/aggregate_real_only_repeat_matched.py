"""Aggregate the three formal Real-only-repeat-matched validation reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (20260820, 20260821, 20260822)
CLASSES = ("inclusion", "patches", "scratches")


def main() -> int:
    rows: list[dict[str, object]] = []
    per_class: dict[str, list[dict[str, float]]] = {name: [] for name in CLASSES}
    for seed in SEEDS:
        name = f"real_only_repeat_matched_final_s{seed}_100e"
        source = PROJECT / "reports" / "yolo_eval" / f"{name}_val" / "metrics.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing completed validation report: {source}")
        document = json.loads(source.read_text(encoding="utf-8"))
        evaluation = document.get("evaluation", {})
        if evaluation.get("split") != "val" or evaluation.get("test_access_authorized", False):
            raise RuntimeError(f"report is not validation-only: {source}")
        overall = document["metrics"]["overall"]
        worst = document["metrics"]["worst_class"]
        class_rows = {item["class_name"]: item for item in document["metrics"]["per_class"]}
        if set(class_rows) != set(CLASSES):
            raise RuntimeError(f"unexpected evaluated classes in {source}: {sorted(class_rows)}")
        for class_name in CLASSES:
            per_class[class_name].append(
                {
                    "ap50": float(class_rows[class_name]["ap50"]),
                    "recall": float(class_rows[class_name]["recall"]),
                    "precision": float(class_rows[class_name]["precision"]),
                }
            )
        rows.append(
            {
                "seed": seed,
                "precision": overall["precision"],
                "recall": overall["recall"],
                "map50": overall["ap50"],
                "map50_95": overall["ap50_95"],
                "worst_class": worst["class_name"],
                "worst_class_ap50": worst["ap50"],
                "weights_sha256": document["inputs"]["weights"]["sha256"],
                "content_lock_sha256": document["inputs"]["content_lock"]["sha256"],
            }
        )

    numeric = ("precision", "recall", "map50", "map50_95", "worst_class_ap50")
    aggregate = {
        key: {
            "mean": float(np.mean([float(row[key]) for row in rows])),
            "std_ddof0": float(np.std([float(row[key]) for row in rows], ddof=0)),
        }
        for key in numeric
    }
    per_class_aggregate = {
        class_name: {
            metric: {
                "mean": float(np.mean([item[metric] for item in values])),
                "std_ddof0": float(np.std([item[metric] for item in values], ddof=0)),
            }
            for metric in ("ap50", "recall", "precision")
        }
        for class_name, values in per_class.items()
    }
    lock_hashes = {str(row["content_lock_sha256"]) for row in rows}
    if len(lock_hashes) != 1:
        raise RuntimeError(f"content lock differs across seeds: {sorted(lock_hashes)}")

    result = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "condition": "real_only_repeat_matched",
        "formal_protocol": "configs/real_only_repeat_matched_final_v1.yaml",
        "seeds": list(SEEDS),
        "test_accessed": False,
        "runs": rows,
        "aggregate": aggregate,
        "per_class_aggregate": per_class_aggregate,
    }
    output_root = PROJECT / "reports" / "real_only_repeat_matched_final_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Real-only-repeat-matched formal validation summary",
        "",
        "> Three frozen seeds; validation only; locked test was not accessed.",
        "",
        "| Metric | Mean | Std (ddof=0) |",
        "|---|---:|---:|",
    ]
    for key in numeric:
        lines.append(f"| {key} | {aggregate[key]['mean']:.6f} | {aggregate[key]['std_ddof0']:.6f} |")
    lines.extend(["", "| Class | AP50 mean | AP50 std | Recall mean | Recall std |", "|---|---:|---:|---:|---:|"])
    for class_name in CLASSES:
        values = per_class_aggregate[class_name]
        lines.append(
            f"| {class_name} | {values['ap50']['mean']:.6f} | {values['ap50']['std_ddof0']:.6f} | "
            f"{values['recall']['mean']:.6f} | {values['recall']['std_ddof0']:.6f} |"
        )
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
