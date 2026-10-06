"""Aggregate the three independent Real-only validation reports."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
SEEDS = (20260820, 20260821, 20260822)


def main() -> int:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        name = f"real_only_final_s{seed}_100e"
        source = PROJECT / "reports" / "yolo_eval" / f"{name}_val" / "metrics.json"
        if not source.is_file():
            raise FileNotFoundError(f"missing completed validation report: {source}")
        document = json.loads(source.read_text(encoding="utf-8"))
        overall = document["metrics"]["overall"]
        worst = document["metrics"]["worst_class"]
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
    result = {
        "schema_version": "1.0",
        "created_at": datetime.now().astimezone().isoformat(),
        "condition": "real_only",
        "formal_protocol": "configs/real_only_final_v1.yaml",
        "seeds": list(SEEDS),
        "test_accessed": False,
        "runs": rows,
        "aggregate": aggregate,
    }
    output_root = PROJECT / "reports" / "real_only_final_v1"
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    with (output_root / "runs.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Real-only formal validation summary",
        "",
        "> Three frozen seeds; validation only; locked test is not included in this bundle.",
        "",
        "| Metric | Mean | Std (ddof=0) |",
        "|---|---:|---:|",
    ]
    for key in numeric:
        lines.append(f"| {key} | {aggregate[key]['mean']:.6f} | {aggregate[key]['std_ddof0']:.6f} |")
    (output_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

