"""Aggregate the six formal conditions into reports/results_summary.json + .md.

For multi-round conditions the LAST round's metrics are the condition result
(r3 trains on the full 150 accepted samples).  Reads every
reports/yolo_eval/<name>_val/metrics.json, computes per-condition mean/std
across the three seeds, and writes machine + human readable summaries.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT / "reports" / "yolo_eval"
SEEDS = (20260820, 20260821, 20260822)
CLASSES = ("inclusion", "patches", "scratches")

RUNS = {
    "real_only": [f"real_only_4090_s{s}_100e" for s in SEEDS],
    "real_only_repeat_matched": [f"real_only_repeat_matched_4090_s{s}_100e" for s in SEEDS],
    "uniform_fixed": [f"uniform_fixed_4090_s{s}_100e" for s in SEEDS],
    "formula_one_shot": [f"formula_one_shot_4090_s{s}_100e" for s in SEEDS],
    "formula_multi_round": [f"formula_multi_round_4090_s{s}_r3_100e" for s in SEEDS],
    "agent_multi_round": [f"agent_multi_round_4090_s{s}_r3_100e" for s in SEEDS],
}


def load_metrics(run_name: str, split: str = "val") -> dict | None:
    path = EVAL_ROOT / f"{run_name}_{split}" / "metrics.json"
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    m = doc["metrics"]
    per = {c["class_name"]: c for c in m["per_class"]}
    return {
        "map50": m["overall"]["ap50"],
        "map95": m["overall"]["ap50_95"],
        "precision": m["overall"]["precision"],
        "recall": m["overall"]["recall"],
        "worst_ap50": m["worst_class"]["ap50"],
        "worst_class": m["worst_class"]["class_name"],
        "std_ap50": m["class_metric_std"]["ap50"],
        "per_class": {
            name: {"ap50": per[name]["ap50"], "recall": per[name]["recall"],
                   "ap50_95": per[name]["ap50_95"], "precision": per[name]["precision"]}
            for name in CLASSES
        },
    }


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def aggregate(split: str) -> tuple[dict[str, dict], dict[str, list[str]]]:
    summary: dict[str, dict] = {}
    missing: dict[str, list[str]] = {}
    for cond, runs in RUNS.items():
        per_seed = {}
        for run in runs:
            m = load_metrics(run, split)
            if m is None:
                missing.setdefault(cond, []).append(run)
                continue
            per_seed[run] = m
        if not per_seed:
            continue
        keys = ("map50", "map95", "precision", "recall", "worst_ap50", "std_ap50")
        agg: dict[str, float] = {}
        for k in keys:
            mu, sd = mean_std([v[k] for v in per_seed.values()])
            agg[k] = mu
            agg[f"{k}_std"] = sd
        for cls in CLASSES:
            for k in ("ap50", "recall"):
                mu, sd = mean_std([v["per_class"][cls][k] for v in per_seed.values()])
                agg[f"{cls}_{k}"] = mu
                agg[f"{cls}_{k}_std"] = sd
        agg["n_seeds"] = len(per_seed)
        agg["per_seed"] = {run: {k: round(v[k], 5) for k in keys} for run, v in per_seed.items()}
        summary[cond] = agg
    return summary, missing


def render_table(title: str, summary: dict[str, dict]) -> list[str]:
    lines = [f"## {title}", "",
             "| 条件 | n | mAP50 | mAP50-95 | macro P | macro R | 最弱类 AP50 | AP50 类间 std |",
             "|---|---|---|---|---|---|---|---|"]
    for cond in RUNS:
        if cond not in summary:
            lines.append(f"| {cond} | 0 | — | — | — | — | — | — |")
            continue
        a = summary[cond]
        lines.append(
            f"| {cond} | {a['n_seeds']} | {a['map50']:.4f}±{a['map50_std']:.4f} "
            f"| {a['map95']:.4f}±{a['map95_std']:.4f} | {a['precision']:.4f} "
            f"| {a['recall']:.4f} | {a['worst_ap50']:.4f} | {a['std_ap50']:.4f} |"
        )
    lines += ["", f"### 每类 AP50（均值，{title}）", "",
              "| 条件 | inclusion | patches | scratches |", "|---|---|---|---|"]
    for cond in RUNS:
        if cond not in summary:
            continue
        a = summary[cond]
        lines.append(
            f"| {cond} | {a['inclusion_ap50']:.4f} | {a['patches_ap50']:.4f} | {a['scratches_ap50']:.4f} |"
        )
    lines.append("")
    return lines


def main() -> int:
    val_summary, val_missing = aggregate("val")
    test_summary, test_missing = aggregate("test")

    out_json = PROJECT / "reports" / "results_summary.json"
    out_json.write_text(
        json.dumps({"val": val_summary, "test": test_summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = ["# 六组对照结果汇总（seeds 均值±std）", ""]
    lines += render_table("验证集（反馈决策用）", val_summary)
    lines += render_table("锁定测试集（协议冻结后一次性评估）", test_summary)
    for split, missing in (("val", val_missing), ("test", test_missing)):
        if missing:
            lines += [f"## 未完成 run（{split}）", ""]
            for cond, runs in missing.items():
                lines.append(f"- {cond}: {', '.join(runs)}")

    out_md = PROJECT / "reports" / "results_summary.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
