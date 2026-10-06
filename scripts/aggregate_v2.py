"""Aggregate v2 final results (6 conditions x 3 seeds) for val and test.

Model set comes from reports/v2/final/protocol_freeze.json (r3 for multi-round).
Writes reports/v2/final/results_summary.json + .md.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
FINAL = PROJECT / "reports" / "v2" / "final"
EVAL_ROOT = FINAL / "yolo_eval"
CLASSES = ("inclusion", "patches", "scratches")
KEYS = ("map50", "map95", "precision", "recall", "worst_ap50")


def load(run: str, split: str) -> dict | None:
    path = EVAL_ROOT / f"{run}_{split}" / "metrics.json"
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))["metrics"]
    per = {c["class_name"]: c for c in m["per_class"]}
    out = {"map50": m["overall"]["ap50"], "map95": m["overall"]["ap50_95"],
           "precision": m["overall"]["precision"], "recall": m["overall"]["recall"],
           "worst_ap50": m["worst_class"]["ap50"]}
    for c in CLASSES:
        out[f"{c}_ap50"] = per[c]["ap50"]
        out[f"{c}_recall"] = per[c]["recall"]
    return out


def ms(v: list[float]) -> tuple[float, float]:
    return statistics.mean(v), (statistics.stdev(v) if len(v) > 1 else 0.0)


def main() -> int:
    freeze = json.loads((FINAL / "protocol_freeze.json").read_text(encoding="utf-8"))
    conds: dict[str, list[dict]] = {}
    for m in freeze["models"]:
        conds.setdefault(m["condition"], []).append(m)
    result: dict = {"protocol_sha256": freeze["protocol_sha256"]}
    lines = ["# DefectAug v2 正式结果（3 seeds，均值±std）", "",
             f"协议 sha256 `{freeze['protocol_sha256']}`；多轮条件取 r3 模型。", ""]
    for split, title in (("val", "验证集（反馈决策用）"), ("test", "锁定测试集（冻结后一次性评估）")):
        summ: dict = {}
        for cond, models in conds.items():
            per_seed = {m["seed"]: load(m["run"], split) for m in models}
            if any(v is None for v in per_seed.values()):
                raise SystemExit(f"missing {split} metrics for {cond}")
            agg = {k: ms([v[k] for v in per_seed.values()]) for k in per_seed[models[0]["seed"]]}
            summ[cond] = {"mean_std": agg, "per_seed": per_seed}
        result[split] = summ
        lines += [f"## {title}", "",
                  "| 条件 | mAP50 | mAP50-95 | macro P | macro R | 最弱类 AP50 | inclusion AP50 | patches AP50 | scratches AP50 |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for cond, s in summ.items():
            a = s["mean_std"]
            cells = [f"{a[k][0]:.4f}±{a[k][1]:.4f}" for k in ("map50", "map95")]
            cells += [f"{a[k][0]:.4f}" for k in ("precision", "recall", "worst_ap50")]
            cells += [f"{a[f'{c}_ap50'][0]:.4f}" for c in CLASSES]
            lines.append(f"| {cond} | " + " | ".join(cells) + " |")
        lines += ["", f"### 每 seed mAP50（{split}）", "",
                  "| 条件 | " + " | ".join(str(m["seed"]) for m in conds["real_only"]) + " |",
                  "|---|---|---|---|"]
        for cond, s in summ.items():
            lines.append(f"| {cond} | " + " | ".join(f"{s['per_seed'][m['seed']]['map50']:.4f}"
                                                   for m in conds["real_only"]) + " |")
        lines.append("")
    (FINAL / "results_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    (FINAL / "results_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
