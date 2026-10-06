"""Bar chart of the six conditions: mAP50 and worst-class AP50 on val and test."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
SUMMARY = PROJECT / "reports" / "results_summary.json"
OUT = PROJECT / "reports" / "figures" / "six_conditions_map50.png"

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

LABELS = {
    "real_only": "Real-only",
    "real_only_repeat_matched": "Repeat-matched",
    "uniform_fixed": "Uniform-fixed",
    "formula_one_shot": "Formula-1shot",
    "formula_multi_round": "Formula-multi",
    "agent_multi_round": "Agent-multi",
}
ORDER = list(LABELS)


def panel(ax, data, title):
    names = [LABELS[c] for c in ORDER if c in data]
    map50 = [data[c]["map50"] for c in ORDER if c in data]
    map50_e = [data[c].get("map50_std", 0) for c in ORDER if c in data]
    worst = [data[c]["worst_ap50"] for c in ORDER if c in data]
    worst_e = [data[c].get("worst_ap50_std", 0) for c in ORDER if c in data]
    x = np.arange(len(names))
    w = 0.38
    ax.bar(x - w / 2, map50, w, yerr=map50_e, capsize=3, color="#2e75b6", label="mAP50")
    ax.bar(x + w / 2, worst, w, yerr=worst_e, capsize=3, color="#a9c4e8", label="最弱类 AP50")
    for xi, v in zip(x - w / 2, map50):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + w / 2, worst):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=7.5)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, max(map50) * 1.3 + 0.05)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title(title, fontsize=11.5)
    ax.legend(fontsize=9)


def main() -> int:
    doc = json.loads(SUMMARY.read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6), dpi=200)
    panel(axes[0], doc["val"], "验证集（反馈决策依据）")
    panel(axes[1], doc["test"], "锁定测试集（冻结后一次性评估）")
    fig.suptitle("六组对照：mAP50 与最弱类 AP50（3 seeds 均值±std）", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
