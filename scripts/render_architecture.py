"""Render the closed-loop system architecture diagram for RP/paper/PPT."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "reports" / "figures" / "architecture_v1.png"

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False

BLUE = "#1f4e79"
MID = "#2e75b6"
LIGHT = "#deebf7"
GREEN = "#548235"
GREEN_L = "#e2efda"
GRAY = "#7f7f7f"
GRAY_L = "#f2f2f2"

fig, ax = plt.subplots(figsize=(14.5, 8.2), dpi=200)
ax.set_xlim(0, 100)
ax.set_ylim(0, 58)
ax.axis("off")


def box(x, y, w, h, title, body, fc=LIGHT, ec=MID, title_c=BLUE, fs=10.5, bfs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6,rounding_size=1.4",
                              fc=fc, ec=ec, lw=1.6))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="center",
            fontsize=fs, fontweight="bold", color=title_c)
    ax.text(x + w / 2, y + (h - 4.6) / 2, body, ha="center", va="center",
            fontsize=bfs, color="#333333", linespacing=1.45)


def arrow(x1, y1, x2, y2, label="", color=MID, style="-|>", lw=2.0, ls="-",
          lx=0, ly=1.6, connectionstyle="arc3,rad=0"):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                               mutation_scale=16, lw=lw, color=color, ls=ls,
                               connectionstyle=connectionstyle))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, ha="center",
                va="center", fontsize=8.5, color=color, fontweight="bold")


# ---- main loop, left to right ----
box(1.5, 38, 15, 14, "冻结训练集", "100 张真实样本\ninclusion 10\npatches 30\nscratches 60\n+ 已接受合成样本")
box(20.5, 38, 14, 14, "检测器", "YOLO11n\nSGD · 100 epochs\nimgsz 640\n固定种子×3")
box(39, 38, 15, 14, "验证集评估", "独立 val（150 张）\n类别级 recall / AP50\nmAP50 / mAP50-95\n最弱类定位")
box(58.5, 38, 17, 14, "控制器（配额决策）", "Uniform 固定均匀\nFormula 透明公式\nAgent LLM 多轮\n预算 150，每轮 ≤50")
box(79.5, 38, 18, 14, "生成工具\npatch_copy_gauss5", "真实缺陷连通域\nMask 引导 Copy-Paste\n缩放/旋转/翻转\n5px 高斯边界羽化")

# bottom row
box(64, 6, 20, 14, "自动质量门", "硬不变量：\n· Mask 外像素零改动\n· 空掩膜/越界即拒\n标签由变换后 Mask\n自动生成，无需人工", fc=GREEN_L, ec=GREEN, title_c=GREEN)
box(38, 6, 20, 14, "合成样本入库", "接受样本写回训练集\n候选上限 450\n每源 ≤8 接受\n全台账可复现", fc=GREEN_L, ec=GREEN, title_c=GREEN)
box(6, 6, 22, 14, "锁定测试集", "150 张，SHA-256 内容锁\n闭环决策不可见\n仅在协议冻结后评估一次", fc=GRAY_L, ec=GRAY, title_c="#555555")

# arrows main row
arrow(16.5, 45, 20.5, 45)
arrow(34.5, 45, 39, 45, "类别级指标")
arrow(54, 45, 58.5, 45, "弱类 + 历史")
arrow(75.5, 45, 79.5, 45, "类别配额")
# generator -> quality gate
arrow(88.5, 38, 84, 20.4, "候选样本", color=GREEN)
# quality gate -> synth pool
arrow(64, 13, 58, 13, "接受", color=GREEN)
# synth pool -> train set (loop back)
arrow(38, 13, 16, 13, "", color=GREEN)
arrow(16, 13, 9, 37.5, "加入训练", color=GREEN, connectionstyle="arc3,rad=-0.25")
# multi-round feedback hint (val metrics -> controller already shown)
# test set gate: detector -> locked test set (single post-freeze evaluation)
arrow(27.5, 37.5, 17, 20.5, "仅冻结后一次性评估", color=GRAY, ls="--", lw=1.5,
      connectionstyle="arc3,rad=0.15", lx=-6)

ax.text(50, 55.8, "检测器反馈闭环智能体数据增强 — 系统架构",
        ha="center", va="center", fontsize=15, fontweight="bold", color=BLUE)
ax.text(50, 1.5, "Agent 仅见验证集类别级指标与生成台账；配额经 schema 校验、预算受限、工具白名单 patch_copy_gauss5",
        ha="center", va="center", fontsize=9, color="#666666")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, bbox_inches="tight", facecolor="white")
print(OUT)
