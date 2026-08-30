from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "results/data_quality/structure_first_strict_180/mp_hzw_structure_first_summary.json"
OUT_PDF = Path(__file__).with_suffix(".pdf")
OUT_PNG = Path(__file__).with_suffix(".png")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def box(ax, x, y, w, h, title, value, color, subtitle=""):
    patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.025",
                           facecolor=color, edgecolor="white", linewidth=1.5)
    ax.add_patch(patch)
    ax.text(x+w/2, y+h*0.65, title, ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.text(x+w/2, y+h*0.35, f"{value:,}", ha="center", va="center", fontsize=18, color="white", fontweight="bold")
    if subtitle:
        ax.text(x+w/2, y-0.035, subtitle, ha="center", va="top", fontsize=8.5, color="#475569")


def main() -> None:
    data = json.loads(SOURCE.read_text(encoding="utf-8"))
    total = int(data["total_rows"])
    matched = int(data["structure_matched_rows"])
    unmatched = int(data["structure_unmatched_rows"])
    missing = int(data["missing_mp_record_rows"])
    conflicts = data["matched_label_conflict_counts"]
    conflict_values = [
        int(conflicts["mp_metal_hzw_nonmetal"]),
        int(conflicts["mp_nonmetal_hzw_metal"]),
        int(conflicts["same_class_gap_diff_gt_1ev"]),
    ]
    conflict_labels = ["MP 金属 / HZW 非金属", "MP 非金属 / HZW 金属", "同类别但 gap 差 > 1 eV"]

    fig = plt.figure(figsize=(12.2, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.08, 1.0])
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])

    ax0.set_xlim(0, 1)
    ax0.set_ylim(0, 1)
    ax0.axis("off")
    ax0.set_title("Structure-first audit：先确认结构，再讨论标签", fontsize=13, fontweight="bold", pad=8)
    box(ax0, 0.05, 0.68, 0.38, 0.20, "可审计记录", total, "#1e3a8a")
    box(ax0, 0.57, 0.68, 0.38, 0.20, "结构匹配", matched, "#15803d", f"占总记录 {matched/total*100:.2f}%")
    ax0.add_patch(FancyArrowPatch((0.43, 0.78), (0.57, 0.78), arrowstyle="-|>", mutation_scale=18,
                                  linewidth=2, color="#64748b"))
    box(ax0, 0.09, 0.27, 0.34, 0.18, "缺 MP 记录", missing, "#f59e0b", f"占总记录 {missing/total*100:.2f}%")
    box(ax0, 0.57, 0.27, 0.34, 0.18, "结构不匹配", unmatched, "#dc2626", f"占总记录 {unmatched/total*100:.3f}%")
    ax0.text(0.5, 0.06, "关键转折：结构对应关系很强，但 band gap 标签仍可能冲突。",
             ha="center", va="center", fontsize=11, color="#7f1d1d", fontweight="bold")

    colors = ["#ef4444", "#f97316", "#8b5cf6"]
    bars = ax1.barh(range(3), conflict_values, color=colors, height=0.62)
    ax1.set_yticks(range(3), conflict_labels, fontsize=10)
    ax1.invert_yaxis()
    ax1.set_xlabel("结构匹配样本中的冲突数量", fontsize=10)
    ax1.set_title("结构匹配不等于标签一致", fontsize=13, fontweight="bold", pad=8)
    ax1.grid(axis="x", linestyle="--", alpha=0.28)
    ax1.spines[["top", "right", "left"]].set_visible(False)
    ax1.tick_params(axis="y", length=0)
    xmax = max(conflict_values) * 1.22
    ax1.set_xlim(0, xmax)
    for bar, value in zip(bars, conflict_values):
        ax1.text(value + xmax*0.025, bar.get_y()+bar.get_height()/2, f"{value:,}",
                 va="center", fontsize=11, fontweight="bold")
    no_conflict = int(conflicts["no_label_conflict"])
    ax1.text(0.02, -0.18, f"无显式标签冲突：{no_conflict:,} 条；但审计仍显示跨源监督目标不完全等价。",
             transform=ax1.transAxes, fontsize=9.5, color="#334155")

    fig.suptitle("MP–HZW 数据质量审计：模型排序变化首先是数据问题，而不只是模型问题",
                 fontsize=15, fontweight="bold")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"total={total}")
    print(f"matched={matched}")
    print(f"unmatched={unmatched}")
    print(f"missing={missing}")
    print(f"mp_metal_hzw_nonmetal={conflict_values[0]}")
    print(f"mp_nonmetal_hzw_metal={conflict_values[1]}")
    print(f"same_class_gap_diff_gt_1ev={conflict_values[2]}")
    print(f"wrote={OUT_PDF}")


if __name__ == "__main__":
    main()
