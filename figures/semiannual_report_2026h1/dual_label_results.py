from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MP_CSV = ROOT / "results/mp_hzw_relaxed_clean_equal_atoms_mp_t/comparison_summary.csv"
HZW_CSV = ROOT / "results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/comparison_summary.csv"
OUT_PDF = Path(__file__).with_suffix(".pdf")
OUT_PNG = Path(__file__).with_suffix(".png")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Microsoft YaHei", "SimHei", "DejaVu Sans"],
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

WANTED = {
    "T2": "T2_ALIGNN_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}
LABELS = ["T2\nALIGNN", "T4\n+ single VN", "T7\n+ VN + Huber", "T10\n+ decay + Huber"]
COLORS = ["#64748b", "#3b82f6", "#16a34a", "#f97316"]


def load_mae(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    by_name = {row["experiment_name"]: float(row["mae"]) for row in rows}
    return {short: by_name[full] for short, full in WANTED.items()}


def improvement(baseline: float, value: float) -> float:
    return (baseline - value) / baseline * 100.0


def draw_panel(ax, title: str, values: dict[str, float], ylim: tuple[float, float]):
    xs = np.arange(len(WANTED))
    ys = [values[k] for k in WANTED]
    bars = ax.bar(xs, ys, color=COLORS, width=0.68, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks(xs, LABELS, fontsize=9)
    ax.set_ylim(*ylim)
    ax.set_ylabel("Test MAE (eV，越低越好)", fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.28)
    ax.spines[["top", "right"]].set_visible(False)
    for i, (bar, value) in enumerate(zip(bars, ys)):
        ax.text(bar.get_x() + bar.get_width()/2, value + 0.0015, f"{value:.6f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold" if i in (2,3) else "normal")
        if i > 0:
            imp = improvement(ys[0], value)
            ax.text(bar.get_x() + bar.get_width()/2, ylim[0] + 0.005,
                    f"较 T2 ↓{imp:.2f}%", ha="center", va="bottom", fontsize=7.5,
                    color="#14532d" if imp >= 5 else "#334155")
    best_idx = int(np.argmin(ys))
    bars[best_idx].set_edgecolor("#7c2d12" if best_idx == 3 else "#14532d")
    bars[best_idx].set_linewidth(2.4)


def main() -> None:
    mp = load_mae(MP_CSV)
    hzw = load_mae(HZW_CSV)

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.5), constrained_layout=True)
    draw_panel(axes[0], "同一批结构 · MP label", mp, (0.18, 0.255))
    draw_panel(axes[1], "同一批结构 · HZW label", hzw, (0.16, 0.225))
    fig.suptitle("最终双标签验证：简单 virtual node 有效，Huber 是跨源稳定收益",
                 fontsize=15, fontweight="bold")
    fig.text(0.5, -0.015,
             "结论：MP 上 T10 最低；HZW 上 T7 最低。跨标签源最稳方案为 T7（single VN + Huber）。",
             ha="center", fontsize=10, color="#1e3a8a")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print("MP", " ".join(f"{k}={mp[k]:.6f}" for k in WANTED))
    print("HZW", " ".join(f"{k}={hzw[k]:.6f}" for k in WANTED))
    print(f"MP_T7_improvement={improvement(mp['T2'], mp['T7']):.2f}%")
    print(f"MP_T10_improvement={improvement(mp['T2'], mp['T10']):.2f}%")
    print(f"HZW_T7_improvement={improvement(hzw['T2'], hzw['T7']):.2f}%")
    print(f"wrote={OUT_PDF}")


if __name__ == "__main__":
    main()
