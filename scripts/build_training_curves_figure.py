"""Build the per-epoch training/validation curve figure for the Supplementary Information.

Reads the seed-42 ``history.json`` files written by the training runners; no
metric is recomputed here. Output is written as a PDF and a PNG.

Two deliberate design constraints:

1. The loss panels (a, b) show only the four ALIGNN-backbone variants, because
   they are the models that share one runner and one target scale. T2/T4
   minimise MSE and T7/T10 minimise the Huber (smooth-L1, beta=0.5) objective,
   so the like-for-like loss comparisons are T2 vs T4 and T7 vs T10; the
   CGCNN (T1) and M3GNet (T3) runners define their own objectives on their own
   scales and would not be comparable on the same axis.
2. The MAE panels (c, d) carry all six core models, because MAE is
   objective-independent and therefore comparable across every runner.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Same Okabe-Ito roles as build_academic_paper_figures.py: T2 = BLUE,
# T4 = TEAL, T7 = ORANGE, T10 = PURPLE. T1 and T3 are drawn as MUTED and RED
# here rather than both MUTED, because two line series cannot share a colour;
# neither colour carries a model role in any other panel of the paper.
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
PURPLE = "#CC79A7"
RED = "#D55E00"
MUTED = "#6B7280"
INK = "#222222"

RUN_DIRS = {
    "T1": "T1_CGCNN_baseline",
    "T2": "T2_ALIGNN_baseline",
    "T3": "T3_M3GNet_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}
LABELS = {
    "T1": "CGCNN",
    "T2": "ALIGNN",
    "T3": "M3GNet",
    "T4": "ALIGNN+VN",
    "T7": "ALIGNN+VN+Huber",
    "T10": "ALIGNN+VN+Huber+decay",
}
COLORS = {
    "T1": MUTED,
    "T2": BLUE,
    "T3": RED,
    "T4": TEAL,
    "T7": ORANGE,
    "T10": PURPLE,
}
LOSS_MODELS = ["T2", "T4", "T7", "T10"]
CORE_MODELS = ["T1", "T2", "T3", "T4", "T7", "T10"]
SOURCES = {"MP": "mp", "HZW": "hzw"}


def load_history(source: str, tcode: str) -> pd.DataFrame:
    path = (
        ROOT
        / "results"
        / f"mp_hzw_relaxed_clean_equal_atoms_{SOURCES[source]}_t"
        / RUN_DIRS[tcode]
        / "history.json"
    )
    with path.open("r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.13,
        1.10,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def style_axes(ax) -> None:
    ax.grid(True, which="major", linestyle="--", linewidth=0.6, alpha=0.35)
    ax.grid(True, which="minor", linestyle=":", linewidth=0.4, alpha=0.20)
    ax.set_xlim(1, 50)
    ax.set_xticks([1, 10, 20, 30, 40, 50])


def build() -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.0))

    # --- Row 1: training and validation loss, ALIGNN-backbone variants only.
    for ax, source, letter in [
        (axes[0, 0], "MP", "(a)"),
        (axes[0, 1], "HZW", "(b)"),
    ]:
        for tcode in LOSS_MODELS:
            h = load_history(source, tcode)
            ax.plot(
                h["epoch"],
                h["train_loss"],
                color=COLORS[tcode],
                linewidth=1.3,
                linestyle="--",
                alpha=0.85,
            )
            ax.plot(
                h["epoch"],
                h["val_loss"],
                color=COLORS[tcode],
                linewidth=1.6,
                linestyle="-",
            )
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training-objective loss")
        ax.set_title(f"{source} labels: loss curves (ALIGNN backbone)")
        ax.text(
            0.985,
            0.965,
            "ALIGNN / +VN: MSE\n+Huber / +Huber+decay: Huber ($\\beta=0.5$)",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.4,
            color=INK,
            bbox={
                "facecolor": "white",
                "edgecolor": "#D1D5DB",
                "linewidth": 0.6,
                "pad": 2.4,
            },
        )
        style_axes(ax)
        panel_label(ax, letter)

    handles = [
        Line2D([], [], color=COLORS[t], linewidth=1.8, label=LABELS[t])
        for t in LOSS_MODELS
    ]
    handles += [
        Line2D([], [], color=INK, linewidth=1.6, linestyle="-", label="validation"),
        Line2D([], [], color=INK, linewidth=1.3, linestyle="--", label="training"),
    ]
    axes[0, 0].legend(
        handles=handles, loc="lower left", frameon=False, ncol=2, columnspacing=1.1
    )

    # --- Row 2: validation MAE, all six core models (objective-independent).
    for ax, source, letter in [
        (axes[1, 0], "MP", "(c)"),
        (axes[1, 1], "HZW", "(d)"),
    ]:
        for tcode in CORE_MODELS:
            h = load_history(source, tcode)
            ax.plot(
                h["epoch"],
                h["val_mae"],
                color=COLORS[tcode],
                linewidth=1.5,
                label=LABELS[tcode],
            )
            best = h.loc[h["val_mae"].idxmin()]
            ax.plot(
                best["epoch"],
                best["val_mae"],
                marker="*",
                markersize=8,
                color=COLORS[tcode],
                markeredgecolor="white",
                markeredgewidth=0.5,
                linestyle="none",
            )
        ax.set_yscale("log")
        ax.set_yticks([0.2, 0.3, 0.4, 0.6, 1.0])
        ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
        # Headroom so the legend / annotation box never sits on top of the
        # CGCNN curve, whose early epochs are the highest series in the panel.
        ax.set_ylim(top=1.75)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (eV)")
        ax.set_title(f"{source} labels: validation MAE (all core models)")
        style_axes(ax)
        panel_label(ax, letter)

    axes[1, 0].legend(loc="upper right", frameon=False, ncol=2, columnspacing=1.1)
    axes[1, 1].text(
        0.985,
        0.965,
        "$\\star$ best-validation epoch\n(checkpoint scored on the test set)",
        transform=axes[1, 1].transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=INK,
        bbox={
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "linewidth": 0.6,
            "pad": 2.4,
        },
    )

    fig.tight_layout(pad=1.1, h_pad=2.4, w_pad=2.2)
    fig.savefig(OUT / "figS_training_curves.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        OUT / "figS_training_curves.png", dpi=240, bbox_inches="tight", pad_inches=0.04
    )
    plt.close(fig)


if __name__ == "__main__":
    build()
    print("wrote", OUT / "figS_training_curves.pdf")
