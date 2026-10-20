from __future__ import annotations

"""A4: broadcast-decay training dynamics (reviewer comment 5).

Reviewer comment 5 asks for mechanistic evidence that the VN->atom broadcast
decay is what *differentiates* T10 from T7 across the two label sources. This
figure isolates the decay signal by overlaying, per label source, the T7 (no
decay) and T10 (decay) validation-MAE trajectories together with the per-epoch
gap curve (T10 val_mae - T7 val_mae) on a secondary axis, and shades the decay
window (epoch 10 -> 50).

Decay mechanism (fixed facts): T10 applies a broadcast scale that is held at
1.0 through epoch 10, then decays linearly to min_scale=0.25 at epoch 50. The
scale multiplies ONLY the virtual-node -> atom broadcast. T7 uses no decay
(scale == 1.0 for every epoch). The "deep-water" phase is the late window where
the scale is small (approaching 0.25). Test to read off the gap curve: on HZW,
does T10 degrade relative to T7 in the deep-water phase (late gap turns
positive / grows), while on MP the gap stays flat or goes negative (T10 no
worse, possibly better)?
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
OUT_DIR = ROOT / "figures" / "decay_dynamics"

# Decay schedule for T10 (broadcast VN->atom scale).
DECAY_START_EPOCH = 10
DECAY_END_EPOCH = 50
DECAY_MIN_SCALE = 0.25

DATASETS = {
    "MP label": "mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "HZW label": "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}

T7_DIR = "T7_ALIGNN_huber_virtualnode"
T10_DIR = "T10_ALIGNN_decay_huber_virtualnode"

COLOR_T7 = "#D97706"  # amber, matches huber-diagnostics T7
COLOR_T10 = "#2563EB"  # blue, matches huber-diagnostics T10
COLOR_GAP = "#059669"  # green for the difference curve
COLOR_DECAY = "#9CA3AF"


def load_history(result_root: Path, dataset_dir: str, exp_dir: str) -> pd.DataFrame:
    path = result_root / dataset_dir / exp_dir / "history.json"
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)


def broadcast_scale(epoch: np.ndarray) -> np.ndarray:
    """T10 VN->atom broadcast scale as a function of epoch."""
    frac = (epoch - DECAY_START_EPOCH) / (DECAY_END_EPOCH - DECAY_START_EPOCH)
    frac = np.clip(frac, 0.0, 1.0)
    return 1.0 - frac * (1.0 - DECAY_MIN_SCALE)


def make_figure(result_root: Path, out_dir: Path, dpi: int) -> tuple[Path, dict]:
    fig, axes = plt.subplots(2, 1, figsize=(9.6, 8.2), sharex=True)
    readouts: dict[str, dict] = {}

    for ax, (dataset_name, dataset_dir) in zip(axes, DATASETS.items()):
        t7 = load_history(result_root, dataset_dir, T7_DIR)
        t10 = load_history(result_root, dataset_dir, T10_DIR)

        # Align on the common epoch index so the gap is well defined.
        merged = t7[["epoch", "val_mae"]].merge(
            t10[["epoch", "val_mae"]], on="epoch", suffixes=("_t7", "_t10")
        )
        epoch = merged["epoch"].to_numpy()
        gap = merged["val_mae_t10"].to_numpy() - merged["val_mae_t7"].to_numpy()

        # Shade the decay window and annotate the deep-water phase.
        ax.axvspan(
            DECAY_START_EPOCH,
            DECAY_END_EPOCH,
            color=COLOR_DECAY,
            alpha=0.16,
            zorder=0,
            label=f"decay window (scale 1.0->{DECAY_MIN_SCALE})",
        )
        ax.axvline(DECAY_START_EPOCH, color=COLOR_DECAY, linewidth=1.0, linestyle=":")

        ax.plot(
            t7["epoch"],
            t7["val_mae"],
            color=COLOR_T7,
            linewidth=2.0,
            label="T7 (no decay) val MAE",
        )
        ax.plot(
            t10["epoch"],
            t10["val_mae"],
            color=COLOR_T10,
            linewidth=2.0,
            label="T10 (decay) val MAE",
        )
        ax.set_ylabel("Validation MAE (eV)")
        ax.set_title(dataset_name, loc="left", fontsize=11, fontweight="bold")
        style_axes(ax)

        # Secondary axis: per-epoch gap (T10 - T7).
        ax2 = ax.twinx()
        ax2.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
        ax2.plot(
            epoch,
            gap,
            color=COLOR_GAP,
            linewidth=1.6,
            linestyle="--",
            label="T10 - T7 val MAE (gap)",
        )
        ax2.fill_between(
            epoch,
            0.0,
            gap,
            where=gap > 0,
            color=COLOR_GAP,
            alpha=0.12,
            interpolate=True,
        )
        ax2.set_ylabel("T10 - T7 val MAE (eV)", color=COLOR_GAP)
        ax2.tick_params(axis="y", labelcolor=COLOR_GAP)
        ax2.spines["top"].set_visible(False)

        # Merge legends from both axes into one box.
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8, loc="upper right")

        # Quantitative read-out: early (pre/at decay start) vs deep-water (late).
        early_mask = epoch <= DECAY_START_EPOCH
        deep_mask = epoch >= (DECAY_START_EPOCH + DECAY_END_EPOCH) // 2  # >= 30
        last_mask = epoch >= (DECAY_END_EPOCH - 10)  # final 10 epochs
        readouts[dataset_name] = {
            "gap_early_mean": float(np.mean(gap[early_mask]))
            if early_mask.any()
            else float("nan"),
            "gap_deepwater_mean": float(np.mean(gap[deep_mask]))
            if deep_mask.any()
            else float("nan"),
            "gap_final10_mean": float(np.mean(gap[last_mask]))
            if last_mask.any()
            else float("nan"),
            "gap_final_epoch": float(gap[-1]),
            "deepwater_min_scale": float(broadcast_scale(epoch[deep_mask]).min())
            if deep_mask.any()
            else float("nan"),
        }

    axes[-1].set_xlabel("Epoch")
    fig.suptitle(
        "Validation-MAE Dynamics of Broadcast Decay on MP and HZW Labels",
        y=0.995,
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "decay_val_mae_dynamics.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path, readouts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="root directory holding the per-dataset result folders",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=OUT_DIR, help="directory for the output figure"
    )
    parser.add_argument("--dpi", type=int, default=200, help="figure DPI (>=150)")
    args = parser.parse_args()

    path, readouts = make_figure(args.result_root, args.out_dir, max(150, args.dpi))
    print(f"Saved figure: {path}")
    print("\nGap read-out (T10 val_mae - T7 val_mae), eV:")
    for dataset_name, r in readouts.items():
        print(
            f"  {dataset_name:10s} | early(<=e10) {r['gap_early_mean']:+.4f} | "
            f"deep-water(>=e30) {r['gap_deepwater_mean']:+.4f} | "
            f"final10 {r['gap_final10_mean']:+.4f} | final-epoch {r['gap_final_epoch']:+.4f}"
        )


if __name__ == "__main__":
    main()
