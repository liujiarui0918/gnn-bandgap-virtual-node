from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = ROOT / "results"
OUT_DIR = ROOT / "figures" / "huber_diagnostics"

DATASETS = {
    "MP label": RESULT_ROOT / "mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "HZW label": RESULT_ROOT / "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}

EXPERIMENTS = {
    "T4 MSE": "T4_ALIGNN_plain_single_virtualnode",
    "T7 Huber": "T7_ALIGNN_huber_virtualnode",
    "T10 Decay+Huber": "T10_ALIGNN_decay_huber_virtualnode",
}

COLORS = {
    "T4 MSE": "#6B7280",
    "T7 Huber": "#D97706",
    "T10 Decay+Huber": "#2563EB",
}


def load_history(dataset_dir: Path, exp_dir: str) -> pd.DataFrame:
    path = dataset_dir / exp_dir / "history.json"
    with path.open("r", encoding="utf-8") as f:
        rows = json.load(f)
    return pd.DataFrame(rows)


def load_predictions(dataset_dir: Path, exp_dir: str) -> pd.DataFrame:
    path = dataset_dir / exp_dir / "predictions.csv"
    df = pd.read_csv(path)
    df["residual"] = df["prediction"] - df["target"]
    df["abs_error"] = df["residual"].abs()
    return df


def load_metrics(dataset_dir: Path, exp_dir: str) -> dict:
    path = dataset_dir / exp_dir / "metrics.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def style_axes(ax) -> None:
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def savefig(fig, name: str) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return path


def make_t7_training_curves() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 6.4), sharex=True)
    for row, (dataset_name, dataset_dir) in enumerate(DATASETS.items()):
        hist = load_history(dataset_dir, EXPERIMENTS["T7 Huber"])
        ax_loss = axes[row, 0]
        ax_mae = axes[row, 1]
        ax_loss.plot(hist["epoch"], hist["train_loss"], label="train loss", color="#B45309")
        ax_loss.plot(hist["epoch"], hist["val_loss"], label="val loss", color="#F59E0B")
        ax_loss.set_title(f"{dataset_name}: T7 Huber objective")
        ax_loss.set_ylabel("Huber loss")
        ax_loss.legend(frameon=False, fontsize=8)
        style_axes(ax_loss)

        ax_mae.plot(hist["epoch"], hist["train_mae"], label="train MAE", color="#1D4ED8")
        ax_mae.plot(hist["epoch"], hist["val_mae"], label="val MAE", color="#60A5FA")
        ax_mae.set_title(f"{dataset_name}: T7 MAE")
        ax_mae.set_ylabel("MAE (eV)")
        ax_mae.legend(frameon=False, fontsize=8)
        style_axes(ax_mae)

    for ax in axes[-1, :]:
        ax.set_xlabel("Epoch")
    fig.suptitle("T7 ALIGNN + single virtual node + Huber loss: training dynamics", y=1.02)
    return savefig(fig, "t7_huber_training_curves.png")


def make_val_mae_curves() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True)
    for ax, (dataset_name, dataset_dir) in zip(axes, DATASETS.items()):
        for label, exp_dir in EXPERIMENTS.items():
            hist = load_history(dataset_dir, exp_dir)
            metrics = load_metrics(dataset_dir, exp_dir)
            ax.plot(
                hist["epoch"],
                hist["val_mae"],
                label=f"{label} (test {metrics['mae']:.3f})",
                color=COLORS[label],
                linewidth=2.0 if "Huber" in label else 1.6,
            )
        ax.set_title(dataset_name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation MAE (eV)")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    fig.suptitle("Validation MAE curves: MSE virtual node vs Huber variants", y=1.03)
    return savefig(fig, "val_mae_curves_t4_t7_t10.png")


def make_residual_histograms() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.1), sharey=True)
    bins = np.linspace(-2.0, 2.0, 81)
    for ax, (dataset_name, dataset_dir) in zip(axes, DATASETS.items()):
        for label in ["T4 MSE", "T7 Huber"]:
            df = load_predictions(dataset_dir, EXPERIMENTS[label])
            metrics = load_metrics(dataset_dir, EXPERIMENTS[label])
            clipped = df["residual"].clip(-2.0, 2.0)
            ax.hist(
                clipped,
                bins=bins,
                density=True,
                histtype="stepfilled",
                alpha=0.34 if label == "T4 MSE" else 0.42,
                linewidth=1.5,
                color=COLORS[label],
                label=f"{label}: MAE {metrics['mae']:.3f}",
            )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(dataset_name)
        ax.set_xlabel("Residual = prediction - target (eV), clipped to [-2, 2]")
        ax.set_ylabel("Density")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    fig.suptitle("Residual distributions: Huber reduces average error without changing backbone", y=1.03)
    return savefig(fig, "residual_hist_t4_vs_t7.png")


def make_abs_error_quantiles() -> Path:
    quantiles = [0.50, 0.75, 0.90, 0.95, 0.99]
    qlabels = ["P50", "P75", "P90", "P95", "P99"]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2), sharey=True)
    width = 0.24
    x = np.arange(len(quantiles))
    for ax, (dataset_name, dataset_dir) in zip(axes, DATASETS.items()):
        for idx, (label, exp_dir) in enumerate(EXPERIMENTS.items()):
            df = load_predictions(dataset_dir, exp_dir)
            vals = df["abs_error"].quantile(quantiles).to_numpy()
            ax.bar(x + (idx - 1) * width, vals, width=width, label=label, color=COLORS[label])
        ax.set_xticks(x)
        ax.set_xticklabels(qlabels)
        ax.set_title(dataset_name)
        ax.set_xlabel("Absolute-error quantile")
        ax.set_ylabel("Absolute error (eV)")
        ax.legend(frameon=False, fontsize=8)
        style_axes(ax)
    fig.suptitle("Error-tail diagnostics: Huber improves median/upper-tail behavior differently by label source", y=1.03)
    return savefig(fig, "abs_error_quantiles_t4_t7_t10.png")


def make_pred_vs_target_t7() -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), sharex=True, sharey=True)
    for ax, (dataset_name, dataset_dir) in zip(axes, DATASETS.items()):
        df = load_predictions(dataset_dir, EXPERIMENTS["T7 Huber"])
        lim = float(np.nanpercentile(np.r_[df["target"].to_numpy(), df["prediction"].to_numpy()], 99.5))
        lim = max(4.0, min(8.0, lim))
        hb = ax.hexbin(
            df["target"],
            df["prediction"],
            gridsize=55,
            extent=(0, lim, 0, lim),
            mincnt=1,
            cmap="YlOrBr",
            linewidths=0,
        )
        ax.plot([0, lim], [0, lim], color="black", linewidth=1.0, linestyle="--")
        ax.set_title(dataset_name)
        ax.set_xlabel("Target band gap (eV)")
        ax.set_ylabel("Predicted band gap (eV)")
        style_axes(ax)
        fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03, label="count")
    fig.suptitle("T7 Huber predictions: target-vs-prediction density", y=1.03)
    return savefig(fig, "pred_vs_target_t7_huber.png")


def make_summary_csv() -> Path:
    rows = []
    for dataset_name, dataset_dir in DATASETS.items():
        for label, exp_dir in EXPERIMENTS.items():
            df = load_predictions(dataset_dir, exp_dir)
            metrics = load_metrics(dataset_dir, exp_dir)
            rows.append(
                {
                    "dataset": dataset_name,
                    "experiment": label,
                    "mae": metrics["mae"],
                    "rmse": metrics["rmse"],
                    "r2": metrics["r2"],
                    "metal_f1": metrics["metal_f1"],
                    "bias_mean_residual": df["residual"].mean(),
                    "median_abs_error": df["abs_error"].median(),
                    "p90_abs_error": df["abs_error"].quantile(0.90),
                    "p95_abs_error": df["abs_error"].quantile(0.95),
                    "p99_abs_error": df["abs_error"].quantile(0.99),
                    "error_gt_1ev_ratio": (df["abs_error"] > 1.0).mean(),
                    "n": len(df),
                }
            )
    summary = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / "huber_residual_summary.csv"
    summary.to_csv(path, index=False)
    return path


def main() -> None:
    paths = [
        make_t7_training_curves(),
        make_val_mae_curves(),
        make_residual_histograms(),
        make_abs_error_quantiles(),
        make_pred_vs_target_t7(),
        make_summary_csv(),
    ]
    print("Generated:")
    for path in paths:
        print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
