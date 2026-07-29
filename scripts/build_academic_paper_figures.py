"""Build compact, publication-style composite figures for the paper.

The figures use the verified result CSV files rather than duplicating training
or recomputing metrics. Each output is written as a PDF and a PNG.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LogNorm


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

BLUE = "#2D5F8B"
TEAL = "#3A8D8F"
ORANGE = "#D9772A"
GREEN = "#4C8C5A"
RED = "#B84A4A"
INK = "#222222"
MUTED = "#6B7280"

# Reviewer comment 7: in-figure series codes are replaced by descriptive
# variant names. Data lookups still key on the T-codes (they are the CSV
# index/keys); only the displayed text uses these names. Kept identical across
# fig2/fig3/fig4 for consistency. LABELS is the single-line form; AXIS_LABELS is
# the line-wrapped form for tight tick axes (same descriptor, wrapped only).
LABELS = {
    "T1": "CGCNN",
    "T2": "ALIGNN",
    "T3": "M3GNet",
    "T4": "ALIGNN+VN",
    "T7": "ALIGNN+VN+Huber",
    "T10": "ALIGNN+VN+Huber+decay",
}
AXIS_LABELS = {
    "T1": "CGCNN",
    "T2": "ALIGNN",
    "T3": "M3GNet",
    "T4": "ALIGNN\n+VN",
    "T7": "ALIGNN\n+VN+Huber",
    "T10": "ALIGNN+VN\n+Huber+decay",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / f"{name}.png", dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.08,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


def load_results(dataset: str) -> pd.DataFrame:
    path = ROOT / "results" / f"mp_hzw_relaxed_clean_equal_atoms_{dataset}_t" / "comparison_summary.csv"
    df = pd.read_csv(path)
    df["id"] = df["experiment_name"].str.extract(r"^(T\d+)")
    return df.set_index("id")


RUN_DIRS = {
    "T2": "T2_ALIGNN_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}


def load_test_predictions(dataset: str, tcode: str) -> pd.DataFrame:
    """Per-sample seed-42 canonical run predictions, test split only."""
    path = (
        ROOT
        / "results"
        / f"mp_hzw_relaxed_clean_equal_atoms_{dataset}_t"
        / RUN_DIRS[tcode]
        / "predictions.csv"
    )
    df = pd.read_csv(path)
    return df[df["split"] == "test"].reset_index(drop=True)


def load_multiseed() -> pd.DataFrame:
    """Multi-seed MAE stats indexed by (source, T-code)."""
    df = pd.read_csv(ROOT / "results" / "analysis" / "multiseed_summary.csv")
    df["id"] = df["model"].str.extract(r"^(T\d+)")
    return df.set_index(["source", "id"])


def build_data_audit() -> None:
    """Two self-contained audit accounts (self-check issue A).

    The two accounts use different thresholds and MUST NOT be mixed on a single
    waterfall. Each panel closes on its own numbers:

    Panel (a) -- audit conflict statistics (metal-like <=0.05 eV or is_metal,
    nonmetal >=0.10 eV):
        95,462 entered = 92,787 matched + 2,636 no-MP + 39 unmatched
        92,787 matched = 92,507 exact + 280 supercell
        92,787 matched = 3,142 + 3,046 + 1,121 conflicts + 85,478 no-conflict

    Panel (b) -- T-series actual filtering (zero-gap threshold 1e-6 eV):
        92,787 -> -7,984 gap conflicts (2,935 + 5,049) -> 84,803
               -> -252 non equal-atom -> 84,551
               -> -1 export safety check -> 84,550
        84,550 -> split 67,640 / 8,455 / 8,455
    """
    # Panel (a): audit conflict account. Bars are grouped as
    # matching (top) then conflict breakdown within the matched set.
    audit = [
        ("Entered audit", 95462, BLUE),
        ("Structure matched", 92787, TEAL),
        ("No MP counterpart", 2636, MUTED),
        ("Structure unmatched", 39, MUTED),
        ("MP metal / HZW nonmetal", 3142, RED),
        ("MP nonmetal / HZW metal", 3046, RED),
        ("Same class, gap $\\Delta$ >1 eV", 1121, ORANGE),
        ("No conflict (kept)", 85478, GREEN),
    ]

    # Panel (b): T-series filtering waterfall, then the train/val/test split.
    filt = [
        ("Structure matched", 92787, TEAL),
        ("Gap-conflict removed", 84803, ORANGE),
        ("Equal-atom subset", 84551, GREEN),
        ("Export-safe final", 84550, GREEN),
    ]
    filt_notes = ["", "$-$7,984", "$-$252", "$-$1"]
    split = {"Train": 67640, "Validation": 8455, "Test": 8455}

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.4, 4.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # --- Panel (a) ---
    ax = axes[0]
    labels = [x[0].replace(" / ", " /\n").replace("Same class, ", "Same class,\n") for x in audit]
    values = [x[1] for x in audit]
    colors = [x[2] for x in audit]
    y = np.arange(len(audit))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 95462 * 1.20)
    ax.set_xlabel("Records")
    ax.set_title("Audit conflict statistics\n(metal-like $\\leq$0.05 eV or is_metal / nonmetal $\\geq$0.10 eV)", fontsize=8.5)
    for yi, value in zip(y, values):
        ax.text(value + 95462 * 0.015, yi, f"{value:,}", va="center", fontsize=7.5)
    ax.axhline(3.5, color=INK, linewidth=0.6, alpha=0.35)
    ax.text(
        0.985,
        0.44,
        "95,462 = 92,787 + 2,636 + 39\n"
        "matched = 92,507 exact + 280 supercell\n"
        "92,787 = 3,142 + 3,046 + 1,121 + 85,478",
        transform=ax.transAxes,
        ha="right",
        va="center",
        fontsize=6.3,
        color=MUTED,
    )
    ax.grid(axis="x", alpha=0.2)
    panel_label(ax, "(a)")

    # --- Panel (b) ---
    ax = axes[1]
    labels = [x[0] for x in filt]
    values = [x[1] for x in filt]
    colors = [x[2] for x in filt]
    y = np.arange(len(filt))
    ax.barh(y, values, color=colors, height=0.62)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 92787 * 1.16)
    ax.set_xlabel("Records")
    ax.set_title("T-series actual filtering\n(zero-gap threshold 1e-6 eV)", fontsize=8.5)
    for yi, value, note in zip(y, values, filt_notes):
        tag = f"{value:,}" + (f"  {note}" if note else "")
        ax.text(value + 92787 * 0.012, yi, tag, va="center", fontsize=7.5)
    ax.text(
        0.0,
        -0.16,
        "$-$7,984 gap conflicts = MP zero / HZW nonzero 2,935 + HZW zero / MP nonzero 5,049\n"
        f"Split (of 84,550): Train {split['Train']:,} / Val {split['Validation']:,} / Test {split['Test']:,}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=6.3,
        color=MUTED,
        clip_on=False,
    )
    ax.grid(axis="x", alpha=0.2)
    panel_label(ax, "(b)")

    fig.subplots_adjust(wspace=0.62, bottom=0.22, top=0.86, left=0.16, right=0.98)
    save(fig, "fig2_data_audit")


def build_main_results() -> None:
    mp = load_results("mp")
    hzw = load_results("hzw")
    seeds = load_multiseed()
    core = ["T1", "T2", "T3", "T4", "T7", "T10"]
    labels = AXIS_LABELS
    bar_colors = [MUTED, BLUE, MUTED, TEAL, ORANGE, GREEN]

    fig, axes = plt.subplots(
        2,
        3,
        figsize=(12.6, 7.7),
        gridspec_kw={"width_ratios": [1.1, 1.1, 1.15]},
    )

    # --- Row 1 (a)(b): per-source MAE bars (canonical seed-42 run) with
    # multi-seed error bars (+/- 1 s.d. from multiseed_summary.csv). Bar values
    # stay on the comparison_summary pipeline; only the error bars are new.
    for ax, df, source, title, letter in [
        (axes[0, 0], mp, "mp", "MP labels", "(a)"),
        (axes[0, 1], hzw, "hzw", "HZW labels", "(b)"),
    ]:
        values = [df.loc[x, "mae"] for x in core]
        errs = [seeds.loc[(source, x), "mae_std"] for x in core]
        n_seeds = sorted({int(seeds.loc[(source, x), "n_seeds"]) for x in core})
        y = np.arange(len(core))
        ax.barh(y, values, color=bar_colors, height=0.58)
        ax.errorbar(
            values,
            y,
            xerr=errs,
            fmt="none",
            ecolor=INK,
            elinewidth=0.9,
            capsize=2.4,
            capthick=0.9,
        )
        ax.set_yticks(y, [labels[x] for x in core])
        ax.invert_yaxis()
        ax.set_xlabel("Test MAE (eV)")
        ax.set_title(title)
        ax.axvline(df.loc["T2", "mae"], color=BLUE, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0.16, 0.40)
        for yi, value, err in zip(y, values, errs):
            ax.text(value + err + 0.005, yi, f"{value:.3f}", va="center", fontsize=8)
        ax.text(
            0.98,
            0.03,
            f"error bars: $\\pm$1 s.d. over {'/'.join(str(n) for n in n_seeds)} seeds",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=6.5,
            color=MUTED,
        )
        ax.grid(axis="x", alpha=0.2)
        panel_label(ax, letter)

    comparisons = ["ALIGNN$\\to$+VN\n(global token)", "+VN$\\to$+Huber\n(robust loss)", "+Huber$\\to$+decay\n(broadcast)"]
    gains = {
        "MP": [
            mp.loc["T2", "mae"] - mp.loc["T4", "mae"],
            mp.loc["T4", "mae"] - mp.loc["T7", "mae"],
            mp.loc["T7", "mae"] - mp.loc["T10", "mae"],
        ],
        "HZW": [
            hzw.loc["T2", "mae"] - hzw.loc["T4", "mae"],
            hzw.loc["T4", "mae"] - hzw.loc["T7", "mae"],
            hzw.loc["T7", "mae"] - hzw.loc["T10", "mae"],
        ],
    }
    ax = axes[0, 2]
    x = np.arange(len(comparisons))
    width = 0.34
    ax.bar(x - width / 2, gains["MP"], width, label="MP", color=BLUE)
    ax.bar(x + width / 2, gains["HZW"], width, label="HZW", color=ORANGE)
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, comparisons)
    # Self-check issue I: make the sign convention unambiguous. gains = MAE(prev)
    # - MAE(next), so positive means the newer variant has the lower MAE. This is
    # the OPPOSITE sign of the SI decay table (which reports positive = T10 worse);
    # the data direction is unchanged, only the label is disambiguated.
    ax.set_ylabel("MAE reduction vs\nprevious variant (eV)")
    ax.set_title("Factorized cross-source gains")
    ax.text(
        0.5,
        0.97,
        "positive = improvement (lower MAE)",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=MUTED,
    )
    ax.legend(frameon=False, loc="lower left")
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(c)")
    for xpos, value in zip(x - width / 2, gains["MP"]):
        offset = 0.001 if value >= 0 else 0.00035
        ax.text(xpos, value + offset, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    for xpos, value in zip(x + width / 2, gains["HZW"]):
        offset = 0.001 if value >= 0 else 0.00035
        ax.text(xpos, value + offset, f"{value:+.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=7)
    ax.set_ylim(-0.0045, 0.014)

    # --- Row 2 (d)(e): T7 parity density (test split, log-count hexbin).
    for ax, dataset, summary, title, letter in [
        (axes[1, 0], "mp", mp, "ALIGNN+VN+Huber, MP labels", "(d)"),
        (axes[1, 1], "hzw", hzw, "ALIGNN+VN+Huber, HZW labels", "(e)"),
    ]:
        pred = load_test_predictions(dataset, "T7")
        mae = (pred["prediction"] - pred["target"]).abs().mean()
        table_mae = summary.loc["T7", "mae"]
        if abs(mae - table_mae) > 5e-4:
            raise ValueError(
                f"Sanity check failed ({dataset} T7): predictions MAE {mae:.6f} "
                f"vs comparison_summary {table_mae:.6f}"
            )
        print(
            f"[sanity] T7 {dataset.upper()} test MAE from predictions.csv = {mae:.6f} "
            f"(comparison_summary {table_mae:.6f}, n={len(pred)})"
        )
        hb = ax.hexbin(
            pred["target"],
            pred["prediction"],
            gridsize=55,
            extent=(0, 10, 0, 10),
            cmap="viridis",
            norm=LogNorm(),
            mincnt=1,
            linewidths=0.1,
        )
        ax.plot([0, 10], [0, 10], color=INK, linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.set_aspect("equal")
        ax.set_xlabel("DFT band gap (eV)")
        ax.set_ylabel("Predicted band gap (eV)")
        ax.set_title(title)
        ax.text(
            0.04,
            0.96,
            f"MAE = {mae:.3f} eV\n$n$ = {len(pred):,} (test)",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.5,
        )
        cb = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.9)
        cb.set_label("Count", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        panel_label(ax, letter)

    # --- (f): residual histograms, MSE vs Huber virtual-node variants (MP).
    ax = axes[1, 2]
    bins = np.linspace(-1.5, 1.5, 76)
    for tcode, color, label in [
        ("T4", TEAL, "ALIGNN+VN (MSE)"),
        ("T7", ORANGE, "ALIGNN+VN+Huber"),
    ]:
        pred = load_test_predictions("mp", tcode)
        resid = pred["prediction"] - pred["target"]
        ax.hist(
            resid,
            bins=bins,
            color=color,
            alpha=0.55,
            label=label,
            edgecolor=color,
            linewidth=0.5,
        )
        n_shown = ((resid >= -1.5) & (resid <= 1.5)).sum()
        print(
            f"[sanity] fig3(f) {tcode} MP residuals: median {resid.median():+.4f} eV, "
            f"{n_shown}/{len(resid)} within [-1.5, 1.5] eV"
        )
    ax.axvline(0, color=INK, linewidth=0.8, alpha=0.6)
    ax.set_xlim(-1.5, 1.5)
    ax.set_box_aspect(1)
    ax.set_xlabel("Prediction $-$ target (eV)")
    ax.set_ylabel("Test-set count")
    ax.set_title("Residual distribution, MP labels")
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(f)")

    fig.subplots_adjust(wspace=0.46, hspace=0.42, bottom=0.09, top=0.94)
    save(fig, "fig3_main_results")


def build_mechanism_diagnostics() -> None:
    summary = pd.read_csv(ROOT / "figures" / "huber_diagnostics" / "huber_residual_summary.csv")
    # Full-test-set diagnostic (8,455 samples/source); the 240-sample pilot
    # lives in the un-suffixed oversmoothing_hooks_{mp,hzw} directories.
    smooth_mp = pd.read_csv(ROOT / "results" / "analysis" / "oversmoothing_hooks_mp_full" / "oversmoothing_final_layer_summary.csv")
    smooth_hzw = pd.read_csv(ROOT / "results" / "analysis" / "oversmoothing_hooks_hzw_full" / "oversmoothing_final_layer_summary.csv")

    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0))

    ax = axes[0, 0]
    quantiles = ["median_abs_error", "p90_abs_error", "p95_abs_error", "p99_abs_error"]
    qlabels = ["Median", "P90", "P95", "P99"]
    x = np.arange(len(quantiles))
    styles = [
        ("MP label", "T4 MSE", BLUE, "-", "o"),
        ("MP label", "T7 Huber", ORANGE, "-", "s"),
        ("HZW label", "T4 MSE", TEAL, "--", "o"),
        ("HZW label", "T7 Huber", GREEN, "--", "s"),
    ]
    for dataset, experiment, color, linestyle, marker in styles:
        row = summary[(summary["dataset"] == dataset) & (summary["experiment"] == experiment)].iloc[0]
        ax.plot(
            x,
            [row[q] for q in quantiles],
            marker=marker,
            linestyle=linestyle,
            color=color,
            linewidth=1.5,
            label=f"{dataset.split()[0]} {experiment.split()[1]}",
        )
    ax.set_xticks(x, qlabels)
    ax.set_ylabel("Absolute error (eV)")
    ax.set_title("Robust loss reduces typical error")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(a)")

    ax = axes[0, 1]
    decay = pd.DataFrame(
        {
            "Dataset": ["MP", "HZW"],
            "MAE": [-0.001245, 0.003032],
            "RMSE": [-0.006213, 0.007755],
            "Median AE": [-0.002057, 0.001005],
            "P90 AE": [0.006355, 0.009106],
        }
    )
    x = np.arange(len(decay.columns) - 1)
    width = 0.36
    ax.bar(x - width / 2, decay.loc[0, decay.columns[1:]], width, color=BLUE, label="MP")
    ax.bar(x + width / 2, decay.loc[1, decay.columns[1:]], width, color=ORANGE, label="HZW")
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set_xticks(x, decay.columns[1:])
    ax.set_ylabel("$(+$decay$)$ $-$ $(+$Huber$)$ (eV)")
    ax.set_title("Broadcast decay is label-source dependent")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(b)")

    def smooth_deltas(df: pd.DataFrame, metric: str) -> list[float]:
        sub = df[df["num_atoms_bucket"] == ">80"].set_index("experiment")
        return [sub.loc[m, metric] - sub.loc["T2", metric] for m in ["T4", "T7", "T10"]]

    models = ["T4", "T7", "T10"]
    x = np.arange(len(models))
    width = 0.36
    for ax, metric, ylabel, title, letter in [
        (axes[1, 0], "mean_pairwise_cosine", "Delta cosine", "Large-structure embedding similarity", "(c)"),
        (axes[1, 1], "node_feature_variance", "Delta variance", "Large-structure node diversity", "(d)"),
    ]:
        mp_values = smooth_deltas(smooth_mp, metric)
        hzw_values = smooth_deltas(smooth_hzw, metric)
        ax.bar(x - width / 2, mp_values, width, color=BLUE, label="MP")
        ax.bar(x + width / 2, hzw_values, width, color=ORANGE, label="HZW")
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(x, [AXIS_LABELS[m] for m in models])
        ax.set_ylabel(ylabel + " vs ALIGNN")
        ax.set_title(title + " (>80 atoms)")
        ax.legend(frameon=False, loc="best")
        ax.grid(axis="y", alpha=0.2)
        panel_label(ax, letter)

    fig.subplots_adjust(hspace=0.34, wspace=0.26, top=0.96)
    save(fig, "fig4_mechanism_diagnostics")


if __name__ == "__main__":
    build_data_audit()
    build_main_results()
    build_mechanism_diagnostics()
    print(f"Wrote composite figures to {OUT}")
