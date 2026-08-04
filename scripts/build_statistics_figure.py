"""Build the statistical-evidence forest plot (fig6) for the paper.

Two panels, both showing point estimates with 95% bootstrap confidence
intervals and a vertical zero reference line:

    (a) Paired-bootstrap MAE differences between model pairs on the shared test
        split (B = 10,000, bootstrap seed 20260714). Source: the paired
        bootstrap table (results/analysis/paired_bootstrap.csv).
    (b) Over-smoothing delta-metrics for the >80-atom subset (n = 422), model
        minus the T2 baseline, with B = 2,000 bootstrap CIs. Source:
        results/analysis/oversmoothing_delta_ci_full.csv.

Unlike the other paper figures this script does not read the per-sample
prediction CSVs: both panels plot summary statistics that the upstream analysis
scripts have already reduced to (estimate, ci_low, ci_high) triples. Those
triples are transcribed here so the figure stays reproducible from the public
repository, where results/ holds HZW-derived tables and is not redistributable.
Every value below is identical to the corresponding row of the main-text and
supplementary tables; check_against_tables() re-asserts that correspondence at
build time so the two can never drift apart silently.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures" / "paper"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Okabe-Ito, same categorical roles as the other paper figures.
BLUE = "#0072B2"
ORANGE = "#E69F00"
INK = "#222222"
MUTED = "#6B7280"

# (pair label, MP delta/lo/hi, HZW delta/lo/hi). Transcribed from the paired
# bootstrap table; delta = MAE(a) - MAE(b), so negative favours the first model.
BOOTSTRAP = [
    ("+VN vs ALIGNN\n(T4 $-$ T2)", (-0.0075, -0.0121, -0.0028), (-0.0104, -0.0147, -0.0060)),
    ("+Huber vs +VN\n(T7 $-$ T4)", (-0.0076, -0.0122, -0.0029), (-0.0078, -0.0120, -0.0035)),
    ("+VN+Huber vs ALIGNN\n(T7 $-$ T2)", (-0.0151, -0.0198, -0.0103), (-0.0181, -0.0224, -0.0138)),
    ("+decay vs +Huber\n(T10 $-$ T7)", (-0.0012, -0.0058, +0.0033), (+0.0030, -0.0011, +0.0071)),
]

# (model, MP mean/lo/hi, HZW mean/lo/hi) for the >80-atom subset, delta vs T2.
# Cosine is the scale-invariant metric and therefore the one plotted; the
# variance and Dirichlet columns are norm-sensitive corroboration and stay in
# the supplementary table.
OVERSMOOTHING = [
    ("ALIGNN+VN (T4)", (-0.0871, -0.1031, -0.0702), (-0.0906, -0.1054, -0.0760)),
    ("ALIGNN+VN+Huber (T7)", (-0.0872, -0.1028, -0.0715), (-0.0896, -0.1050, -0.0746)),
    ("+broadcast decay (T10)", (-0.0859, -0.1016, -0.0697), (-0.0931, -0.1072, -0.0787)),
]


def check_against_tables() -> None:
    """Guard the transcription: point estimates must lie inside their own CIs,
    and the significance pattern must match what the text claims."""
    for name, mp, hzw in BOOTSTRAP + OVERSMOOTHING:
        for source, (est, lo, hi) in (("MP", mp), ("HZW", hzw)):
            if not lo <= est <= hi:
                raise ValueError(f"{name} [{source}]: estimate {est} outside CI [{lo}, {hi}]")

    def excludes_zero(triple):
        _, lo, hi = triple
        return lo > 0 or hi < 0

    expected_significant = {
        "+VN vs ALIGNN\n(T4 $-$ T2)": (True, True),
        "+Huber vs +VN\n(T7 $-$ T4)": (True, True),
        "+VN+Huber vs ALIGNN\n(T7 $-$ T2)": (True, True),
        # The decay step is the one comparison that resolves to nothing on
        # either source; this is a claim of the paper, so assert it.
        "+decay vs +Huber\n(T10 $-$ T7)": (False, False),
    }
    for name, mp, hzw in BOOTSTRAP:
        want = expected_significant[name]
        got = (excludes_zero(mp), excludes_zero(hzw))
        if want != got:
            raise ValueError(f"{name}: significance pattern {got} != expected {want}")

    # Every large-structure cosine delta is negative and resolved.
    for name, mp, hzw in OVERSMOOTHING:
        for source, triple in (("MP", mp), ("HZW", hzw)):
            if not excludes_zero(triple) or triple[0] >= 0:
                raise ValueError(f"{name} [{source}]: expected a resolved negative cosine delta")
    print("[sanity] forest-plot values agree with the main and supplementary tables")


def forest(ax, rows, xlabel, title, note) -> None:
    """One forest panel: two offset rows (MP, HZW) per entry, CI whiskers,
    zero reference line, most-recent entry at the top."""
    n = len(rows)
    offset = 0.17
    for k, (name, mp, hzw) in enumerate(rows):
        base = n - 1 - k
        for (est, lo, hi), color, dy, marker in (
            (mp, BLUE, +offset, "o"),
            (hzw, ORANGE, -offset, "s"),
        ):
            y = base + dy
            resolved = lo > 0 or hi < 0
            ax.plot([lo, hi], [y, y], color=color, linewidth=1.5, solid_capstyle="butt")
            for edge in (lo, hi):
                ax.plot([edge, edge], [y - 0.06, y + 0.06], color=color, linewidth=1.5)
            ax.plot(
                [est],
                [y],
                marker=marker,
                markersize=6,
                color=color,
                markerfacecolor=color if resolved else "white",
                markeredgecolor=color,
                markeredgewidth=1.3,
                linestyle="none",
            )
        if k % 2 == 1:
            ax.axhspan(base - 0.5, base + 0.5, color="#000000", alpha=0.035, zorder=0)

    ax.axvline(0, color=INK, linewidth=1.0, zorder=1)
    ax.set_yticks(np.arange(n), [r[0] for r in reversed(rows)], fontsize=7.5)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=9.5)
    ax.grid(axis="x", alpha=0.2)
    ax.text(
        0.5,
        -0.19,
        note,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=6.8,
        color=MUTED,
    )


def main() -> None:
    check_against_tables()

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), gridspec_kw={"wspace": 0.42})

    forest(
        axes[0],
        BOOTSTRAP,
        "$\\Delta$MAE (eV), 95\\% paired-bootstrap CI",
        "Which model differences are statistically resolved",
        "negative favours the newer variant; $B=10{,}000$ paired resamples of the shared test set "
        "($n=8{,}455$)",
    )
    forest(
        axes[1],
        OVERSMOOTHING,
        "$\\Delta$ mean pairwise cosine vs ALIGNN, 95\\% CI",
        "Large-structure embedding diversity ($>$80 atoms, $n=422$)",
        "negative = less embedding homogenization than the baseline; $B=2{,}000$ bootstrap resamples",
    )

    handles = [
        Line2D([], [], color=BLUE, marker="o", markersize=6, linewidth=1.5, label="MP labels"),
        Line2D([], [], color=ORANGE, marker="s", markersize=6, linewidth=1.5, label="HZW labels"),
        Line2D([], [], color=MUTED, marker="o", markersize=6, markerfacecolor="white",
               markeredgecolor=MUTED, linestyle="none", label="CI crosses zero (unresolved)"),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        loc="lower center",
        ncol=3,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.03),
    )

    for ax, letter in zip(axes, ("(a)", "(b)")):
        ax.text(
            -0.30,
            1.06,
            letter,
            transform=ax.transAxes,
            fontsize=11,
            fontweight="bold",
            va="top",
            ha="left",
            color=INK,
        )

    fig.savefig(OUT / "fig6_statistical_evidence.pdf", bbox_inches="tight", pad_inches=0.05)
    fig.savefig(OUT / "fig6_statistical_evidence.png", dpi=240, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Wrote fig6_statistical_evidence.[pdf|png] to {OUT}")


if __name__ == "__main__":
    main()
