"""Build the physics-analysis composite figure (fig5) for the paper.

Three main-text panels driven by the verified per-sample prediction CSVs
(seed-42 canonical runs, test split only):

    (a) periodic-table heatmap of the per-element MAE change T7 - T2 (MP);
    (b) MAE by crystal system (T2/T4/T7, MP labels);
    (c) MAE by true band-gap bucket (T2/T4/T7, MP solid / HZW dashed).

The absolute per-element MAE heatmap that used to occupy the main-text panel
(a) is now written separately as figures/paper/figS5_element_mae_absolute for
the Supplementary Information: at \\textwidth two full periodic tables side by
side left each cell about 3 mm wide, and the absolute map is descriptive
context rather than evidence for a claim in the main text. The delta map, which
carries the claim, is drawn at roughly double the cell size in its place.

A sample containing element X contributes its absolute error to X. Elements
with fewer than MIN_ELEMENT_N test samples are greyed out. No randomness is
used anywhere in this script. Outputs: figures/paper/fig5_physics_analysis and
figures/paper/figS5_element_mae_absolute, each as PDF + PNG.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm, to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

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

# Same Okabe-Ito palette as build_academic_paper_figures.py (fixed categorical
# order: T2 = BLUE, T4 = TEAL, T7 = ORANGE across all paper figures).
BLUE = "#0072B2"
TEAL = "#009E73"
ORANGE = "#E69F00"
INK = "#222222"
MUTED = "#6B7280"
GREY_CELL = "#D6D6D6"

MODEL_COLORS = {"T2": BLUE, "T4": TEAL, "T7": ORANGE}
MODEL_LABELS = {"T2": "ALIGNN", "T4": "ALIGNN+VN", "T7": "ALIGNN+VN+Huber"}

RUN_DIRS = {
    "T2": "T2_ALIGNN_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
}

MIN_ELEMENT_N = 30

# ---------------------------------------------------------------------------
# Periodic table layout: symbol -> (row, col) in the standard 18-column form,
# lanthanides / actinides pulled out into two extra rows. Built from atomic
# number so no external dependency (pymatgen etc.) is needed.
# ---------------------------------------------------------------------------
ELEMENTS = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
    "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
    "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
    "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
    "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
    "Md", "No", "Lr",
]
ELEMENT_SET = set(ELEMENTS)


def _position(z: int) -> tuple[float, int]:
    """(row, col) for atomic number z, 1-based; La/Ac series on rows 8.7/9.7."""
    if z == 1:
        return (1, 1)
    if z == 2:
        return (1, 18)
    if z <= 10:  # Li..Ne
        return (2, z - 2) if z <= 4 else (2, z + 8)
    if z <= 18:  # Na..Ar
        return (3, z - 10) if z <= 12 else (3, z)
    if z <= 36:  # K..Kr
        return (4, z - 18)
    if z <= 54:  # Rb..Xe
        return (5, z - 36)
    if z <= 56:  # Cs, Ba
        return (6, z - 54)
    if z <= 71:  # La..Lu -> lanthanide row
        return (8.7, z - 57 + 3)
    if z <= 86:  # Hf..Rn
        return (6, z - 68)
    if z <= 88:  # Fr, Ra
        return (7, z - 86)
    if z <= 103:  # Ac..Lr -> actinide row
        return (9.7, z - 89 + 3)
    raise ValueError(f"No layout position for Z={z}")


POSITIONS = {symbol: _position(z) for z, symbol in enumerate(ELEMENTS, start=1)}

_TOKEN_RE = re.compile(r"[A-Z][a-z]?")


def parse_elements(formula: str) -> frozenset[str]:
    """Element set of a formula string, e.g. 'Li2FeSiO4' -> {Li, Fe, Si, O}.

    Two-letter symbols win automatically because the regex consumes a
    following lowercase letter when present ('NO6' -> N, O; 'Nb2O5' -> Nb, O).
    Digits and parentheses are ignored. Tokens that are not element symbols
    are dropped (none occur in the audit table; verified by the sanity print).
    Non-string input (NaN rows for records without an MP counterpart) maps to
    an empty set; such rows never appear in the T-series test split.
    """
    if not isinstance(formula, str):
        return frozenset()
    return frozenset(t for t in _TOKEN_RE.findall(formula) if t in ELEMENT_SET)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_predictions(dataset: str, tcode: str) -> pd.DataFrame:
    path = (
        ROOT
        / "results"
        / f"mp_hzw_relaxed_clean_equal_atoms_{dataset}_t"
        / RUN_DIRS[tcode]
        / "predictions.csv"
    )
    df = pd.read_csv(path)
    df = df[df["split"] == "test"].reset_index(drop=True)
    df["abs_err"] = (df["prediction"] - df["target"]).abs()
    return df


def load_metadata() -> pd.DataFrame:
    """sample_id -> element set (from mp_formula) and crystal system."""
    audit = pd.read_csv(
        ROOT / "results" / "data_quality" / "mp_hzw_full_audit_table.remote.csv",
        usecols=["sample_id", "hzw_hc_id", "mp_formula"],
    )
    struct = pd.read_csv(ROOT / "results" / "analysis" / "hzw_structmeta_lookup.csv")
    meta = audit.merge(struct, left_on="hzw_hc_id", right_on="hc_id", how="left")
    meta["elements"] = meta["mp_formula"].map(parse_elements)

    # Sanity: every token in every formula must be a known element symbol.
    unknown = set()
    for formula in meta["mp_formula"].dropna():
        unknown.update(t for t in _TOKEN_RE.findall(formula) if t not in ELEMENT_SET)
    if unknown:
        raise ValueError(f"Unrecognised element tokens in mp_formula: {sorted(unknown)}")
    return meta[["sample_id", "elements", "crystal_system"]]


def check_global_mae(dataset: str, tcode: str, pred: pd.DataFrame) -> None:
    summary = pd.read_csv(
        ROOT / "results" / f"mp_hzw_relaxed_clean_equal_atoms_{dataset}_t" / "comparison_summary.csv"
    )
    table_mae = summary.loc[
        summary["experiment_name"].str.startswith(f"{tcode}_"), "mae"
    ].iloc[0]
    mae = pred["abs_err"].mean()
    flag = "OK" if abs(mae - table_mae) <= 5e-4 else "MISMATCH"
    print(
        f"[sanity] {tcode} {dataset.upper()} global test MAE {mae:.6f} "
        f"vs comparison_summary {table_mae:.6f} -> {flag} (n={len(pred)})"
    )
    if flag != "OK":
        raise ValueError(f"Global MAE sanity check failed for {tcode} {dataset}")


# ---------------------------------------------------------------------------
# Per-element aggregation
# ---------------------------------------------------------------------------

def element_stats(pred: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """MAE and n per element; a sample counts for every element it contains."""
    merged = pred.merge(meta, on="sample_id", how="left", validate="one_to_one")
    if merged["elements"].isna().any():
        raise ValueError("Some test samples missing from the audit table")
    rows = merged[["abs_err", "elements"]].explode("elements")
    grouped = rows.groupby("elements")["abs_err"].agg(mae="mean", n="size")
    grouped.index.name = "element"
    return grouped


def _cell_text_color(facecolor) -> str:
    r, g, b = to_rgb(facecolor)
    return INK if (0.299 * r + 0.587 * g + 0.114 * b) > 0.55 else "white"


def draw_periodic_table(
    ax,
    fig,
    values,
    counts,
    cmap,
    norm,
    cbar_label,
    symbol_fontsize: float = 5.6,
    cbar_ticks=None,
    note_fontsize: float = 6.8,
) -> None:
    """values/counts: dict symbol -> value / n. Cells with n < MIN_ELEMENT_N grey."""
    for symbol in ELEMENTS:
        row, col = POSITIONS[symbol]
        n = counts.get(symbol, 0)
        if n >= MIN_ELEMENT_N:
            face = cmap(norm(values[symbol]))
            text_color = _cell_text_color(face)
        elif n > 0:
            face = GREY_CELL
            text_color = MUTED
        else:
            face = "white"
            text_color = "#C0C0C0"
        ax.add_patch(
            Rectangle(
                (col, -row),
                1.0,
                1.0,
                facecolor=face,
                edgecolor="#BBBBBB" if n == 0 else "white",
                linewidth=0.4,
            )
        )
        ax.text(
            col + 0.5,
            -row + 0.5,
            symbol,
            ha="center",
            va="center",
            fontsize=symbol_fontsize,
            color=text_color,
        )
    ax.set_xlim(0.7, 19.3)
    ax.set_ylim(-10.2, 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    cbar = fig.colorbar(
        ScalarMappable(norm=norm, cmap=cmap),
        ax=ax,
        orientation="horizontal",
        fraction=0.055,
        pad=0.02,
        shrink=0.62,
    )
    if cbar_ticks is not None:
        cbar.set_ticks(cbar_ticks)
    cbar.set_label(cbar_label, fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    # Two distinct greys are in play and reviewers conflate them: an element
    # present but under-sampled, versus an element absent from the test split.
    # Both are spelled out with a colour swatch rather than left to the caption.
    legend_items = [
        (GREY_CELL, f"$n<{MIN_ELEMENT_N}$ test structures (not scored)"),
        ("white", "element absent from the test split"),
    ]
    for k, (face, text) in enumerate(legend_items):
        y = 0.985 - k * 0.055
        ax.add_patch(
            Rectangle(
                (0.615, y - 0.030),
                0.030,
                0.034,
                transform=ax.transAxes,
                facecolor=face,
                edgecolor="#BBBBBB",
                linewidth=0.4,
                clip_on=False,
            )
        )
        ax.text(
            0.655,
            y - 0.013,
            text,
            transform=ax.transAxes,
            ha="left",
            va="center",
            fontsize=note_fontsize,
            color=MUTED,
        )


def panel_label(ax, label: str, x: float = -0.06) -> None:
    ax.text(
        x,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        va="top",
        ha="left",
        color=INK,
    )


# ---------------------------------------------------------------------------
# Gap buckets
# ---------------------------------------------------------------------------
BUCKET_LABELS = ["0\n(metal)", "(0,0.5]", "(0.5,1]", "(1,2]", "(2,3]", "(3,4]", ">4"]


def bucket_index(target: pd.Series) -> pd.Series:
    edges = [0.5, 1.0, 2.0, 3.0, 4.0]
    idx = pd.Series(np.searchsorted(edges, target.to_numpy(), side="left") + 1, index=target.index)
    idx[target <= 1e-6] = 0
    return idx


def main() -> None:
    meta = load_metadata()
    preds = {
        (src, t): load_test_predictions(src, t)
        for src in ("mp", "hzw")
        for t in ("T2", "T4", "T7")
    }
    for (src, t), pred in preds.items():
        check_global_mae(src, t, pred)

    # --- Per-element stats (MP labels) for panels (a)/(b) ---
    el7 = element_stats(preds[("mp", "T7")], meta)
    el2 = element_stats(preds[("mp", "T2")], meta)
    assert (el7["n"] == el2["n"]).all(), "T2/T7 share the test split; counts must match"

    cov = el7.sort_values("n", ascending=False)
    print("[sanity] element coverage top5:", [(i, int(r.n)) for i, r in cov.head(5).iterrows()])
    print("[sanity] element coverage bottom5:", [(i, int(r.n)) for i, r in cov.tail(5).iterrows()])
    qual = el7[el7["n"] >= MIN_ELEMENT_N]
    print(
        f"[sanity] elements present: {len(el7)}, with n>={MIN_ELEMENT_N}: {len(qual)}; "
        f"T7 per-element MAE range {qual['mae'].min():.3f}-{qual['mae'].max():.3f} eV "
        f"(min {qual['mae'].idxmin()}, max {qual['mae'].idxmax()})"
    )

    # Main text: one periodic table (the delta map, which carries the claim)
    # spanning the full width on top, with the two aggregate panels below. The
    # absolute map is written separately for the SI.
    fig = plt.figure(figsize=(12.8, 9.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.45, 1.0], hspace=0.26, wspace=0.20)
    ax_delta = fig.add_subplot(gs[0, :])
    ax_sys = fig.add_subplot(gs[1, 0])
    ax_bucket = fig.add_subplot(gs[1, 1])

    # --- (a) per-element delta MAE T7 - T2, main text ---
    delta = (el7["mae"] - el2["mae"]).loc[qual.index]
    lim = float(np.abs(delta).max())
    # TwoSlopeNorm already pins white at 0; the tick makes the neutral point
    # readable on the colorbar itself instead of implied by the caption.
    step = lim / 2.0
    delta_ticks = [-lim, -step, 0.0, step, lim]
    norm_b = TwoSlopeNorm(vmin=-lim, vcenter=0.0, vmax=lim)
    draw_periodic_table(
        ax_delta,
        fig,
        delta.to_dict(),
        el7["n"].to_dict(),
        plt.get_cmap("RdBu_r"),
        norm_b,
        "$\\Delta$MAE = (VN+Huber) $-$ ALIGNN (eV);  blue = improvement, white = no change",
        symbol_fontsize=8.0,
        cbar_ticks=delta_ticks,
        note_fontsize=7.5,
    )
    ax_delta.set_title(
        "Where the VN+Huber gain lives: per-element MAE change vs ALIGNN, MP labels",
        fontsize=10,
    )
    panel_label(ax_delta, "(a)", x=0.0)
    print(
        f"[sanity] per-element dMAE range {delta.min():+.3f} ({delta.idxmin()}) to "
        f"{delta.max():+.3f} ({delta.idxmax()}) eV; improved elements: "
        f"{int((delta < 0).sum())}/{len(delta)}"
    )

    # --- (b) MAE by crystal system (MP labels) ---
    systems = ["triclinic", "monoclinic", "orthorhombic", "tetragonal", "trigonal", "hexagonal", "cubic"]
    merged = {
        t: preds[("mp", t)].merge(meta, on="sample_id", how="left") for t in ("T2", "T4", "T7")
    }
    counts = merged["T7"]["crystal_system"].value_counts()
    dropped = merged["T7"]["crystal_system"].isna().sum()
    print("[sanity] crystal-system test counts:", {s: int(counts.get(s, 0)) for s in systems},
          f"(no metadata: {dropped})")

    ax = ax_sys
    x = np.arange(len(systems))
    width = 0.26
    for k, t in enumerate(("T2", "T4", "T7")):
        by_sys = merged[t].groupby("crystal_system")["abs_err"].mean()
        ax.bar(
            x + (k - 1) * width,
            [by_sys[s] for s in systems],
            width,
            color=MODEL_COLORS[t],
            label=MODEL_LABELS[t],
        )
    ax.set_xticks(
        x,
        [f"{s}\n$n$={int(counts.get(s, 0)):,}" for s in systems],
        fontsize=7,
    )
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("MAE by crystal system (symmetry ascending), MP labels", fontsize=9.5)
    ax.legend(frameon=False, loc="upper right", fontsize=7.5)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(b)", x=-0.10)

    # --- (c) MAE by true band-gap bucket, MP solid / HZW dashed ---
    # Model identity is carried by colour AND marker shape; label source by
    # line style AND marker fill, so the panel survives greyscale printing.
    ax = ax_bucket
    xb = np.arange(len(BUCKET_LABELS))
    markers = {"T2": "o", "T4": "s", "T7": "^"}
    for src, linestyle, fill in (("mp", "-", "full"), ("hzw", "--", "none")):
        for t in ("T2", "T4", "T7"):
            pred = preds[(src, t)]
            by_bucket = pred.groupby(bucket_index(pred["target"]))["abs_err"].mean()
            ax.plot(
                xb,
                [by_bucket.get(i, np.nan) for i in range(len(BUCKET_LABELS))],
                color=MODEL_COLORS[t],
                linestyle=linestyle,
                marker=markers[t],
                markersize=5,
                fillstyle=fill,
                markerfacecolor=MODEL_COLORS[t] if fill == "full" else "white",
                markeredgecolor=MODEL_COLORS[t],
                markeredgewidth=1.1,
                linewidth=1.4,
            )
        n_per = bucket_index(preds[(src, "T7")]["target"]).value_counts().sort_index()
        print(f"[sanity] {src.upper()} gap-bucket test counts:",
              {BUCKET_LABELS[i].split(chr(10))[0]: int(n_per.get(i, 0)) for i in range(len(BUCKET_LABELS))})
    ax.set_xticks(xb, BUCKET_LABELS, fontsize=7)
    ax.set_xlabel("True band gap (eV)")
    ax.set_ylabel("Test MAE (eV)")
    ax.set_title("MAE by true band-gap bucket", fontsize=9.5)
    handles = [
        Line2D([], [], color=MODEL_COLORS[t], marker=markers[t], markersize=5, linewidth=1.4,
               label=MODEL_LABELS[t])
        for t in ("T2", "T4", "T7")
    ] + [
        Line2D([], [], color=INK, linestyle="-", marker="o", markersize=5,
               markerfacecolor=INK, linewidth=1.4, label="MP labels (filled)"),
        Line2D([], [], color=INK, linestyle="--", marker="o", markersize=5,
               markerfacecolor="white", markeredgecolor=INK, linewidth=1.4,
               label="HZW labels (open)"),
    ]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.2)
    panel_label(ax, "(c)", x=-0.10)

    fig.subplots_adjust(top=0.95, bottom=0.07, left=0.06, right=0.98)
    fig.savefig(OUT / "fig5_physics_analysis.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(OUT / "fig5_physics_analysis.png", dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print(f"Wrote fig5_physics_analysis.[pdf|png] to {OUT}")

    # --- Supplementary: absolute per-element MAE for T7 ---
    figs, axs = plt.subplots(1, 1, figsize=(9.0, 6.4))
    norm_a = Normalize(vmin=qual["mae"].min(), vmax=qual["mae"].max())
    draw_periodic_table(
        axs,
        figs,
        qual["mae"].to_dict(),
        el7["n"].to_dict(),
        plt.get_cmap("viridis"),
        norm_a,
        "Per-element test MAE (eV)",
        symbol_fontsize=7.0,
        note_fontsize=7.0,
    )
    axs.set_title(
        "Absolute per-element test MAE, ALIGNN+VN+Huber, MP labels",
        fontsize=10,
    )
    figs.savefig(OUT / "figS5_element_mae_absolute.pdf", bbox_inches="tight", pad_inches=0.04)
    figs.savefig(OUT / "figS5_element_mae_absolute.png", dpi=240, bbox_inches="tight", pad_inches=0.04)
    plt.close(figs)
    print(f"Wrote figS5_element_mae_absolute.[pdf|png] to {OUT}")


if __name__ == "__main__":
    main()
