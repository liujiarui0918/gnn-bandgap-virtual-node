from __future__ import annotations

# =============================================================================
# A6 -- Chemical/structural profiling of decay-harmed vs decay-benefited HZW
# cohorts (response to reviewer comment 5).
#
# Question: which crystals depend on *persistent* global calibration? We split
# HZW test samples by per-sample delta-AE = abs_error(T10_DecayHuber) -
# abs_error(T7_Huber). Worst top 10% (delta>0, T10 hurt them = "decay-harmed")
# vs best top 10% (delta<0, T10 helped them = "decay-benefited"), then contrast
# element families / crystal system / space group / gap x atom / label gap-diff.
# Proportion gaps carry bootstrap CIs (default_rng(20260714), B=2000).
#
# INPUTS
#   1. results/analysis/global_decay_transfer_records.csv
#         per-(sample,model) predictions; we pair T7_Huber vs T10_DecayHuber on
#         dataset==HZW to get per-sample delta-AE + atom/gap/label-diff buckets.
#   2. results/data_quality/mp_hzw_full_audit_table.remote.csv
#         sample_id -> hzw_hc_id, hzw_formula, ... (join key sample_id).
#   3. results/analysis/hzw_structmeta_lookup.csv   (built by
#         scripts/extract_hzw_structmeta.py)
#         hc_id -> space_group_number, crystal_system. Joined via audit.hzw_hc_id.
#
# OUTPUTS (results/analysis/decay_mechanism/)
#   decay_cohorts_profile.csv     categorical proportions + numeric means with
#                                 harmed/benefited values, diff, bootstrap CI.
#   decay_cohorts_crosstab.csv    target_gap_bucket x atom_bucket counts/props.
#   decay_cohorts_profile_table.md  human-readable side-by-side comparison.
# =============================================================================

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "analysis"
OUT_DIR = ANALYSIS_DIR / "decay_mechanism"
OUT_DIR.mkdir(parents=True, exist_ok=True)

RECORDS_CSV = ANALYSIS_DIR / "global_decay_transfer_records.csv"
AUDIT_CSV = ROOT / "results" / "data_quality" / "mp_hzw_full_audit_table.remote.csv"
STRUCT_CSV = ANALYSIS_DIR / "hzw_structmeta_lookup.csv"

SEED = 20260714
N_BOOT = 2000
TOP_FRAC = 0.10

# --- element families ---------------------------------------------------------
TRANSITION_METALS = {
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
}
HALOGENS = {"F", "Cl", "Br", "I", "At"}
CHALCOGENS = {"O", "S", "Se", "Te", "Po"}  # 氧族 (group 16)

_ELEMENT_RE = re.compile(r"([A-Z][a-z]?)")

ATOM_ORDER = ["<=10", "11-20", "21-40", "41-80", "81-160", ">160"]
GAP_ORDER = ["0", "(0,0.5)", "[0.5,1)", "[1,2)", "[2,4)", ">=4"]
DIFF_ORDER = ["<0.1", "[0.1,0.5)", "[0.5,1)", ">=1"]
CRYSTAL_ORDER = [
    "cubic", "hexagonal", "trigonal", "tetragonal",
    "orthorhombic", "monoclinic", "triclinic", "unknown",
]


def parse_elements(formula: str) -> set[str]:
    if not formula:
        return set()
    return set(_ELEMENT_RE.findall(formula))


def symmetry_class(sg: int | None) -> str:
    # coarse crystallographic symmetry tier by space-group number
    if sg is None:
        return "unknown"
    if sg <= 74:
        return "low(1-74)"
    if sg <= 167:
        return "medium(75-167)"
    return "high(168-230)"


def load_struct_lookup() -> dict[str, dict[str, str]]:
    with STRUCT_CSV.open(newline="", encoding="utf-8") as fh:
        return {row["hc_id"]: row for row in csv.DictReader(fh)}


def load_audit() -> dict[str, dict[str, str]]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        return {row["sample_id"]: row for row in csv.DictReader(fh)}


def build_samples() -> list[dict]:
    """One row per HZW test sample with delta-AE + chem/struct features."""
    audit = load_audit()
    struct = load_struct_lookup()

    # pair T7 vs T10 on HZW
    paired: dict[str, dict] = defaultdict(dict)
    with RECORDS_CSV.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["dataset"] != "HZW":
                continue
            sid = row["sample_id"]
            paired[sid]["meta"] = row
            paired[sid][row["model"]] = float(row["abs_error"])

    samples: list[dict] = []
    n_no_struct = 0
    for sid, rec in paired.items():
        if "T7_Huber" not in rec or "T10_DecayHuber" not in rec:
            continue
        meta = rec["meta"]
        arow = audit.get(sid, {})
        hc_id = arow.get("hzw_hc_id", "")
        srow = struct.get(hc_id)
        if srow is None:
            n_no_struct += 1
        formula = arow.get("hzw_formula", "")
        elems = parse_elements(formula)

        sg_raw = (srow or {}).get("space_group_number", "")
        try:
            sg = int(float(sg_raw)) if sg_raw not in ("", None) else None
        except (TypeError, ValueError):
            sg = None
        cs = (srow or {}).get("crystal_system") or "unknown"

        try:
            gap_diff = abs(float(meta.get("mp_hzw_gap_diff") or 0.0))
        except ValueError:
            gap_diff = 0.0

        samples.append(
            {
                "sample_id": sid,
                "delta_ae": rec["T10_DecayHuber"] - rec["T7_Huber"],
                "t7_ae": rec["T7_Huber"],
                "t10_ae": rec["T10_DecayHuber"],
                "has_transition_metal": bool(elems & TRANSITION_METALS),
                "has_halogen": bool(elems & HALOGENS),
                "has_chalcogen": bool(elems & CHALCOGENS),
                "crystal_system": cs,
                "space_group_number": sg,
                "symmetry_class": symmetry_class(sg),
                "atom_bucket": meta.get("atom_bucket", ""),
                "target_gap_bucket": meta.get("target_gap_bucket", ""),
                "mp_hzw_gap_diff": gap_diff,
                "mp_hzw_gap_diff_bucket": meta.get("mp_hzw_gap_diff_bucket", ""),
                "mp_is_metal": meta.get("mp_is_metal", "") == "True",
                "hzw_is_metal": meta.get("hzw_is_metal", "") == "True",
            }
        )
    print(
        f"[build] HZW paired samples={len(samples)} "
        f"(without struct join={n_no_struct})"
    )
    return samples


# --- cohort split -------------------------------------------------------------
def split_cohorts(samples: list[dict]) -> tuple[list[dict], list[dict], dict]:
    ordered = sorted(samples, key=lambda s: s["delta_ae"])
    n = len(ordered)
    k = max(1, round(TOP_FRAC * n))
    benefited = ordered[:k]                 # most negative delta -> T10 helped
    harmed = ordered[-k:]                   # most positive delta -> T10 hurt
    info = {
        "n_total": n,
        "cohort_size": k,
        "benefited_delta_max": benefited[-1]["delta_ae"],
        "harmed_delta_min": harmed[0]["delta_ae"],
        "benefited_mean_delta": float(np.mean([s["delta_ae"] for s in benefited])),
        "harmed_mean_delta": float(np.mean([s["delta_ae"] for s in harmed])),
    }
    return harmed, benefited, info


# --- bootstrap ----------------------------------------------------------------
def bootstrap_diff(
    harmed_vec: np.ndarray, benefit_vec: np.ndarray,
    h_idx: np.ndarray, b_idx: np.ndarray,
) -> tuple[float, float, float]:
    """Diff of means (harmed - benefited) + percentile CI over shared resamples."""
    point = float(harmed_vec.mean() - benefit_vec.mean())
    boot = harmed_vec[h_idx].mean(axis=1) - benefit_vec[b_idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def categorical_rows(
    dimension: str, values: list[str], harmed: list[dict], benefited: list[dict],
    field: str, h_idx: np.ndarray, b_idx: np.ndarray,
) -> list[dict]:
    nh, nb = len(harmed), len(benefited)
    h_field = np.array([s[field] for s in harmed])
    b_field = np.array([s[field] for s in benefited])
    rows = []
    for val in values:
        h_vec = (h_field == val).astype(float)
        b_vec = (b_field == val).astype(float)
        if h_vec.sum() == 0 and b_vec.sum() == 0:
            continue
        diff, lo, hi = bootstrap_diff(h_vec, b_vec, h_idx, b_idx)
        rows.append(
            {
                "dimension": dimension,
                "category": val,
                "harmed_n": int(h_vec.sum()),
                "harmed_prop": float(h_vec.mean()),
                "benefited_n": int(b_vec.sum()),
                "benefited_prop": float(b_vec.mean()),
                "diff_harmed_minus_benefited": diff,
                "ci_low": lo,
                "ci_high": hi,
                "significant": (lo > 0) or (hi < 0),
            }
        )
    return rows


def boolean_row(
    dimension: str, harmed: list[dict], benefited: list[dict], field: str,
    h_idx: np.ndarray, b_idx: np.ndarray,
) -> dict:
    h_vec = np.array([1.0 if s[field] else 0.0 for s in harmed])
    b_vec = np.array([1.0 if s[field] else 0.0 for s in benefited])
    diff, lo, hi = bootstrap_diff(h_vec, b_vec, h_idx, b_idx)
    return {
        "dimension": dimension,
        "category": field,
        "harmed_n": int(h_vec.sum()),
        "harmed_prop": float(h_vec.mean()),
        "benefited_n": int(b_vec.sum()),
        "benefited_prop": float(b_vec.mean()),
        "diff_harmed_minus_benefited": diff,
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0) or (hi < 0),
    }


def numeric_row(
    dimension: str, harmed: list[dict], benefited: list[dict], field: str,
    h_idx: np.ndarray, b_idx: np.ndarray,
) -> dict:
    h_vec = np.array([float(s[field]) for s in harmed])
    b_vec = np.array([float(s[field]) for s in benefited])
    diff, lo, hi = bootstrap_diff(h_vec, b_vec, h_idx, b_idx)
    return {
        "dimension": dimension,
        "category": field + " (mean)",
        "harmed_n": len(harmed),
        "harmed_prop": float(h_vec.mean()),
        "benefited_n": len(benefited),
        "benefited_prop": float(b_vec.mean()),
        "diff_harmed_minus_benefited": diff,
        "ci_low": lo,
        "ci_high": hi,
        "significant": (lo > 0) or (hi < 0),
    }


def crosstab(cohort: list[dict]) -> dict[tuple[str, str], int]:
    tab: dict[tuple[str, str], int] = defaultdict(int)
    for s in cohort:
        tab[(s["target_gap_bucket"], s["atom_bucket"])] += 1
    return tab


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def fmt(x) -> str:
    if isinstance(x, bool):
        return "yes" if x else ""
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main() -> None:
    samples = build_samples()
    harmed, benefited, info = split_cohorts(samples)
    rng = np.random.default_rng(SEED)
    nh, nb = len(harmed), len(benefited)
    h_idx = rng.integers(0, nh, size=(N_BOOT, nh))
    b_idx = rng.integers(0, nb, size=(N_BOOT, nb))

    rows: list[dict] = []
    rows.append(boolean_row("element_family", harmed, benefited, "has_transition_metal", h_idx, b_idx))
    rows.append(boolean_row("element_family", harmed, benefited, "has_halogen", h_idx, b_idx))
    rows.append(boolean_row("element_family", harmed, benefited, "has_chalcogen", h_idx, b_idx))
    rows.append(boolean_row("label", harmed, benefited, "mp_is_metal", h_idx, b_idx))
    rows.append(boolean_row("label", harmed, benefited, "hzw_is_metal", h_idx, b_idx))
    rows += categorical_rows("crystal_system", CRYSTAL_ORDER, harmed, benefited, "crystal_system", h_idx, b_idx)
    rows += categorical_rows(
        "symmetry_class", ["low(1-74)", "medium(75-167)", "high(168-230)", "unknown"],
        harmed, benefited, "symmetry_class", h_idx, b_idx,
    )
    rows += categorical_rows("atom_bucket", ATOM_ORDER, harmed, benefited, "atom_bucket", h_idx, b_idx)
    rows += categorical_rows("target_gap_bucket", GAP_ORDER, harmed, benefited, "target_gap_bucket", h_idx, b_idx)
    rows += categorical_rows("gap_diff_bucket", DIFF_ORDER, harmed, benefited, "mp_hzw_gap_diff_bucket", h_idx, b_idx)
    rows.append(numeric_row("numeric", harmed, benefited, "mp_hzw_gap_diff", h_idx, b_idx))
    rows.append(numeric_row_sg(harmed, benefited, h_idx, b_idx))

    write_csv(OUT_DIR / "decay_cohorts_profile.csv", rows)

    # crosstab CSV (gap x atom)
    h_tab, b_tab = crosstab(harmed), crosstab(benefited)
    keys = sorted(set(h_tab) | set(b_tab), key=lambda k: (GAP_ORDER.index(k[0]) if k[0] in GAP_ORDER else 99,
                                                          ATOM_ORDER.index(k[1]) if k[1] in ATOM_ORDER else 99))
    cross_rows = []
    for g, a in keys:
        cross_rows.append(
            {
                "target_gap_bucket": g,
                "atom_bucket": a,
                "harmed_count": h_tab.get((g, a), 0),
                "harmed_prop": h_tab.get((g, a), 0) / nh,
                "benefited_count": b_tab.get((g, a), 0),
                "benefited_prop": b_tab.get((g, a), 0) / nb,
            }
        )
    write_csv(OUT_DIR / "decay_cohorts_crosstab.csv", cross_rows)

    write_markdown(info, rows, cross_rows)

    sig = [r for r in rows if r["significant"]]
    print(f"[cohorts] size={info['cohort_size']} each of {info['n_total']} HZW samples")
    print(f"[cohorts] harmed mean delta-AE={info['harmed_mean_delta']:.4f}, "
          f"benefited mean delta-AE={info['benefited_mean_delta']:.4f}")
    print(f"[cohorts] significant dimensions (CI excludes 0): {len(sig)}/{len(rows)}")
    for r in sig:
        print(f"   {r['dimension']:>18} :: {r['category']:<22} "
              f"harmed={r['harmed_prop']:.3f} benefited={r['benefited_prop']:.3f} "
              f"diff={r['diff_harmed_minus_benefited']:+.3f} "
              f"CI[{r['ci_low']:+.3f},{r['ci_high']:+.3f}]")


def numeric_row_sg(harmed, benefited, h_idx, b_idx) -> dict:
    # space_group_number restricted to samples with a known sg, but bootstrap
    # indices span the full cohort; use NaN-safe means so unknowns don't bias.
    h_all = np.array([s["space_group_number"] if s["space_group_number"] is not None else np.nan for s in harmed])
    b_all = np.array([s["space_group_number"] if s["space_group_number"] is not None else np.nan for s in benefited])
    point = float(np.nanmean(h_all) - np.nanmean(b_all))
    boot = np.nanmean(h_all[h_idx], axis=1) - np.nanmean(b_all[b_idx], axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "dimension": "numeric",
        "category": "space_group_number (mean)",
        "harmed_n": int(np.sum(~np.isnan(h_all))),
        "harmed_prop": float(np.nanmean(h_all)),
        "benefited_n": int(np.sum(~np.isnan(b_all))),
        "benefited_prop": float(np.nanmean(b_all)),
        "diff_harmed_minus_benefited": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "significant": (lo > 0) or (hi < 0),
    }


def write_markdown(info: dict, rows: list[dict], cross_rows: list[dict]) -> None:
    path = OUT_DIR / "decay_cohorts_profile_table.md"
    lines = [
        "# HZW decay-harmed vs decay-benefited cohort profile (A6)",
        "",
        "Per-sample delta-AE = abs_error(T10_DecayHuber) - abs_error(T7_Huber) on HZW.",
        f"Cohort size = top {int(TOP_FRAC*100)}% each = {info['cohort_size']} of "
        f"{info['n_total']} HZW test samples.",
        "",
        f"- **decay-harmed** (T10 worse): mean delta-AE = {info['harmed_mean_delta']:.4f} "
        f"(min in cohort = {info['harmed_delta_min']:.4f})",
        f"- **decay-benefited** (T10 better): mean delta-AE = {info['benefited_mean_delta']:.4f} "
        f"(max in cohort = {info['benefited_delta_max']:.4f})",
        "",
        "`significant` = 95% bootstrap CI (default_rng(20260714), B=2000) of the "
        "harmed-minus-benefited proportion/mean gap excludes 0.",
        "",
        "| Dimension | Category | Harmed | Benefited | Diff | 95% CI | Sig |",
        "| --- | --- | ---: | ---: | ---: | :---: | :---: |",
    ]
    for r in rows:
        lines.append(
            "| " + " | ".join([
                str(r["dimension"]),
                str(r["category"]),
                fmt(r["harmed_prop"]),
                fmt(r["benefited_prop"]),
                f"{r['diff_harmed_minus_benefited']:+.4f}",
                f"[{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]",
                fmt(r["significant"]),
            ]) + " |"
        )
    lines += [
        "",
        "## gap x atom crosstab (cohort proportions)",
        "",
        "| target_gap_bucket | atom_bucket | Harmed n | Harmed % | Benefited n | Benefited % |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for c in cross_rows:
        lines.append(
            "| " + " | ".join([
                c["target_gap_bucket"], c["atom_bucket"],
                str(c["harmed_count"]), f"{c['harmed_prop']*100:.1f}%",
                str(c["benefited_count"]), f"{c['benefited_prop']*100:.1f}%",
            ]) + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
