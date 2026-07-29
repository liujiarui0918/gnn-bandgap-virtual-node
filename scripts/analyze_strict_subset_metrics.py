"""Script A7: strict-consistent test-subset re-evaluation (reviewer comment 6).

Question addressed: does relaxed-label noise mask the architectural gain, when
measured under an *evaluation* lens (as opposed to a *training* lens)?

Positioning (writing note):
- Training stays on the RELAXED data for every model. We do NOT retrain here.
- We take the fixed 8455 test samples and split them, by the S-series strict
  threshold, into a "clean" subset and its "noisy" complement, then recompute
  MAE / RMSE / metal_F1 for each model on each subset.
- This isolates the EVALUATION-lens effect only. It is complementary to a
  "strict training lens" experiment (which requires a GPU re-run and is out of
  scope for this script).
- Clean vs noisy subsets have different sample sizes, so absolute numbers are
  not directly comparable across subsets. Only the WITHIN-lens relative
  architectural gain (T4->T7->T10 deltas) is compared across the two lenses.

Inputs (verified):
- Per-sample predictions:
    results/mp_hzw_relaxed_clean_equal_atoms_{mp,hzw}_t/<MODEL>/predictions.csv
    columns: sample_id,target,prediction,split   (we keep split == "test")
    Local models: T4, T7, T10. T2 is absent -> gracefully skipped.
- Audit master table:
    results/data_quality/mp_hzw_full_audit_table.remote.csv
    per sample_id: mp_is_metal, hzw_is_metal, abs_gap_diff,
    volume_per_atom_rel_diff, mp_band_gap, hzw_band_gap, ...

S-series strict-clean definition (used for the clean/noisy label):
    clean  <=>  mp_is_metal == hzw_is_metal
                AND abs_gap_diff <= 0.5 eV
                AND volume_per_atom_rel_diff <= 0.08
    otherwise noisy.

metal_F1 definition (verified against crystal_gnn/utils.py:62-91, non-symmetric
double threshold):
    target_is_metal = target      <= 1e-8
    pred_is_metal   = prediction   <= 0.1 eV
    F1 = 2*P*R/(P+R) over these boolean labels.

Output: results/analysis/strict_subset_metrics.csv
    columns: label_source,subset,model,n,mae,rmse,metal_f1
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_CSV = ROOT / "results" / "data_quality" / "mp_hzw_full_audit_table.remote.csv"
OUT_CSV = OUT_DIR / "strict_subset_metrics.csv"

# Strict-clean thresholds (S-series).
ABS_GAP_DIFF_MAX = 0.5  # eV
VOL_REL_DIFF_MAX = 0.08

# metal_F1 thresholds (crystal_gnn/utils.py).
TARGET_METAL_MAX = 1e-8
PRED_METAL_MAX = 0.1

# label_source -> results subdir. Each source is one target-label series.
SOURCES = {
    "mp": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "hzw": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}

# Ordered model roster. T2 is intentionally listed to detect its absence.
MODELS = {
    "T2": "T2_ALIGNN_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}

# Architectural gain pairs: (better, baseline) -> delta MAE = better - baseline.
GAIN_PAIRS = [("T4", "T2"), ("T7", "T4")]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_audit() -> dict[str, dict[str, str]]:
    with AUDIT_CSV.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        return {row["sample_id"]: row for row in reader}


def strict_label(audit_row: dict[str, str]) -> str:
    """Return 'clean' or 'noisy' per the S-series strict definition."""
    mp_metal = parse_bool(audit_row.get("mp_is_metal", ""))
    hzw_metal = parse_bool(audit_row.get("hzw_is_metal", ""))
    abs_gap_diff = float(audit_row.get("abs_gap_diff") or "nan")
    vol_rel_diff = float(audit_row.get("volume_per_atom_rel_diff") or "nan")
    is_clean = (
        mp_metal == hzw_metal
        and abs_gap_diff <= ABS_GAP_DIFF_MAX
        and vol_rel_diff <= VOL_REL_DIFF_MAX
    )
    return "clean" if is_clean else "noisy"


def load_predictions(path: Path) -> dict[str, dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows: dict[str, dict[str, float]] = {}
        for row in reader:
            if row.get("split") != "test":
                continue
            target = float(row["target"])
            pred = float(row["prediction"])
            rows[row["sample_id"]] = {"target": target, "prediction": pred}
        return rows


def metal_f1(values: list[dict[str, float]]) -> float:
    tp = fp = fn = 0
    for v in values:
        target_is_metal = v["target"] <= TARGET_METAL_MAX
        pred_is_metal = v["prediction"] <= PRED_METAL_MAX
        if target_is_metal and pred_is_metal:
            tp += 1
        elif (not target_is_metal) and pred_is_metal:
            fp += 1
        elif target_is_metal and (not pred_is_metal):
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def summarize(values: list[dict[str, float]]) -> dict[str, float | int]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mae": float("nan"), "rmse": float("nan"), "metal_f1": float("nan")}
    errors = [v["prediction"] - v["target"] for v in values]
    mae = sum(abs(e) for e in errors) / n
    rmse = (sum(e * e for e in errors) / n) ** 0.5
    return {"n": n, "mae": mae, "rmse": rmse, "metal_f1": metal_f1(values)}


def write_csv(path: Path, rows: list[dict[str, object]], header_comment: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        for line in header_comment:
            fh.write(f"# {line}\n")
        writer = csv.DictWriter(fh, fieldnames=["label_source", "subset", "model", "n", "mae", "rmse", "metal_f1"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "label_source": row["label_source"],
                    "subset": row["subset"],
                    "model": row["model"],
                    "n": row["n"],
                    "mae": "" if row["n"] == 0 else f"{float(row['mae']):.6f}",
                    "rmse": "" if row["n"] == 0 else f"{float(row['rmse']):.6f}",
                    "metal_f1": "" if row["n"] == 0 else f"{float(row['metal_f1']):.6f}",
                }
            )


def build() -> tuple[list[dict[str, object]], dict[str, object]]:
    audit = load_audit()

    # subset counts per source (labels are per-sample, independent of model, but
    # the actual test sample set is fixed per source; count over union of models).
    rows: list[dict[str, object]] = []
    diagnostics: dict[str, object] = {"missing_models": [], "subset_counts": {}, "gains": {}}

    # per (source, model) -> loaded predictions
    loaded: dict[tuple[str, str], dict[str, dict[str, float]]] = {}
    for source, base in SOURCES.items():
        for model, subdir in MODELS.items():
            path = base / subdir / "predictions.csv"
            if not path.exists():
                diagnostics["missing_models"].append(f"{source}/{model}")
                continue
            loaded[(source, model)] = load_predictions(path)

    # subset sizes per source from any available model's test id set (they share it).
    for source in SOURCES:
        any_model = next((m for m in MODELS if (source, m) in loaded), None)
        if any_model is None:
            diagnostics["subset_counts"][source] = {"clean": 0, "noisy": 0, "unaudited": 0}
            continue
        counts = {"clean": 0, "noisy": 0, "unaudited": 0}
        for sample_id in loaded[(source, any_model)]:
            audit_row = audit.get(sample_id)
            if audit_row is None:
                counts["unaudited"] += 1
                continue
            counts[strict_label(audit_row)] += 1
        diagnostics["subset_counts"][source] = counts

    # metrics per (source, subset, model)
    metrics: dict[tuple[str, str, str], dict[str, float | int]] = {}
    for (source, model), preds in loaded.items():
        by_subset: dict[str, list[dict[str, float]]] = defaultdict(list)
        for sample_id, pv in preds.items():
            audit_row = audit.get(sample_id)
            if audit_row is None:
                continue  # samples absent from audit table cannot be labeled
            by_subset[strict_label(audit_row)].append(pv)
        for subset in ("clean", "noisy"):
            stats = summarize(by_subset.get(subset, []))
            metrics[(source, subset, model)] = stats
            rows.append({"label_source": source, "subset": subset, "model": model, **stats})

    # architectural gain contrast: delta MAE per pair, on clean vs noisy.
    for source in SOURCES:
        diagnostics["gains"][source] = {}
        for better, baseline in GAIN_PAIRS:
            pair = f"{better}-{baseline}"
            entry: dict[str, object] = {}
            for subset in ("clean", "noisy"):
                b = metrics.get((source, subset, better))
                a = metrics.get((source, subset, baseline))
                if b is None or a is None or b["n"] == 0 or a["n"] == 0:
                    entry[subset] = None  # missing model (e.g. T2) -> blank
                else:
                    entry[subset] = float(b["mae"]) - float(a["mae"])
            diagnostics["gains"][source][pair] = entry

    return rows, diagnostics


def main() -> None:
    rows, diagnostics = build()

    header_comment = [
        "Script A7 -- strict-consistent test-subset re-evaluation (reviewer comment 6).",
        "Training on RELAXED data (no retrain here); this measures the EVALUATION lens only.",
        "Complementary to a strict TRAINING lens (needs GPU re-run).",
        "Subset sizes differ; compare only WITHIN-lens relative gains, not absolute numbers.",
        "Inputs: results/mp_hzw_relaxed_clean_equal_atoms_{mp,hzw}_t/<MODEL>/predictions.csv (split==test);",
        "        results/data_quality/mp_hzw_full_audit_table.remote.csv .",
        f"Strict-clean: mp_is_metal==hzw_is_metal AND abs_gap_diff<={ABS_GAP_DIFF_MAX} eV "
        f"AND volume_per_atom_rel_diff<={VOL_REL_DIFF_MAX}; else noisy.",
        f"metal_F1: target_is_metal=target<={TARGET_METAL_MAX}, pred_is_metal=prediction<={PRED_METAL_MAX} eV.",
    ]
    write_csv(OUT_CSV, rows, header_comment)

    if diagnostics["missing_models"]:
        print(f"[warn] missing model predictions (skipped): {sorted(set(diagnostics['missing_models']))}")
        for source, gains in diagnostics["gains"].items():
            for pair, entry in gains.items():
                if entry.get("clean") is None and entry.get("noisy") is None:
                    print(f"[warn] gain pair {source}/{pair} is blank (baseline/better model absent).")

    print(json.dumps(
        {"out_csv": str(OUT_CSV), "subset_counts": diagnostics["subset_counts"], "gains": diagnostics["gains"]},
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
