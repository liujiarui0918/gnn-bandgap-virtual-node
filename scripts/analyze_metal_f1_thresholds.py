"""A8 — Metal/non-metal F1 threshold sensitivity (reviewer comment 8).

Question addressed
-------------------
The metal_f1 reported in comparison_summary.csv uses a single fixed prediction
threshold (0.1 eV). A reviewer asked whether the model ranking depends on that
arbitrary choice. This script recomputes the metal-class F1 for a sweep of
prediction thresholds and checks that the T4/T7/T10 ordering is stable.

F1 definition (faithful to crystal_gnn/utils.py:62-91)
-----------------------------------------------------
The positive ("metal") class is defined per sample as:
    ground-truth metal : target     <= 1e-8 eV   (fixed)
    predicted   metal : prediction <= metal_threshold
    tp = target_is_metal AND pred_is_metal
    fp = (NOT target_is_metal) AND pred_is_metal
    fn = target_is_metal AND (NOT pred_is_metal)
    precision = tp / (tp + fp)  (0 if denom == 0)
    recall    = tp / (tp + fn)  (0 if denom == 0)
    f1        = 2 * p * r / (p + r)  (0 if denom == 0)
Only the test split is used.

Inputs
------
results/mp_hzw_relaxed_clean_equal_atoms_mp_t/<MODEL>/predictions.csv
results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/<MODEL>/predictions.csv
  columns: sample_id, target, prediction, split (test rows only)
Models: T4_ALIGNN_plain_single_virtualnode,
        T7_ALIGNN_huber_virtualnode,
        T10_ALIGNN_decay_huber_virtualnode
(T2 is absent from these run folders and is skipped gracefully.)

Thresholds
----------
prediction threshold sweep : {0.01, 0.05, 0.1} eV
ground-truth metal threshold : 1e-8 eV (fixed)

Output
------
results/analysis/metal_f1_thresholds.csv
  columns: label_source, model, pred_threshold,
           n_metal_true, precision, recall, f1
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "metal_f1_thresholds.csv"

# Fixed ground-truth metal threshold, faithful to crystal_gnn/utils.py.
TARGET_METAL_THRESHOLD = 1e-8
PRED_THRESHOLDS = [0.01, 0.05, 0.1]

LABEL_SOURCES = {
    "mp": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "hzw": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}

MODELS = [
    "T2_ALIGNN_baseline",
    "T4_ALIGNN_plain_single_virtualnode",
    "T7_ALIGNN_huber_virtualnode",
    "T10_ALIGNN_decay_huber_virtualnode",
]


def load_test_predictions(path: Path) -> list[tuple[float, float]]:
    """Return (target, prediction) pairs for the test split only."""
    pairs: list[tuple[float, float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("split") != "test":
                continue
            pairs.append((float(row["target"]), float(row["prediction"])))
    return pairs


def metal_prf(
    pairs: list[tuple[float, float]], pred_threshold: float
) -> dict[str, float | int]:
    """Precision/recall/F1 for the metal class, faithful to regression_metrics."""
    tp = fp = fn = n_metal_true = 0
    for target, pred in pairs:
        target_is_metal = target <= TARGET_METAL_THRESHOLD
        pred_is_metal = pred <= pred_threshold
        if target_is_metal:
            n_metal_true += 1
        if target_is_metal and pred_is_metal:
            tp += 1
        elif (not target_is_metal) and pred_is_metal:
            fp += 1
        elif target_is_metal and (not pred_is_metal):
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "n_metal_true": n_metal_true,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for label_source, base in LABEL_SOURCES.items():
        for model in MODELS:
            pred_path = base / model / "predictions.csv"
            if not pred_path.is_file():
                print(f"[skip] missing predictions: {pred_path}")
                continue
            pairs = load_test_predictions(pred_path)
            for threshold in PRED_THRESHOLDS:
                stats = metal_prf(pairs, threshold)
                rows.append(
                    {
                        "label_source": label_source,
                        "model": model,
                        "pred_threshold": threshold,
                        **stats,
                    }
                )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    header_comment = [
        "# A8 metal/non-metal F1 threshold sensitivity (reviewer comment 8)",
        "# inputs:",
        "#   results/mp_hzw_relaxed_clean_equal_atoms_mp_t/<MODEL>/predictions.csv (test split)",
        "#   results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/<MODEL>/predictions.csv (test split)",
        "# models: T4_ALIGNN_plain_single_virtualnode, T7_ALIGNN_huber_virtualnode, T10_ALIGNN_decay_huber_virtualnode",
        "# metal positive class: ground-truth metal target<=1e-8 eV (fixed); predicted metal prediction<=pred_threshold",
        "# pred_threshold sweep: 0.01, 0.05, 0.1 eV",
    ]
    fieldnames = [
        "label_source",
        "model",
        "pred_threshold",
        "n_metal_true",
        "precision",
        "recall",
        "f1",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        for line in header_comment:
            fh.write(line + "\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_ranking(rows: list[dict[str, object]]) -> None:
    """Print, per source and threshold, the F1 ranking of the models."""
    for label_source in LABEL_SOURCES:
        print(f"\n== label_source={label_source} :: metal F1 by threshold ==")
        for threshold in PRED_THRESHOLDS:
            subset = [
                r
                for r in rows
                if r["label_source"] == label_source and r["pred_threshold"] == threshold
            ]
            ranked = sorted(subset, key=lambda r: float(r["f1"]), reverse=True)  # type: ignore[arg-type]
            order = " > ".join(
                f"{str(r['model']).split('_')[0]}({r['f1']})" for r in ranked
            )
            print(f"  thr={threshold}: {order}")


def main() -> None:
    rows = build_rows()
    write_csv(OUT_CSV, rows)
    print_ranking(rows)
    print(f"\n[written] {OUT_CSV} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
