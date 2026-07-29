from __future__ import annotations

# Cross-label evaluation (2x2 matrix) for reviewer comment 5.
#
# Goal: decompose whether the global-broadcast decay (T10) penalty comes from
# the *training signal* (which label the model was trained on) or from the
# *evaluation label* (which label the MAE is scored against). Because the test
# structures and the split are identical across the two label sources, we can
# reuse each model's predictions and re-score them against the *other* source's
# target column at zero extra cost.
#
# Inputs (verified):
#   results/mp_hzw_relaxed_clean_equal_atoms_mp_t/<MODEL>/predictions.csv
#       columns: sample_id,target,prediction,split   (target = MP band gap)
#   results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/<MODEL>/predictions.csv
#       columns: sample_id,target,prediction,split   (target = HZW band gap)
#   Models: T4_ALIGNN_plain_single_virtualnode,
#           T7_ALIGNN_huber_virtualnode,
#           T10_ALIGNN_decay_huber_virtualnode
#   (T2_ALIGNN_baseline is absent and is skipped gracefully.)
#
# For each model we form 4 combinations (train_source x eval_source):
#   {train=MP,  eval=MP }: prediction from mp_t,  target from mp_t
#   {train=MP,  eval=HZW}: prediction from mp_t,  target from hzw_t
#   {train=HZW, eval=MP }: prediction from hzw_t, target from mp_t
#   {train=HZW, eval=HZW}: prediction from hzw_t, target from hzw_t
# The eval label is always the target column of the eval source, aligned by
# sample_id. If the two sources' sample_id sets differ, we take the intersection
# and report its size.
#
# Output: results/analysis/cross_label_eval.csv
#   columns: model,train_source,eval_source,n_samples,mae,rmse

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "MP": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_mp_t",
    "HZW": ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
}

MODELS = [
    "T2_ALIGNN_baseline",
    "T4_ALIGNN_plain_single_virtualnode",
    "T7_ALIGNN_huber_virtualnode",
    "T10_ALIGNN_decay_huber_virtualnode",
]


def load_predictions(path: Path) -> dict[str, dict[str, float]]:
    """sample_id -> {target, prediction} for test-split rows."""
    rows: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("split") != "test":
                continue
            rows[row["sample_id"]] = {
                "target": float(row["target"]),
                "prediction": float(row["prediction"]),
            }
    return rows


def metrics(pairs: list[tuple[float, float]]) -> tuple[float, float]:
    """pairs of (prediction, eval_target) -> (mae, rmse)."""
    n = len(pairs)
    if n == 0:
        return float("nan"), float("nan")
    abs_sum = 0.0
    sq_sum = 0.0
    for pred, tgt in pairs:
        err = pred - tgt
        abs_sum += abs(err)
        sq_sum += err * err
    return abs_sum / n, (sq_sum / n) ** 0.5


def main() -> None:
    records: list[dict[str, object]] = []
    alignment: dict[str, dict[str, int]] = {}

    for model in MODELS:
        paths = {src: base / model / "predictions.csv" for src, base in SOURCES.items()}
        missing = [src for src, p in paths.items() if not p.exists()]
        if missing:
            print(f"[skip] {model}: missing sources {missing}")
            continue

        preds = {src: load_predictions(p) for src, p in paths.items()}
        # eval label for a source = that source's target column, aligned by id.
        eval_target = {src: {sid: r["target"] for sid, r in preds[src].items()} for src in SOURCES}

        shared = sorted(set(preds["MP"]) & set(preds["HZW"]))
        alignment[model] = {
            "mp_ids": len(preds["MP"]),
            "hzw_ids": len(preds["HZW"]),
            "shared_ids": len(shared),
        }

        for train_source in ("MP", "HZW"):
            for eval_source in ("MP", "HZW"):
                pairs = [
                    (preds[train_source][sid]["prediction"], eval_target[eval_source][sid])
                    for sid in shared
                ]
                mae, rmse = metrics(pairs)
                records.append(
                    {
                        "model": model,
                        "train_source": train_source,
                        "eval_source": eval_source,
                        "n_samples": len(pairs),
                        "mae": mae,
                        "rmse": rmse,
                    }
                )

    out_path = OUT_DIR / "cross_label_eval.csv"
    header_comment = [
        "# Cross-label evaluation (2x2: train_source x eval_source) for reviewer comment 5.",
        "# Inputs:",
        "#   results/mp_hzw_relaxed_clean_equal_atoms_mp_t/<MODEL>/predictions.csv  (target = MP band gap)",
        "#   results/mp_hzw_relaxed_clean_equal_atoms_hzw_t/<MODEL>/predictions.csv (target = HZW band gap)",
        "#   Models: " + ", ".join(MODELS),
        "#   prediction = the train_source model's output; eval label = eval_source target column, aligned by sample_id (intersection).",
    ]
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        for line in header_comment:
            fh.write(line + "\n")
        writer = csv.DictWriter(
            fh,
            fieldnames=["model", "train_source", "eval_source", "n_samples", "mae", "rmse"],
        )
        writer.writeheader()
        writer.writerows(records)

    print(json.dumps({"alignment": alignment, "records": records}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
