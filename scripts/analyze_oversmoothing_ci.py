from __future__ import annotations

# Over-smoothing final-layer delta diagnostic with bootstrap confidence intervals.
#
# Purpose (reviewer comment 3 partial mitigation): reproduce the final-layer
# "delta vs T2" over-smoothing signature from
#   docs/oversmoothing_global_token_diagnostic.md
# and attach a 95% bootstrap CI to every delta so the 240-sample evidence can be
# read with its uncertainty instead of point estimates alone.
#
# Method:
#   - final layer = the maximum layer_index captured for each model.
#   - delta is sample-paired: for a given sample_id, (model_metric - T2_metric)
#     at the final layer, for model in {T4, T7, T10}.
#   - bootstrap resamples the paired samples with replacement (B iterations) and
#     reports the 2.5 / 97.5 percentiles of the resampled mean delta.
#
# Inputs (per-sample per-layer over-smoothing metrics, 240 samples x 16 layers x 4 models):
#   - results/analysis/oversmoothing_hooks_mp/oversmoothing_sample_metrics.csv   (label_source MP)
#   - results/analysis/oversmoothing_hooks_hzw/oversmoothing_sample_metrics.csv  (label_source HZW)
#   columns: experiment,sample_id,num_atoms,num_atoms_bucket,layer_index,layer_name,
#            stage,class_name,mean_pairwise_cosine,node_feature_variance,dirichlet_energy

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASELINE = "T2"
GLOBAL_MODELS = ["T4", "T7", "T10"]

# metric label -> source column name (higher-is-relief noted in the doc)
METRICS = {
    "cosine": "mean_pairwise_cosine",
    "variance": "node_feature_variance",
    "dirichlet": "dirichlet_energy",
}

SEED = 20260714
BOOTSTRAP_ITERS = 2000
GT80_THRESHOLD = 80


def load_final_layer_metrics(path: Path) -> dict[str, dict]:
    """Return {experiment: {sample_id: {"num_atoms": int, metric_label: float, ...}}}
    keeping only rows at each experiment's maximum layer_index."""
    max_layer: dict[str, int] = {}
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            exp = row["experiment"]
            li = int(row["layer_index"])
            if exp not in max_layer or li > max_layer[exp]:
                max_layer[exp] = li
            rows.append(row)

    out: dict[str, dict] = defaultdict(dict)
    for row in rows:
        exp = row["experiment"]
        if int(row["layer_index"]) != max_layer[exp]:
            continue
        rec = {"num_atoms": int(row["num_atoms"])}
        for label, col in METRICS.items():
            rec[label] = float(row[col])
        out[exp][row["sample_id"]] = rec
    return out


def paired_deltas(
    final: dict[str, dict], model: str, metric: str, subset: str
) -> np.ndarray:
    """Per-sample (model - T2) deltas at the final layer for one metric/subset."""
    base = final[BASELINE]
    mod = final[model]
    deltas: list[float] = []
    for sample_id in base.keys() & mod.keys():
        if subset == "gt80" and base[sample_id]["num_atoms"] <= GT80_THRESHOLD:
            continue
        deltas.append(mod[sample_id][metric] - base[sample_id][metric])
    return np.asarray(deltas, dtype=float)


def bootstrap_ci(
    deltas: np.ndarray, rng: np.random.Generator, iters: int
) -> tuple[float, float, float]:
    """Return (mean, ci_low, ci_high) via percentile bootstrap over paired samples."""
    n = deltas.shape[0]
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        idx = rng.integers(0, n, size=n)
        means[i] = deltas[idx].mean()
    ci_low, ci_high = np.percentile(means, [2.5, 97.5])
    return float(deltas.mean()), float(ci_low), float(ci_high)


def build_rows(sources: dict[str, Path]) -> list[dict[str, object]]:
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []
    for label_source, path in sources.items():
        final = load_final_layer_metrics(path)
        for subset in ["all", "gt80"]:
            for model in GLOBAL_MODELS:
                for metric in METRICS:
                    deltas = paired_deltas(final, model, metric, subset)
                    mean, ci_low, ci_high = bootstrap_ci(deltas, rng, BOOTSTRAP_ITERS)
                    rows.append(
                        {
                            "label_source": label_source,
                            "subset": subset,
                            "model": model,
                            "metric": metric,
                            "n_samples": int(deltas.shape[0]),
                            "delta_mean": mean,
                            "ci_low": ci_low,
                            "ci_high": ci_high,
                            "ci_excludes_0": (ci_low > 0.0) or (ci_high < 0.0),
                        }
                    )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mp-csv",
        type=Path,
        default=ROOT
        / "results"
        / "analysis"
        / "oversmoothing_hooks_mp"
        / "oversmoothing_sample_metrics.csv",
    )
    parser.add_argument(
        "--hzw-csv",
        type=Path,
        default=ROOT
        / "results"
        / "analysis"
        / "oversmoothing_hooks_hzw"
        / "oversmoothing_sample_metrics.csv",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "results" / "analysis" / "oversmoothing_delta_ci.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sources = {"MP": args.mp_csv, "HZW": args.hzw_csv}
    rows = build_rows(sources)
    write_csv(args.out_csv, rows)

    print(f"wrote {len(rows)} rows -> {args.out_csv}")
    header = ["label_source", "subset", "model", "metric", "n", "delta_mean", "ci_low", "ci_high", "excl0"]
    print("  ".join(header))
    for r in rows:
        print(
            f"{r['label_source']:>3}  {r['subset']:>4}  {r['model']:>3}  "
            f"{r['metric']:>9}  {r['n_samples']:>3}  "
            f"{r['delta_mean']:+.4f}  [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}]  "
            f"{r['ci_excludes_0']}"
        )


if __name__ == "__main__":
    main()
