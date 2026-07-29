from __future__ import annotations

"""Aggregate multi-seed transfer results (整改计划 1.4).

Combines per-seed run directories into two summary tables:

  1. --output (default results/analysis/multiseed_summary.csv)
     Per model x label-source, mean +/- sample std (ddof=1, n = #seeds) of
     mae / rmse / r2 / metal_f1, drawn from each seed's comparison_summary.csv.

  2. results/analysis/multiseed_pairs.csv
     Model-pair analysis (T4-T2, T7-T4, T10-T7, T7-T2, T10-T4):
       (a) per-seed delta-MAE and direction consistency (k/n seeds same sign);
       (b) per-seed paired bootstrap CI on delta-MAE when BOTH sides have a
           predictions.csv for that seed (B=2000, rng=default_rng(20260714)).

Deliberately NO t-test: with n=3 seeds the power is far too low. The evidence
standard for 1.4 is: mean +/- std, per-seed paired bootstrap CI, and direction
consistency across seeds -- consistent with plan document section 1.4.

Directory convention:
  base seed (42) -> results/mp_hzw_relaxed_clean_equal_atoms_{source}_t
  other seeds    -> results/mp_hzw_relaxed_clean_equal_atoms_{source}_t_seed{N}

Usage examples:
  # run right now with only the existing seed-42 data (n_seeds=1, std=NaN)
  py -3 scripts/aggregate_multiseed.py

  # once remote seeds 1042 / 2042 have landed
  py -3 scripts/aggregate_multiseed.py --seeds 42 1042 2042

  # custom roots / sources / output
  py -3 scripts/aggregate_multiseed.py \
      --base-root results --label-sources mp hzw \
      --seeds 42 1042 2042 --output results/analysis/multiseed_summary.csv
"""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

BASE_SEED = 42  # the seed whose directory carries no _seed{N} suffix
DIR_STEM = "mp_hzw_relaxed_clean_equal_atoms_{source}_t"

METRICS = ["mae", "rmse", "r2", "metal_f1"]

# Model pairs written as "A-B" -> delta = metric_A - metric_B. A is the newer /
# candidate model, B the baseline. Negative delta-MAE => A has lower error.
PAIRS = [
    ("T4", "T2"),
    ("T7", "T4"),
    ("T10", "T7"),
    ("T7", "T2"),
    ("T10", "T4"),
]

BOOTSTRAP_B = 2000
BOOTSTRAP_SEED = 20260714


def warn(msg: str) -> None:
    print(f"[warn] {msg}", file=sys.stderr)


def seed_dir(base_root: Path, source: str, seed: int) -> Path:
    stem = DIR_STEM.format(source=source)
    if seed != BASE_SEED:
        stem = f"{stem}_seed{seed}"
    return base_root / stem


def load_comparison(path: Path) -> dict[str, dict[str, float]]:
    """experiment_name -> {metric: value} from a comparison_summary.csv."""
    out: dict[str, dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            name = row["experiment_name"]
            vals: dict[str, float] = {}
            for m in METRICS:
                raw = row.get(m, "")
                try:
                    vals[m] = float(raw)
                except (TypeError, ValueError):
                    vals[m] = float("nan")
            out[name] = vals
    return out


def load_abs_errors(path: Path) -> dict[str, float]:
    """sample_id -> |prediction - target| over the test split."""
    out: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("split") != "test":
                continue
            try:
                err = abs(float(row["prediction"]) - float(row["target"]))
            except (TypeError, ValueError, KeyError):
                continue
            out[row["sample_id"]] = err
    return out


def find_model_dir(source_dir: Path, tnum: str) -> Path | None:
    """Locate the subdirectory whose name starts with 'T{n}_'."""
    if not source_dir.is_dir():
        return None
    prefix = f"{tnum}_"
    for child in sorted(source_dir.iterdir()):
        if child.is_dir() and child.name.startswith(prefix):
            return child
    return None


def find_experiment_name(comparison: dict[str, dict[str, float]], tnum: str) -> str | None:
    """Find the comparison_summary experiment_name starting with 'T{n}_'."""
    prefix = f"{tnum}_"
    for name in comparison:
        if name.startswith(prefix):
            return name
    return None


def mean_std(values: list[float]) -> tuple[float, float]:
    """Mean and sample std (ddof=1). std is NaN when n < 2."""
    arr = np.asarray(values, dtype=float)
    arr = arr[~np.isnan(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else float("nan")
    return mean, std


def fmt_pm(mean: float, std: float) -> str:
    if np.isnan(mean):
        return ""
    if np.isnan(std):
        return f"{mean:.6f}"
    return f"{mean:.6f} +/- {std:.6f}"


def paired_bootstrap_delta(
    ae_a: dict[str, float], ae_b: dict[str, float]
) -> dict[str, float] | None:
    """Per-seed paired bootstrap over shared sample_ids.

    Statistic = mean(|err_a| - |err_b|) = mae_a - mae_b on the shared samples.
    Returns point estimate, 95% percentile CI, and paired-sample count.
    """
    shared = sorted(set(ae_a) & set(ae_b))
    if not shared:
        return None
    diff = np.array([ae_a[s] - ae_b[s] for s in shared], dtype=float)
    n = diff.size
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    idx = rng.integers(0, n, size=(BOOTSTRAP_B, n))
    boot_means = diff[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {
        "n_paired": float(n),
        "delta_mae_shared": float(diff.mean()),
        "boot_ci_low": float(lo),
        "boot_ci_high": float(hi),
    }


def build_summary(
    base_root: Path, sources: list[str], seeds: list[int]
) -> tuple[list[dict[str, object]], dict[tuple[str, int], dict[str, dict[str, float]]]]:
    """Per model x source aggregation. Also returns cached per-seed comparisons."""
    # cache[(source, seed)] = comparison dict (may be empty if missing)
    cache: dict[tuple[str, int], dict[str, dict[str, float]]] = {}
    # per source: ordered union of experiment names + which seeds hold each
    per_source_models: dict[str, list[str]] = {}
    model_seed_vals: dict[tuple[str, str], dict[int, dict[str, float]]] = {}

    for source in sources:
        seen: list[str] = []
        for seed in seeds:
            sdir = seed_dir(base_root, source, seed)
            cpath = sdir / "comparison_summary.csv"
            if not cpath.is_file():
                warn(f"missing comparison_summary: {cpath} (source={source}, seed={seed}) -> skipped")
                cache[(source, seed)] = {}
                continue
            comparison = load_comparison(cpath)
            cache[(source, seed)] = comparison
            for name, vals in comparison.items():
                if name not in seen:
                    seen.append(name)
                model_seed_vals.setdefault((source, name), {})[seed] = vals
        per_source_models[source] = seen

    rows: list[dict[str, object]] = []
    for source in sources:
        for name in per_source_models[source]:
            seed_vals = model_seed_vals.get((source, name), {})
            found_seeds = [s for s in seeds if s in seed_vals]
            missing = [s for s in seeds if s not in seed_vals]
            if missing:
                warn(
                    f"model {name} (source={source}) absent for seeds {missing}; "
                    f"aggregating over {found_seeds} (n_seeds={len(found_seeds)})"
                )
            row: dict[str, object] = {
                "source": source,
                "model": name,
                "n_seeds": len(found_seeds),
                "seeds_found": "|".join(str(s) for s in found_seeds),
            }
            for m in METRICS:
                vals = [seed_vals[s][m] for s in found_seeds]
                mean, std = mean_std(vals)
                row[f"{m}_mean"] = mean
                row[f"{m}_std"] = std
                row[f"{m}_mean_pm_std"] = fmt_pm(mean, std)
            rows.append(row)
    return rows, cache


def build_pairs(
    base_root: Path,
    sources: list[str],
    seeds: list[int],
    cache: dict[tuple[str, int], dict[str, dict[str, float]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    for source in sources:
        for a, b in PAIRS:
            pair = f"{a}-{b}"
            per_seed: list[dict[str, object]] = []
            deltas: list[float] = []

            for seed in seeds:
                comparison = cache.get((source, seed), {})
                name_a = find_experiment_name(comparison, a)
                name_b = find_experiment_name(comparison, b)
                if name_a is None or name_b is None:
                    if comparison:
                        warn(
                            f"pair {pair} (source={source}, seed={seed}): "
                            f"missing metrics for {a if name_a is None else b} -> seed skipped"
                        )
                    continue
                mae_a = comparison[name_a]["mae"]
                mae_b = comparison[name_b]["mae"]
                delta = mae_a - mae_b
                deltas.append(delta)

                srow: dict[str, object] = {
                    "source": source,
                    "pair": pair,
                    "scope": str(seed),
                    "model_a": name_a,
                    "model_b": name_b,
                    "n_seeds": "",
                    "mae_a": mae_a,
                    "mae_b": mae_b,
                    "delta_mae": delta,
                    "delta_sign": "+" if delta > 0 else ("-" if delta < 0 else "0"),
                    "direction_consistency": "",
                    "mean_delta_mae": "",
                    "std_delta_mae": "",
                    "has_both_predictions": False,
                    "n_paired": "",
                    "delta_mae_shared": "",
                    "boot_ci_low": "",
                    "boot_ci_high": "",
                    "bootstrap_B": "",
                }

                # (b) per-seed paired bootstrap when both predictions exist
                sdir = seed_dir(base_root, source, seed)
                dir_a = find_model_dir(sdir, a)
                dir_b = find_model_dir(sdir, b)
                pred_a = dir_a / "predictions.csv" if dir_a else None
                pred_b = dir_b / "predictions.csv" if dir_b else None
                if pred_a and pred_a.is_file() and pred_b and pred_b.is_file():
                    boot = paired_bootstrap_delta(
                        load_abs_errors(pred_a), load_abs_errors(pred_b)
                    )
                    if boot is not None:
                        srow["has_both_predictions"] = True
                        srow["n_paired"] = int(boot["n_paired"])
                        srow["delta_mae_shared"] = boot["delta_mae_shared"]
                        srow["boot_ci_low"] = boot["boot_ci_low"]
                        srow["boot_ci_high"] = boot["boot_ci_high"]
                        srow["bootstrap_B"] = BOOTSTRAP_B
                per_seed.append(srow)

            if not per_seed:
                warn(f"pair {pair} (source={source}): no usable seeds -> skipped")
                continue

            # direction consistency: k/n seeds sharing the majority sign
            n = len(deltas)
            n_pos = sum(1 for d in deltas if d > 0)
            n_neg = sum(1 for d in deltas if d < 0)
            k = max(n_pos, n_neg)
            consistency = f"{k}/{n}"
            mean_d, std_d = mean_std(deltas)

            rows.extend(per_seed)
            rows.append(
                {
                    "source": source,
                    "pair": pair,
                    "scope": "ALL",
                    "model_a": per_seed[0]["model_a"],
                    "model_b": per_seed[0]["model_b"],
                    "n_seeds": n,
                    "mae_a": "",
                    "mae_b": "",
                    "delta_mae": "",
                    "delta_sign": "",
                    "direction_consistency": consistency,
                    "mean_delta_mae": mean_d,
                    "std_delta_mae": std_d,
                    "has_both_predictions": "",
                    "n_paired": "",
                    "delta_mae_shared": "",
                    "boot_ci_low": "",
                    "boot_ci_high": "",
                    "bootstrap_B": "",
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        warn(f"no rows to write for {path}")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate multi-seed transfer results (整改计划 1.4).")
    p.add_argument("--base-root", type=Path, default=ROOT / "results",
                   help="Directory holding the per-seed run folders (default: results).")
    p.add_argument("--label-sources", nargs="+", default=["mp", "hzw"],
                   help="Label sources (default: mp hzw).")
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 1042, 2042],
                   help="Seeds; %d maps to the no-suffix directory (default: 42 1042 2042)." % BASE_SEED)
    p.add_argument("--output", type=Path, default=ROOT / "results" / "analysis" / "multiseed_summary.csv",
                   help="Summary CSV path (pairs CSV is written alongside it).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    base_root: Path = args.base_root
    sources: list[str] = args.label_sources
    seeds: list[int] = args.seeds

    summary_rows, cache = build_summary(base_root, sources, seeds)
    pair_rows = build_pairs(base_root, sources, seeds, cache)

    summary_path: Path = args.output
    pairs_path = summary_path.parent / "multiseed_pairs.csv"
    write_csv(summary_path, summary_rows)
    write_csv(pairs_path, pair_rows)

    n_found = sum(1 for k, v in cache.items() if v)
    print(
        f"seeds={seeds} sources={sources} | comparison files found={n_found}/{len(cache)}\n"
        f"summary rows={len(summary_rows)} -> {summary_path}\n"
        f"pairs rows={len(pair_rows)} -> {pairs_path}"
    )


if __name__ == "__main__":
    main()
