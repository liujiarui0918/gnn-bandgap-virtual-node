from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

# Fixed bootstrap seed for reproducibility (reviewer comment 1: statistical significance).
BOOTSTRAP_SEED = 20260714

# Model id -> results sub-directory name.
MODEL_DIRS = {
    "T2": "T2_ALIGNN_baseline",
    "T4": "T4_ALIGNN_plain_single_virtualnode",
    "T7": "T7_ALIGNN_huber_virtualnode",
    "T10": "T10_ALIGNN_decay_huber_virtualnode",
}

# Ordered pairs (model_a, model_b); delta = MAE(model_a) - MAE(model_b).
MODEL_PAIRS = [
    ("T4", "T2"),
    ("T7", "T4"),
    ("T10", "T7"),
    ("T7", "T2"),
    ("T10", "T4"),
]


def load_abs_errors(path: Path) -> dict[str, float]:
    """Read a predictions.csv and return {sample_id: abs_error} for test rows only."""
    errors: dict[str, float] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            if row.get("split") != "test":
                continue
            target = float(row["target"])
            pred = float(row["prediction"])
            errors[row["sample_id"]] = abs(pred - target)
    return errors


def paired_bootstrap(abs_a: np.ndarray, abs_b: np.ndarray, n_boot: int) -> dict[str, float]:
    """Paired bootstrap over shared test-sample indices.

    The same resampled indices are applied to both models on every draw, which is
    what makes the comparison paired. delta = mean(|err_a|) - mean(|err_b|).
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = abs_a.shape[0]
    deltas = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[i] = abs_a[idx].mean() - abs_b[idx].mean()
    ci_low, ci_high = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_mae": float(abs_a.mean() - abs_b.mean()),
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "p_delta_lt_0": float(np.mean(deltas < 0.0)),
        "significant_95": bool(ci_low > 0.0 or ci_high < 0.0),
    }


def predictions_path(results_root: Path, model_id: str) -> Path:
    return results_root / MODEL_DIRS[model_id] / "predictions.csv"


def build_rows(
    label_source: str,
    results_root: Path,
    n_boot: int,
    used_inputs: list[Path],
) -> list[dict[str, object]]:
    cache: dict[str, dict[str, float]] = {}

    def get_errors(model_id: str) -> dict[str, float] | None:
        if model_id in cache:
            return cache[model_id]
        path = predictions_path(results_root, model_id)
        if not path.exists():
            print(f"[warn] {label_source}: missing predictions for {model_id} at {path}")
            return None
        errors = load_abs_errors(path)
        cache[model_id] = errors
        used_inputs.append(path)
        return errors

    rows: list[dict[str, object]] = []
    for model_a, model_b in MODEL_PAIRS:
        errors_a = get_errors(model_a)
        errors_b = get_errors(model_b)
        if errors_a is None or errors_b is None:
            print(
                f"[warn] {label_source}: skipping pair {model_a}-{model_b} "
                "(one or both model directories unavailable)"
            )
            continue
        shared = sorted(set(errors_a) & set(errors_b))
        if not shared:
            print(f"[warn] {label_source}: no shared test samples for {model_a}-{model_b}")
            continue
        abs_a = np.array([errors_a[s] for s in shared], dtype=np.float64)
        abs_b = np.array([errors_b[s] for s in shared], dtype=np.float64)
        stats = paired_bootstrap(abs_a, abs_b, n_boot)
        rows.append(
            {
                "label_source": label_source,
                "pair": f"{model_a}-{model_b}",
                "model_a": model_a,
                "model_b": model_b,
                "n_samples": len(shared),
                "delta_mae": stats["delta_mae"],
                "ci_low": stats["ci_low"],
                "ci_high": stats["ci_high"],
                "p_delta_lt_0": stats["p_delta_lt_0"],
                "significant_95": stats["significant_95"],
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]], input_files: list[Path], n_boot: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "label_source",
        "pair",
        "model_a",
        "model_b",
        "n_samples",
        "delta_mae",
        "ci_low",
        "ci_high",
        "p_delta_lt_0",
        "significant_95",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        fh.write("# Paired bootstrap MAE comparison (reviewer comment 1: statistical significance)\n")
        fh.write(f"# bootstrap_seed={BOOTSTRAP_SEED} n_boot={n_boot}\n")
        fh.write("# delta_mae = MAE(model_a) - MAE(model_b); 95% CI from 2.5/97.5 percentiles\n")
        fh.write("# input predictions files used:\n")
        for src in input_files:
            try:
                rel = src.relative_to(ROOT)
            except ValueError:
                rel = src
            fh.write(f"#   {rel.as_posix()}\n")
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root-mp",
        type=Path,
        default=ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_mp_t",
        help="Directory holding per-model MP-label prediction sub-directories.",
    )
    parser.add_argument(
        "--results-root-hzw",
        type=Path,
        default=ROOT / "results" / "mp_hzw_relaxed_clean_equal_atoms_hzw_t",
        help="Directory holding per-model HZW-label prediction sub-directories.",
    )
    parser.add_argument("--n-boot", type=int, default=10000, help="Number of bootstrap resamples.")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "analysis" / "paired_bootstrap.csv",
        help="Output CSV path.",
    )
    args = parser.parse_args()

    used_inputs: list[Path] = []
    rows: list[dict[str, object]] = []
    rows += build_rows("mp", args.results_root_mp, args.n_boot, used_inputs)
    rows += build_rows("hzw", args.results_root_hzw, args.n_boot, used_inputs)

    write_csv(args.output, rows, used_inputs, args.n_boot)

    print(f"[done] wrote {len(rows)} pair rows to {args.output}")
    for row in rows:
        flag = "SIGNIFICANT" if row["significant_95"] else "n.s."
        print(
            f"  {row['label_source']:>3} {row['pair']:>7} n={row['n_samples']} "
            f"delta_mae={row['delta_mae']:+.6f} "
            f"CI=[{row['ci_low']:+.6f}, {row['ci_high']:+.6f}] "
            f"P(d<0)={row['p_delta_lt_0']:.4f} {flag}"
        )


if __name__ == "__main__":
    main()
