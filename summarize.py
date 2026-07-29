from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics into CSV.")
    parser.add_argument("--results_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--experiment_names", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    include_names = set(args.experiment_names or [])
    rows = []
    for metrics_path in sorted(results_dir.glob("*/metrics.json")):
        if metrics_path.parent.name.startswith("smoke"):
            continue
        if include_names and metrics_path.parent.name not in include_names:
            continue
        with metrics_path.open("r", encoding="utf-8") as handle:
            rows.append(json.load(handle))
    if not rows:
        raise FileNotFoundError(f"no metrics.json found under {results_dir}")

    fieldnames = [
        "experiment_name",
        "model",
        "mae",
        "rmse",
        "r2",
        "metal_f1",
        "best_epoch",
        "epochs_ran",
        "train_seconds",
        "peak_gpu_gb",
    ]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
