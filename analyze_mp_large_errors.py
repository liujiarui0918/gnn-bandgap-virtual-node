from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


PREDICTION_FILES = {
    "MP_E4_full": "results/mp_bandgap_standard/MP_E4_ALIGNN_virtualnode/predictions.csv",
    "MP_E19_full": "results/mp_bandgap_standard/MP_E19_ALIGNN_metalaware_virtualnode/predictions.csv",
    "MP_E20b_large_only": "results/mp_size_split/MP_E20b_ALIGNN_virtualnode_large_gt40/predictions.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze MP large-system test errors against num_atoms.")
    parser.add_argument("--dataset_root", default="data/mp_bandgap_official")
    parser.add_argument("--output_dir", default="results/mp_size_split/analysis")
    parser.add_argument("--large_threshold", type=int, default=40)
    return parser.parse_args()


def load_num_atoms_map(dataset_root: Path) -> dict[str, int]:
    num_atoms_by_id: dict[str, int] = {}
    with (dataset_root / "manifest.jsonl").open("r", encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            structure = record.get("structure") or {}
            num_atoms = int(record.get("num_atoms") or len(structure.get("sites", [])))
            num_atoms_by_id[str(record["sample_id"])] = num_atoms
    return num_atoms_by_id


def load_prediction_rows(path: Path, num_atoms_by_id: dict[str, int], large_threshold: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != "test":
                continue
            sample_id = str(row["sample_id"])
            num_atoms = int(row.get("num_atoms") or num_atoms_by_id[sample_id])
            if num_atoms <= large_threshold:
                continue
            target = float(row["target"])
            prediction = float(row["prediction"])
            error = prediction - target
            rows.append(
                {
                    "sample_id": sample_id,
                    "num_atoms": num_atoms,
                    "target": target,
                    "prediction": prediction,
                    "error": error,
                    "abs_error": abs(error),
                    "squared_error": error * error,
                    "target_is_metal": target <= 1e-8,
                }
            )
    return rows


def atom_bucket(num_atoms: int) -> str:
    if num_atoms <= 60:
        return "41-60"
    if num_atoms <= 80:
        return "61-80"
    if num_atoms <= 120:
        return "81-120"
    if num_atoms <= 160:
        return "121-160"
    return ">160"


def gap_bucket(target: float) -> str:
    if target <= 1e-8:
        return "metal=0"
    if target <= 0.5:
        return "(0,0.5]"
    if target <= 1.0:
        return "(0.5,1]"
    if target <= 2.0:
        return "(1,2]"
    return ">2"


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return 0.0
    return numerator / math.sqrt(x_var * y_var)


def rankdata(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(indexed):
        end = cursor + 1
        while end < len(indexed) and indexed[end][1] == indexed[cursor][1]:
            end += 1
        rank = (cursor + end - 1) / 2.0 + 1.0
        for index in range(cursor, end):
            ranks[indexed[index][0]] = rank
        cursor = end
    return ranks


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(rankdata(xs), rankdata(ys))


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[index]


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    abs_errors = [float(row["abs_error"]) for row in rows]
    errors = [float(row["error"]) for row in rows]
    return {
        "n": len(rows),
        "mae": round(sum(abs_errors) / len(rows), 6),
        "rmse": round(math.sqrt(sum(error * error for error in errors) / len(rows)), 6),
        "bias": round(sum(errors) / len(rows), 6),
        "p50_abs_error": round(percentile(abs_errors, 0.50), 6),
        "p90_abs_error": round(percentile(abs_errors, 0.90), 6),
        "p95_abs_error": round(percentile(abs_errors, 0.95), 6),
        "p99_abs_error": round(percentile(abs_errors, 0.99), 6),
        "metal_ratio": round(sum(1 for row in rows if row["target_is_metal"]) / len(rows), 6),
        "num_atoms_mean": round(sum(float(row["num_atoms"]) for row in rows) / len(rows), 6),
    }


def grouped_summary(rows: list[dict[str, Any]], key_fn) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(key_fn(row))].append(row)
    return [
        {"bucket": bucket, **summarize_rows(group)}
        for bucket, group in sorted(groups.items(), key=lambda item: item[0])
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    num_atoms_by_id = load_num_atoms_map(project_root / args.dataset_root)

    model_rows: dict[str, list[dict[str, Any]]] = {}
    overall_rows: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    gap_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    top_error_rows: list[dict[str, Any]] = []

    for model_name, prediction_path in PREDICTION_FILES.items():
        rows = load_prediction_rows(project_root / prediction_path, num_atoms_by_id, args.large_threshold)
        model_rows[model_name] = rows
        overall = {"model": model_name, **summarize_rows(rows)}
        overall_rows.append(overall)

        num_atoms = [float(row["num_atoms"]) for row in rows]
        abs_errors = [float(row["abs_error"]) for row in rows]
        correlation_rows.append(
            {
                "model": model_name,
                "pearson_num_atoms_abs_error": round(pearson(num_atoms, abs_errors), 6),
                "spearman_num_atoms_abs_error": round(spearman(num_atoms, abs_errors), 6),
            }
        )

        for summary in grouped_summary(rows, lambda row: atom_bucket(int(row["num_atoms"]))):
            bucket_rows.append({"model": model_name, **summary})
        for summary in grouped_summary(rows, lambda row: gap_bucket(float(row["target"]))):
            gap_rows.append({"model": model_name, **summary})

        for row in sorted(rows, key=lambda item: float(item["abs_error"]), reverse=True)[:30]:
            top_error_rows.append({"model": model_name, **row})

    write_csv(output_dir / "mp_large_error_overall.csv", overall_rows)
    write_csv(output_dir / "mp_large_error_by_num_atoms.csv", bucket_rows)
    write_csv(output_dir / "mp_large_error_by_gap.csv", gap_rows)
    write_csv(output_dir / "mp_large_error_correlations.csv", correlation_rows)
    write_csv(output_dir / "mp_large_error_top30.csv", top_error_rows)

    summary_md = "\n\n".join(
        [
            "# MP large-system error analysis",
            f"Large-system threshold: `num_atoms > {args.large_threshold}`.",
            "## Overall",
            markdown_table(
                overall_rows,
                ["model", "n", "mae", "rmse", "bias", "p90_abs_error", "p95_abs_error", "p99_abs_error", "metal_ratio"],
            ),
            "## Correlation With Num Atoms",
            markdown_table(
                correlation_rows,
                ["model", "pearson_num_atoms_abs_error", "spearman_num_atoms_abs_error"],
            ),
            "## By Num-Atoms Bucket",
            markdown_table(
                bucket_rows,
                ["model", "bucket", "n", "mae", "rmse", "bias", "p90_abs_error", "metal_ratio"],
            ),
            "## By Band-Gap Bucket",
            markdown_table(
                gap_rows,
                ["model", "bucket", "n", "mae", "rmse", "bias", "p90_abs_error"],
            ),
        ]
    )
    (output_dir / "mp_large_error_analysis.md").write_text(summary_md, encoding="utf-8")
    print(summary_md)


if __name__ == "__main__":
    main()
