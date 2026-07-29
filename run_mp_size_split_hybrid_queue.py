from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from crystal_gnn.utils import load_yaml, regression_metrics
from faithful_gnn.common import load_manifest, save_json, save_predictions
from faithful_gnn.router_runner import _num_atoms_from_record


SMALL_EXPERT_DIR = "results/mp_size_split/MP_E20a_ALIGNN_virtualnode_small_le40"
LARGE_CONFIG = "configs/mp/MP_E21b_alignn_multi_virtualnode_large_gt40.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MP hybrid size split: E4 small expert + E8 large expert.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--results_dir", default="results/mp_size_split_hybrid")
    parser.add_argument("--threshold", type=int, default=40)
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def build_num_atoms_map(dataset_root: Path) -> dict[str, int]:
    _records, split_map = load_manifest(dataset_root)
    return {
        str(record["sample_id"]): _num_atoms_from_record(record)
        for split_name in ("train", "val", "test")
        for record in split_map.get(split_name, [])
    }


def load_prediction_rows(path: Path, source_model: str, num_atoms_map: dict[str, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = str(row["sample_id"])
            rows.append(
                {
                    "sample_id": sample_id,
                    "target": float(row["target"]),
                    "prediction": float(row["prediction"]),
                    "split": row["split"],
                    "num_atoms": int(num_atoms_map[sample_id]),
                    "source_model": source_model,
                }
            )
    return rows


def metric_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    targets = torch.tensor([float(row["target"]) for row in rows], dtype=torch.float32)
    predictions = torch.tensor([float(row["prediction"]) for row in rows], dtype=torch.float32)
    return regression_metrics(targets, predictions)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    results_root = project_root / args.results_dir
    results_root.mkdir(parents=True, exist_ok=True)

    large_config = load_yaml(project_root / LARGE_CONFIG)
    large_dir = results_root / large_config["experiment_name"]
    metrics_path = large_dir / "metrics.json"
    if metrics_path.exists() and not args.no_resume:
        print(f"[skip] metrics already exist for {large_config['experiment_name']}: {metrics_path}", flush=True)
    else:
        run_command(
            [
                args.python,
                "run_faithful_experiment.py",
                "--config",
                str(project_root / LARGE_CONFIG),
                "--output_dir",
                str(large_dir),
            ],
            cwd=project_root,
        )

    dataset_root = project_root / large_config["data"]["dataset_root"]
    num_atoms_map = build_num_atoms_map(dataset_root)
    small_dir = project_root / SMALL_EXPERT_DIR
    merged_dir = results_root / f"MP_E21_ALIGNN_size_split_e4small_e8large_le{args.threshold}_gt{args.threshold}"

    small_rows = load_prediction_rows(small_dir / "predictions.csv", "small_e4_le40", num_atoms_map)
    large_rows = load_prediction_rows(large_dir / "predictions.csv", "large_e8_gt40", num_atoms_map)
    merged_rows = sorted(small_rows + large_rows, key=lambda row: str(row["sample_id"]))
    test_rows = [row for row in merged_rows if row["split"] == "test"]
    test_metrics = metric_rows(test_rows)
    metrics = {
        "experiment_name": merged_dir.name,
        "model": "ALIGNN_SizeSplit_E4Small_E8Large",
        "threshold": int(args.threshold),
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "metal_f1": test_metrics["metal_f1"],
        "small_expert": "MP_E20a_ALIGNN_virtualnode_small_le40",
        "large_expert": large_config["experiment_name"],
        "test_samples": len(test_rows),
    }
    save_json(merged_dir / "metrics.json", metrics)
    save_predictions(merged_dir / "predictions.csv", merged_rows)
    print(json.dumps(metrics, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
