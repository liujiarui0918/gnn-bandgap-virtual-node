from __future__ import annotations

from pathlib import Path
from typing import Any
import time

import torch

from crystal_gnn.utils import load_yaml, regression_metrics
from faithful_gnn.alignn_runner import build_dataloaders, collect_prediction_rows
from faithful_gnn.alignn_variants import build_alignn_model
from faithful_gnn.common import (
    device_from_config,
    elapsed_seconds,
    load_manifest,
    peak_gpu_gb,
    save_json,
    save_predictions,
    seed_everything,
    setup_output_dir,
)


def _num_atoms_from_record(record: dict[str, Any]) -> int:
    structure = record.get("structure", {})
    if isinstance(structure, dict):
        sites = structure.get("sites")
        if isinstance(sites, list):
            return len(sites)
    try:
        from pymatgen.core import Structure

        structure_obj = Structure.from_dict(structure)
        if hasattr(structure_obj, "sites"):
            return len(structure_obj.sites)
        if hasattr(structure_obj, "__len__"):
            return len(structure_obj)  # type: ignore[arg-type]
    except Exception as exc:  # pragma: no cover - fallback for dataset shape drift
        raise ValueError(f"unable to infer num_atoms for sample {record.get('sample_id')}") from exc
    raise ValueError(f"unable to infer num_atoms for sample {record.get('sample_id')}")


def _num_atoms_by_id(dataset_root: str | Path) -> dict[str, int]:
    _records, split_map = load_manifest(dataset_root)
    return {
        record["sample_id"]: _num_atoms_from_record(record)
        for split_name in ("train", "val", "test")
        for record in split_map.get(split_name, [])
    }


def _rows_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["sample_id"]): row for row in rows}


def _metrics_from_rows(rows: list[dict[str, Any]]) -> dict[str, float]:
    targets = torch.tensor([float(row["target"]) for row in rows], dtype=torch.float32)
    predictions = torch.tensor([float(row["prediction"]) for row in rows], dtype=torch.float32)
    return regression_metrics(targets, predictions)


def route_rows_by_threshold(
    primary_rows: list[dict[str, Any]],
    secondary_rows: list[dict[str, Any]],
    num_atoms_by_id: dict[str, int],
    threshold: int,
    primary_name: str = "E4",
    secondary_name: str = "E8",
) -> list[dict[str, Any]]:
    primary_by_id = _rows_by_id(primary_rows)
    secondary_by_id = _rows_by_id(secondary_rows)
    sample_ids = list(primary_by_id.keys())
    if sample_ids != list(secondary_by_id.keys()):
        raise ValueError("expert predictions must cover the same samples in the same order")

    routed_rows: list[dict[str, Any]] = []
    for sample_id in sample_ids:
        num_atoms = int(num_atoms_by_id[sample_id])
        use_primary = num_atoms <= threshold
        expert_name = primary_name if use_primary else secondary_name
        expert_row = primary_by_id[sample_id] if use_primary else secondary_by_id[sample_id]
        routed_row = dict(expert_row)
        routed_row["num_atoms"] = num_atoms
        routed_row["router_threshold"] = int(threshold)
        routed_row["source_model"] = expert_name
        routed_rows.append(routed_row)
    return routed_rows


def select_best_num_atoms_threshold(
    primary_val_rows: list[dict[str, Any]],
    secondary_val_rows: list[dict[str, Any]],
    num_atoms_by_id: dict[str, int],
    primary_name: str = "E4",
    secondary_name: str = "E8",
) -> tuple[int, dict[str, float], list[dict[str, float | int]]]:
    sample_ids = [str(row["sample_id"]) for row in primary_val_rows]
    unique_num_atoms = sorted({int(num_atoms_by_id[sample_id]) for sample_id in sample_ids})
    if not unique_num_atoms:
        raise ValueError("validation predictions are empty")
    candidate_thresholds = sorted({unique_num_atoms[0] - 1, *unique_num_atoms})

    best_threshold = candidate_thresholds[0]
    best_metrics: dict[str, float] | None = None
    sweep: list[dict[str, float | int]] = []
    for threshold in candidate_thresholds:
        routed_rows = route_rows_by_threshold(
            primary_val_rows,
            secondary_val_rows,
            num_atoms_by_id,
            threshold=threshold,
            primary_name=primary_name,
            secondary_name=secondary_name,
        )
        metrics = _metrics_from_rows(routed_rows)
        sweep.append({"threshold": int(threshold), **metrics})
        if best_metrics is None or metrics["mae"] < best_metrics["mae"] or (
            metrics["mae"] == best_metrics["mae"] and threshold < best_threshold
        ):
            best_threshold = threshold
            best_metrics = metrics

    assert best_metrics is not None
    return int(best_threshold), best_metrics, sweep


def _load_alignn_expert(expert_config_path: str | Path, checkpoint_path: str | Path, device: torch.device) -> torch.nn.Module:
    expert_config = load_yaml(expert_config_path)
    model = build_alignn_model(expert_config["model"]).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir, logger = setup_output_dir(output_dir, "alignn_router")
    seed_everything(int(config.get("seed", 42)))
    device = device_from_config(config["training"].get("device"))

    logger.info("loading router dataloaders")
    _train_loader, val_loader, test_loader = build_dataloaders(config, physics_variant=False)
    logger.info(
        "router dataloader sizes | val=%s test=%s",
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    experts = config["router"]
    primary_name = str(experts["primary_expert"]["name"])
    secondary_name = str(experts["secondary_expert"]["name"])
    num_atoms_map = _num_atoms_by_id(Path(config["data"]["dataset_root"]).resolve())

    wall_start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    logger.info("loading primary expert %s", primary_name)
    primary_model = _load_alignn_expert(
        experts["primary_expert"]["config_path"],
        experts["primary_expert"]["checkpoint_path"],
        device,
    )
    logger.info("loading secondary expert %s", secondary_name)
    secondary_model = _load_alignn_expert(
        experts["secondary_expert"]["config_path"],
        experts["secondary_expert"]["checkpoint_path"],
        device,
    )

    primary_val_rows = collect_prediction_rows(primary_model, val_loader, device, physics_variant=False, split="val")
    secondary_val_rows = collect_prediction_rows(secondary_model, val_loader, device, physics_variant=False, split="val")
    best_threshold, best_val_metrics, threshold_sweep = select_best_num_atoms_threshold(
        primary_val_rows,
        secondary_val_rows,
        num_atoms_map,
        primary_name=primary_name,
        secondary_name=secondary_name,
    )

    logger.info("best validation threshold=%s | val mae=%.4f", best_threshold, best_val_metrics["mae"])

    primary_test_rows = collect_prediction_rows(primary_model, test_loader, device, physics_variant=False, split="test")
    secondary_test_rows = collect_prediction_rows(secondary_model, test_loader, device, physics_variant=False, split="test")

    routed_val_rows = route_rows_by_threshold(
        primary_val_rows,
        secondary_val_rows,
        num_atoms_map,
        threshold=best_threshold,
        primary_name=primary_name,
        secondary_name=secondary_name,
    )
    routed_test_rows = route_rows_by_threshold(
        primary_test_rows,
        secondary_test_rows,
        num_atoms_map,
        threshold=best_threshold,
        primary_name=primary_name,
        secondary_name=secondary_name,
    )

    test_metrics = _metrics_from_rows(routed_test_rows)
    metrics = {
        "experiment_name": config["experiment_name"],
        "model": config["router"].get("runner_name", "ALIGNN_ValRouted_E4_E8"),
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "metal_f1": test_metrics["metal_f1"],
        "best_threshold": int(best_threshold),
        "val_mae": best_val_metrics["mae"],
        "val_rmse": best_val_metrics["rmse"],
        "val_r2": best_val_metrics["r2"],
        "val_metal_f1": best_val_metrics["metal_f1"],
        "primary_expert": primary_name,
        "secondary_expert": secondary_name,
        "train_seconds": elapsed_seconds(wall_start_time),
        "peak_gpu_gb": peak_gpu_gb(device),
    }
    history = {
        "threshold_sweep": threshold_sweep,
        "best_threshold": int(best_threshold),
        "primary_expert": primary_name,
        "secondary_expert": secondary_name,
    }

    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    save_predictions(output_dir / "predictions.csv", routed_val_rows + routed_test_rows)
    logger.info("completed %s | threshold=%s | test mae=%.4f", config["experiment_name"], best_threshold, test_metrics["mae"])
    return metrics
