from __future__ import annotations

import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch

from crystal_gnn.utils import regression_metrics


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # numpy and dgl are guarded so seeding never becomes a hard dependency:
    # a missing library must not turn a reproducibility helper into a crash.
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import dgl

        dgl.random.seed(seed)
    except ImportError:
        pass


def setup_output_dir(output_dir: str | Path, logger_name: str) -> tuple[Path, logging.Logger]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_path / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return output_path, logger


def save_json(path: str | Path, payload: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_predictions(path: str | Path, predictions: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["sample_id", "target", "prediction", "split"]
    extra_keys = sorted({key for row in predictions for key in row.keys() if key not in fieldnames})
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames + extra_keys)
        writer.writeheader()
        for row in predictions:
            writer.writerow(row)


def load_manifest(dataset_root: str | Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    dataset_root = Path(dataset_root)
    manifest_path = dataset_root / "manifest.jsonl"
    records: list[dict[str, Any]] = []
    split_map: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            records.append(record)
            split_map.setdefault(record["split"], []).append(record)
    return records, split_map


def ordered_split_records(split_map: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    return [*split_map.get("train", []), *split_map.get("val", []), *split_map.get("test", [])]


def device_from_config(device_name: str | None) -> torch.device:
    if device_name and device_name != "auto":
        return torch.device(device_name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def tensor_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, (list, tuple)):
        converted = [tensor_to_device(item, device) for item in value]
        return type(value)(converted)
    if isinstance(value, dict):
        return {key: tensor_to_device(item, device) for key, item in value.items()}
    return value


def peak_gpu_gb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return round(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3), 3)


def start_timer() -> float:
    return time.time()


def elapsed_seconds(start_time: float) -> float:
    return round(time.time() - start_time, 3)


def summarize_predictions(
    sample_ids: list[str],
    targets: list[float],
    predictions: list[float],
    split: str,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    target_tensor = torch.tensor(targets, dtype=torch.float32)
    prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
    metrics = regression_metrics(target_tensor, prediction_tensor)
    rows = [
        {
            "sample_id": sample_id,
            "target": float(target),
            "prediction": float(prediction),
            "split": split,
        }
        for sample_id, target, prediction in zip(sample_ids, targets, predictions)
    ]
    return metrics, rows

