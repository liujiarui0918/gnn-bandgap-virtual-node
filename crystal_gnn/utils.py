from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import torch
import yaml


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.ndim == 1:
        out = src.new_zeros(dim_size)
    else:
        out = src.new_zeros(dim_size, src.size(-1))
    if index.numel() == 0:
        return out
    out.index_add_(0, index, src)
    return out


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    out = scatter_sum(src, index, dim_size)
    counts = src.new_zeros(dim_size)
    if index.numel() > 0:
        counts.index_add_(0, index, torch.ones_like(index, dtype=src.dtype))
    counts = counts.clamp_min(1.0)
    if src.ndim == 1:
        return out / counts
    return out / counts.unsqueeze(-1)


def sinusoidal_encoding(coords: torch.Tensor, num_frequencies: int = 8) -> torch.Tensor:
    if coords.numel() == 0:
        return coords.new_zeros(coords.size(0), coords.size(1) * num_frequencies * 2)
    frequencies = (2 ** torch.arange(num_frequencies, device=coords.device, dtype=coords.dtype)) * math.pi
    angles = coords.unsqueeze(-1) * frequencies
    return torch.cat([angles.sin(), angles.cos()], dim=-1).flatten(1)


def regression_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    metal_threshold: float = 0.1,
) -> dict[str, float]:
    targets = targets.float()
    predictions = predictions.float()
    errors = predictions - targets
    mae = errors.abs().mean().item()
    rmse = torch.sqrt((errors.square()).mean()).item()
    target_mean = targets.mean()
    denominator = torch.sum((targets - target_mean).square()).item()
    numerator = torch.sum(errors.square()).item()
    r2 = 1.0 - numerator / denominator if denominator > 0 else 0.0

    target_is_metal = targets <= 1e-8
    pred_is_metal = predictions <= metal_threshold
    tp = torch.logical_and(target_is_metal, pred_is_metal).sum().item()
    fp = torch.logical_and(~target_is_metal, pred_is_metal).sum().item()
    fn = torch.logical_and(target_is_metal, ~pred_is_metal).sum().item()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "r2": round(r2, 6),
        "metal_f1": round(f1, 6),
    }

