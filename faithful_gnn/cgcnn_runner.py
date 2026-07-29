from __future__ import annotations

import contextlib
import hashlib
import json
import random
import time
import warnings
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.multiprocessing as torch_mp
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from faithful_gnn.common import (
    elapsed_seconds,
    peak_gpu_gb,
    save_json,
    save_predictions,
    seed_everything,
    setup_output_dir,
    device_from_config,
)
from faithful_gnn.vendor import configure_vendor_paths

configure_vendor_paths()

from cgcnn.data import CIFData, collate_pool  # noqa: E402
from cgcnn.model import CrystalGraphConvNet  # noqa: E402

try:
    torch_mp.set_sharing_strategy("file_system")
except RuntimeError:
    pass


class Normalizer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)
        if float(self.std) == 0.0:
            self.std = torch.tensor(1.0, dtype=tensor.dtype)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=tensor.device, dtype=tensor.dtype)
        std = self.std.to(device=tensor.device, dtype=tensor.dtype)
        return (tensor - mean) / std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        mean = self.mean.to(device=normed_tensor.device, dtype=normed_tensor.dtype)
        std = self.std.to(device=normed_tensor.device, dtype=normed_tensor.dtype)
        return normed_tensor * std + mean


class InMemoryCIFData(Dataset):
    def __init__(self, samples: list[tuple[Any, torch.Tensor, str]]) -> None:
        self.samples = samples
        self.id_prop_data = [(str(cif_id), float(target.view(-1)[0])) for _, target, cif_id in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[Any, torch.Tensor, str]:
        return self.samples[index]


def list_collate(batch: list[Any]) -> list[Any]:
    return batch


def build_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = training_config.get("optimizer", "adam").lower()
    lr = float(training_config["lr"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    train_loader: DataLoader,
) -> tuple[Any, str]:
    scheduler_name = training_config.get("scheduler", "none").lower()
    if scheduler_name == "onecycle":
        return (
            OneCycleLR(
                optimizer,
                max_lr=float(training_config.get("onecycle_max_lr", training_config["lr"])),
                epochs=int(training_config["epochs"]),
                steps_per_epoch=max(1, len(train_loader)),
            ),
            scheduler_name,
        )
    if scheduler_name == "plateau":
        return (
            ReduceLROnPlateau(
                optimizer,
                mode="min",
                patience=int(training_config.get("scheduler_patience", 5)),
                factor=float(training_config.get("scheduler_factor", 0.5)),
            ),
            scheduler_name,
        )
    if scheduler_name == "multistep":
        return (
            MultiStepLR(
                optimizer,
                milestones=[int(item) for item in training_config.get("lr_milestones", [100])],
                gamma=float(training_config.get("scheduler_gamma", 0.1)),
            ),
            scheduler_name,
        )
    if scheduler_name == "cosine":
        return (CosineAnnealingLR(optimizer, T_max=int(training_config["epochs"])), scheduler_name)
    return None, "none"


def build_dataset(split_root: Path, model_config: dict[str, Any]) -> CIFData:
    return CIFData(
        str(split_root),
        max_num_nbr=int(model_config.get("max_num_nbr", 12)),
        radius=float(model_config.get("radius", 8.0)),
        dmin=float(model_config.get("dmin", 0.0)),
        step=float(model_config.get("step", 0.2)),
        random_seed=int(model_config.get("random_seed", 42)),
    )


def cgcnn_cache_path(split_root: Path, model_config: dict[str, Any], cache_dir: Path, split_name: str) -> Path:
    id_prop_path = split_root / "id_prop.csv"
    atom_init_path = split_root / "atom_init.json"
    payload = {
        "split_root": str(split_root.resolve()),
        "split_name": split_name,
        "id_prop_mtime_ns": id_prop_path.stat().st_mtime_ns if id_prop_path.exists() else None,
        "id_prop_size": id_prop_path.stat().st_size if id_prop_path.exists() else None,
        "atom_init_mtime_ns": atom_init_path.stat().st_mtime_ns if atom_init_path.exists() else None,
        "max_num_nbr": int(model_config.get("max_num_nbr", 12)),
        "radius": float(model_config.get("radius", 8.0)),
        "dmin": float(model_config.get("dmin", 0.0)),
        "step": float(model_config.get("step", 0.2)),
        "random_seed": int(model_config.get("random_seed", 42)),
    }
    digest = hashlib.sha1(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]
    return cache_dir / f"{split_name}_{digest}.pt"


def preload_dataset(
    dataset: CIFData,
    split_root: Path,
    split_name: str,
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    training_config: dict[str, Any],
) -> Dataset:
    if not bool(training_config.get("preload_to_memory", False)):
        return dataset

    cache_dir_value = data_config.get("cgcnn_graph_cache_dir")
    cache_dir = Path(cache_dir_value) if cache_dir_value else split_root.parent / "cgcnn_graph_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cgcnn_cache_path(split_root, model_config, cache_dir, split_name)
    rebuild = bool(training_config.get("rebuild_graph_cache", False))

    if cache_path.exists() and not rebuild:
        print(f"[cgcnn-cache] loading {split_name}: {cache_path}", flush=True)
        samples = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"[cgcnn-cache] loaded {split_name}: {len(samples)} samples", flush=True)
        return InMemoryCIFData(samples)

    workers = int(training_config.get("preload_num_workers", max(1, int(training_config.get("num_workers", 0)))))
    batch_size = int(training_config.get("preload_batch_size", 128))
    print(
        f"[cgcnn-cache] building {split_name}: {len(dataset)} samples, workers={workers}, batch_size={batch_size}",
        flush=True,
    )
    started = time.time()
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": False,
        "collate_fn": list_collate,
    }
    if workers > 0:
        loader_kwargs["persistent_workers"] = False
    loader = DataLoader(dataset, **loader_kwargs)
    samples: list[tuple[Any, torch.Tensor, str]] = []
    for batch_index, batch in enumerate(loader, start=1):
        samples.extend(batch)
        if batch_index == 1 or batch_index % 50 == 0:
            print(f"[cgcnn-cache] {split_name}: cached {len(samples)}/{len(dataset)}", flush=True)

    tmp_path = cache_path.with_suffix(f".{int(time.time())}.tmp")
    torch.save(samples, tmp_path)
    tmp_path.replace(cache_path)
    seconds = time.time() - started
    print(f"[cgcnn-cache] built {split_name}: {len(samples)} samples in {seconds:.1f}s -> {cache_path}", flush=True)
    return InMemoryCIFData(samples)


def build_dataloaders(config: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_root = Path(config["data"]["dataset_root"]) / "cgcnn"
    train_root = dataset_root / "train"
    val_root = dataset_root / "val"
    test_root = dataset_root / "test"
    train_dataset = build_dataset(train_root, config["model"])
    val_dataset = build_dataset(val_root, config["model"])
    test_dataset = build_dataset(test_root, config["model"])

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", True))
    test_batch_size = int(config["training"].get("test_batch_size", batch_size))

    train_dataset = preload_dataset(train_dataset, train_root, "train", config["model"], config["data"], config["training"])
    val_dataset = preload_dataset(val_dataset, val_root, "val", config["model"], config["data"], config["training"])
    test_dataset = preload_dataset(test_dataset, test_root, "test", config["model"], config["data"], config["training"])

    loader_kwargs: dict[str, Any] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": collate_pool,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(config["training"].get("persistent_workers", True))
        loader_kwargs["prefetch_factor"] = int(config["training"].get("prefetch_factor", 2))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        **loader_kwargs,
    )
    return train_loader, val_loader, test_loader


def build_normalizer(dataset: CIFData, seed: int) -> Normalizer:
    rng = random.Random(seed)
    sample_size = min(len(dataset), 500)
    sample_indices = list(range(len(dataset)))
    rng.shuffle(sample_indices)
    selected = [dataset[index] for index in sample_indices[:sample_size]]
    targets = torch.stack([item[1] for item in selected], dim=0)
    return Normalizer(targets)


def move_input_to_device(batch_input: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]:
    atom_features, neighbor_features, neighbor_indices, crystal_atom_idx = batch_input
    return (
        atom_features.to(device),
        neighbor_features.to(device),
        neighbor_indices.to(device),
        [index.to(device) for index in crystal_atom_idx],
    )


def regression_summary(targets: list[float], predictions: list[float]) -> dict[str, float]:
    from crystal_gnn.utils import regression_metrics

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
    return regression_metrics(target_tensor, prediction_tensor)


def run_epoch(
    model: CrystalGraphConvNet,
    loader: DataLoader,
    device: torch.device,
    normalizer: Normalizer,
    training_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scheduler_name: str = "none",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    amp_enabled = device.type == "cuda" and bool(training_config.get("mixed_precision", False))
    grad_clip = float(training_config.get("grad_clip_norm", 0.0))
    losses: list[float] = []
    targets: list[float] = []
    predictions: list[float] = []
    prediction_rows: list[dict[str, Any]] = []

    for batch_input, target, batch_cif_ids in loader:
        batch_input = move_input_to_device(batch_input, device)
        target = target.view(-1).to(device)
        target_norm = normalizer.norm(target)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        with autocast_context:
            output = model(*batch_input).view(-1)
            loss = nn.functional.mse_loss(output, target_norm)

        if train_mode:
            if scaler is not None and amp_enabled:
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
            if scheduler_name == "onecycle" and scheduler is not None:
                scheduler.step()

        denorm_output = normalizer.denorm(output.detach().cpu())
        losses.append(float(loss.detach().cpu().item()))
        targets.extend(target.detach().cpu().tolist())
        predictions.extend(denorm_output.tolist())
        prediction_rows.extend(
            {
                "sample_id": str(cif_id),
                "target": float(true_value),
                "prediction": float(predicted_value),
                "split": "",
            }
            for cif_id, true_value, predicted_value in zip(batch_cif_ids, target.detach().cpu().tolist(), denorm_output.tolist())
        )

    metrics = regression_summary(targets, predictions)
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 6)
    return metrics, prediction_rows


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir, logger = setup_output_dir(output_dir, "cgcnn_faithful")
    warnings.filterwarnings("ignore", message=r".*not find enough neighbors to build graph.*")
    warnings.filterwarnings("ignore", message=r"Issues encountered while parsing CIF:.*")
    seed_everything(int(config.get("seed", 42)))
    device = device_from_config(config["training"].get("device"))
    train_loader, val_loader, test_loader = build_dataloaders(config)
    normalizer = build_normalizer(train_loader.dataset, int(config.get("seed", 42)))

    sample_input, _, _ = train_loader.dataset[0]
    orig_atom_fea_len = sample_input[0].shape[-1]
    nbr_fea_len = sample_input[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len=orig_atom_fea_len,
        nbr_fea_len=nbr_fea_len,
        atom_fea_len=int(config["model"].get("atom_fea_len", 64)),
        n_conv=int(config["model"].get("n_conv", 3)),
        h_fea_len=int(config["model"].get("h_fea_len", 128)),
        n_h=int(config["model"].get("n_h", 1)),
        classification=False,
    ).to(device)

    optimizer = build_optimizer(model, config["training"])
    scheduler, scheduler_name = build_scheduler(optimizer, config["training"], train_loader)
    amp_enabled = device.type == "cuda" and bool(config["training"].get("mixed_precision", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None
    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    best_checkpoint = output_dir / "best_model.pt"

    import time

    start_time = time.time()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    epochs = int(config["training"]["epochs"])
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 0) or 0)
    epochs_without_improvement = 0
    for epoch in range(1, epochs + 1):
        train_metrics, _train_predictions = run_epoch(
            model,
            train_loader,
            device,
            normalizer,
            config["training"],
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics, _val_predictions = run_epoch(
                model,
                val_loader,
                device,
                normalizer,
                config["training"],
            )

        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["mae"])
        elif scheduler_name in {"cosine", "multistep"} and scheduler is not None:
            scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "lr": optimizer.param_groups[0]["lr"],
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
            }
        )
        logger.info(
            "epoch %02d | lr %.6g | train loss %.4f mae %.4f | val loss %.4f mae %.4f",
            epoch,
            optimizer.param_groups[0]["lr"],
            train_metrics["loss"],
            train_metrics["mae"],
            val_metrics["loss"],
            val_metrics["mae"],
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save({"model_state_dict": model.state_dict(), "config": config}, best_checkpoint)
        else:
            epochs_without_improvement += 1

        if early_stopping_patience > 0 and epochs_without_improvement >= early_stopping_patience:
            logger.info(
                "early stopping at epoch %02d (no val-MAE improvement for %d epochs; best epoch %02d, best val mae %.4f)",
                epoch,
                epochs_without_improvement,
                best_epoch,
                best_val_mae,
            )
            break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        test_metrics, prediction_rows = run_epoch(model, test_loader, device, normalizer, config["training"])
    for row in prediction_rows:
        row["split"] = "test"

    metrics = {
        "experiment_name": config["experiment_name"],
        "model": config["model"].get("runner_name", "CGCNN"),
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "metal_f1": test_metrics["metal_f1"],
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "train_seconds": elapsed_seconds(start_time),
        "peak_gpu_gb": peak_gpu_gb(device),
    }
    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    save_predictions(output_dir / "predictions.csv", prediction_rows)
    logger.info("completed %s | test mae=%.4f rmse=%.4f", config["experiment_name"], test_metrics["mae"], test_metrics["rmse"])
    return metrics
