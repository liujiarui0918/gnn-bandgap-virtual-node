from __future__ import annotations

import argparse
import contextlib
import logging
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader

from crystal_gnn.data import CrystalShardDataset, graph_collate
from crystal_gnn.models import build_model
from crystal_gnn.utils import load_yaml, regression_metrics, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train crystal graph models.")
    parser.add_argument("--model", required=True, help="Model name override.")
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--output_dir", required=True, help="Training artifact directory.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epochs override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional batch size override.")
    parser.add_argument("--processed_dir", default=None, help="Optional processed dataset directory override.")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional dataloader worker override.")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def make_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    lr = float(training_config["lr"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    optimizer_name = training_config.get("optimizer", "adam").lower()
    if optimizer_name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    steps_per_epoch: int,
) -> tuple[Any, str]:
    scheduler_name = training_config.get("scheduler", "none").lower()
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
    if scheduler_name == "onecycle":
        return (
            OneCycleLR(
                optimizer,
                max_lr=float(training_config.get("onecycle_max_lr", training_config["lr"])),
                epochs=int(training_config["epochs"]),
                steps_per_epoch=max(1, steps_per_epoch),
            ),
            scheduler_name,
        )
    if scheduler_name == "cosine":
        return (CosineAnnealingLR(optimizer, T_max=int(training_config["epochs"])), scheduler_name)
    return None, "none"


def compute_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, float]]:
    training_config = config["training"]
    model_config = config["model"]

    if model_config["name"] == "m3gnet_phys":
        main = F.mse_loss(outputs["band_gap"], batch["band_gap"])
        aux = F.mse_loss(outputs["cbm"], batch["cbm"]) + F.mse_loss(outputs["vbm"], batch["vbm"])
        phys = F.mse_loss(outputs["band_gap"], outputs["cbm"] - outputs["vbm"])
        nonneg = torch.relu(-outputs["band_gap"]).square().mean()
        loss = (
            main
            + float(training_config.get("lambda_aux", 0.3)) * aux
            + float(training_config.get("lambda_phys", 0.5)) * phys
            + float(training_config.get("lambda_nonneg", 0.1)) * nonneg
        )
        return loss, {
            "main": round(main.item(), 6),
            "aux": round(aux.item(), 6),
            "phys": round(phys.item(), 6),
            "nonneg": round(nonneg.item(), 6),
        }

    loss_name = training_config.get("loss", "mse").lower()
    if loss_name == "l1":
        loss = F.l1_loss(outputs["band_gap"], batch["band_gap"])
    elif loss_name == "smooth_l1":
        loss = F.smooth_l1_loss(outputs["band_gap"], batch["band_gap"])
    else:
        loss = F.mse_loss(outputs["band_gap"], batch["band_gap"])
    return loss, {"main": round(loss.item(), 6)}


def run_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scheduler_name: str = "none",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, float], list[dict[str, float]]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    epoch_losses: list[float] = []
    loss_components: list[dict[str, float]] = []
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    amp_enabled = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    grad_clip = float(config["training"].get("grad_clip_norm", 0.0))

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        with autocast_context:
            outputs = model(batch)
            loss, components = compute_loss(outputs, batch, config)

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

        epoch_losses.append(loss.detach().item())
        loss_components.append(components)
        predictions.append(outputs["band_gap"].detach().cpu())
        targets.append(batch["band_gap"].detach().cpu())

    prediction_tensor = torch.cat(predictions) if predictions else torch.empty(0)
    target_tensor = torch.cat(targets) if targets else torch.empty(0)
    metrics = regression_metrics(
        target_tensor,
        prediction_tensor,
        metal_threshold=float(config["training"].get("metal_threshold", 0.1)),
    )
    metrics["loss"] = round(sum(epoch_losses) / max(1, len(epoch_losses)), 6)
    return metrics, loss_components


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: dict[str, Any],
) -> dict[str, float]:
    with torch.no_grad():
        metrics, _ = run_epoch(model, loader, device, config)
    return metrics


def plot_history(history: list[dict[str, float]], output_path: Path) -> None:
    epochs = [item["epoch"] for item in history]
    train_mae = [item["train_mae"] for item in history]
    val_mae = [item["val_mae"] for item in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_mae, label="train_mae")
    plt.plot(epochs, val_mae, label="val_mae")
    plt.xlabel("Epoch")
    plt.ylabel("MAE (eV)")
    plt.title("Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    config["model"]["name"] = args.model
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.processed_dir is not None:
        config["data"]["processed_dir"] = args.processed_dir
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    output_dir = Path(args.output_dir)
    logger = setup_logging(output_dir)
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False, allow_unicode=True)

    seed_everything(int(config.get("seed", 42)))
    device_name = config["training"].get("device", "cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name if device_name != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    data_dir = Path(config["data"]["processed_dir"])
    train_dataset = CrystalShardDataset(data_dir, "train")
    val_dataset = CrystalShardDataset(data_dir, "val")
    test_dataset = CrystalShardDataset(data_dir, "test")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=graph_collate,
        pin_memory=bool(config["training"].get("pin_memory", True)),
    )
    eval_loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "collate_fn": graph_collate,
        "pin_memory": bool(config["training"].get("pin_memory", True)),
    }
    val_loader = DataLoader(val_dataset, **eval_loader_kwargs)
    test_loader = DataLoader(test_dataset, **eval_loader_kwargs)

    model = build_model(config["model"]).to(device)
    optimizer = make_optimizer(model, config["training"])
    scheduler, scheduler_name = make_scheduler(optimizer, config["training"], len(train_loader))
    amp_enabled = device.type == "cuda" and bool(config["training"].get("mixed_precision", True))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    logger.info("training on %s", device)
    logger.info("train/val/test sizes = %s / %s / %s", len(train_dataset), len(val_dataset), len(test_dataset))

    best_val_mae = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    best_checkpoint = output_dir / "best_model.pt"
    history: list[dict[str, float]] = []
    start_time = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    for epoch in range(1, int(config["training"]["epochs"]) + 1):
        train_metrics, _ = run_epoch(
            model,
            train_loader,
            device,
            config,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scaler=scaler,
        )
        val_metrics = evaluate(model, val_loader, device, config)

        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["mae"])
        elif scheduler_name == "cosine" and scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics["loss"],
                "train_mae": train_metrics["mae"],
                "val_loss": val_metrics["loss"],
                "val_mae": val_metrics["mae"],
                "lr": lr,
            }
        )
        logger.info(
            "epoch %02d | lr %.6g | train loss %.4f mae %.4f | val loss %.4f mae %.4f",
            epoch,
            lr,
            train_metrics["loss"],
            train_metrics["mae"],
            val_metrics["loss"],
            val_metrics["mae"],
        )

        if val_metrics["mae"] < best_val_mae:
            best_val_mae = val_metrics["mae"]
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                },
                best_checkpoint,
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= int(config["training"].get("early_stopping_patience", 10)):
            logger.info("early stopping triggered at epoch %s", epoch)
            break

    checkpoint = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device, config)

    elapsed = time.time() - start_time
    peak_gpu_gb = (
        round(torch.cuda.max_memory_allocated(device=device) / (1024 ** 3), 3) if device.type == "cuda" else 0.0
    )

    metrics = {
        "experiment_name": config.get("experiment_name", args.model),
        "model": args.model,
        "best_epoch": best_epoch,
        "epochs_ran": len(history),
        "train_seconds": round(elapsed, 2),
        "peak_gpu_gb": peak_gpu_gb,
        **test_metrics,
    }
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "history.json", {"history": history})
    plot_history(history, output_dir / "learning_curve.png")
    logger.info("test metrics: %s", metrics)


if __name__ == "__main__":
    main()
