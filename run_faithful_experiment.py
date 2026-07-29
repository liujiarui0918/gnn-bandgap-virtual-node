from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from crystal_gnn.utils import load_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one faithful crystal GNN experiment.")
    parser.add_argument("--config", required=True, help="Path to the experiment YAML config.")
    parser.add_argument("--output_dir", default=None, help="Optional output directory override.")
    parser.add_argument("--batch_size", type=int, default=None, help="Optional training batch size override.")
    parser.add_argument("--device", default=None, help="Optional device override.")
    parser.add_argument("--num_workers", type=int, default=None, help="Optional dataloader worker override.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override.")
    parser.add_argument("--lr", type=float, default=None, help="Optional learning-rate override.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if torch.cuda.is_available() and os.environ.get("GNN_ENABLE_TF32", "1").lower() not in {"0", "false", "no"}:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    config = load_yaml(args.config)
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.device is not None:
        config["training"]["device"] = args.device
    if args.num_workers is not None:
        config["training"]["num_workers"] = args.num_workers
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.lr is not None:
        config["training"]["lr"] = args.lr
        if "onecycle_max_lr" in config["training"]:
            config["training"]["onecycle_max_lr"] = args.lr
    output_dir = Path(args.output_dir) if args.output_dir else Path("results") / config["experiment_name"]
    runner = config["runner"].lower()

    if runner == "cgcnn":
        from faithful_gnn.cgcnn_runner import run_experiment
    elif runner == "alignn":
        from faithful_gnn.alignn_runner import run_experiment
    elif runner == "router":
        from faithful_gnn.router_runner import run_experiment
    elif runner == "m3gnet":
        from faithful_gnn.m3gnet_runner import run_experiment
    else:
        raise ValueError(f"unsupported faithful runner: {runner}")

    metrics = run_experiment(config, output_dir)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
