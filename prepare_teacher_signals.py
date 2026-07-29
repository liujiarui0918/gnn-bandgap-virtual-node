from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import torch

from crystal_gnn.utils import load_yaml
from faithful_gnn.alignn_runner import build_dataloaders, collect_prediction_rows
from faithful_gnn.alignn_variants import build_alignn_model
from faithful_gnn.common import device_from_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare per-sample E4/E8 teacher gain targets for adaptive dual-slot training.")
    parser.add_argument("--teacher_a_config", required=True, help="Config path for the conservative teacher, typically E4.")
    parser.add_argument("--teacher_a_dir", required=True, help="Output directory containing best_model.pt for teacher A.")
    parser.add_argument("--teacher_b_config", required=True, help="Config path for the enhanced teacher, typically E8.")
    parser.add_argument("--teacher_b_dir", required=True, help="Output directory containing best_model.pt for teacher B.")
    parser.add_argument("--output", required=True, help="CSV path for the generated gain targets.")
    parser.add_argument("--temperature", type=float, default=0.15, help="Soft target temperature on the teacher error delta.")
    return parser.parse_args()


def physics_variant_from_config(config: dict) -> bool:
    return config["model"].get("variant", "baseline").lower() in {
        "physconstraint",
        "sizesupervisedphysvirtualnode",
    }


def predict_all_splits(config: dict, checkpoint_dir: Path) -> list[dict[str, float | str]]:
    physics_variant = physics_variant_from_config(config)
    device = device_from_config(config["training"].get("device"))
    train_loader, val_loader, test_loader = build_dataloaders(config, physics_variant=physics_variant)
    loader_cls = type(train_loader)

    def ordered_eval_loader(loader, batch_size: int | None = None):
        return loader_cls(
            loader.dataset,
            batch_size=batch_size or loader.batch_size,
            shuffle=False,
            collate_fn=loader.collate_fn,
            drop_last=False,
            num_workers=getattr(loader, "num_workers", 0),
            pin_memory=getattr(loader, "pin_memory", False),
        )

    train_eval_loader = ordered_eval_loader(
        train_loader,
        batch_size=int(config["training"].get("test_batch_size", config["training"]["batch_size"])),
    )
    val_eval_loader = ordered_eval_loader(val_loader)
    test_eval_loader = ordered_eval_loader(test_loader)
    model = build_alignn_model(config["model"]).to(device)
    checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    rows: list[dict[str, float | str]] = []
    with torch.no_grad():
        rows.extend(
            collect_prediction_rows(
                model,
                train_eval_loader,
                device,
                physics_variant=physics_variant,
                split="train",
            )
        )
        rows.extend(
            collect_prediction_rows(
                model,
                val_eval_loader,
                device,
                physics_variant=physics_variant,
                split="val",
            )
        )
        rows.extend(
            collect_prediction_rows(
                model,
                test_eval_loader,
                device,
                physics_variant=physics_variant,
                split="test",
            )
        )
    return rows


def sigmoid(value: float) -> float:
    if value >= 0.0:
        z = math.exp(-value)
        return 1.0 / (1.0 + z)
    z = math.exp(value)
    return z / (1.0 + z)


def main() -> None:
    args = parse_args()
    teacher_a_config = load_yaml(args.teacher_a_config)
    teacher_b_config = load_yaml(args.teacher_b_config)

    teacher_a_rows = predict_all_splits(teacher_a_config, Path(args.teacher_a_dir))
    teacher_b_rows = predict_all_splits(teacher_b_config, Path(args.teacher_b_dir))
    teacher_a_map = {str(row["sample_id"]): row for row in teacher_a_rows}
    teacher_b_map = {str(row["sample_id"]): row for row in teacher_b_rows}

    missing = sorted(set(teacher_a_map).symmetric_difference(teacher_b_map))
    if missing:
        raise ValueError(f"teacher prediction sets do not align, first mismatches: {missing[:10]}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "target",
        "teacher_a_prediction",
        "teacher_b_prediction",
        "teacher_a_abs_error",
        "teacher_b_abs_error",
        "teacher_delta",
        "gain_target",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample_id in sorted(teacher_a_map):
            row_a = teacher_a_map[sample_id]
            row_b = teacher_b_map[sample_id]
            target = float(row_a["target"])
            prediction_a = float(row_a["prediction"])
            prediction_b = float(row_b["prediction"])
            abs_error_a = abs(prediction_a - target)
            abs_error_b = abs(prediction_b - target)
            teacher_delta = abs_error_a - abs_error_b
            gain_target = sigmoid(teacher_delta / max(args.temperature, 1e-6))
            writer.writerow(
                {
                    "sample_id": sample_id,
                    "split": row_a["split"],
                    "target": target,
                    "teacher_a_prediction": prediction_a,
                    "teacher_b_prediction": prediction_b,
                    "teacher_a_abs_error": abs_error_a,
                    "teacher_b_abs_error": abs_error_b,
                    "teacher_delta": teacher_delta,
                    "gain_target": gain_target,
                }
            )

    print(f"saved teacher gain targets to {output_path}")


if __name__ == "__main__":
    main()
