from __future__ import annotations

import contextlib
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import Dataset

from crystal_gnn.utils import scatter_mean
from faithful_gnn.alignn_variants import build_alignn_model, graph_node_batch
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
from faithful_gnn.phys_features import GRAPH_EXTRA_FEATURE_INDEX, build_graph_extra_features, load_teacher_signal_map
from faithful_gnn.vendor import configure_vendor_paths

configure_vendor_paths()

from dgl.dataloading import GraphDataLoader  # noqa: E402
from jarvis.core.atoms import pmg_to_atoms  # noqa: E402
from pymatgen.core import Structure  # noqa: E402
from alignn.data import get_train_val_loaders  # noqa: E402


def graph_edge_batch(graph, device: torch.device) -> torch.Tensor:
    batch_sizes = graph.batch_num_edges().to(device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), batch_sizes)


def graph_level_ndata(graph: Any | None, key: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
    if graph is None or key not in graph.ndata:
        return None
    feature = graph.ndata[key].to(device=device, dtype=dtype)
    node_batch = graph_node_batch(graph, device)
    return scatter_mean(feature, node_batch, int(graph.batch_size))


class AlignnMetadataDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        records_by_id: dict[str, dict[str, Any]],
        teacher_signal_map: dict[str, float],
    ) -> None:
        self.base_dataset = base_dataset
        self.ids = [str(sample_id) for sample_id in getattr(base_dataset, "ids")]
        self.prepare_batch = getattr(base_dataset, "prepare_batch", None)
        self.extra_feature_map: dict[str, torch.Tensor] = {}
        self.teacher_signal_map = teacher_signal_map
        for sample_id in self.ids:
            record = records_by_id[sample_id]
            self.extra_feature_map[sample_id] = torch.tensor(
                build_graph_extra_features(record),
                dtype=torch.get_default_dtype(),
            )

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        sample = self.base_dataset[index]
        sample_id = self.ids[index]
        extra_features = self.extra_feature_map[sample_id]

        if len(sample) == 4:
            graph, line_graph, lattice, label = sample
            graph = graph.local_var()
            graph.ndata["extra_features"] = extra_features.view(1, -1).repeat(graph.num_nodes(), 1)
            if self.teacher_signal_map:
                gain_target = float(self.teacher_signal_map.get(sample_id, 0.5))
                graph.ndata["teacher_gain_target"] = graph.ndata["extra_features"].new_full(
                    (graph.num_nodes(), 1),
                    gain_target,
                )
            return graph, line_graph, lattice, label

        graph, lattice, label = sample
        graph = graph.local_var()
        graph.ndata["extra_features"] = extra_features.view(1, -1).repeat(graph.num_nodes(), 1)
        if self.teacher_signal_map:
            gain_target = float(self.teacher_signal_map.get(sample_id, 0.5))
            graph.ndata["teacher_gain_target"] = graph.ndata["extra_features"].new_full(
                (graph.num_nodes(), 1),
                gain_target,
            )
        return graph, lattice, label


class FilteredMetadataDataset(Dataset):
    def __init__(self, base_dataset: AlignnMetadataDataset, indices: list[int]) -> None:
        self.base_dataset = base_dataset
        self.indices = indices
        self.ids = [base_dataset.ids[index] for index in indices]
        self.prepare_batch = getattr(base_dataset, "prepare_batch", None)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        return self.base_dataset[self.indices[index]]


def num_atoms_from_record(record: dict[str, Any]) -> int:
    if "num_atoms" in record:
        return int(record["num_atoms"])
    structure = record.get("structure", {})
    if isinstance(structure, dict):
        sites = structure.get("sites")
        if isinstance(sites, list):
            return len(sites)
    return len(Structure.from_dict(structure))


def apply_num_atoms_filter(
    dataset: AlignnMetadataDataset,
    records_by_id: dict[str, dict[str, Any]],
    data_config: dict[str, Any],
    split_name: str,
) -> Dataset:
    min_atoms = data_config.get("num_atoms_min")
    max_atoms = data_config.get("num_atoms_max")
    if min_atoms is None and max_atoms is None:
        return dataset

    min_atoms_int = int(min_atoms) if min_atoms is not None else None
    max_atoms_int = int(max_atoms) if max_atoms is not None else None
    indices: list[int] = []
    for index, sample_id in enumerate(dataset.ids):
        num_atoms = num_atoms_from_record(records_by_id[str(sample_id)])
        if min_atoms_int is not None and num_atoms < min_atoms_int:
            continue
        if max_atoms_int is not None and num_atoms > max_atoms_int:
            continue
        indices.append(index)
    if not indices:
        raise ValueError(
            f"num_atoms filter removed all {split_name} samples: "
            f"min={min_atoms_int} max={max_atoms_int}"
        )
    return FilteredMetadataDataset(dataset, indices)


def build_optimizer(model: torch.nn.Module, training_config: dict[str, Any]) -> torch.optim.Optimizer:
    optimizer_name = training_config.get("optimizer", "adamw").lower()
    lr = float(training_config["lr"])
    weight_decay = float(training_config.get("weight_decay", 0.0))
    if optimizer_name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    return AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    training_config: dict[str, Any],
    train_loader: GraphDataLoader,
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
    if scheduler_name == "cosine":
        return (CosineAnnealingLR(optimizer, T_max=int(training_config["epochs"])), scheduler_name)
    return None, "none"


def manifest_to_alignn_record(record: dict[str, Any], physics_variant: bool) -> dict[str, Any]:
    structure = Structure.from_dict(record["structure"])
    atoms = pmg_to_atoms(structure)
    target: float | list[float]
    if physics_variant:
        target = [float(record["band_gap"]), float(record["cbm"]), float(record["vbm"])]
    else:
        target = float(record["band_gap"])
    return {
        "jid": record["sample_id"],
        "atoms": atoms.to_dict(),
        "target": target,
    }


def build_dataloaders(config: dict[str, Any], physics_variant: bool) -> tuple[GraphDataLoader, GraphDataLoader, GraphDataLoader]:
    data_config = config["data"]
    dataset_root = Path(data_config["dataset_root"]).resolve()
    raw_cache_dir = Path(str(data_config.get("cache_dir", dataset_root / "alignn_cache")))
    local_cache_root = data_config.get("local_cache_root") or os.environ.get("ALIGNN_LOCAL_CACHE_ROOT")
    cache_namespace = str(data_config.get("cache_namespace", config["experiment_name"]))
    if local_cache_root:
        cache_dir = Path(str(local_cache_root)).resolve() / cache_namespace
    elif os.name != "nt" and not raw_cache_dir.is_absolute():
        # LMDB is more reliable on a local scratch disk than on mounted project storage.
        cache_dir = Path("/tmp/gnn_alignn_cache").resolve() / cache_namespace
    else:
        cache_dir = raw_cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename_prefix = str((cache_dir / "alignn_faithful_").resolve())

    _records, split_map = load_manifest(dataset_root)
    ordered_records = [*split_map["train"], *split_map["val"], *split_map["test"]]
    records_by_id = {str(record["sample_id"]): record for record in ordered_records}
    teacher_signal_map = load_teacher_signal_map(data_config.get("teacher_signal_path"))
    dataset_array = [manifest_to_alignn_record(record, physics_variant=physics_variant) for record in ordered_records]

    train_loader, val_loader, test_loader, _prepare_batch = get_train_val_loaders(
        dataset_array=dataset_array,
        target="target",
        n_train=len(split_map["train"]),
        n_val=len(split_map["val"]),
        n_test=len(split_map["test"]),
        train_ratio=None,
        val_ratio=0.0,
        test_ratio=0.0,
        batch_size=int(config["training"]["batch_size"]),
        standardize=False,
        line_graph=True,
        split_seed=int(config.get("seed", 42)),
        workers=int(config["training"].get("num_workers", 0)),
        pin_memory=bool(config["training"].get("pin_memory", True)),
        save_dataloader=False,
        filename=filename_prefix,
        id_tag="jid",
        cutoff=float(data_config.get("cutoff", 8.0)),
        cutoff_extra=float(data_config.get("cutoff_extra", 3.0)),
        max_neighbors=int(data_config.get("max_neighbors", 12)),
        keep_data_order=True,
        output_dir=str(cache_dir),
        use_lmdb=bool(data_config.get("use_lmdb", True)),
    )

    collate_fn = train_loader.collate_fn
    num_workers = int(config["training"].get("num_workers", 0))
    pin_memory = bool(config["training"].get("pin_memory", True))
    batch_size = int(config["training"]["batch_size"])
    test_batch_size = int(config["training"].get("test_batch_size", batch_size))

    train_dataset = AlignnMetadataDataset(train_loader.dataset, records_by_id, teacher_signal_map)
    val_dataset = AlignnMetadataDataset(val_loader.dataset, records_by_id, teacher_signal_map)
    test_dataset = AlignnMetadataDataset(test_loader.dataset, records_by_id, teacher_signal_map)
    train_dataset = apply_num_atoms_filter(train_dataset, records_by_id, data_config, "train")
    val_dataset = apply_num_atoms_filter(val_dataset, records_by_id, data_config, "val")
    test_dataset = apply_num_atoms_filter(test_dataset, records_by_id, data_config, "test")

    train_loader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = GraphDataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader


def move_batch(batch: tuple[Any, ...], device: torch.device) -> tuple[Any, ...]:
    return tuple(item.to(device) if hasattr(item, "to") else item for item in batch)


def compute_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    training_config: dict[str, Any],
    physics_variant: bool,
    graph: Any | None = None,
    line_graph: Any | None = None,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor]:
    def masked_mse_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        if bool(valid_mask.any().item()):
            return F.mse_loss(prediction[valid_mask], target[valid_mask])
        return prediction.sum() * 0.0

    def size_supervision_loss() -> tuple[torch.Tensor, dict[str, float]]:
        lambda_size = float(training_config.get("lambda_size", 0.0))
        if lambda_size <= 0.0 or graph is None or "slot2_gate_logit" not in outputs:
            return outputs["band_gap"].sum() * 0.0, {}
        size_threshold = float(training_config.get("size_loss_threshold", 80.0))
        size_temperature = max(float(training_config.get("size_loss_temperature", 20.0)), 1e-6)
        batch_sizes = graph.batch_num_nodes().to(
            device=outputs["band_gap"].device,
            dtype=outputs["band_gap"].dtype,
        )
        size_target = torch.sigmoid((batch_sizes - size_threshold) / size_temperature)
        slot2_size = F.binary_cross_entropy_with_logits(
            outputs["slot2_gate_logit"].view(-1),
            size_target.view(-1),
        )
        loss = lambda_size * slot2_size
        components = {"size": round(slot2_size.item(), 6)}

        lambda_size_fusion = float(training_config.get("lambda_size_fusion", 0.0))
        if lambda_size_fusion > 0.0 and "global_fusion_logit" in outputs:
            fusion_size = F.binary_cross_entropy_with_logits(
                outputs["global_fusion_logit"].view(-1),
                size_target.view(-1),
            )
            loss = loss + lambda_size_fusion * fusion_size
            components["size_fusion"] = round(fusion_size.item(), 6)
        return loss, components

    def gain_supervision_loss() -> tuple[torch.Tensor, dict[str, float]]:
        lambda_gain = float(training_config.get("lambda_gain", 0.0))
        if lambda_gain <= 0.0 or graph is None or "slot2_gate_logit" not in outputs:
            return outputs["band_gap"].sum() * 0.0, {}
        gain_target = graph_level_ndata(
            graph,
            "teacher_gain_target",
            device=outputs["band_gap"].device,
            dtype=outputs["band_gap"].dtype,
        )
        if gain_target is None:
            return outputs["band_gap"].sum() * 0.0, {}
        gain_target = gain_target.view(-1).clamp(0.0, 1.0)
        slot2_gain = F.binary_cross_entropy_with_logits(outputs["slot2_gate_logit"].view(-1), gain_target)
        loss = lambda_gain * slot2_gain
        components = {
            "gain": round(slot2_gain.item(), 6),
            "gain_target_mean": round(gain_target.mean().item(), 6),
        }
        lambda_gain_fusion = float(training_config.get("lambda_gain_fusion", 0.0))
        if lambda_gain_fusion > 0.0 and "global_fusion_logit" in outputs:
            fusion_gain = F.binary_cross_entropy_with_logits(outputs["global_fusion_logit"].view(-1), gain_target)
            loss = loss + lambda_gain_fusion * fusion_gain
            components["gain_fusion"] = round(fusion_gain.item(), 6)
        return loss, components

    def complexity_supervision_loss() -> tuple[torch.Tensor, dict[str, float]]:
        lambda_complexity = float(training_config.get("lambda_complexity", 0.0))
        if lambda_complexity <= 0.0 or graph is None or "slot2_gate_logit" not in outputs:
            return outputs["band_gap"].sum() * 0.0, {}
        extra_features = graph_level_ndata(
            graph,
            "extra_features",
            device=outputs["band_gap"].device,
            dtype=outputs["band_gap"].dtype,
        )
        if extra_features is None:
            return outputs["band_gap"].sum() * 0.0, {}

        batch_size = int(graph.batch_size)
        edge_lengths = torch.norm(graph.edata["r"].to(device=outputs["band_gap"].device, dtype=outputs["band_gap"].dtype), dim=1)
        if edge_lengths.numel() > 0:
            edge_batch = graph_edge_batch(graph, outputs["band_gap"].device)
            edge_mean = scatter_mean(edge_lengths.unsqueeze(-1), edge_batch, batch_size).view(-1)
            centered = edge_lengths - edge_mean[edge_batch]
            edge_var = scatter_mean((centered * centered).unsqueeze(-1), edge_batch, batch_size).view(-1)
            edge_cv = edge_var.sqrt() / edge_mean.clamp_min(1e-6)
        else:
            edge_cv = outputs["band_gap"].new_zeros((batch_size,))

        edge_cv_norm = (edge_cv / float(training_config.get("complexity_edge_cv_scale", 0.35))).clamp(0.0, 1.0)
        lattice_cv_norm = (
            extra_features[:, GRAPH_EXTRA_FEATURE_INDEX["lattice_length_cv"]]
            / float(training_config.get("complexity_lattice_cv_scale", 0.35))
        ).clamp(0.0, 1.0)
        lattice_angle_norm = (
            extra_features[:, GRAPH_EXTRA_FEATURE_INDEX["lattice_angle_std"]]
            / float(training_config.get("complexity_angle_std_scale", 0.15))
        ).clamp(0.0, 1.0)
        complexity_target = (
            float(training_config.get("complexity_edge_weight", 0.35)) * edge_cv_norm
            + float(training_config.get("complexity_entropy_weight", 0.25))
            * extra_features[:, GRAPH_EXTRA_FEATURE_INDEX["composition_entropy"]].clamp(0.0, 1.0)
            + float(training_config.get("complexity_element_weight", 0.2))
            * extra_features[:, GRAPH_EXTRA_FEATURE_INDEX["element_count_norm"]].clamp(0.0, 1.0)
            + float(training_config.get("complexity_lattice_weight", 0.1)) * lattice_cv_norm
            + float(training_config.get("complexity_angle_weight", 0.1)) * lattice_angle_norm
        ).clamp(0.0, 1.0)
        slot2_complexity = F.binary_cross_entropy_with_logits(outputs["slot2_gate_logit"].view(-1), complexity_target)
        loss = lambda_complexity * slot2_complexity
        components = {
            "complexity": round(slot2_complexity.item(), 6),
            "complexity_target_mean": round(complexity_target.mean().item(), 6),
            "complexity_edge_cv_mean": round(edge_cv.mean().item(), 6),
        }
        lambda_complexity_fusion = float(training_config.get("lambda_complexity_fusion", 0.0))
        if lambda_complexity_fusion > 0.0 and "global_fusion_logit" in outputs:
            fusion_complexity = F.binary_cross_entropy_with_logits(
                outputs["global_fusion_logit"].view(-1),
                complexity_target,
            )
            loss = loss + lambda_complexity_fusion * fusion_complexity
            components["complexity_fusion"] = round(fusion_complexity.item(), 6)
        return loss, components

    def orthogonality_loss() -> tuple[torch.Tensor, dict[str, float]]:
        lambda_orth = float(training_config.get("lambda_orth", 0.0))
        if lambda_orth <= 0.0 or "primary_slot_state" not in outputs or "secondary_slot_state" not in outputs:
            return outputs["band_gap"].sum() * 0.0, {}
        primary_slot = F.normalize(outputs["primary_slot_state"], dim=-1)
        secondary_slot = F.normalize(outputs["secondary_slot_state"], dim=-1)
        slot_cosine = (primary_slot * secondary_slot).sum(dim=-1)
        orth = slot_cosine.square().mean()
        loss = lambda_orth * orth
        components = {
            "orth": round(orth.item(), 6),
            "slot_cosine_abs_mean": round(slot_cosine.abs().mean().item(), 6),
        }
        lambda_slot_collapse = float(training_config.get("lambda_slot_collapse", 0.0))
        min_slot2_mean = float(training_config.get("min_slot2_mean", 0.15))
        if lambda_slot_collapse > 0.0 and "slot2_enable" in outputs:
            collapse = torch.relu(outputs["slot2_enable"].new_tensor(min_slot2_mean) - outputs["slot2_enable"].mean())
            loss = loss + lambda_slot_collapse * collapse
            components["slot_collapse"] = round(collapse.item(), 6)
        return loss, components

    def regime_partition_loss(band_gap_target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        lambda_regime = float(training_config.get("lambda_regime", 0.0))
        if lambda_regime <= 0.0 or "regime_logits" not in outputs:
            return outputs["band_gap"].sum() * 0.0, {}
        metal_epsilon = float(training_config.get("metal_epsilon", 1e-8))
        small_gap_threshold = float(training_config.get("small_gap_threshold", 1.0))
        regime_target = torch.zeros_like(band_gap_target, dtype=torch.long)
        regime_target = regime_target + ((band_gap_target > metal_epsilon) & (band_gap_target <= small_gap_threshold)).long()
        regime_target = regime_target + (band_gap_target > small_gap_threshold).long() * 2
        regime_ce = F.cross_entropy(outputs["regime_logits"], regime_target)
        loss = lambda_regime * regime_ce
        components = {
            "regime_ce": round(regime_ce.item(), 6),
            "regime_acc": round((outputs["regime_logits"].argmax(dim=-1) == regime_target).float().mean().item(), 6),
        }
        lambda_regime_expert = float(training_config.get("lambda_regime_expert", 0.0))
        if lambda_regime_expert > 0.0 and "regime_band_gaps" in outputs:
            batch_index = torch.arange(regime_target.size(0), device=regime_target.device)
            selected_predictions = outputs["regime_band_gaps"][batch_index, regime_target]
            regime_expert = F.mse_loss(selected_predictions.view(-1), band_gap_target.view(-1))
            loss = loss + lambda_regime_expert * regime_expert
            components["regime_expert"] = round(regime_expert.item(), 6)
        return loss, components

    size_loss, size_components = size_supervision_loss()
    gain_loss, gain_components = gain_supervision_loss()
    complexity_loss, complexity_components = complexity_supervision_loss()
    orth_loss_value, orth_components = orthogonality_loss()

    if not physics_variant:
        target = labels.view(-1)
        prediction = outputs["band_gap"].view(-1)
        final_loss_name = str(training_config.get("final_gap_loss", "mse")).lower()
        if final_loss_name in {"huber", "smooth_l1"}:
            main = F.smooth_l1_loss(
                prediction,
                target,
                beta=float(training_config.get("final_gap_huber_beta", 0.5)),
            )
        elif final_loss_name == "mae":
            main = F.l1_loss(prediction, target)
        else:
            main = F.mse_loss(prediction, target)
        regime_loss, regime_components = regime_partition_loss(target)
        metal_loss = prediction.sum() * 0.0
        metal_components: dict[str, float] = {}
        if "metal_logit" in outputs and "positive_log_gap" in outputs:
            metal_epsilon = float(training_config.get("metal_epsilon", 1e-8))
            metal_target = (target <= metal_epsilon).to(dtype=prediction.dtype)
            metal_bce_target = metal_target
            label_smoothing = float(training_config.get("metal_label_smoothing", 0.0))
            if label_smoothing > 0.0:
                metal_bce_target = metal_bce_target * (1.0 - label_smoothing) + 0.5 * label_smoothing

            pos_weight_setting = training_config.get("metal_pos_weight")
            pos_weight = None
            if pos_weight_setting is not None:
                if str(pos_weight_setting).lower() == "auto":
                    positive = metal_target.sum().clamp_min(1.0)
                    negative = (1.0 - metal_target).sum().clamp_min(1.0)
                    pos_weight = (negative / positive).clamp(0.25, 4.0).to(device=prediction.device)
                else:
                    pos_weight = prediction.new_tensor(float(pos_weight_setting))
            metal_bce = F.binary_cross_entropy_with_logits(
                outputs["metal_logit"].view(-1),
                metal_bce_target,
                pos_weight=pos_weight,
            )

            nonmetal_mask = target > metal_epsilon
            if bool(nonmetal_mask.any().item()):
                log_gap = F.smooth_l1_loss(
                    outputs["positive_log_gap"].view(-1)[nonmetal_mask],
                    torch.log1p(target[nonmetal_mask].clamp_min(0.0)),
                    beta=float(training_config.get("log_gap_huber_beta", 0.2)),
                )
            else:
                log_gap = prediction.sum() * 0.0

            metal_loss = (
                float(training_config.get("lambda_metal_cls", 0.2)) * metal_bce
                + float(training_config.get("lambda_nonmetal_log_gap", 0.6)) * log_gap
            )
            metal_components = {
                "metal_bce": round(metal_bce.item(), 6),
                "log_gap": round(log_gap.item(), 6),
                "metal_target_mean": round(metal_target.mean().item(), 6),
            }
            if "metal_probability" in outputs:
                metal_components["metal_probability_mean"] = round(
                    outputs["metal_probability"].detach().float().mean().item(),
                    6,
                )
            if "positive_gap" in outputs:
                metal_components["positive_gap_mean"] = round(
                    outputs["positive_gap"].detach().float().mean().item(),
                    6,
                )

        loss = (
            float(training_config.get("lambda_final_gap", 1.0)) * main
            + metal_loss
            + size_loss
            + gain_loss
            + complexity_loss
            + orth_loss_value
            + regime_loss
        )
        components = {"main": round(main.item(), 6)}
        components.update(metal_components)
        components.update(size_components)
        components.update(gain_components)
        components.update(complexity_components)
        components.update(orth_components)
        components.update(regime_components)
        return loss, components, target

    labels = labels.view(-1, 3)
    band_gap_target = labels[:, 0]
    cbm_target = labels[:, 1]
    vbm_target = labels[:, 2]
    valid_edge_epsilon = float(training_config.get("valid_edge_epsilon", 1e-8))
    valid_edge_mask = (cbm_target.abs() > valid_edge_epsilon) | (vbm_target.abs() > valid_edge_epsilon)
    main = F.mse_loss(outputs["band_gap"].view(-1), band_gap_target)
    aux = masked_mse_loss(outputs["cbm"].view(-1), cbm_target, valid_edge_mask) + masked_mse_loss(
        outputs["vbm"].view(-1), vbm_target, valid_edge_mask
    )
    phys = masked_mse_loss(
        outputs["band_gap"].view(-1),
        outputs["cbm"].view(-1) - outputs["vbm"].view(-1),
        valid_edge_mask,
    )
    nonneg = torch.relu(-outputs["band_gap"].view(-1)).square().mean()
    loss = (
        main
        + float(training_config.get("lambda_aux", 0.3)) * aux
        + float(training_config.get("lambda_phys", 0.5)) * phys
        + float(training_config.get("lambda_nonneg", 0.1)) * nonneg
        + size_loss
        + gain_loss
        + complexity_loss
        + orth_loss_value
    )
    components = {
        "main": round(main.item(), 6),
        "aux": round(aux.item(), 6),
        "phys": round(phys.item(), 6),
        "nonneg": round(nonneg.item(), 6),
        "edge_valid_ratio": round(valid_edge_mask.float().mean().item(), 6),
    }
    components.update(size_components)
    components.update(gain_components)
    components.update(complexity_components)
    components.update(orth_components)
    return loss, components, band_gap_target


def regression_summary(targets: list[float], predictions: list[float]) -> dict[str, float]:
    from crystal_gnn.utils import regression_metrics

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
    return regression_metrics(target_tensor, prediction_tensor)


def run_epoch(
    model: torch.nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    training_config: dict[str, Any],
    physics_variant: bool,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scheduler_name: str = "none",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, float], list[dict[str, float]], list[float], list[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    amp_enabled = device.type == "cuda" and bool(training_config.get("mixed_precision", False))
    grad_clip = float(training_config.get("grad_clip_norm", 0.0))
    losses: list[float] = []
    components_log: list[dict[str, float]] = []
    targets: list[float] = []
    predictions: list[float] = []
    metal_probabilities: list[float] = []
    positive_gaps: list[float] = []
    gate_logs: dict[str, list[float]] = {
        "slot2_enable_mean": [],
        "global_fusion_strength_mean": [],
    }

    for batch in loader:
        graph, line_graph, lattice, labels = move_batch(batch, device)
        labels = labels.float()
        if train_mode:
            optimizer.zero_grad(set_to_none=True)
        autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        with autocast_context:
            outputs = model([graph, line_graph, lattice])
            loss, components, band_gap_target = compute_loss(
                outputs,
                labels,
                training_config,
                physics_variant,
                graph=graph,
                line_graph=line_graph,
            )

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

        band_gap_prediction = outputs["band_gap"].detach().view(-1).cpu()
        targets.extend(band_gap_target.detach().view(-1).cpu().tolist())
        predictions.extend(band_gap_prediction.tolist())
        if "metal_probability" in outputs:
            metal_probabilities.extend(outputs["metal_probability"].detach().view(-1).float().cpu().tolist())
        if "positive_gap" in outputs:
            positive_gaps.extend(outputs["positive_gap"].detach().view(-1).float().cpu().tolist())
        losses.append(float(loss.detach().cpu().item()))
        components_log.append(components)
        if "slot2_enable" in outputs:
            gate_logs["slot2_enable_mean"].append(float(outputs["slot2_enable"].detach().float().mean().cpu().item()))
        if "global_fusion_strength" in outputs:
            gate_logs["global_fusion_strength_mean"].append(
                float(outputs["global_fusion_strength"].detach().float().mean().cpu().item())
            )

    metrics = regression_summary(targets, predictions)
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 6)
    if metal_probabilities:
        target_tensor = torch.tensor(targets, dtype=torch.float32)
        probability_tensor = torch.tensor(metal_probabilities, dtype=torch.float32)
        metal_epsilon = float(training_config.get("metal_epsilon", 1e-8))
        metal_probability_threshold = float(training_config.get("metal_probability_threshold", 0.5))
        target_is_metal = target_tensor <= metal_epsilon
        pred_is_metal = probability_tensor >= metal_probability_threshold
        tp = torch.logical_and(target_is_metal, pred_is_metal).sum().item()
        fp = torch.logical_and(~target_is_metal, pred_is_metal).sum().item()
        fn = torch.logical_and(target_is_metal, ~pred_is_metal).sum().item()
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        metrics["metal_head_precision"] = round(precision, 6)
        metrics["metal_head_recall"] = round(recall, 6)
        metrics["metal_head_f1"] = round(f1, 6)
        metrics["metal_probability_mean"] = round(probability_tensor.mean().item(), 6)
    if positive_gaps:
        metrics["positive_gap_mean"] = round(torch.tensor(positive_gaps, dtype=torch.float32).mean().item(), 6)
    for key, values in gate_logs.items():
        if values:
            metrics[key] = round(sum(values) / len(values), 6)
    return metrics, components_log, targets, predictions


def collect_prediction_rows(
    model: torch.nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    physics_variant: bool,
    split: str,
) -> list[dict[str, Any]]:
    model.eval()
    rows: list[dict[str, Any]] = []
    cursor = 0
    dataset_ids = list(loader.dataset.ids)
    with torch.no_grad():
        for batch in loader:
            graph, line_graph, lattice, labels = move_batch(batch, device)
            outputs = model([graph, line_graph, lattice])
            batch_size = len(outputs["band_gap"].view(-1))
            sample_ids = dataset_ids[cursor : cursor + batch_size]
            cursor += batch_size
            if physics_variant:
                labels = labels.view(-1, 3).cpu()
                for index, sample_id in enumerate(sample_ids):
                    rows.append(
                        {
                            "sample_id": sample_id,
                            "target": float(labels[index, 0].item()),
                            "prediction": float(outputs["band_gap"][index].detach().cpu().item()),
                            "target_cbm": float(labels[index, 1].item()),
                            "prediction_cbm": float(outputs["cbm"][index].detach().cpu().item()),
                            "target_vbm": float(labels[index, 2].item()),
                            "prediction_vbm": float(outputs["vbm"][index].detach().cpu().item()),
                            "split": split,
                        }
                    )
            else:
                labels = labels.view(-1).cpu()
                for index, sample_id in enumerate(sample_ids):
                    row = {
                        "sample_id": sample_id,
                        "target": float(labels[index].item()),
                        "prediction": float(outputs["band_gap"][index].detach().cpu().item()),
                        "split": split,
                    }
                    if "metal_probability" in outputs:
                        row["metal_probability"] = float(outputs["metal_probability"][index].detach().cpu().item())
                    if "positive_gap" in outputs:
                        row["positive_gap"] = float(outputs["positive_gap"][index].detach().cpu().item())
                    if "positive_log_gap" in outputs:
                        row["positive_log_gap"] = float(outputs["positive_log_gap"][index].detach().cpu().item())
                    rows.append(row)
    return rows


def evaluate_test_predictions(
    model: torch.nn.Module,
    loader: GraphDataLoader,
    device: torch.device,
    physics_variant: bool,
) -> list[dict[str, Any]]:
    return collect_prediction_rows(model, loader, device, physics_variant=physics_variant, split="test")


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir, logger = setup_output_dir(output_dir, "alignn_faithful")
    seed_everything(int(config.get("seed", 42)))
    device = device_from_config(config["training"].get("device"))
    physics_variant = config["model"].get("variant", "baseline").lower() in {
        "physconstraint",
        "sizesupervisedphysvirtualnode",
    }

    logger.info("loading faithful ALIGNN dataloaders")
    train_loader, val_loader, test_loader = build_dataloaders(config, physics_variant=physics_variant)
    logger.info(
        "dataloader sizes | train=%s val=%s test=%s",
        len(train_loader.dataset),
        len(val_loader.dataset),
        len(test_loader.dataset),
    )

    model = build_alignn_model(config["model"]).to(device)
    optimizer = build_optimizer(model, config["training"])
    scheduler, scheduler_name = build_scheduler(optimizer, config["training"], train_loader)
    amp_enabled = device.type == "cuda" and bool(config["training"].get("mixed_precision", False))
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled) if device.type == "cuda" else None

    history: list[dict[str, Any]] = []
    best_val_mae = float("inf")
    best_epoch = 0
    final_best_checkpoint = output_dir / "best_model.pt"
    local_checkpoint_root = config["training"].get("local_checkpoint_root") or os.environ.get(
        "FAITHFUL_LOCAL_CHECKPOINT_ROOT"
    )
    if local_checkpoint_root:
        checkpoint_dir = Path(str(local_checkpoint_root)).resolve() / config["experiment_name"]
    elif os.name != "nt" and str(output_dir).startswith("/root/lanyun-fs/"):
        checkpoint_dir = Path("/root/lanyun-tmp/gnn_checkpoints").resolve() / config["experiment_name"]
    else:
        checkpoint_dir = output_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_checkpoint = checkpoint_dir / "best_model.pt"
    import time

    wall_start = time.time()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device=device)

    epochs = int(config["training"]["epochs"])
    early_stopping_patience = int(config["training"].get("early_stopping_patience", 0) or 0)
    epochs_without_improvement = 0

    def set_model_epoch(epoch_value: int) -> None:
        if hasattr(model, "set_training_epoch"):
            model.set_training_epoch(epoch_value, epochs)

    def component_average(log_rows: list[dict[str, float]], key: str) -> float | None:
        values = [row[key] for row in log_rows if key in row]
        if not values:
            return None
        return round(sum(values) / len(values), 6)

    for epoch in range(1, epochs + 1):
        set_model_epoch(epoch)
        train_metrics, train_components, _train_targets, _train_predictions = run_epoch(
            model,
            train_loader,
            device,
            config["training"],
            physics_variant=physics_variant,
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics, val_components, _val_targets, _val_predictions = run_epoch(
                model,
                val_loader,
                device,
                config["training"],
                physics_variant=physics_variant,
            )

        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["mae"])
        elif scheduler_name == "cosine" and scheduler is not None:
            scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        epoch_summary = {
            "epoch": epoch,
            "lr": lr,
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
        }
        if "slot2_enable_mean" in train_metrics and "slot2_enable_mean" in val_metrics:
            epoch_summary["train_slot2_enable_mean"] = train_metrics["slot2_enable_mean"]
            epoch_summary["val_slot2_enable_mean"] = val_metrics["slot2_enable_mean"]
        if "global_fusion_strength_mean" in train_metrics and "global_fusion_strength_mean" in val_metrics:
            epoch_summary["train_global_fusion_strength_mean"] = train_metrics["global_fusion_strength_mean"]
            epoch_summary["val_global_fusion_strength_mean"] = val_metrics["global_fusion_strength_mean"]
        if physics_variant and train_components and val_components:
            for train_key, val_key, out_train, out_val in [
                ("main", "main", "train_phys_main", "val_phys_main"),
                ("aux", "aux", "train_phys_aux", "val_phys_aux"),
                ("phys", "phys", "train_phys_consistency", "val_phys_consistency"),
            ]:
                train_value = component_average(train_components, train_key)
                val_value = component_average(val_components, val_key)
                if train_value is not None and val_value is not None:
                    epoch_summary[out_train] = train_value
                    epoch_summary[out_val] = val_value
        for component_key, train_name, val_name in [
            ("size", "train_size_loss", "val_size_loss"),
            ("size_fusion", "train_size_fusion_loss", "val_size_fusion_loss"),
            ("gain", "train_gain_loss", "val_gain_loss"),
            ("gain_fusion", "train_gain_fusion_loss", "val_gain_fusion_loss"),
            ("complexity", "train_complexity_loss", "val_complexity_loss"),
            ("complexity_fusion", "train_complexity_fusion_loss", "val_complexity_fusion_loss"),
            ("orth", "train_orth_loss", "val_orth_loss"),
            ("slot_collapse", "train_slot_collapse_loss", "val_slot_collapse_loss"),
            ("regime_ce", "train_regime_ce", "val_regime_ce"),
            ("regime_expert", "train_regime_expert", "val_regime_expert"),
            ("regime_acc", "train_regime_acc", "val_regime_acc"),
            ("metal_bce", "train_metal_bce", "val_metal_bce"),
            ("log_gap", "train_log_gap", "val_log_gap"),
            ("metal_target_mean", "train_metal_target_mean", "val_metal_target_mean"),
            ("metal_probability_mean", "train_metal_probability_component_mean", "val_metal_probability_component_mean"),
            ("positive_gap_mean", "train_positive_gap_component_mean", "val_positive_gap_component_mean"),
            ("gain_target_mean", "train_gain_target_mean", "val_gain_target_mean"),
            ("complexity_target_mean", "train_complexity_target_mean", "val_complexity_target_mean"),
            ("slot_cosine_abs_mean", "train_slot_cosine_abs_mean", "val_slot_cosine_abs_mean"),
        ]:
            train_value = component_average(train_components, component_key)
            val_value = component_average(val_components, component_key)
            if train_value is not None and val_value is not None:
                epoch_summary[train_name] = train_value
                epoch_summary[val_name] = val_value
        history.append(epoch_summary)

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
            torch.save({"model_state_dict": model.state_dict(), "config": config, "epoch": epoch}, best_checkpoint)
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
    set_model_epoch(int(checkpoint.get("epoch", best_epoch)))
    model.eval()
    with torch.no_grad():
        test_metrics, _test_components, _targets, _predictions = run_epoch(
            model,
            test_loader,
            device,
            config["training"],
            physics_variant=physics_variant,
        )
    prediction_rows = evaluate_test_predictions(model, test_loader, device, physics_variant=physics_variant)

    if best_checkpoint != final_best_checkpoint:
        try:
            shutil.copy2(best_checkpoint, final_best_checkpoint)
        except OSError as exc:
            logger.warning("failed to copy best checkpoint from %s to %s: %s", best_checkpoint, final_best_checkpoint, exc)

    metrics = {
        "experiment_name": config["experiment_name"],
        "model": config["model"].get("runner_name", f"ALIGNN_{config['model'].get('variant', 'baseline')}"),
        "mae": test_metrics["mae"],
        "rmse": test_metrics["rmse"],
        "r2": test_metrics["r2"],
        "metal_f1": test_metrics["metal_f1"],
        "best_epoch": best_epoch,
        "epochs_ran": epoch,
        "train_seconds": elapsed_seconds(wall_start),
        "peak_gpu_gb": peak_gpu_gb(device),
    }
    if config["data"].get("num_atoms_min") is not None or config["data"].get("num_atoms_max") is not None:
        metrics["num_atoms_filter"] = {
            "min": config["data"].get("num_atoms_min"),
            "max": config["data"].get("num_atoms_max"),
        }
    if "slot2_enable_mean" in test_metrics:
        metrics["slot2_enable_mean"] = test_metrics["slot2_enable_mean"]
    if "global_fusion_strength_mean" in test_metrics:
        metrics["global_fusion_strength_mean"] = test_metrics["global_fusion_strength_mean"]
    for metric_key in [
        "metal_head_precision",
        "metal_head_recall",
        "metal_head_f1",
        "metal_probability_mean",
        "positive_gap_mean",
    ]:
        if metric_key in test_metrics:
            metrics[metric_key] = test_metrics[metric_key]
    if physics_variant:
        metrics["physics_constraint"] = {
            "lambda_aux": float(config["training"].get("lambda_aux", 0.3)),
            "lambda_phys": float(config["training"].get("lambda_phys", 0.5)),
            "lambda_nonneg": float(config["training"].get("lambda_nonneg", 0.1)),
            "valid_edge_epsilon": float(config["training"].get("valid_edge_epsilon", 1e-8)),
            "valid_edge_rule": "abs(cbm) > eps or abs(vbm) > eps",
        }

    save_json(output_dir / "history.json", history)
    save_json(output_dir / "metrics.json", metrics)
    save_predictions(output_dir / "predictions.csv", prediction_rows)
    logger.info("completed %s | test mae=%.4f rmse=%.4f", config["experiment_name"], test_metrics["mae"], test_metrics["rmse"])
    return metrics
