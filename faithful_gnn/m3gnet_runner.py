from __future__ import annotations

import contextlib
import gc
import json
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset, Sampler

from faithful_gnn.common import (
    device_from_config,
    elapsed_seconds,
    peak_gpu_gb,
    save_json,
    save_predictions,
    seed_everything,
    setup_output_dir,
)
from faithful_gnn.vendor import configure_vendor_paths

os.environ.setdefault("MATGL_BACKEND", "DGL")
configure_vendor_paths()

from pymatgen.core import Structure  # noqa: E402
from pymatgen.core.periodic_table import Element  # noqa: E402
from dgl.data.utils import load_graphs, save_graphs  # noqa: E402
from matgl.ext.pymatgen import Structure2Graph  # noqa: E402
from matgl.graph._compute_dgl import compute_pair_vector_and_distance, create_line_graph  # noqa: E402
from matgl.graph.data import collate_fn_graph  # noqa: E402
from matgl.models import M3GNet  # noqa: E402


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
    train_loader: Any,
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


def iter_manifest_records(dataset_root: str | Path, split: str | None = None):
    manifest_path = Path(dataset_root) / "manifest.jsonl"
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if split is not None and record.get("split") != split:
                continue
            yield record


def _format_float_token(value: float) -> str:
    return f"{value:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _split_cache_dir(
    cache_root: Path,
    split_name: str,
    shard_size: int,
    include_line_graph: bool,
    cutoff: float,
    threebody_cutoff: float,
) -> Path:
    return cache_root / (
        f"{split_name}_m3gnet_shards_"
        f"cutoff{_format_float_token(cutoff)}_"
        f"threebody{_format_float_token(threebody_cutoff)}_"
        f"line{int(include_line_graph)}_"
        f"size{shard_size}"
    )


def _shard_paths(split_cache_dir: Path, shard_id: int) -> dict[str, Path]:
    prefix = f"shard_{shard_id:05d}"
    return {
        "graphs": split_cache_dir / f"{prefix}_graphs.bin",
        "line_graphs": split_cache_dir / f"{prefix}_line_graphs.bin",
        "lattices": split_cache_dir / f"{prefix}_lattices.pt",
        "state_attr": split_cache_dir / f"{prefix}_state_attr.pt",
        "labels": split_cache_dir / f"{prefix}_labels.pt",
    }


def _shard_complete(paths: dict[str, Path], include_line_graph: bool) -> bool:
    required = [paths["graphs"], paths["lattices"], paths["state_attr"], paths["labels"]]
    if include_line_graph:
        required.append(paths["line_graphs"])
    return all(path.exists() for path in required)


def _cache_ready(
    split_cache_dir: Path,
    include_line_graph: bool,
    split_count: int,
    shard_size: int,
) -> bool:
    metadata_path = split_cache_dir / "metadata.json"
    if not metadata_path.exists():
        return False
    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("num_samples") != split_count or metadata.get("shard_size") != shard_size:
        return False
    if bool(metadata.get("include_line_graph", False)) != include_line_graph:
        return False
    num_shards = int(metadata.get("num_shards", 0))
    return all(_shard_complete(_shard_paths(split_cache_dir, shard_id), include_line_graph) for shard_id in range(num_shards))


def _load_or_compute_manifest_stats(
    dataset_root: Path,
    cache_root: Path,
) -> tuple[dict[str, int], list[str], tuple[str, ...]]:
    stats_path = cache_root / "manifest_stats.json"
    if stats_path.exists():
        with stats_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        split_counts = {key: int(value) for key, value in payload["split_counts"].items()}
        test_ids = list(payload["test_ids"])
        element_types = tuple(payload["element_types"])
        return split_counts, test_ids, element_types

    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    test_ids: list[str] = []
    elements: set[str] = set()
    for record in iter_manifest_records(dataset_root):
        split_name = record.get("split", "train")
        split_counts[split_name] = split_counts.get(split_name, 0) + 1
        if split_name == "test":
            test_ids.append(record["sample_id"])
        structure = Structure.from_dict(record["structure"])
        elements.update(structure.composition.get_el_amt_dict().keys())

    element_types = tuple(sorted(elements, key=lambda element: Element(element).Z))
    save_json(
        stats_path,
        {
            "split_counts": split_counts,
            "test_ids": test_ids,
            "element_types": list(element_types),
        },
    )
    return split_counts, test_ids, element_types


def _build_graph_entry(
    structure_dict: dict[str, Any],
    converter: Structure2Graph,
    include_line_graph: bool,
    threebody_cutoff: float,
) -> tuple[Any, torch.Tensor, Any, torch.Tensor]:
    structure = Structure.from_dict(structure_dict)
    graph, lattice, state_attr = converter.get_graph(structure)
    graph.ndata["pos"] = torch.tensor(structure.cart_coords, dtype=torch.float32)
    graph.edata["pbc_offshift"] = torch.matmul(graph.edata["pbc_offset"], lattice[0])
    bond_vec, bond_dist = compute_pair_vector_and_distance(graph)
    graph.edata["bond_vec"] = bond_vec
    graph.edata["bond_dist"] = bond_dist
    line_graph = None
    if include_line_graph:
        line_graph = create_line_graph(graph, threebody_cutoff, directed=False)
        for name in ["bond_vec", "bond_dist", "pbc_offset"]:
            line_graph.ndata.pop(name)
    graph.ndata.pop("pos")
    graph.edata.pop("pbc_offshift")
    return graph, lattice, line_graph, torch.tensor(state_attr, dtype=torch.float32)


def _write_shard_cache(
    records: list[dict[str, Any]],
    split_cache_dir: Path,
    shard_id: int,
    converter: Structure2Graph,
    include_line_graph: bool,
    threebody_cutoff: float,
    logger: Any | None,
) -> None:
    paths = _shard_paths(split_cache_dir, shard_id)
    if _shard_complete(paths, include_line_graph):
        return

    graphs: list[Any] = []
    line_graphs: list[Any] = []
    lattices: list[torch.Tensor] = []
    state_attr: list[torch.Tensor] = []
    labels: list[float] = []
    for record in records:
        graph, lattice, line_graph, state_value = _build_graph_entry(
            record["structure"],
            converter,
            include_line_graph=include_line_graph,
            threebody_cutoff=threebody_cutoff,
        )
        graphs.append(graph)
        lattices.append(lattice.cpu())
        state_attr.append(state_value.cpu())
        labels.append(float(record["band_gap"]))
        if include_line_graph and line_graph is not None:
            line_graphs.append(line_graph)

    save_graphs(str(paths["graphs"]), graphs)
    torch.save(lattices, paths["lattices"])
    torch.save(torch.stack(state_attr), paths["state_attr"])
    torch.save(labels, paths["labels"])
    if include_line_graph:
        save_graphs(str(paths["line_graphs"]), line_graphs)

    del graphs, line_graphs, lattices, state_attr, labels
    gc.collect()
    if logger is not None:
        logger.info("wrote M3GNet shard %s with %d samples", shard_id, len(records))


def _build_sharded_split_cache(
    dataset_root: Path,
    cache_root: Path,
    split_name: str,
    converter: Structure2Graph,
    include_line_graph: bool,
    cutoff: float,
    threebody_cutoff: float,
    shard_size: int,
    split_count: int,
    logger: Any | None,
) -> Path:
    split_cache_dir = _split_cache_dir(
        cache_root,
        split_name,
        shard_size=shard_size,
        include_line_graph=include_line_graph,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
    )
    split_cache_dir.mkdir(parents=True, exist_ok=True)
    if _cache_ready(split_cache_dir, include_line_graph=include_line_graph, split_count=split_count, shard_size=shard_size):
        if logger is not None:
            logger.info("reusing M3GNet shard cache for %s at %s", split_name, split_cache_dir)
        return split_cache_dir

    if logger is not None:
        logger.info(
            "building M3GNet shard cache for %s | samples=%d | shard_size=%d",
            split_name,
            split_count,
            shard_size,
        )

    shard_lengths: list[int] = []
    shard_records: list[dict[str, Any]] = []
    shard_id = 0
    for record in iter_manifest_records(dataset_root, split=split_name):
        shard_records.append(record)
        if len(shard_records) < shard_size:
            continue
        _write_shard_cache(
            shard_records,
            split_cache_dir,
            shard_id=shard_id,
            converter=converter,
            include_line_graph=include_line_graph,
            threebody_cutoff=threebody_cutoff,
            logger=logger,
        )
        shard_lengths.append(len(shard_records))
        shard_id += 1
        shard_records = []

    if shard_records:
        _write_shard_cache(
            shard_records,
            split_cache_dir,
            shard_id=shard_id,
            converter=converter,
            include_line_graph=include_line_graph,
            threebody_cutoff=threebody_cutoff,
            logger=logger,
        )
        shard_lengths.append(len(shard_records))

    save_json(
        split_cache_dir / "metadata.json",
        {
            "split": split_name,
            "num_samples": split_count,
            "num_shards": len(shard_lengths),
            "shard_size": shard_size,
            "shard_lengths": shard_lengths,
            "include_line_graph": include_line_graph,
            "cutoff": cutoff,
            "threebody_cutoff": threebody_cutoff,
        },
    )
    return split_cache_dir


class CachedM3GNetShardDataset(Dataset):
    def __init__(self, split_cache_dir: Path, include_line_graph: bool) -> None:
        self.split_cache_dir = Path(split_cache_dir)
        self.include_line_graph = include_line_graph
        with (self.split_cache_dir / "metadata.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        self.num_samples = int(metadata["num_samples"])
        self.num_shards = int(metadata["num_shards"])
        self.shard_size = int(metadata["shard_size"])
        self._active_shard_id = -1
        self._graphs: list[Any] = []
        self._line_graphs: list[Any] = []
        self._lattices: list[torch.Tensor] = []
        self._state_attr = torch.empty(0)
        self._labels: list[float] = []

    def _load_shard(self, shard_id: int) -> None:
        if shard_id == self._active_shard_id:
            return
        paths = _shard_paths(self.split_cache_dir, shard_id)
        self._graphs, _ = load_graphs(str(paths["graphs"]))
        self._lattices = torch.load(paths["lattices"], map_location="cpu", weights_only=False)
        self._state_attr = torch.load(paths["state_attr"], map_location="cpu", weights_only=False)
        self._labels = torch.load(paths["labels"], map_location="cpu", weights_only=False)
        if self.include_line_graph:
            self._line_graphs, _ = load_graphs(str(paths["line_graphs"]))
        else:
            self._line_graphs = []
        self._active_shard_id = shard_id

    def __getitem__(self, idx: int):
        shard_id = idx // self.shard_size
        offset = idx % self.shard_size
        self._load_shard(shard_id)
        items = [
            self._graphs[offset],
            self._lattices[offset],
            self._state_attr[offset],
            {"band_gap": torch.tensor(self._labels[offset], dtype=torch.float32)},
        ]
        if self.include_line_graph:
            items.insert(2, self._line_graphs[offset])
        return tuple(items)

    def __len__(self) -> int:
        return self.num_samples


class ShardBatchSampler(Sampler[list[int]]):
    def __init__(self, dataset_size: int, batch_size: int, shard_size: int, shuffle: bool) -> None:
        self.dataset_size = dataset_size
        self.batch_size = batch_size
        self.shard_size = shard_size
        self.shuffle = shuffle

    def __iter__(self):
        shard_ids = list(range(math.ceil(self.dataset_size / self.shard_size)))
        if self.shuffle:
            random.shuffle(shard_ids)
        for shard_id in shard_ids:
            start = shard_id * self.shard_size
            end = min(start + self.shard_size, self.dataset_size)
            indices = list(range(start, end))
            if self.shuffle:
                random.shuffle(indices)
            for offset in range(0, len(indices), self.batch_size):
                yield indices[offset : offset + self.batch_size]

    def __len__(self) -> int:
        return math.ceil(self.dataset_size / self.batch_size)


def build_dataloaders(config: dict[str, Any], logger: Any | None = None) -> tuple[Any, Any, Any, list[str], tuple[str, ...]]:
    data_config = config["data"]
    dataset_root = Path(data_config["dataset_root"])
    cache_root = Path(data_config.get("cache_dir", dataset_root / "m3gnet_cache"))
    cache_root.mkdir(parents=True, exist_ok=True)

    split_counts, test_ids, element_types = _load_or_compute_manifest_stats(dataset_root, cache_root)
    cutoff = float(config["model"].get("cutoff", 5.0))
    threebody_cutoff = float(config["model"].get("threebody_cutoff", 4.0))
    shard_size = int(data_config.get("preprocess_shard_size", data_config.get("shard_size", 2048)))
    converter = Structure2Graph(element_types=element_types, cutoff=cutoff)

    train_cache_dir = _build_sharded_split_cache(
        dataset_root=dataset_root,
        cache_root=cache_root,
        split_name="train",
        converter=converter,
        include_line_graph=True,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        shard_size=shard_size,
        split_count=int(split_counts.get("train", 0)),
        logger=logger,
    )
    val_cache_dir = _build_sharded_split_cache(
        dataset_root=dataset_root,
        cache_root=cache_root,
        split_name="val",
        converter=converter,
        include_line_graph=True,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        shard_size=shard_size,
        split_count=int(split_counts.get("val", 0)),
        logger=logger,
    )
    test_cache_dir = _build_sharded_split_cache(
        dataset_root=dataset_root,
        cache_root=cache_root,
        split_name="test",
        converter=converter,
        include_line_graph=True,
        cutoff=cutoff,
        threebody_cutoff=threebody_cutoff,
        shard_size=shard_size,
        split_count=int(split_counts.get("test", 0)),
        logger=logger,
    )

    train_data = CachedM3GNetShardDataset(train_cache_dir, include_line_graph=True)
    val_data = CachedM3GNetShardDataset(val_cache_dir, include_line_graph=True)
    test_data = CachedM3GNetShardDataset(test_cache_dir, include_line_graph=True)

    collate = partial(collate_fn_graph, include_line_graph=True)
    if int(config["training"].get("num_workers", 0)) != 0 and logger is not None:
        logger.info("forcing num_workers=0 for sharded M3GNet dataset to avoid per-worker shard-cache duplication")
    num_workers = 0
    pin_memory = bool(config["training"].get("pin_memory", False))
    batch_size = int(config["training"]["batch_size"])
    test_batch_size = int(config["training"].get("test_batch_size", batch_size))

    train_loader = DataLoader(
        train_data,
        batch_sampler=ShardBatchSampler(len(train_data), batch_size, shard_size=shard_size, shuffle=True),
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_data,
        batch_sampler=ShardBatchSampler(len(val_data), batch_size, shard_size=shard_size, shuffle=False),
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_data,
        batch_sampler=ShardBatchSampler(len(test_data), test_batch_size, shard_size=shard_size, shuffle=False),
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader, test_loader, test_ids, element_types


def prepare_lat(lat: torch.Tensor, batch_size: int) -> torch.Tensor:
    if lat.dim() == 2:
        return lat.unsqueeze(0)
    if lat.dim() == 3 and lat.size(0) == batch_size:
        return lat
    return lat.view(batch_size, 3, 3)


def forward_batch(model: M3GNet, graph, lat: torch.Tensor, line_graph, state_attr: torch.Tensor) -> torch.Tensor:
    graph = graph.local_var()
    batch_lattice = prepare_lat(lat, graph.batch_size)
    graph.edata["lattice"] = torch.repeat_interleave(batch_lattice, graph.batch_num_edges(), dim=0)
    graph.edata["pbc_offshift"] = (graph.edata["pbc_offset"].unsqueeze(dim=-1) * graph.edata["lattice"]).sum(dim=1)
    graph.ndata["pos"] = (
        graph.ndata["frac_coords"].unsqueeze(dim=-1) * torch.repeat_interleave(batch_lattice, graph.batch_num_nodes(), dim=0)
    ).sum(dim=1)
    return model(g=graph, l_g=line_graph, state_attr=state_attr).view(-1)


def regression_summary(targets: list[float], predictions: list[float]) -> dict[str, float]:
    from crystal_gnn.utils import regression_metrics

    target_tensor = torch.tensor(targets, dtype=torch.float32)
    prediction_tensor = torch.tensor(predictions, dtype=torch.float32)
    return regression_metrics(target_tensor, prediction_tensor)


def run_epoch(
    model: M3GNet,
    loader: Any,
    device: torch.device,
    training_config: dict[str, Any],
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scheduler_name: str = "none",
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> tuple[dict[str, float], list[float], list[float]]:
    train_mode = optimizer is not None
    model.train(train_mode)
    amp_enabled = device.type == "cuda" and bool(training_config.get("mixed_precision", False))
    grad_clip = float(training_config.get("grad_clip_norm", 0.0))
    losses: list[float] = []
    targets: list[float] = []
    predictions: list[float] = []

    for graph, lat, line_graph, state_attr, labels in loader:
        graph = graph.to(device)
        lat = lat.to(device)
        line_graph = line_graph.to(device)
        state_attr = state_attr.to(device)
        labels = labels.view(-1).to(device)
        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        autocast_context = torch.cuda.amp.autocast(enabled=amp_enabled) if amp_enabled else contextlib.nullcontext()
        with autocast_context:
            output = forward_batch(model, graph, lat, line_graph, state_attr)
            loss = F.mse_loss(output, labels)

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

        losses.append(float(loss.detach().cpu().item()))
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(output.detach().cpu().tolist())

    metrics = regression_summary(targets, predictions)
    metrics["loss"] = round(sum(losses) / max(1, len(losses)), 6)
    return metrics, targets, predictions


def evaluate_predictions(model: M3GNet, loader: Any, device: torch.device, test_ids: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cursor = 0
    model.eval()
    with torch.no_grad():
        for graph, lat, line_graph, state_attr, labels in loader:
            graph = graph.to(device)
            lat = lat.to(device)
            line_graph = line_graph.to(device)
            state_attr = state_attr.to(device)
            output = forward_batch(model, graph, lat, line_graph, state_attr).detach().cpu()
            labels = labels.view(-1).cpu()
            batch_size = len(labels)
            sample_ids = test_ids[cursor : cursor + batch_size]
            cursor += batch_size
            for index, sample_id in enumerate(sample_ids):
                rows.append(
                    {
                        "sample_id": sample_id,
                        "target": float(labels[index].item()),
                        "prediction": float(output[index].item()),
                        "split": "test",
                    }
                )
    return rows


def run_experiment(config: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    output_dir, logger = setup_output_dir(output_dir, "m3gnet_faithful")
    seed_everything(int(config.get("seed", 42)))
    device = device_from_config(config["training"].get("device"))
    train_loader, val_loader, test_loader, test_ids, element_types = build_dataloaders(config, logger=logger)

    model = M3GNet(
        element_types=element_types,
        dim_node_embedding=int(config["model"].get("dim_node_embedding", 64)),
        dim_edge_embedding=int(config["model"].get("dim_edge_embedding", 64)),
        max_n=int(config["model"].get("max_n", 3)),
        max_l=int(config["model"].get("max_l", 3)),
        nblocks=int(config["model"].get("nblocks", 3)),
        cutoff=float(config["model"].get("cutoff", 5.0)),
        threebody_cutoff=float(config["model"].get("threebody_cutoff", 4.0)),
        units=int(config["model"].get("units", 128)),
        is_intensive=bool(config["model"].get("is_intensive", True)),
        readout_type=config["model"].get("readout_type", "weighted_atom"),
        task_type="regression",
        ntargets=1,
        dropout=config["model"].get("dropout"),
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
        train_metrics, _targets, _predictions = run_epoch(
            model,
            train_loader,
            device,
            config["training"],
            optimizer=optimizer,
            scheduler=scheduler,
            scheduler_name=scheduler_name,
            scaler=scaler,
        )
        with torch.no_grad():
            val_metrics, _targets, _predictions = run_epoch(model, val_loader, device, config["training"])

        if scheduler_name == "plateau" and scheduler is not None:
            scheduler.step(val_metrics["mae"])
        elif scheduler_name == "cosine" and scheduler is not None:
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
        test_metrics, _targets, _predictions = run_epoch(model, test_loader, device, config["training"])
    prediction_rows = evaluate_predictions(model, test_loader, device, test_ids)

    metrics = {
        "experiment_name": config["experiment_name"],
        "model": config["model"].get("runner_name", "M3GNet"),
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
