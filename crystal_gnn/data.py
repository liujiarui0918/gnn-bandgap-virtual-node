from __future__ import annotations

import gzip
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Callable, Iterator

import numpy as np
import torch
from pymatgen.core import Structure
from tqdm import tqdm


def _is_gzip(path: Path) -> bool:
    with path.open("rb") as handle:
        return handle.read(2) == b"\x1f\x8b"


def iter_json_records(
    path: str | Path,
    on_error: Callable[[int, json.JSONDecodeError, str], None] | None = None,
) -> Iterator[dict]:
    path = Path(path)
    opener = gzip.open if _is_gzip(path) else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                if on_error is not None:
                    on_error(line_number, exc, line[:200])
                continue


def _extract_targets(record: dict) -> dict | None:
    crystal_band = record.get("crystal_band") or []
    if not crystal_band:
        return None
    band_info = crystal_band[0]
    band_gap = band_info.get("band_gap")
    if band_gap is None or band_gap < 0 or band_gap > 15:
        return None
    return {
        "band_gap": float(band_gap),
        "cbm": float(band_info.get("cbm", 0.0) or 0.0),
        "vbm": float(band_info.get("vbm", 0.0) or 0.0),
        "is_metal": 1.0 if abs(float(band_gap)) < 1e-8 else 0.0,
    }


def _build_line_graph(edge_index: np.ndarray, edge_vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if edge_index.shape[1] == 0:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    by_source: dict[int, list[int]] = defaultdict(list)
    for edge_id, source in enumerate(edge_index[0].tolist()):
        by_source[int(source)].append(edge_id)

    line_sources: list[int] = []
    line_targets: list[int] = []
    line_angles: list[float] = []

    for edge_ids in by_source.values():
        if len(edge_ids) < 2:
            continue
        for source_edge in edge_ids:
            vec_a = edge_vectors[source_edge]
            norm_a = np.linalg.norm(vec_a)
            if norm_a < 1e-8:
                continue
            for target_edge in edge_ids:
                if source_edge == target_edge:
                    continue
                vec_b = edge_vectors[target_edge]
                norm_b = np.linalg.norm(vec_b)
                if norm_b < 1e-8:
                    continue
                cosine = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
                cosine = max(-1.0, min(1.0, cosine))
                line_sources.append(source_edge)
                line_targets.append(target_edge)
                line_angles.append(math.acos(cosine))

    if not line_sources:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)

    return (
        np.asarray([line_sources, line_targets], dtype=np.int64),
        np.asarray(line_angles, dtype=np.float32),
    )


def build_graph(record: dict, cutoff: float, max_neighbors: int) -> dict | None:
    targets = _extract_targets(record)
    if targets is None:
        return None

    structure = Structure.from_dict(record["structure"])
    center_indices, point_indices, offsets, distances = structure.get_neighbor_list(cutoff)

    lattice_matrix = np.asarray(structure.lattice.matrix, dtype=np.float32)
    frac_coords = np.asarray(structure.frac_coords, dtype=np.float32)
    cart_coords = np.asarray(structure.cart_coords, dtype=np.float32)
    atomic_numbers = np.asarray([site.specie.Z for site in structure], dtype=np.int64)
    num_nodes = int(len(structure))

    neighbors_by_source: dict[int, list[tuple[float, int, np.ndarray]]] = defaultdict(list)
    for source, target, offset, distance in zip(center_indices, point_indices, offsets, distances):
        if distance < 1e-8:
            continue
        neighbors_by_source[int(source)].append(
            (
                float(distance),
                int(target),
                np.asarray(offset, dtype=np.float32),
            )
        )

    edge_sources: list[int] = []
    edge_targets: list[int] = []
    edge_vectors: list[np.ndarray] = []
    edge_distances: list[float] = []

    for source in range(num_nodes):
        ranked = sorted(neighbors_by_source[source], key=lambda item: item[0])[:max_neighbors]
        for distance, target, offset in ranked:
            translated_target = cart_coords[target] + offset @ lattice_matrix
            edge_sources.append(source)
            edge_targets.append(target)
            edge_vectors.append(translated_target - cart_coords[source])
            edge_distances.append(distance)

    if not edge_sources:
        return None

    edge_index = np.asarray([edge_sources, edge_targets], dtype=np.int64)
    edge_vectors_np = np.asarray(edge_vectors, dtype=np.float32)
    line_edge_index, line_angles = _build_line_graph(edge_index, edge_vectors_np)

    graph_features = np.asarray(
        [
            structure.lattice.a,
            structure.lattice.b,
            structure.lattice.c,
            structure.lattice.alpha,
            structure.lattice.beta,
            structure.lattice.gamma,
            float(record.get("density", structure.density)),
            float(record.get("num_atoms", num_nodes)),
        ],
        dtype=np.float32,
    )

    return {
        "atomic_numbers": torch.tensor(atomic_numbers, dtype=torch.long),
        "frac_coords": torch.tensor(frac_coords, dtype=torch.float32),
        "edge_index": torch.tensor(edge_index, dtype=torch.long),
        "edge_vectors": torch.tensor(edge_vectors_np, dtype=torch.float32),
        "edge_distances": torch.tensor(np.asarray(edge_distances, dtype=np.float32)),
        "line_edge_index": torch.tensor(line_edge_index, dtype=torch.long),
        "line_angles": torch.tensor(line_angles, dtype=torch.float32),
        "graph_features": torch.tensor(graph_features, dtype=torch.float32),
        "band_gap": torch.tensor(targets["band_gap"], dtype=torch.float32),
        "cbm": torch.tensor(targets["cbm"], dtype=torch.float32),
        "vbm": torch.tensor(targets["vbm"], dtype=torch.float32),
        "is_metal": torch.tensor(targets["is_metal"], dtype=torch.float32),
        "num_nodes": num_nodes,
        "formula": record.get("formula", ""),
        "source_id": record.get("source_id", record.get("hc_id", "")),
    }


def preprocess_records(
    raw_path: str | Path,
    cutoff: float,
    max_neighbors: int,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    limit: int | None = None,
) -> tuple[dict[str, list[dict]], dict]:
    accepted_records: list[dict] = []
    seen_records = 0
    skipped_missing = 0
    skipped_graph = 0
    skipped_bad_json = 0
    skipped_record_errors = 0
    bad_json_examples: list[dict[str, str | int]] = []
    record_error_examples: list[dict[str, str | int]] = []

    def on_json_error(line_number: int, exc: json.JSONDecodeError, sample: str) -> None:
        nonlocal skipped_bad_json
        skipped_bad_json += 1
        if len(bad_json_examples) < 5:
            bad_json_examples.append(
                {
                    "line_number": line_number,
                    "message": exc.msg,
                    "sample": sample,
                }
            )

    iterator = iter_json_records(raw_path, on_error=on_json_error)
    progress = tqdm(iterator, desc="preprocess", unit="record")
    for record in progress:
        seen_records += 1
        try:
            graph = build_graph(record, cutoff=cutoff, max_neighbors=max_neighbors)
        except Exception as exc:
            skipped_record_errors += 1
            if len(record_error_examples) < 5:
                record_error_examples.append(
                    {
                        "source_id": str(record.get("source_id", record.get("hc_id", ""))),
                        "formula": str(record.get("formula", "")),
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            continue
        if graph is None:
            crystal_band = record.get("crystal_band") or []
            if not crystal_band or crystal_band[0].get("band_gap") is None:
                skipped_missing += 1
            else:
                skipped_graph += 1
            continue
        accepted_records.append(graph)
        if limit is not None and len(accepted_records) >= limit:
            break

    rng = random.Random(seed)
    rng.shuffle(accepted_records)

    total = len(accepted_records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    splits = {
        "train": accepted_records[:train_end],
        "val": accepted_records[train_end:val_end],
        "test": accepted_records[val_end:],
    }
    metadata = {
        "raw_path": str(raw_path),
        "seen_records": seen_records,
        "accepted_records": total,
        "skipped_missing_or_invalid": skipped_missing,
        "skipped_graph_build": skipped_graph,
        "skipped_malformed_json": skipped_bad_json,
        "skipped_record_exceptions": skipped_record_errors,
        "split_sizes": {name: len(records) for name, records in splits.items()},
        "cutoff": cutoff,
        "max_neighbors": max_neighbors,
        "seed": seed,
    }
    if bad_json_examples:
        metadata["bad_json_examples"] = bad_json_examples
    if record_error_examples:
        metadata["record_error_examples"] = record_error_examples
    return splits, metadata


def save_processed_splits(output_dir: str | Path, splits: dict[str, list[dict]], shard_size: int = 2048) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_name, records in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        for existing_shard in split_dir.glob("shard_*.pt"):
            existing_shard.unlink()
        for shard_id in range(0, len(records), shard_size):
            shard = records[shard_id : shard_id + shard_size]
            shard_path = split_dir / f"shard_{shard_id // shard_size:05d}.pt"
            torch.save(shard, shard_path)


class CrystalShardDataset(torch.utils.data.Dataset):
    def __init__(self, processed_dir: str | Path, split: str) -> None:
        split_dir = Path(processed_dir) / split
        shard_paths = sorted(split_dir.glob("shard_*.pt"))
        if not shard_paths:
            raise FileNotFoundError(f"no shards found under {split_dir}")
        self.records: list[dict] = []
        for shard_path in shard_paths:
            self.records.extend(torch.load(shard_path, map_location="cpu", weights_only=False))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        return self.records[index]


def graph_collate(records: list[dict]) -> dict[str, torch.Tensor]:
    atomic_numbers = []
    frac_coords = []
    edge_index = []
    edge_vectors = []
    edge_distances = []
    line_edge_index = []
    line_angles = []
    graph_features = []
    band_gap = []
    cbm = []
    vbm = []
    is_metal = []
    node_batch = []
    edge_batch = []

    node_offset = 0
    edge_offset = 0
    for graph_id, record in enumerate(records):
        num_nodes = int(record["atomic_numbers"].size(0))
        num_edges = int(record["edge_index"].size(1))

        atomic_numbers.append(record["atomic_numbers"])
        frac_coords.append(record["frac_coords"])
        edge_index.append(record["edge_index"] + node_offset)
        edge_vectors.append(record["edge_vectors"])
        edge_distances.append(record["edge_distances"])
        line_edge_index.append(record["line_edge_index"] + edge_offset)
        line_angles.append(record["line_angles"])
        graph_features.append(record["graph_features"])
        band_gap.append(record["band_gap"])
        cbm.append(record["cbm"])
        vbm.append(record["vbm"])
        is_metal.append(record["is_metal"])
        node_batch.append(torch.full((num_nodes,), graph_id, dtype=torch.long))
        edge_batch.append(torch.full((num_edges,), graph_id, dtype=torch.long))

        node_offset += num_nodes
        edge_offset += num_edges

    def _cat(items: list[torch.Tensor], dim: int) -> torch.Tensor:
        if not items:
            return torch.empty(0)
        return torch.cat(items, dim=dim)

    return {
        "atomic_numbers": _cat(atomic_numbers, 0),
        "frac_coords": _cat(frac_coords, 0),
        "edge_index": _cat(edge_index, 1),
        "edge_vectors": _cat(edge_vectors, 0),
        "edge_distances": _cat(edge_distances, 0),
        "line_edge_index": _cat(line_edge_index, 1),
        "line_angles": _cat(line_angles, 0),
        "graph_features": torch.stack(graph_features),
        "band_gap": torch.stack(band_gap),
        "cbm": torch.stack(cbm),
        "vbm": torch.stack(vbm),
        "is_metal": torch.stack(is_metal),
        "node_batch": _cat(node_batch, 0),
        "edge_batch": _cat(edge_batch, 0),
        "num_graphs": len(records),
    }
