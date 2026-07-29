from __future__ import annotations

import argparse
import csv
import json
import sys
import types
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

# DGL 2.x imports GraphBolt at module import time. The remote login node keeps
# CUDA-enabled DGL packages but no driver library, so GraphBolt import fails
# before we can run CPU diagnostics. This dummy module is enough because this
# script only uses ordinary DGL graphs and GraphDataLoader.
sys.modules.setdefault("dgl.graphbolt", types.ModuleType("dgl.graphbolt"))

# Allow `python scripts/analyze_oversmoothing_hooks.py` from anywhere: the
# faithful_gnn package lives in the repo root, one level above this file.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from faithful_gnn.alignn_runner import build_dataloaders, move_batch
from faithful_gnn.alignn_variants import build_alignn_model
from faithful_gnn.common import load_manifest


LAYER_CLASS_NAMES = {"ALIGNNConv", "EdgeGatedGraphConv"}


@dataclass(frozen=True)
class ExperimentSpec:
    key: str
    checkpoint: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run hook-based over-smoothing diagnostics for ALIGNN checkpoints. "
            "The script computes layer-wise mean pairwise cosine similarity, node-feature "
            "variance, and graph Dirichlet energy on held-out test samples."
        )
    )
    parser.add_argument("--results-root", required=True, help="Directory containing T-series experiment folders.")
    parser.add_argument("--output-dir", required=True, help="Directory for CSV/PNG/Markdown outputs.")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=["T2", "T4", "T7", "T10"],
        help="Experiment keys to evaluate. Defaults to T2 T4 T7 T10.",
    )
    parser.add_argument("--max-samples", type=int, default=2048, help="Maximum test samples per model.")
    parser.add_argument("--mini-train-size", type=int, default=8, help="Train records kept in the temporary mini dataset.")
    parser.add_argument("--mini-val-size", type=int, default=8, help="Validation records kept in the temporary mini dataset.")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size for diagnostics.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Inference device.")
    return parser.parse_args()


def experiment_dirs(results_root: Path) -> dict[str, str]:
    return {
        "T2": "T2_ALIGNN_baseline",
        "T4": "T4_ALIGNN_plain_single_virtualnode",
        "T7": "T7_ALIGNN_huber_virtualnode",
        "T10": "T10_ALIGNN_decay_huber_virtualnode",
    }


def resolve_experiments(results_root: Path, keys: list[str]) -> list[ExperimentSpec]:
    mapping = experiment_dirs(results_root)
    specs: list[ExperimentSpec] = []
    for raw_key in keys:
        key = raw_key.upper()
        if key not in mapping:
            raise ValueError(f"unsupported experiment key for this diagnostic: {raw_key}")
        checkpoint = results_root / mapping[key] / "best_model.pt"
        if not checkpoint.exists():
            raise FileNotFoundError(f"checkpoint not found for {key}: {checkpoint}")
        specs.append(ExperimentSpec(key=key, checkpoint=checkpoint))
    return specs


def choose_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def num_atoms_bucket(num_atoms: int) -> str:
    if num_atoms <= 10:
        return "<=10"
    if num_atoms <= 20:
        return "11-20"
    if num_atoms <= 40:
        return "21-40"
    if num_atoms <= 80:
        return "41-80"
    if num_atoms <= 160:
        return "81-160"
    return ">160"


def num_atoms_from_record(record: dict[str, Any]) -> int:
    if "num_atoms" in record:
        return int(record["num_atoms"])
    structure = record.get("structure", {})
    if isinstance(structure, dict):
        sites = structure.get("sites")
        if isinstance(sites, list):
            return len(sites)
    return 0


def stratified_test_records(test_records: list[dict[str, Any]], max_samples: int) -> list[dict[str, Any]]:
    bucket_order = ["<=10", "11-20", "21-40", "41-80", "81-160", ">160"]
    by_bucket: dict[str, list[dict[str, Any]]] = {bucket: [] for bucket in bucket_order}
    for record in test_records:
        by_bucket[num_atoms_bucket(num_atoms_from_record(record))].append(record)

    selected: list[dict[str, Any]] = []
    cursors = {bucket: 0 for bucket in bucket_order}
    while len(selected) < max_samples:
        progressed = False
        for bucket in bucket_order:
            cursor = cursors[bucket]
            records = by_bucket[bucket]
            if cursor >= len(records):
                continue
            selected.append(records[cursor])
            cursors[bucket] += 1
            progressed = True
            if len(selected) >= max_samples:
                break
        if not progressed:
            break
    return selected


def write_mini_dataset_root(
    *,
    original_root: Path,
    output_dir: Path,
    max_test_samples: int,
    train_size: int,
    val_size: int,
) -> Path:
    mini_root = output_dir / f"_mini_dataset_test{max_test_samples}_train{train_size}_val{val_size}"
    marker = mini_root / "metadata.json"
    if marker.exists() and (mini_root / "manifest.jsonl").exists():
        return mini_root

    _records, split_map = load_manifest(original_root)
    train_records = [dict(record, split="train") for record in split_map.get("train", [])[:train_size]]
    val_records = [dict(record, split="val") for record in split_map.get("val", [])[:val_size]]
    test_records = [
        dict(record, split="test")
        for record in stratified_test_records(split_map.get("test", []), max_test_samples)
    ]
    manifest_records = [*train_records, *val_records, *test_records]
    if not train_records or not val_records or not test_records:
        raise ValueError(
            f"cannot create mini dataset from {original_root}: "
            f"train={len(train_records)} val={len(val_records)} test={len(test_records)}"
        )

    mini_root.mkdir(parents=True, exist_ok=True)
    with (mini_root / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for record in manifest_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    (mini_root / "splits.json").write_text(
        json.dumps(
            {
                "train": [record["sample_id"] for record in train_records],
                "val": [record["sample_id"] for record in val_records],
                "test": [record["sample_id"] for record in test_records],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    test_bucket_counts: dict[str, int] = defaultdict(int)
    for record in test_records:
        test_bucket_counts[num_atoms_bucket(num_atoms_from_record(record))] += 1
    marker.write_text(
        json.dumps(
            {
                "source_dataset_root": str(original_root),
                "split_sizes": {"train": len(train_records), "val": len(val_records), "test": len(test_records)},
                "test_bucket_counts": dict(test_bucket_counts),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return mini_root


def layer_stage(layer_name: str, class_name: str) -> str:
    if class_name == "ALIGNNConv" or "alignn" in layer_name.lower():
        return "alignn"
    if class_name == "EdgeGatedGraphConv" or "gcn" in layer_name.lower():
        return "gcn"
    return "unknown"


def tensor_from_hook_output(output: Any) -> torch.Tensor | None:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    return None


def register_layer_hooks(model: torch.nn.Module, captured: list[dict[str, Any]]) -> list[Any]:
    handles: list[Any] = []
    layer_index = 0
    for name, module in model.named_modules():
        class_name = module.__class__.__name__
        if class_name not in LAYER_CLASS_NAMES:
            continue
        current_index = layer_index
        current_name = name
        current_class_name = class_name
        current_stage = layer_stage(current_name, current_class_name)
        layer_index += 1

        def hook(_module: torch.nn.Module, _inputs: tuple[Any, ...], output: Any, *, idx=current_index, lname=current_name, cname=current_class_name, stage=current_stage) -> None:
            tensor = tensor_from_hook_output(output)
            if tensor is None:
                return
            captured.append(
                {
                    "layer_index": idx,
                    "layer_name": lname,
                    "class_name": cname,
                    "stage": stage,
                    "node_features": tensor.detach(),
                }
            )

        handles.append(module.register_forward_hook(hook))
    if not handles:
        raise RuntimeError("no ALIGNNConv/EdgeGatedGraphConv modules found for hook diagnostics")
    return handles


def graph_edge_offsets(batch_num_nodes: list[int]) -> list[int]:
    offsets = [0]
    for count in batch_num_nodes[:-1]:
        offsets.append(offsets[-1] + int(count))
    return offsets


def batch_edge_slices(graph: Any) -> list[tuple[int, int]]:
    edge_counts = [int(x) for x in graph.batch_num_edges().detach().cpu().tolist()]
    slices: list[tuple[int, int]] = []
    cursor = 0
    for count in edge_counts:
        slices.append((cursor, cursor + count))
        cursor += count
    return slices


def mean_pairwise_cosine(x: torch.Tensor) -> float:
    n = int(x.size(0))
    if n <= 1:
        return 1.0
    normalized = torch.nn.functional.normalize(x.float(), dim=-1, eps=1e-8)
    summed = normalized.sum(dim=0)
    total_pair_sum = float((summed * summed).sum().item() - n)
    return total_pair_sum / float(n * (n - 1))


def node_feature_variance(x: torch.Tensor) -> float:
    if int(x.size(0)) <= 1:
        return 0.0
    return float(x.float().var(dim=0, unbiased=False).mean().item())


def graph_dirichlet_energy(x: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> float:
    if src.numel() == 0:
        return 0.0
    diff = x.float()[src] - x.float()[dst]
    return float(diff.square().sum(dim=-1).mean().item())


def per_graph_rows(
    *,
    experiment: str,
    sample_ids: list[str],
    graph: Any,
    captured_layers: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    batch_num_nodes = [int(x) for x in graph.batch_num_nodes().detach().cpu().tolist()]
    node_offsets = graph_edge_offsets(batch_num_nodes)
    edge_slices = batch_edge_slices(graph)
    src_all, dst_all = graph.edges()
    src_all = src_all.detach().cpu()
    dst_all = dst_all.detach().cpu()

    for layer in captured_layers:
        features = layer["node_features"].detach().cpu()
        for graph_index, num_nodes in enumerate(batch_num_nodes):
            node_start = node_offsets[graph_index]
            node_end = node_start + num_nodes
            edge_start, edge_end = edge_slices[graph_index]
            local_src = src_all[edge_start:edge_end] - node_start
            local_dst = dst_all[edge_start:edge_end] - node_start
            node_features = features[node_start:node_end]
            rows.append(
                {
                    "experiment": experiment,
                    "sample_id": sample_ids[graph_index],
                    "num_atoms": num_nodes,
                    "num_atoms_bucket": num_atoms_bucket(num_nodes),
                    "layer_index": int(layer["layer_index"]),
                    "layer_name": layer["layer_name"],
                    "stage": layer["stage"],
                    "class_name": layer["class_name"],
                    "mean_pairwise_cosine": mean_pairwise_cosine(node_features),
                    "node_feature_variance": node_feature_variance(node_features),
                    "dirichlet_energy": graph_dirichlet_energy(node_features, local_src, local_dst),
                }
            )
    return rows


def load_model_and_test_loader(
    spec: ExperimentSpec,
    device: torch.device,
    batch_size: int,
    output_dir: Path,
    max_samples: int,
    mini_train_size: int,
    mini_val_size: int,
) -> tuple[torch.nn.Module, Any, dict[str, Any]]:
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    original_dataset_root = Path(config["data"]["dataset_root"]).resolve()
    mini_dataset_root = write_mini_dataset_root(
        original_root=original_dataset_root,
        output_dir=output_dir,
        max_test_samples=max_samples,
        train_size=mini_train_size,
        val_size=mini_val_size,
    )
    config["data"]["dataset_root"] = str(mini_dataset_root)
    config["data"]["cache_dir"] = str(mini_dataset_root / "alignn_cache")
    config["data"]["local_cache_root"] = str(output_dir / "_alignn_cache")
    config["data"]["cache_namespace"] = f"{spec.key}_oversmoothing_test{max_samples}"
    config.setdefault("training", {})["batch_size"] = batch_size
    config.setdefault("training", {})["test_batch_size"] = batch_size
    config["training"]["num_workers"] = 0
    physics_variant = config["model"].get("variant", "baseline").lower() in {
        "physconstraint",
        "sizesupervisedphysvirtualnode",
    }
    _train_loader, _val_loader, test_loader = build_dataloaders(config, physics_variant=physics_variant)
    model = build_alignn_model(config["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if hasattr(model, "set_training_epoch"):
        model.set_training_epoch(int(checkpoint.get("epoch", config["training"].get("epochs", 50))), int(config["training"].get("epochs", 50)))
    model.eval()
    return model, test_loader, config


def run_experiment(
    spec: ExperimentSpec,
    device: torch.device,
    batch_size: int,
    max_samples: int,
    output_dir: Path,
    mini_train_size: int,
    mini_val_size: int,
) -> list[dict[str, Any]]:
    model, test_loader, _config = load_model_and_test_loader(
        spec,
        device=device,
        batch_size=batch_size,
        output_dir=output_dir,
        max_samples=max_samples,
        mini_train_size=mini_train_size,
        mini_val_size=mini_val_size,
    )
    dataset_ids = [str(sample_id) for sample_id in getattr(test_loader.dataset, "ids")]
    rows: list[dict[str, Any]] = []
    processed = 0
    cursor = 0
    with torch.no_grad():
        for batch in test_loader:
            if processed >= max_samples:
                break
            graph, line_graph, lattice, _labels = move_batch(batch, device)
            batch_size_actual = int(graph.batch_size)
            take = min(batch_size_actual, max_samples - processed)
            sample_ids = dataset_ids[cursor : cursor + batch_size_actual]
            cursor += batch_size_actual
            captured: list[dict[str, Any]] = []
            handles = register_layer_hooks(model, captured)
            try:
                _ = model([graph, line_graph, lattice])
            finally:
                for handle in handles:
                    handle.remove()
            batch_rows = per_graph_rows(
                experiment=spec.key,
                sample_ids=sample_ids,
                graph=graph,
                captured_layers=captured,
            )
            if take < batch_size_actual:
                keep_ids = set(sample_ids[:take])
                batch_rows = [row for row in batch_rows if row["sample_id"] in keep_ids]
            rows.extend(batch_rows)
            processed += take
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                row["experiment"],
                row["layer_index"],
                row["stage"],
                row["class_name"],
                row["num_atoms_bucket"],
            )
        ].append(row)

    output: list[dict[str, Any]] = []
    for (experiment, layer_index, stage, class_name, bucket), group in sorted(grouped.items()):
        output.append(
            {
                "experiment": experiment,
                "layer_index": layer_index,
                "stage": stage,
                "class_name": class_name,
                "num_atoms_bucket": bucket,
                "n": len(group),
                "mean_num_atoms": sum(float(row["num_atoms"]) for row in group) / len(group),
                "mean_pairwise_cosine": sum(float(row["mean_pairwise_cosine"]) for row in group) / len(group),
                "node_feature_variance": sum(float(row["node_feature_variance"]) for row in group) / len(group),
                "dirichlet_energy": sum(float(row["dirichlet_energy"]) for row in group) / len(group),
            }
        )
    return output


def aggregate_all_buckets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    all_rows = [dict(row, num_atoms_bucket="ALL") for row in rows]
    large_rows = [dict(row, num_atoms_bucket=">80") for row in rows if int(row["num_atoms"]) > 80]
    return aggregate_rows([*all_rows, *large_rows])


def final_layer_summary(aggregates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    max_layer_by_experiment: dict[str, int] = {}
    for row in aggregates:
        if row["num_atoms_bucket"] != "ALL":
            continue
        exp = str(row["experiment"])
        max_layer_by_experiment[exp] = max(max_layer_by_experiment.get(exp, -1), int(row["layer_index"]))

    summary = [
        row
        for row in aggregates
        if int(row["layer_index"]) == max_layer_by_experiment.get(str(row["experiment"]), -999)
        and row["num_atoms_bucket"] in {"ALL", ">80", "81-160", ">160"}
    ]
    return sorted(summary, key=lambda row: (str(row["num_atoms_bucket"]), str(row["experiment"])))


def write_markdown(path: Path, aggregates: list[dict[str, Any]], args: argparse.Namespace) -> None:
    final_rows = final_layer_summary(aggregates)
    by_key = {(row["experiment"], row["num_atoms_bucket"]): row for row in final_rows}
    lines: list[str] = []
    lines.append("# Over-smoothing Hook Diagnostic")
    lines.append("")
    lines.append(f"- Results root: `{args.results_root}`")
    lines.append(f"- Max test samples per model: `{args.max_samples}`")
    lines.append(f"- Batch size: `{args.batch_size}`")
    lines.append("")
    lines.append("Metrics are computed from atom embeddings captured after each ALIGNN/GatedGCN layer.")
    lines.append("For over-smoothing, the expected signal is lower mean pairwise cosine, higher node-feature variance, and higher Dirichlet energy in later layers.")
    lines.append("")
    lines.append("## Final-Layer Summary")
    lines.append("")
    lines.append("| Bucket | Model | n | cosine ↓ | variance ↑ | Dirichlet ↑ |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in final_rows:
        lines.append(
            f"| {row['num_atoms_bucket']} | {row['experiment']} | {int(row['n'])} | "
            f"{float(row['mean_pairwise_cosine']):.6f} | {float(row['node_feature_variance']):.6f} | "
            f"{float(row['dirichlet_energy']):.6f} |"
        )
    lines.append("")

    if ("T2", "ALL") in by_key and ("T4", "ALL") in by_key:
        t2 = by_key[("T2", "ALL")]
        t4 = by_key[("T4", "ALL")]
        cosine_delta = float(t4["mean_pairwise_cosine"]) - float(t2["mean_pairwise_cosine"])
        var_delta = float(t4["node_feature_variance"]) - float(t2["node_feature_variance"])
        dir_delta = float(t4["dirichlet_energy"]) - float(t2["dirichlet_energy"])
        lines.append("## T4 vs T2 Interpretation")
        lines.append("")
        lines.append(
            f"On all evaluated samples, T4-T2 final-layer deltas are: cosine `{cosine_delta:.6f}`, "
            f"variance `{var_delta:.6f}`, Dirichlet `{dir_delta:.6f}`."
        )
        if cosine_delta < 0 and var_delta > 0 and dir_delta > 0:
            lines.append(
                "This is direct embedding-level evidence consistent with the claim that the single global token alleviates over-smoothing."
            )
        elif cosine_delta < 0 or var_delta > 0 or dir_delta > 0:
            lines.append(
                "This is partial evidence only. Some embedding-collapse metrics improve, but the full over-smoothing signature is not uniformly present."
            )
        else:
            lines.append(
                "This does not support an over-smoothing relief claim for T4 over T2 under this diagnostic."
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metrics(aggregates: list[dict[str, Any]], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    all_rows = [row for row in aggregates if row["num_atoms_bucket"] == "ALL"]
    metrics = [
        ("mean_pairwise_cosine", "Mean pairwise cosine"),
        ("node_feature_variance", "Node-feature variance"),
        ("dirichlet_energy", "Dirichlet energy"),
    ]
    for metric, title in metrics:
        plt.figure(figsize=(8.0, 4.6))
        for experiment in sorted({str(row["experiment"]) for row in all_rows}):
            model_rows = sorted(
                [row for row in all_rows if row["experiment"] == experiment],
                key=lambda row: int(row["layer_index"]),
            )
            plt.plot(
                [int(row["layer_index"]) + 1 for row in model_rows],
                [float(row[metric]) for row in model_rows],
                marker="o",
                linewidth=2,
                label=experiment,
            )
        plt.xlabel("Layer")
        plt.ylabel(title)
        plt.title(f"Layer-wise {title}")
        plt.grid(True, alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"layerwise_{metric}.png", dpi=180)
        plt.close()


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    specs = resolve_experiments(results_root, args.experiments)

    all_rows: list[dict[str, Any]] = []
    for spec in specs:
        print(f"[run] {spec.key}: {spec.checkpoint}")
        rows = run_experiment(
            spec,
            device=device,
            batch_size=args.batch_size,
            max_samples=args.max_samples,
            output_dir=output_dir,
            mini_train_size=args.mini_train_size,
            mini_val_size=args.mini_val_size,
        )
        print(f"[done] {spec.key}: {len(rows)} layer/sample rows")
        all_rows.extend(rows)

    sample_rows_path = output_dir / "oversmoothing_sample_metrics.csv"
    agg_rows = aggregate_rows(all_rows)
    agg_rows.extend(aggregate_all_buckets(all_rows))
    aggregate_rows_path = output_dir / "oversmoothing_aggregate_metrics.csv"
    final_rows_path = output_dir / "oversmoothing_final_layer_summary.csv"

    write_csv(sample_rows_path, all_rows)
    write_csv(aggregate_rows_path, agg_rows)
    write_csv(final_rows_path, final_layer_summary(agg_rows))
    plot_metrics(agg_rows, output_dir)
    write_markdown(output_dir / "oversmoothing_hook_diagnostic.md", agg_rows, args)
    (output_dir / "run_config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"[ok] wrote diagnostics to {output_dir}")


if __name__ == "__main__":
    main()
