from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from crystal_gnn.utils import load_yaml


EXPERIMENTS: dict[str, dict[str, str]] = {
    "E1": {
        "name": "MPHZW_E1_CGCNN_baseline",
        "template": "configs/mp/MP_E1_cgcnn_faithful.yaml",
        "python_role": "default",
        "cache_kind": "none",
    },
    "E2": {
        "name": "MPHZW_E2_ALIGNN_baseline",
        "template": "configs/mp/MP_E2_alignn_faithful.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "E3": {
        "name": "MPHZW_E3_M3GNet_baseline",
        "template": "configs/mp/MP_E3_m3gnet_faithful.yaml",
        "python_role": "m3gnet",
        "cache_kind": "m3gnet",
    },
    "E4": {
        "name": "MPHZW_E4_ALIGNN_virtualnode",
        "template": "configs/mp/MP_E4_alignn_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "E5": {
        "name": "MPHZW_E5_ALIGNN_transformer",
        "template": "configs/E5_alignn_transformer_faithful.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "E7": {
        "name": "MPHZW_E7_ALIGNN_attentive_virtualnode",
        "template": "configs/mp/MP_E7_alignn_attentive_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "E8": {
        "name": "MPHZW_E8_ALIGNN_multi_virtualnode",
        "template": "configs/mp/MP_E8_alignn_multi_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "E10": {
        "name": "MPHZW_E10_ALIGNN_valrouted_e4e8",
        "template": "configs/mp/MP_E10_alignn_valrouted_e4e8.yaml",
        "python_role": "alignn",
        "cache_kind": "router",
    },
    "E12": {
        "name": "MPHZW_E12_ALIGNN_sizesupervised_virtualnode",
        "template": "configs/mp/MP_E12_alignn_sizesupervised_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
}


PERFORMANCE_4090: dict[str, dict[str, Any]] = {
    # CGCNN graph construction is expensive. Build the tensor cache in the main
    # process to avoid PyTorch worker shared-memory mmap failures on large splits,
    # then train from RAM with a larger batch.
    "E1": {
        "batch_size": 1536,
        "test_batch_size": 1536,
        "num_workers": 0,
        "mixed_precision": True,
        "preload_to_memory": True,
        "preload_num_workers": 0,
        "preload_batch_size": 256,
    },
    # DGL sparse ops used by ALIGNN produce non-finite gradients under AMP on
    # this CUDA stack. Keep ALIGNN-family models in FP32; M3GNet is stable in AMP.
    "E2": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "E3": {"batch_size": 48, "test_batch_size": 64, "num_workers": 0, "mixed_precision": True},
    "E4": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "E5": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "E7": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "E8": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "E10": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "E12": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
}


# All ALIGNN-family variants in this queue use the same crystal graph
# construction settings. Reusing the E2 graph LMDB avoids creating one 12GB
# cache per model variant on the 50GB scratch disk.
SHARED_ALIGNN_CACHE_NAMESPACE = "MPHZW_CLEAN_E2_ALIGNN_baseline"


OOM_PATTERNS = (
    "CUDA out of memory",
    "CUDA: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "DefaultCPUAllocator: can't allocate memory",
    "Killed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run MP-Hongzhiwei matched E1/E2/E3/E4/E5/E7/E8/E10/E12 queue.")
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--dataset_dir", default="data/mp_hzw_matched")
    parser.add_argument("--results_dir", default="results/mp_hzw_matched_e1_e12")
    parser.add_argument("--cache_root", default="/root/lanyun-tmp/gnn_mp_hzw_matched_cache")
    parser.add_argument("--cgcnn_root", default=None)
    parser.add_argument("--default_python", default=sys.executable)
    parser.add_argument("--alignn_python", default=None)
    parser.add_argument("--m3gnet_python", default=None)
    parser.add_argument("--experiments", nargs="*", default=["E1", "E2", "E3", "E4", "E5", "E7", "E8", "E10", "E12"])
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--export_limit", type=int, default=None)
    parser.add_argument("--export_chunk_size", type=int, default=100)
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--fail_fast", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="Optional epoch override for smoke tests.")
    parser.add_argument("--performance_mode", choices=["safe", "4090"], default="4090")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--mixed_precision", choices=["auto", "true", "false"], default="auto")
    parser.add_argument("--max_batch_retries", type=int, default=3)
    parser.add_argument("--skip_summary", action="store_true")
    return parser.parse_args()


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=True)


def selected_experiments(keys: list[str]) -> list[tuple[str, dict[str, str]]]:
    selected: list[tuple[str, dict[str, str]]] = []
    for raw_key in keys:
        key = raw_key.upper()
        if key not in EXPERIMENTS:
            raise ValueError(f"unknown experiment key: {raw_key}")
        selected.append((key, EXPERIMENTS[key]))
    return selected


def generate_config(
    project_root: Path,
    dataset_dir: Path,
    results_dir: Path,
    cache_root: Path,
    key: str,
    experiment: dict[str, str],
    epochs: int | None,
    performance_mode: str,
    num_workers_override: int | None,
    mixed_precision_override: str,
    batch_divisor: int = 1,
) -> Path:
    config = load_yaml(project_root / experiment["template"])
    config["experiment_name"] = experiment["name"]
    config.setdefault("data", {})["dataset_root"] = str(dataset_dir)
    training_config = config.setdefault("training", {})
    if performance_mode == "4090":
        perf = PERFORMANCE_4090.get(key, {})
        for perf_key, perf_value in perf.items():
            training_config[perf_key] = perf_value
    if batch_divisor > 1:
        for batch_key in ["batch_size", "test_batch_size"]:
            if batch_key in training_config:
                training_config[batch_key] = max(1, int(training_config[batch_key]) // batch_divisor)
    if num_workers_override is not None:
        training_config["num_workers"] = num_workers_override
    if mixed_precision_override != "auto":
        training_config["mixed_precision"] = mixed_precision_override == "true"
    cache_kind = experiment["cache_kind"]
    if cache_kind == "alignn":
        config["data"]["cache_dir"] = str(dataset_dir / "alignn_cache")
        config["data"]["local_cache_root"] = str(cache_root / "alignn")
        config["data"]["cache_namespace"] = SHARED_ALIGNN_CACHE_NAMESPACE
    elif key == "E1":
        config["data"]["cgcnn_graph_cache_dir"] = str(results_dir / "cgcnn_graph_cache")
    elif cache_kind == "m3gnet":
        config["data"]["cache_dir"] = str(cache_root / "m3gnet")
        config["data"]["preprocess_shard_size"] = int(config["data"].get("preprocess_shard_size", 2048))
    elif cache_kind == "router":
        config["data"]["cache_dir"] = str(dataset_dir / "alignn_cache")
        config["data"]["local_cache_root"] = str(cache_root / "alignn")
        config["data"]["cache_namespace"] = SHARED_ALIGNN_CACHE_NAMESPACE
        generated_dir = results_dir / "generated_configs"
        e4_generated = generated_dir / "MPHZW_E4_ALIGNN_virtualnode_batchdiv1.yaml"
        e8_generated = generated_dir / "MPHZW_E8_ALIGNN_multi_virtualnode_batchdiv1.yaml"
        config.setdefault("router", {}).setdefault("primary_expert", {})
        config.setdefault("router", {}).setdefault("secondary_expert", {})
        config["router"]["primary_expert"]["config_path"] = str(
            e4_generated if e4_generated.exists() else project_root / "configs/mp/MP_E4_alignn_virtualnode.yaml"
        )
        config["router"]["primary_expert"]["checkpoint_path"] = str(
            results_dir / "MPHZW_E4_ALIGNN_virtualnode" / "best_model.pt"
        )
        config["router"]["secondary_expert"]["config_path"] = str(
            e8_generated if e8_generated.exists() else project_root / "configs/mp/MP_E8_alignn_multi_virtualnode.yaml"
        )
        config["router"]["secondary_expert"]["checkpoint_path"] = str(
            results_dir / "MPHZW_E8_ALIGNN_multi_virtualnode" / "best_model.pt"
        )
    if epochs is not None:
        training_config["epochs"] = epochs
    config_path = results_dir / "generated_configs" / f"{experiment['name']}_batchdiv{batch_divisor}.yaml"
    write_yaml(config_path, config)
    return config_path


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n===== started {started} =====\n")
        handle.write(" ".join(command) + "\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        returncode = process.wait()
        handle.write(f"===== finished rc={returncode} at {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    return returncode


def log_has_recoverable_oom(log_path: Path) -> bool:
    if not log_path.exists():
        return False
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return any(pattern in text for pattern in OOM_PATTERNS)


def latest_error_excerpt(log_path: Path, max_lines: int = 80) -> str:
    if not log_path.exists():
        return ""
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    interesting = [
        line
        for line in lines
        if re.search(r"error|exception|traceback|out of memory|killed|failed", line, re.IGNORECASE)
    ]
    return "\n".join((interesting or lines)[-max_lines:])


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Materials Project API key is required. Pass --api_key or set MP_API_KEY.")

    project_root = Path(__file__).resolve().parent
    dataset_dir = (project_root / args.dataset_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    cache_root = Path(args.cache_root)
    logs_dir = results_dir / "queue_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MP_API_KEY"] = args.api_key
    env.setdefault("MATGL_BACKEND", "DGL")
    env.setdefault("GNN_ENABLE_TF32", "1")
    env.setdefault("OMP_NUM_THREADS", "8")
    env.setdefault("MKL_NUM_THREADS", "8")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,max_split_size_mb:256")

    default_python = args.default_python
    alignn_python = args.alignn_python or default_python
    m3gnet_python = args.m3gnet_python or default_python
    python_for_role = {"default": default_python, "alignn": alignn_python, "m3gnet": m3gnet_python}

    manifest_path = dataset_dir / "manifest.jsonl"
    metadata_path = dataset_dir / "metadata.json"
    if args.force_export or not manifest_path.exists() or not metadata_path.exists():
        export_command = [
            default_python,
            "export_mp_hzw_matched_dataset.py",
            "--api_key",
            args.api_key,
            "--hzw_raw_path",
            str((project_root / args.hzw_raw_path).resolve()),
            "--output_dir",
            str(dataset_dir),
            "--seed",
            "42",
            "--chunk_size",
            str(args.export_chunk_size),
        ]
        if args.cgcnn_root is not None:
            export_command.extend(["--cgcnn_root", args.cgcnn_root])
        if args.force_export:
            export_command.append("--force")
        if args.export_limit is not None:
            export_command.extend(["--limit", str(args.export_limit)])
        returncode = run_command(export_command, cwd=project_root, env=env, log_path=logs_dir / "export_dataset.log")
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, export_command)
    else:
        print(f"[skip] matched dataset exists at {dataset_dir}")

    failures: list[dict[str, Any]] = []
    for key, experiment in selected_experiments(args.experiments):
        output_dir = results_dir / experiment["name"]
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not args.no_resume:
            print(f"[skip] metrics already exist for {experiment['name']}: {metrics_path}")
            continue

        success = False
        last_failure: dict[str, Any] | None = None
        for retry_index in range(args.max_batch_retries + 1):
            batch_divisor = 2**retry_index
            config_path = generate_config(
                project_root,
                dataset_dir,
                results_dir,
                cache_root,
                key,
                experiment,
                args.epochs,
                args.performance_mode,
                args.num_workers,
                args.mixed_precision,
                batch_divisor=batch_divisor,
            )
            command = [
                python_for_role[experiment["python_role"]],
                "run_faithful_experiment.py",
                "--config",
                str(config_path),
                "--output_dir",
                str(output_dir),
            ]
            log_path = logs_dir / f"{key}_{experiment['name']}_try{retry_index + 1}_batchdiv{batch_divisor}.log"
            print(f"[run] {experiment['name']} try={retry_index + 1} batch_divisor={batch_divisor}")
            returncode = run_command(command, cwd=project_root, env=env, log_path=log_path)
            if returncode == 0:
                success = True
                break

            recoverable = returncode == 137 or log_has_recoverable_oom(log_path)
            last_failure = {
                "experiment_key": key,
                "experiment_name": experiment["name"],
                "config": str(config_path),
                "returncode": int(returncode),
                "command": " ".join(command),
                "log_path": str(log_path),
                "recoverable_oom": recoverable,
                "error_excerpt": latest_error_excerpt(log_path),
            }
            print(f"[fail] {experiment['name']} rc={returncode} recoverable_oom={recoverable}")
            if not recoverable:
                break
        if not success:
            assert last_failure is not None
            failures.append(last_failure)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "failure.json").open("w", encoding="utf-8") as handle:
                json.dump(last_failure, handle, indent=2, ensure_ascii=False)
            if args.fail_fast:
                raise subprocess.CalledProcessError(int(last_failure["returncode"]), str(last_failure["command"]))

    if not args.skip_summary:
        summary_command = [
            default_python,
            "summarize.py",
            "--results_dir",
            str(results_dir),
            "--output",
            str(results_dir / "comparison_summary.csv"),
        ]
        returncode = run_command(summary_command, cwd=project_root, env=env, log_path=logs_dir / "summarize.log")
        if returncode != 0:
            failures.append({"experiment_key": "summary", "returncode": returncode})

    if failures:
        with (results_dir / "failures.json").open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2, ensure_ascii=False)
        print(f"[warn] {len(failures)} failure(s); see {results_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
