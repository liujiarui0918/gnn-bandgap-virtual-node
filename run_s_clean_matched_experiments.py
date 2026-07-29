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


S_EXPERIMENTS: dict[str, dict[str, str]] = {
    "S1": {
        "name": "S1_CGCNN_baseline",
        "template": "configs/mp/MP_E1_cgcnn_faithful.yaml",
        "python_role": "default",
        "cache_kind": "none",
    },
    "S2": {
        "name": "S2_ALIGNN_baseline",
        "template": "configs/mp/MP_E2_alignn_faithful.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S3": {
        "name": "S3_M3GNet_baseline",
        "template": "configs/mp/MP_E3_m3gnet_faithful.yaml",
        "python_role": "m3gnet",
        "cache_kind": "m3gnet",
    },
    "S4": {
        "name": "S4_ALIGNN_virtualnode",
        "template": "configs/mp/MP_E4_alignn_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S14": {
        "name": "S14_ALIGNN_huber_virtualnode",
        "template": "configs/mp/MP_S14_alignn_huber_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S5": {
        "name": "S5_ALIGNN_transformer",
        "template": "configs/E5_alignn_transformer_faithful.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S7": {
        "name": "S7_ALIGNN_attentive_virtualnode",
        "template": "configs/mp/MP_E7_alignn_attentive_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S8": {
        "name": "S8_ALIGNN_multi_virtualnode",
        "template": "configs/mp/MP_E8_alignn_multi_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S10": {
        "name": "S10_ALIGNN_sizeaware_virtualnode_active",
        "template": "configs/E10_alignn_sizeaware_virtualnode_active.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S11": {
        "name": "S11_ALIGNN_valrouted_e4e8",
        "template": "configs/E11_alignn_valrouted_e4e8.yaml",
        "python_role": "alignn",
        "cache_kind": "router",
    },
    "S12": {
        "name": "S12_ALIGNN_sizesupervised_virtualnode",
        "template": "configs/mp/MP_E12_alignn_sizesupervised_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "S13": {
        "name": "S13_ALIGNN_plain_dual_virtualnode",
        "template": "configs/mp/MP_E13_alignn_plain_dual_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
}


PERFORMANCE_4090: dict[str, dict[str, Any]] = {
    "S1": {
        "batch_size": 1536,
        "test_batch_size": 1536,
        "num_workers": 0,
        "mixed_precision": True,
        "preload_to_memory": True,
        "preload_num_workers": 0,
        "preload_batch_size": 256,
    },
    "S2": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "S3": {"batch_size": 48, "test_batch_size": 64, "num_workers": 0, "mixed_precision": True},
    "S4": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "S5": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "S7": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "S8": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "S10": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "S11": {"batch_size": 64, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "S12": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "S13": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
}


SHARED_ALIGNN_CACHE_NAMESPACE = "MPHZW_CLEAN_S2_ALIGNN_baseline"
DEFAULT_EXPERIMENTS = ["S1", "S2", "S3", "S4", "S5", "S7", "S8", "S10", "S11", "S12"]
OOM_PATTERNS = (
    "CUDA out of memory",
    "CUDA: out of memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "CUDACachingAllocator",
    "expandable_segment",
    "INTERNAL ASSERT FAILED",
    "DefaultCPUAllocator: can't allocate memory",
    "Killed",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S1/S2/S3/S4/S5/S7/S8/S10/S11/S12 on MP-HZW clean matched data.")
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--audit_dir", default="results/data_quality")
    parser.add_argument("--dataset_dir", default="data/mp_hzw_clean_matched_s")
    parser.add_argument("--results_dir", default="results/mp_hzw_clean_matched_s")
    parser.add_argument("--cache_root", default="/data/home/tg100025/gnn_work/cache/mp_hzw_clean_s")
    parser.add_argument("--cgcnn_root", default=None)
    parser.add_argument("--default_python", default=sys.executable)
    parser.add_argument("--alignn_python", default=None)
    parser.add_argument("--m3gnet_python", default=None)
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--force_audit", action="store_true")
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--export_limit", type=int, default=None)
    parser.add_argument("--mp_fetch_chunk_size", type=int, default=100)
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
        if key not in S_EXPERIMENTS:
            raise ValueError(f"unknown S experiment key: {raw_key}")
        selected.append((key, S_EXPERIMENTS[key]))
    return selected


def apply_training_overrides(
    config: dict[str, Any],
    key: str,
    epochs: int | None,
    performance_mode: str,
    num_workers_override: int | None,
    mixed_precision_override: str,
    batch_divisor: int,
) -> None:
    training_config = config.setdefault("training", {})
    if performance_mode == "4090":
        for perf_key, perf_value in PERFORMANCE_4090.get(key, {}).items():
            training_config[perf_key] = perf_value
    if batch_divisor > 1:
        for batch_key in ["batch_size", "test_batch_size"]:
            if batch_key in training_config:
                training_config[batch_key] = max(1, int(training_config[batch_key]) // batch_divisor)
    if num_workers_override is not None:
        training_config["num_workers"] = num_workers_override
    if mixed_precision_override != "auto":
        training_config["mixed_precision"] = mixed_precision_override == "true"
    if epochs is not None and config.get("runner", "").lower() != "router":
        training_config["epochs"] = epochs


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
    apply_training_overrides(
        config,
        key,
        epochs,
        performance_mode,
        num_workers_override,
        mixed_precision_override,
        batch_divisor,
    )

    cache_kind = experiment["cache_kind"]
    if cache_kind == "alignn":
        config["data"]["cache_dir"] = str(dataset_dir / "alignn_cache")
        config["data"]["local_cache_root"] = str(cache_root / "alignn")
        config["data"]["cache_namespace"] = SHARED_ALIGNN_CACHE_NAMESPACE
    elif key == "S1":
        config["data"]["cgcnn_graph_cache_dir"] = str(results_dir / "cgcnn_graph_cache")
    elif cache_kind == "m3gnet":
        config["data"]["cache_dir"] = str(cache_root / "m3gnet")
        config["data"]["preprocess_shard_size"] = int(config["data"].get("preprocess_shard_size", 2048))
    elif cache_kind == "router":
        config["data"]["cache_dir"] = str(dataset_dir / "alignn_cache")
        config["data"]["local_cache_root"] = str(cache_root / "alignn")
        config["data"]["cache_namespace"] = SHARED_ALIGNN_CACHE_NAMESPACE
        generated_dir = results_dir / "generated_configs"
        s4_generated = generated_dir / "S4_ALIGNN_virtualnode_batchdiv1.yaml"
        s8_generated = generated_dir / "S8_ALIGNN_multi_virtualnode_batchdiv1.yaml"
        config.setdefault("router", {}).setdefault("primary_expert", {})
        config.setdefault("router", {}).setdefault("secondary_expert", {})
        config["router"]["primary_expert"]["name"] = "S4"
        config["router"]["primary_expert"]["config_path"] = str(
            s4_generated if s4_generated.exists() else project_root / "configs/mp/MP_E4_alignn_virtualnode.yaml"
        )
        config["router"]["primary_expert"]["checkpoint_path"] = str(
            results_dir / "S4_ALIGNN_virtualnode" / "best_model.pt"
        )
        config["router"]["secondary_expert"]["name"] = "S8"
        config["router"]["secondary_expert"]["config_path"] = str(
            s8_generated if s8_generated.exists() else project_root / "configs/mp/MP_E8_alignn_multi_virtualnode.yaml"
        )
        config["router"]["secondary_expert"]["checkpoint_path"] = str(
            results_dir / "S8_ALIGNN_multi_virtualnode" / "best_model.pt"
        )

    config_path = results_dir / "generated_configs" / f"{experiment['name']}_batchdiv{batch_divisor}.yaml"
    write_yaml(config_path, config)
    return config_path


def run_command(command: list[str], cwd: Path, env: dict[str, str], log_path: Path, display_command: list[str] | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    safe_command = display_command if display_command is not None else command
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write(f"\n===== started {started} =====\n")
        handle.write(" ".join(safe_command) + "\n")
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


def ensure_router_dependencies(results_dir: Path) -> None:
    missing = [
        path
        for path in [
            results_dir / "S4_ALIGNN_virtualnode" / "best_model.pt",
            results_dir / "S8_ALIGNN_multi_virtualnode" / "best_model.pt",
        ]
        if not path.exists()
    ]
    if missing:
        missing_text = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"S11 requires finished S4/S8 checkpoints, missing: {missing_text}")


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    audit_dir = (project_root / args.audit_dir).resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    cache_root = Path(args.cache_root)
    logs_dir = results_dir / "queue_logs"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.api_key:
        env["MP_API_KEY"] = args.api_key
    env.setdefault("MATGL_BACKEND", "DGL")
    env.setdefault("DGLBACKEND", "pytorch")
    env.setdefault("GNN_ENABLE_TF32", "1")
    env.setdefault("OMP_NUM_THREADS", "10")
    env.setdefault("MKL_NUM_THREADS", "10")
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.8")

    default_python = args.default_python
    alignn_python = args.alignn_python or default_python
    m3gnet_python = args.m3gnet_python or default_python
    python_for_role = {"default": default_python, "alignn": alignn_python, "m3gnet": m3gnet_python}

    audit_csv = audit_dir / "mp_hzw_full_audit_table.csv"
    mp_cache_path = audit_dir / "mp_hzw_summary_fetch_cache.jsonl"
    if not args.skip_audit and (args.force_audit or not audit_csv.exists() or not mp_cache_path.exists()):
        if not env.get("MP_API_KEY"):
            raise ValueError("MP API key is required for audit. Pass --api_key or set MP_API_KEY.")
        audit_command = [
            default_python,
            "audit_mp_hzw_matched_garbage.py",
            "--hzw_raw_path",
            str((project_root / args.hzw_raw_path).resolve()),
            "--output_dir",
            str(audit_dir),
            "--mp_cache_path",
            str(mp_cache_path),
            "--chunk_size",
            str(args.mp_fetch_chunk_size),
        ]
        if mp_cache_path.exists():
            audit_command.append("--offline_cache")
        returncode = run_command(audit_command, cwd=project_root, env=env, log_path=logs_dir / "audit_garbage.log")
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, audit_command)
    else:
        print(f"[skip] audit exists at {audit_csv}")

    manifest_path = dataset_dir / "manifest.jsonl"
    metadata_path = dataset_dir / "metadata.json"
    if not args.skip_export and (args.force_export or not manifest_path.exists() or not metadata_path.exists()):
        export_command = [
            default_python,
            "export_mp_hzw_clean_matched_dataset.py",
            "--hzw_raw_path",
            str((project_root / args.hzw_raw_path).resolve()),
            "--audit_csv",
            str(audit_csv),
            "--mp_cache_path",
            str(mp_cache_path),
            "--output_dir",
            str(dataset_dir),
            "--force",
        ]
        if args.cgcnn_root is not None:
            export_command.extend(["--cgcnn_root", args.cgcnn_root])
        if args.export_limit is not None:
            export_command.extend(["--limit", str(args.export_limit)])
        returncode = run_command(export_command, cwd=project_root, env=env, log_path=logs_dir / "export_clean_dataset.log")
        if returncode != 0:
            raise subprocess.CalledProcessError(returncode, export_command)
    else:
        print(f"[skip] clean dataset exists at {dataset_dir}")

    failures: list[dict[str, Any]] = []
    for key, experiment in selected_experiments(args.experiments):
        output_dir = results_dir / experiment["name"]
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not args.no_resume:
            print(f"[skip] metrics already exist for {experiment['name']}: {metrics_path}")
            continue
        if key == "S11":
            ensure_router_dependencies(results_dir)

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
                failure_path = output_dir / "failure.json"
                if failure_path.exists():
                    failure_path.unlink()
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
