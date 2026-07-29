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


T_EXPERIMENTS: dict[str, dict[str, str]] = {
    "T1": {
        "name": "T1_CGCNN_baseline",
        "template": "configs/mp/MP_E1_cgcnn_faithful.yaml",
        "python_role": "default",
        "cache_kind": "none",
    },
    "T2": {
        "name": "T2_ALIGNN_baseline",
        "template": "configs/mp/MP_E2_alignn_faithful.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T3": {
        "name": "T3_M3GNet_baseline",
        "template": "configs/mp/MP_E3_m3gnet_faithful.yaml",
        "python_role": "m3gnet",
        "cache_kind": "m3gnet",
    },
    "T4": {
        "name": "T4_ALIGNN_plain_single_virtualnode",
        "template": "configs/mp/MP_E4_alignn_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T5": {
        "name": "T5_ALIGNN_plain_dual_virtualnode",
        "template": "configs/mp/MP_E13_alignn_plain_dual_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T6": {
        "name": "T6_ALIGNN_topk_huber_virtualnode",
        "template": "configs/mp/MP_T6_alignn_topk_huber_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T7": {
        "name": "T7_ALIGNN_huber_virtualnode",
        "template": "configs/mp/MP_T7_alignn_huber_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T8": {
        "name": "T8_ALIGNN_topk_mse_virtualnode",
        "template": "configs/mp/MP_T8_alignn_topk_mse_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T9": {
        "name": "T9_ALIGNN_warmup_topk_huber_virtualnode",
        "template": "configs/mp/MP_T9_alignn_warmup_topk_huber_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
    "T10": {
        "name": "T10_ALIGNN_decay_huber_virtualnode",
        "template": "configs/mp/MP_T10_alignn_decay_huber_virtualnode.yaml",
        "python_role": "alignn",
        "cache_kind": "alignn",
    },
}


PERFORMANCE_4090: dict[str, dict[str, Any]] = {
    "T1": {
        "batch_size": 1536,
        "test_batch_size": 1536,
        "num_workers": 0,
        "mixed_precision": True,
        "preload_to_memory": True,
        "preload_num_workers": 0,
        "preload_batch_size": 256,
    },
    "T2": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T3": {"batch_size": 48, "test_batch_size": 64, "num_workers": 0, "mixed_precision": True},
    "T4": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T5": {"batch_size": 32, "test_batch_size": 48, "num_workers": 6, "mixed_precision": False},
    "T6": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T7": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T8": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T9": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
    "T10": {"batch_size": 48, "test_batch_size": 64, "num_workers": 6, "mixed_precision": False},
}


SHARED_ALIGNN_CACHE_NAMESPACE = "MPHZW_RELAXED_T2_ALIGNN_baseline"
DEFAULT_EXPERIMENTS = ["T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9", "T10"]
DEFAULT_EXPORT_SCRIPT = "export_mp_hzw_final_clean_dataset.py"
DEFAULT_STRUCTURE_MATCHED_CSV = "structure_first_strict_180/mp_hzw_structure_matched_all.csv"
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
    parser = argparse.ArgumentParser(
        description="Run relaxed-clean T-series experiments on MP-HZW equal-atoms matched data."
    )
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument(
        "--hzw_raw_path",
        default="api_downloads/api_downloads/crystal_band_100000.jsonl/crystal_band_100000.jsonl.gz.tmp",
    )
    parser.add_argument("--audit_dir", default="results/data_quality")
    parser.add_argument(
        "--structure_matched_csv",
        default=None,
        help=(
            "Structure-matched MP-HZW CSV. Defaults to "
            "AUDIT_DIR/structure_first_strict_180/mp_hzw_structure_matched_all.csv."
        ),
    )
    parser.add_argument(
        "--mp_cache_path",
        default=None,
        help="MP summary cache JSONL. Defaults to AUDIT_DIR/mp_hzw_summary_fetch_cache.jsonl.",
    )
    parser.add_argument("--dataset_dir", default="data/mp_hzw_relaxed_clean_equal_atoms_mp")
    parser.add_argument("--results_dir", default="results/mp_hzw_relaxed_clean_equal_atoms_mp_t")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed injected into every generated config (default 42 preserves existing runs).",
    )
    parser.add_argument(
        "--results-suffix",
        dest="results_suffix",
        default="",
        help=(
            "Suffix appended to the results root dir (e.g. _seed1042) so different seeds write to "
            "separate directories and resume-skip is isolated per seed. Empty keeps the original path."
        ),
    )
    parser.add_argument(
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="SECTION.KEY=VALUE",
        help=(
            "Set a config value after all other injections, e.g. "
            "--override training.final_gap_huber_beta=0.25 --override model.broadcast_min_scale=0.5. "
            "Repeatable. Values are parsed as int, then float, then true/false, then kept as string. "
            "Applied last, so overrides win over template values and performance-mode settings."
        ),
    )
    parser.add_argument(
        "--early-stopping-patience",
        dest="early_stopping_patience",
        type=int,
        default=0,
        help=(
            "Stop a run once validation MAE has not improved for this many consecutive epochs; "
            "injected into config['training']['early_stopping_patience']. Default 0 disables early "
            "stopping, so runs train the full epoch budget and behave exactly as before."
        ),
    )
    parser.add_argument("--cache_root", default="/data/home/tg100025/gnn_work/cache/mp_hzw_relaxed_clean_equal_atoms_mp_t")
    parser.add_argument("--cgcnn_root", default=None)
    parser.add_argument(
        "--label_source",
        choices=("mp", "hzw"),
        default="mp",
        help="Supervised bandgap label to export and train against.",
    )
    parser.add_argument("--default_python", default=sys.executable)
    parser.add_argument("--alignn_python", default=None)
    parser.add_argument("--m3gnet_python", default=None)
    parser.add_argument("--experiments", nargs="*", default=DEFAULT_EXPERIMENTS)
    parser.add_argument("--force_audit", action="store_true")
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--skip_audit", action="store_true")
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument(
        "--export_script",
        default=DEFAULT_EXPORT_SCRIPT,
        help=(
            "Relaxed-clean exporter to call when dataset files are missing. "
            "The script is expected to accept the same core args as the existing MP-HZW exporters."
        ),
    )
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
        if key not in T_EXPERIMENTS:
            raise ValueError(f"unknown T experiment key: {raw_key}")
        selected.append((key, T_EXPERIMENTS[key]))
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


def parse_override_value(raw: str) -> Any:
    lowered = raw.strip().lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    return raw


def apply_config_overrides(config: dict[str, Any], overrides: list[str]) -> None:
    for spec in overrides:
        if "=" not in spec:
            raise ValueError(f"invalid --override (expected SECTION.KEY=VALUE): {spec}")
        dotted, raw_value = spec.split("=", 1)
        keys = [part for part in dotted.strip().split(".") if part]
        if not keys:
            raise ValueError(f"invalid --override key path: {spec}")
        target: dict[str, Any] = config
        for part in keys[:-1]:
            node = target.setdefault(part, {})
            if not isinstance(node, dict):
                raise ValueError(f"--override path collides with a non-mapping value: {spec}")
            target = node
        target[keys[-1]] = parse_override_value(raw_value)


def generated_config_path(results_dir: Path, experiment_name: str, batch_divisor: int = 1) -> Path:
    return results_dir / "generated_configs" / f"{experiment_name}_batchdiv{batch_divisor}.yaml"


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
    seed: int = 42,
    early_stopping_patience: int = 0,
    overrides: list[str] | None = None,
) -> Path:
    config = load_yaml(project_root / experiment["template"])
    config["experiment_name"] = experiment["name"]
    config["seed"] = seed
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
    # Only inject early stopping when explicitly enabled (>0). The runner treats a
    # missing key and 0 identically, so leaving the config untouched at the default
    # keeps every generated file byte-identical to the full-epoch runs done so far.
    if early_stopping_patience > 0:
        config["training"]["early_stopping_patience"] = early_stopping_patience

    cache_kind = experiment["cache_kind"]
    if cache_kind == "alignn":
        config["data"]["cache_dir"] = str(dataset_dir / "alignn_cache")
        config["data"]["local_cache_root"] = str(cache_root / "alignn")
        config["data"]["cache_namespace"] = SHARED_ALIGNN_CACHE_NAMESPACE
    elif key == "T1":
        config["data"]["cgcnn_graph_cache_dir"] = str(results_dir / "cgcnn_graph_cache")
    elif cache_kind == "m3gnet":
        config["data"]["cache_dir"] = str(cache_root / "m3gnet")
        config["data"]["preprocess_shard_size"] = int(config["data"].get("preprocess_shard_size", 2048))

    # Overrides go last so a sensitivity sweep value (e.g. huber beta, broadcast
    # min scale, m3gnet lr/units) always wins over the template and performance
    # profile, regardless of which section it lives in.
    if overrides:
        apply_config_overrides(config, overrides)

    config_path = generated_config_path(results_dir, experiment["name"], batch_divisor=batch_divisor)
    write_yaml(config_path, config)
    return config_path


def run_command(
    command: list[str],
    cwd: Path,
    env: dict[str, str],
    log_path: Path,
    display_command: list[str] | None = None,
) -> int:
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


def run_audit_if_needed(args: argparse.Namespace, project_root: Path, audit_dir: Path, logs_dir: Path, env: dict[str, str]) -> None:
    audit_csv = audit_dir / "mp_hzw_full_audit_table.csv"
    mp_cache_path = Path(args.mp_cache_path).resolve() if args.mp_cache_path else audit_dir / "mp_hzw_summary_fetch_cache.jsonl"
    if args.skip_audit or (not args.force_audit and audit_csv.exists() and mp_cache_path.exists()):
        print(f"[skip] audit exists at {audit_csv}")
        return

    if not env.get("MP_API_KEY"):
        raise ValueError("MP API key is required for audit. Pass --api_key or set MP_API_KEY.")
    audit_command = [
        args.default_python,
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


def export_dataset_if_needed(
    args: argparse.Namespace,
    project_root: Path,
    audit_dir: Path,
    dataset_dir: Path,
    logs_dir: Path,
    env: dict[str, str],
) -> None:
    manifest_path = dataset_dir / "manifest.jsonl"
    metadata_path = dataset_dir / "metadata.json"
    if args.skip_export or (not args.force_export and manifest_path.exists() and metadata_path.exists()):
        print(f"[skip] relaxed-clean dataset exists at {dataset_dir}")
        return

    export_script = project_root / args.export_script
    if not export_script.exists():
        raise FileNotFoundError(
            f"relaxed-clean exporter is not available: {export_script}. "
            "Create the dataset externally, then pass --skip_export."
        )

    structure_matched_csv = (
        Path(args.structure_matched_csv).resolve()
        if args.structure_matched_csv
        else audit_dir / DEFAULT_STRUCTURE_MATCHED_CSV
    )
    mp_cache_path = Path(args.mp_cache_path).resolve() if args.mp_cache_path else audit_dir / "mp_hzw_summary_fetch_cache.jsonl"
    if not structure_matched_csv.exists():
        raise FileNotFoundError(f"structure-matched CSV not found: {structure_matched_csv}")
    if not mp_cache_path.exists():
        raise FileNotFoundError(f"MP cache not found: {mp_cache_path}")
    export_command = [
        args.default_python,
        str(export_script),
        "--hzw_raw_path",
        str((project_root / args.hzw_raw_path).resolve()),
        "--structure_matched_csv",
        str(structure_matched_csv),
        "--mp_cache_path",
        str(mp_cache_path),
        "--output_dir",
        str(dataset_dir),
        "--label_source",
        args.label_source,
        "--metal_conflict_mode",
        "zero_vs_nonzero",
        "--zero_gap_threshold",
        "1e-6",
        "--disable_gap_diff_filter",
        "--disable_volume_filter",
        "--require_equal_atom_count",
        "--force",
    ]
    if args.cgcnn_root is not None:
        export_command.extend(["--cgcnn_root", args.cgcnn_root])
    if args.export_limit is not None:
        export_command.extend(["--limit", str(args.export_limit)])

    returncode = run_command(export_command, cwd=project_root, env=env, log_path=logs_dir / "export_relaxed_dataset.log")
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, export_command)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    audit_dir = (project_root / args.audit_dir).resolve()
    dataset_dir = (project_root / args.dataset_dir).resolve()
    results_dir = (project_root / (args.results_dir + args.results_suffix)).resolve()
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

    run_audit_if_needed(args, project_root, audit_dir, logs_dir, env)
    export_dataset_if_needed(args, project_root, audit_dir, dataset_dir, logs_dir, env)

    failures: list[dict[str, Any]] = []
    selected = selected_experiments(args.experiments)
    selected_names = [experiment["name"] for _key, experiment in selected]
    for key, experiment in selected:
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
                seed=args.seed,
                early_stopping_patience=args.early_stopping_patience,
                overrides=args.overrides,
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
            "--experiment_names",
            *selected_names,
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
