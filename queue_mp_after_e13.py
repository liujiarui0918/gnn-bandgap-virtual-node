from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wait for Hongzhiwei E13 to finish, then launch the MP standard suite.")
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument("--python", default=sys.executable, help="Python executable used to run the MP launcher.")
    parser.add_argument("--poll_seconds", type=int, default=1800, help="Polling interval while waiting for E13.")
    parser.add_argument(
        "--watched_metrics",
        default="results/E13_ALIGNN_sizesupervised_physconstraint/metrics.json",
    )
    parser.add_argument(
        "--watched_log",
        default="logs/E13_ALIGNN_sizesupervised_physconstraint.nohup.log",
    )
    parser.add_argument(
        "--watched_process_pattern",
        default="run_faithful_experiment.py --config configs/E13_alignn_sizesupervised_physconstraint.yaml",
    )
    parser.add_argument(
        "--launcher_script",
        default="run_mp_standard_experiments.py",
    )
    parser.add_argument(
        "--launcher_process_pattern",
        default="run_mp_standard_experiments.py",
        help="Process pattern used to detect an already running MP launcher.",
    )
    parser.add_argument(
        "--launcher_results_dir",
        default="results/mp_bandgap_standard",
    )
    parser.add_argument(
        "--launcher_dataset_dir",
        default="data/mp_bandgap_official",
    )
    parser.add_argument(
        "--launcher_config_args",
        nargs="*",
        default=[],
        help="Extra arguments forwarded to the MP launcher.",
    )
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def log(message: str) -> None:
    print(f"{timestamp()} | {message}", flush=True)


def process_running(pattern: str) -> bool:
    result = subprocess.run(
        ["pgrep", "-f", pattern],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return result.returncode == 0


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Materials Project API key is required. Pass --api_key or set MP_API_KEY.")

    project_root = Path(args.project_root).resolve()
    watched_metrics = (project_root / args.watched_metrics).resolve()
    watched_log = (project_root / args.watched_log).resolve()
    final_summary = (project_root / args.launcher_results_dir / "comparison_summary.csv").resolve()
    launcher_script = (project_root / args.launcher_script).resolve()

    if final_summary.exists():
        log(f"MP summary already exists, nothing to do: {final_summary}")
        return
    if process_running(args.launcher_process_pattern):
        log(f"MP launcher is already running: {args.launcher_process_pattern}")
        return

    log(f"queue started; waiting for E13 metrics at {watched_metrics}")
    while True:
        metrics_ready = watched_metrics.exists()
        process_active = process_running(args.watched_process_pattern)
        if metrics_ready and not process_active:
            log("E13 metrics detected; starting MP standard suite")
            break

        if not process_active and not metrics_ready:
            log("E13 process is no longer running but metrics.json was not found")
            if watched_log.exists():
                try:
                    tail_lines = watched_log.read_text(encoding="utf-8", errors="ignore").splitlines()[-40:]
                    for line in tail_lines:
                        print(line, flush=True)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    log(f"failed to read E13 log tail: {exc}")
            raise RuntimeError("E13 stopped before producing metrics; refusing to start MP automatically")

        if metrics_ready and process_active:
            log(f"E13 metrics are present but the process is still running; sleeping {args.poll_seconds} seconds")
        else:
            log(f"E13 still running; sleeping {args.poll_seconds} seconds")
        time.sleep(args.poll_seconds)

    env = os.environ.copy()
    env["MP_API_KEY"] = args.api_key
    command = [
        args.python,
        str(launcher_script),
        "--dataset_dir",
        args.launcher_dataset_dir,
        "--results_dir",
        args.launcher_results_dir,
    ]
    if args.launcher_config_args:
        command.extend(args.launcher_config_args)

    log(f"launching MP suite with command: {' '.join(command)}")
    subprocess.run(command, cwd=str(project_root), env=env, check=True)
    log("MP standard suite completed")


if __name__ == "__main__":
    main()
