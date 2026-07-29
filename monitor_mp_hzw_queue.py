from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lightweight monitor for the MP-HZW experiment queue.")
    parser.add_argument("--project_root", default="/root/lanyun-fs/gnn")
    parser.add_argument("--results_dir", default="results/mp_hzw_matched_e1_e12")
    parser.add_argument("--dataset_dir", default="data/mp_hzw_matched")
    parser.add_argument("--pid_file", default="logs/mp_hzw_full_queue.pid")
    parser.add_argument("--interval_seconds", type=int, default=1800)
    parser.add_argument("--log_path", default="logs/mp_hzw_queue_monitor.log")
    parser.add_argument("--state_path", default="logs/mp_hzw_queue_monitor_state.json")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run(command: list[str], timeout: int = 30) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return f"command_failed: {exc}"
    text = completed.stdout.strip()
    if completed.stderr.strip():
        text = f"{text}\n[stderr]\n{completed.stderr.strip()}".strip()
    return text


def process_alive(pid: int) -> bool:
    return subprocess.run(["kill", "-0", str(pid)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0


def read_pid(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def tail(path: Path, line_count: int = 60) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-line_count:]


def latest_export_progress(log_path: Path) -> dict[str, Any] | None:
    for line in reversed(tail(log_path, line_count=500)):
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        if payload.get("event") == "mp_fetch_progress":
            return payload
    return None


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    results_dir = project_root / args.results_dir
    dataset_dir = project_root / args.dataset_dir
    pid_path = project_root / args.pid_file
    logs_dir = results_dir / "queue_logs"
    pid = read_pid(pid_path)
    queue_alive = bool(pid and process_alive(pid))
    metrics = sorted(str(path.relative_to(project_root)) for path in results_dir.glob("*/metrics.json"))
    failures = sorted(str(path.relative_to(project_root)) for path in results_dir.glob("*/failure.json"))
    summary_rows = read_summary(results_dir / "comparison_summary.csv")
    active_processes = run(
        [
            "bash",
            "-lc",
            "ps -eo pid,ppid,stat,etime,pcpu,pmem,rss,cmd | "
            "grep -E 'run_mp_hzw|export_mp_hzw|run_faithful|train.py|python' | grep -v grep || true",
        ],
        timeout=20,
    )
    gpu = run(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu "
            "--format=csv,noheader 2>/dev/null || true",
        ],
        timeout=20,
    )
    disk = run(["bash", "-lc", "df -hT / /root/lanyun-fs /root/lanyun-tmp 2>/dev/null || true"], timeout=20)
    status = "running"
    if summary_rows:
        status = "completed"
    elif failures:
        status = "has_failures"
    elif not queue_alive:
        status = "stopped_before_completion"

    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "queue_pid": pid,
        "queue_alive": queue_alive,
        "export_progress": latest_export_progress(logs_dir / "export_dataset.log"),
        "metadata": load_json(dataset_dir / "metadata.json"),
        "metrics_count": len(metrics),
        "metrics": metrics,
        "failures": failures,
        "summary_rows": summary_rows,
        "active_processes": active_processes,
        "gpu": gpu,
        "disk": disk,
    }


def append_log(log_path: Path, payload: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_state(state_path: Path, payload: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    log_path = project_root / args.log_path
    state_path = project_root / args.state_path
    while True:
        payload = snapshot(args)
        append_log(log_path, payload)
        write_state(state_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if args.once or payload["status"] in {"completed", "stopped_before_completion"}:
            break
        time.sleep(max(30, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
