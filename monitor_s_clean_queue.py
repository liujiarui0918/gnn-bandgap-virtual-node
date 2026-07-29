from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor S-series clean matched training queue.")
    parser.add_argument("--project_root", default="/data/home/tg100025/gnn")
    parser.add_argument("--results_dir", default="/data/home/tg100025/gnn_work/results/mp_hzw_clean_matched_s")
    parser.add_argument("--dataset_dir", default="/data/home/tg100025/gnn_work/data/mp_hzw_clean_matched_s")
    parser.add_argument("--audit_dir", default="/data/home/tg100025/gnn_work/results/data_quality")
    parser.add_argument("--job_file", default="/data/home/tg100025/gnn_work/run/s_clean_queue.jobid")
    parser.add_argument("--interval_seconds", type=int, default=1800)
    parser.add_argument("--log_path", default="/data/home/tg100025/gnn_work/logs/s_clean_queue_monitor.log")
    parser.add_argument("--state_path", default="/data/home/tg100025/gnn_work/logs/s_clean_queue_monitor_state.json")
    parser.add_argument("--error_report_path", default="/data/home/tg100025/gnn_work/logs/last_error_report.md")
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def resolve_path(project_root: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate


def run(command: list[str], timeout: int = 30) -> str:
    try:
        completed = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except Exception as exc:
        return f"command_failed: {exc}"
    text = completed.stdout.strip()
    if completed.stderr.strip():
        text = f"{text}\n[stderr]\n{completed.stderr.strip()}".strip()
    return text


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def read_job_id(path: Path) -> str | None:
    text = read_text(path).strip()
    return text or None


def tail(path: Path, line_count: int = 80) -> list[str]:
    text = read_text(path)
    return text.splitlines()[-line_count:] if text else []


def read_json(path: Path) -> dict[str, Any] | None:
    text = read_text(path)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def read_summary(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def slurm_job_snapshot(job_id: str | None) -> dict[str, Any]:
    if not job_id:
        return {"job_id": None, "squeue": "", "sacct": ""}
    squeue = run(["bash", "-lc", f"squeue -j {job_id} -o '%i %P %j %u %t %M %D %R' 2>&1 || true"], timeout=20)
    sacct = run(["bash", "-lc", f"sacct -j {job_id} --format=JobID,JobName,State,Elapsed,MaxRSS,ExitCode -P 2>&1 | tail -20 || true"], timeout=20)
    return {"job_id": job_id, "squeue": squeue, "sacct": sacct}


def job_running(slurm: dict[str, Any]) -> bool:
    squeue = str(slurm.get("squeue") or "")
    job_id = str(slurm.get("job_id") or "")
    return bool(job_id and re.search(rf"(^|\n){re.escape(job_id)}\s", squeue))


def latest_logs(results_dir: Path) -> list[Path]:
    logs = sorted((results_dir / "queue_logs").glob("*.log"), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    slurm_logs = sorted(results_dir.glob("*.out"), key=lambda path: path.stat().st_mtime if path.exists() else 0)
    return (logs + slurm_logs)[-5:]


def extract_error_report(results_dir: Path, failures: list[str]) -> str:
    lines = ["# Last S queue error report", ""]
    if failures:
        lines.extend(["## Failure files", ""])
        for failure in failures:
            lines.append(f"- `{failure}`")
    lines.extend(["", "## Recent error lines", ""])
    for log_path in latest_logs(results_dir):
        log_lines = tail(log_path, line_count=220)
        interesting = [
            line
            for line in log_lines
            if re.search(r"error|exception|traceback|out of memory|killed|failed", line, re.IGNORECASE)
        ]
        if not interesting:
            interesting = log_lines[-40:]
        lines.append(f"### {log_path}")
        lines.append("")
        lines.append("```text")
        lines.extend(interesting[-120:])
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def snapshot(args: argparse.Namespace) -> dict[str, Any]:
    project_root = Path(args.project_root).resolve()
    results_dir = resolve_path(project_root, args.results_dir)
    dataset_dir = resolve_path(project_root, args.dataset_dir)
    audit_dir = resolve_path(project_root, args.audit_dir)
    job_file = resolve_path(project_root, args.job_file)
    job_id = read_job_id(job_file)
    slurm = slurm_job_snapshot(job_id)

    metrics = sorted(str(path) for path in results_dir.glob("*/metrics.json"))
    failures = sorted(str(path) for path in results_dir.glob("*/failure.json"))
    summary_rows = read_summary(results_dir / "comparison_summary.csv")
    metadata = read_json(dataset_dir / "metadata.json")
    audit_summary = read_json(audit_dir / "mp_hzw_garbage_summary.json")
    running = job_running(slurm)

    status = "running"
    if failures:
        status = "has_failures"
    elif summary_rows:
        status = "completed"
    elif not running:
        status = "stopped_before_completion"

    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": status,
        "slurm": slurm,
        "metrics_count": len(metrics),
        "metrics": metrics,
        "failures": failures,
        "summary_rows": summary_rows,
        "dataset_metadata": metadata,
        "audit_summary": audit_summary,
        "disk": run(["bash", "-lc", "df -hT / /data /tmp 2>/dev/null || true"], timeout=20),
        "processes": run(["bash", "-lc", "ps -u $USER -o pid,ppid,stat,etime,pcpu,pmem,rss,cmd | grep -E 'run_s_clean|audit_mp_hzw|export_mp_hzw|run_faithful|python' | grep -v grep || true"], timeout=20),
    }


def append_log(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    log_path = Path(args.log_path)
    state_path = Path(args.state_path)
    error_report_path = Path(args.error_report_path)
    while True:
        payload = snapshot(args)
        append_log(log_path, payload)
        write_json(state_path, payload)
        print(json.dumps(payload, ensure_ascii=False), flush=True)
        if payload["status"] in {"has_failures", "stopped_before_completion"}:
            report = extract_error_report(Path(args.results_dir), payload.get("failures", []))
            error_report_path.parent.mkdir(parents=True, exist_ok=True)
            error_report_path.write_text(report, encoding="utf-8")
        if args.once or payload["status"] in {"completed", "has_failures", "stopped_before_completion"}:
            break
        time.sleep(max(30, int(args.interval_seconds)))


if __name__ == "__main__":
    main()
