from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor remote faithful training and escalate unknown failures to Codex.")
    parser.add_argument("--host", required=True)
    parser.add_argument("--port", type=int, default=22)
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", default=os.environ.get("REMOTE_SSH_PASSWORD"))
    parser.add_argument("--project-root", default="/root/lanyun-fs/gnn")
    parser.add_argument("--interval-seconds", type=int, default=1800, help="Heartbeat interval while remote training is healthy.")
    parser.add_argument(
        "--incident-interval-seconds",
        type=int,
        default=300,
        help="Polling interval while an unknown-failure bundle exists or a Codex triage run is active.",
    )
    parser.add_argument("--state-path", default="monitoring/remote_monitor_state.json")
    parser.add_argument("--incidents-path", default="monitoring/remote_monitor_incidents.jsonl")
    parser.add_argument("--bundle-dir", default="monitoring/unknown_failures")
    parser.add_argument("--codex-working-dir", default=".")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument("--codex-model", default=None)
    parser.add_argument("--codex-run-dir", default="monitoring/codex_runs")
    parser.add_argument("--ssh-retries", type=int, default=5)
    parser.add_argument("--ssh-retry-sleep-seconds", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "events": [],
        }
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(state, handle, ensure_ascii=False, indent=2)


def append_incident(path: Path, incident: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(incident, ensure_ascii=False) + "\n")


def append_event(state: dict[str, Any], message: str) -> None:
    events = state.setdefault("events", [])
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    events.append(f"{timestamp} | {message}")
    state["events"] = events[-200:]


def print_event(payload: dict[str, Any]) -> None:
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def open_ssh_client(args: argparse.Namespace) -> paramiko.SSHClient:
    if not args.password:
        raise ValueError("SSH password is required via --password or REMOTE_SSH_PASSWORD.")
    last_error: Exception | None = None
    for attempt in range(1, args.ssh_retries + 1):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                hostname=args.host,
                port=args.port,
                username=args.username,
                password=args.password,
                timeout=20,
                banner_timeout=60,
                auth_timeout=30,
            )
            return client
        except Exception as exc:  # pragma: no cover - depends on remote SSH behavior.
            last_error = exc
            try:
                client.close()
            except Exception:
                pass
            if attempt == args.ssh_retries:
                break
            time.sleep(args.ssh_retry_sleep_seconds)
    raise RuntimeError(f"failed to connect to SSH after {args.ssh_retries} attempt(s): {last_error}") from last_error


def run_remote_command(client: paramiko.SSHClient, command: str, timeout: int = 60) -> str:
    stdin, stdout, stderr = client.exec_command(command, timeout=timeout)
    output = stdout.read().decode("utf-8", "replace")
    error = stderr.read().decode("utf-8", "replace")
    if error.strip():
        return f"{output}\n[stderr]\n{error}".strip()
    return output


def parse_json_blob(raw_text: str) -> dict[str, Any]:
    raw_text = raw_text.strip()
    if not raw_text:
        return {}
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        return {"_parse_error": raw_text[-1000:]}


def latest_incomplete_phase(state: dict[str, Any]) -> str | None:
    phases = state.get("phases", {})
    candidates: list[tuple[str, str]] = []
    for phase_name, phase_state in phases.items():
        if phase_state.get("completed"):
            continue
        candidates.append((str(phase_state.get("last_started_at", "")), phase_name))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def state_all_phases_completed(state: dict[str, Any]) -> bool:
    phases = state.get("phases", {})
    return bool(phases) and all(bool(phase_state.get("completed")) for phase_state in phases.values())


def sanitize_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "unknown"


def bundle_key(bundle: dict[str, Any]) -> str:
    return "|".join(
        [
            str(bundle.get("detected_at", "")),
            str(bundle.get("phase", "")),
            str(bundle.get("error_excerpt", ""))[:240],
        ]
    )


def is_process_alive(pid: int | None) -> bool:
    if not pid or pid <= 0:
        return False
    if os.name == "nt":
        probe = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                f"$p = Get-Process -Id {pid} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return probe.returncode == 0 and "alive" in probe.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def read_text_if_exists(path: Path, limit: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-limit:]


def build_snapshot(client: paramiko.SSHClient, project_root: str) -> dict[str, Any]:
    log_root = f"{project_root}/logs"
    commands = {
        "remote_time": "date \"+%F %T %Z\"",
        "summary_exists": f"test -f {project_root}/results/comparison_summary.csv && echo yes || echo no",
        "watchdog_ps": f"ps -ef | grep 'training_watchdog.py --project-root {project_root}' | grep -v grep || true",
        "supervisor_ps": f"ps -ef | grep 'supervise_training.py --project-root {project_root}' | grep -v grep || true",
        "runner_ps": (
            "ps -eo pid=,ppid=,etimes=,cmd= | "
            f"grep '{project_root}' | "
            "grep -E 'run_faithful_experiment.py|export_official_dataset.py|summarize.py' | grep -v grep || true"
        ),
        "watchdog_state": f"cat {log_root}/watchdog_state.json 2>/dev/null || true",
        "supervisor_state": f"cat {log_root}/supervisor_state.json 2>/dev/null || true",
        "unknown_bundle": f"cat {log_root}/watchdog_unknown_failure.json 2>/dev/null || true",
        "watchdog_log": f"tail -c 16000 {log_root}/watchdog.log 2>/dev/null || true",
        "supervisor_log": f"tail -c 16000 {log_root}/supervisor.log 2>/dev/null || true",
        "results": f"find {project_root}/results -maxdepth 2 -name metrics.json -o -name comparison_summary.csv 2>/dev/null | sort || true",
        "gpu": "nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true",
    }
    raw: dict[str, str] = {}
    for key, command in commands.items():
        raw[key] = run_remote_command(client, command, timeout=120)

    supervisor_state = parse_json_blob(raw["supervisor_state"])
    watchdog_state = parse_json_blob(raw["watchdog_state"])
    unknown_bundle = parse_json_blob(raw["unknown_bundle"])
    active_phase = None
    if unknown_bundle.get("phase"):
        active_phase = str(unknown_bundle["phase"])
    if not active_phase:
        active_phase = latest_incomplete_phase(supervisor_state)
    if not active_phase:
        active_phase = latest_incomplete_phase(watchdog_state)

    phase_log_tail = ""
    if active_phase:
        phase_log_tail = run_remote_command(
            client,
            f"tail -c 16000 {log_root}/{active_phase}.log 2>/dev/null || true",
            timeout=120,
        )

    return {
        "checked_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "remote_time": raw["remote_time"].strip(),
        "summary_exists": raw["summary_exists"].strip() == "yes",
        "watchdog_running": bool(raw["watchdog_ps"].strip()),
        "watchdog_ps": raw["watchdog_ps"].strip(),
        "supervisor_running": bool(raw["supervisor_ps"].strip()),
        "supervisor_ps": raw["supervisor_ps"].strip(),
        "runner_processes": [line for line in raw["runner_ps"].splitlines() if line.strip()],
        "watchdog_state": watchdog_state,
        "supervisor_state": supervisor_state,
        "unknown_bundle": unknown_bundle,
        "watchdog_log_tail": raw["watchdog_log"][-16000:],
        "supervisor_log_tail": raw["supervisor_log"][-16000:],
        "active_phase": active_phase,
        "active_phase_log_tail": phase_log_tail[-16000:],
        "gpu": raw["gpu"].strip(),
        "results": [line for line in raw["results"].splitlines() if line.strip()],
    }


def snapshot_all_phases_completed(snapshot: dict[str, Any]) -> bool:
    active_work = bool(
        snapshot.get("watchdog_running")
        or snapshot.get("supervisor_running")
        or snapshot.get("runner_processes")
        or snapshot.get("active_phase")
        or snapshot.get("unknown_bundle")
    )
    if active_work:
        return False
    return state_all_phases_completed(snapshot.get("supervisor_state", {})) or state_all_phases_completed(
        snapshot.get("watchdog_state", {})
    )


def archive_unknown_bundle(base_dir: Path, snapshot: dict[str, Any]) -> Path:
    bundle = snapshot["unknown_bundle"]
    detected_at = sanitize_name(str(bundle.get("detected_at", snapshot["checked_at"])).replace(" ", "_").replace(":", ""))
    phase_name = sanitize_name(str(bundle.get("phase", "unknown")))
    archive_dir = base_dir / f"{detected_at}_{phase_name}"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / "bundle.json").write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (archive_dir / "snapshot.json").write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (archive_dir / "supervisor.log.tail.txt").write_text(snapshot.get("supervisor_log_tail", ""), encoding="utf-8")
    (archive_dir / "watchdog.log.tail.txt").write_text(snapshot.get("watchdog_log_tail", ""), encoding="utf-8")
    (archive_dir / "phase.log.tail.txt").write_text(snapshot.get("active_phase_log_tail", ""), encoding="utf-8")
    return archive_dir


def build_codex_prompt(
    archive_dir: Path,
    snapshot: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    bundle = snapshot["unknown_bundle"]
    return textwrap.dedent(
        f"""
        Background task: monitor the remote training project at {args.project_root} on {args.host}:{args.port} for unknown failures. You are not alone in the codebase; do not revert others' edits.

        A watchdog unknown-failure bundle is waiting now.

        Local workspace:
        - repo root: {Path(args.codex_working_dir).resolve()}
        - archived bundle: {archive_dir / "bundle.json"}
        - archived snapshot: {archive_dir / "snapshot.json"}
        - archived phase tail: {archive_dir / "phase.log.tail.txt"}
        - archived supervisor tail: {archive_dir / "supervisor.log.tail.txt"}

        Remote access:
        - host: {args.host}
        - port: {args.port}
        - user: {args.username}
        - password env var: REMOTE_SSH_PASSWORD
        - remote project root: {args.project_root}

        Current bundle summary:
        - detected_at: {bundle.get("detected_at", "")}
        - phase: {bundle.get("phase", "")}
        - error_excerpt: {bundle.get("error_excerpt", "")}
        - remote_time_when_polled: {snapshot.get("remote_time", "")}

        Your responsibility is remote unknown-failure triage and any required code changes to restore training. Inspect that bundle plus current logs, debug the failure using this repo, edit files directly in your workspace as needed, upload the fix to the remote host, restart the relevant training/watchdog processes, and continue monitoring until the concrete unknown failure is resolved or you are blocked.

        Requirements:
        - Avoid touching unrelated files.
        - Do not revert others' edits.
        - Use the repo in this workspace; do not assume the archived bundle is still current without checking the remote logs.
        - If the failure already self-healed and training is healthy again, verify that and then exit without unnecessary edits.
        - Only send a final message when you either need the main agent because you are blocked or you have resolved a concrete unknown failure.
        - In that final message include exact times, what failed, what you changed, and file paths changed.
        """
    ).strip()


def spawn_codex_triage(
    args: argparse.Namespace,
    snapshot: dict[str, Any],
    archive_dir: Path,
) -> dict[str, Any]:
    run_dir = Path(args.codex_run_dir).resolve() / archive_dir.name
    run_dir.mkdir(parents=True, exist_ok=True)

    prompt_path = run_dir / "prompt.txt"
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    last_message_path = run_dir / "last_message.txt"

    prompt_text = build_codex_prompt(archive_dir=archive_dir, snapshot=snapshot, args=args)
    prompt_path.write_text(prompt_text, encoding="utf-8")

    command = [
        args.codex_bin,
        "exec",
        "--dangerously-bypass-approvals-and-sandbox",
        "-C",
        str(Path(args.codex_working_dir).resolve()),
        "--output-last-message",
        str(last_message_path),
        "--color",
        "never",
    ]
    if args.codex_model:
        command.extend(["-m", args.codex_model])

    env = os.environ.copy()
    if args.password:
        env["REMOTE_SSH_PASSWORD"] = args.password

    prompt_handle = prompt_path.open("r", encoding="utf-8")
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    try:
        process = subprocess.Popen(
            command,
            cwd=Path(args.codex_working_dir).resolve(),
            stdin=prompt_handle,
            stdout=stdout_handle,
            stderr=stderr_handle,
            text=True,
            env=env,
        )
    finally:
        prompt_handle.close()
        stdout_handle.close()
        stderr_handle.close()

    return {
        "pid": process.pid,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "bundle_key": bundle_key(snapshot["unknown_bundle"]),
        "archive_dir": str(archive_dir),
        "prompt_path": str(prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "last_message_path": str(last_message_path),
    }


def build_incident(snapshot: dict[str, Any], incident_type: str, message: str, **extra: Any) -> dict[str, Any]:
    payload = {
        "type": incident_type,
        "message": message,
        "checked_at": snapshot["checked_at"],
        "remote_time": snapshot.get("remote_time"),
        "active_phase": snapshot.get("active_phase"),
        "watchdog_running": snapshot.get("watchdog_running"),
        "supervisor_running": snapshot.get("supervisor_running"),
        "runner_processes": snapshot.get("runner_processes"),
        "gpu": snapshot.get("gpu"),
        "results": snapshot.get("results"),
    }
    payload.update(extra)
    return payload


def summarize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    bundle = snapshot.get("unknown_bundle") or {}
    return {
        "checked_at": snapshot.get("checked_at"),
        "remote_time": snapshot.get("remote_time"),
        "summary_exists": snapshot.get("summary_exists"),
        "watchdog_running": snapshot.get("watchdog_running"),
        "supervisor_running": snapshot.get("supervisor_running"),
        "active_phase": snapshot.get("active_phase"),
        "runner_process_count": len(snapshot.get("runner_processes", [])),
        "results_count": len(snapshot.get("results", [])),
        "gpu": snapshot.get("gpu"),
        "unknown_bundle_phase": bundle.get("phase"),
        "unknown_bundle_detected_at": bundle.get("detected_at"),
        "unknown_bundle_error_excerpt": bundle.get("error_excerpt"),
    }


def finalize_triage_if_needed(
    state: dict[str, Any],
    incidents_path: Path,
    snapshot: dict[str, Any],
) -> None:
    triage = state.get("triage")
    if not triage:
        return
    pid = int(triage.get("pid", 0))
    if is_process_alive(pid):
        return

    last_message = read_text_if_exists(Path(triage.get("last_message_path", "")))
    state["last_triage"] = {
        **triage,
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "last_message_excerpt": last_message,
    }
    append_incident(
        incidents_path,
        build_incident(
            snapshot,
            "codex_triage_finished",
            f"Codex triage pid {pid} finished.",
            bundle_key=triage.get("bundle_key"),
            archive_dir=triage.get("archive_dir"),
            last_message_excerpt=last_message,
        ),
    )
    state["triage"] = None
    append_event(state, f"codex triage pid {pid} finished")


def launch_if_needed(
    args: argparse.Namespace,
    state: dict[str, Any],
    incidents_path: Path,
    bundle_dir: Path,
    snapshot: dict[str, Any],
) -> None:
    bundle = snapshot.get("unknown_bundle") or {}
    if not bundle:
        return

    current_key = bundle_key(bundle)
    triage = state.get("triage")
    if triage and is_process_alive(int(triage.get("pid", 0))) and triage.get("bundle_key") == current_key:
        return

    archive_dir = archive_unknown_bundle(bundle_dir, snapshot)
    triage_info = spawn_codex_triage(args=args, snapshot=snapshot, archive_dir=archive_dir)
    state["triage"] = triage_info
    state["last_bundle_key"] = current_key
    append_event(state, f"spawned codex triage pid {triage_info['pid']} for {archive_dir.name}")
    append_incident(
        incidents_path,
        build_incident(
            snapshot,
            "unknown_failure_detected",
            "Spawned Codex triage for watchdog unknown-failure bundle.",
            bundle_key=current_key,
            archive_dir=str(archive_dir),
            codex_pid=triage_info["pid"],
        ),
    )


def next_sleep_seconds(state: dict[str, Any], snapshot: dict[str, Any], args: argparse.Namespace) -> int:
    triage = state.get("triage")
    if snapshot.get("unknown_bundle"):
        return args.incident_interval_seconds
    if triage and is_process_alive(int(triage.get("pid", 0))):
        return args.incident_interval_seconds
    return args.interval_seconds


def main() -> None:
    args = parse_args()
    state_path = Path(args.state_path)
    incidents_path = Path(args.incidents_path)
    bundle_dir = Path(args.bundle_dir)
    state = load_state(state_path)

    while True:
        client = open_ssh_client(args)
        try:
            snapshot = build_snapshot(client, args.project_root)
        finally:
            client.close()

        state["last_checked_at"] = snapshot["checked_at"]
        state["last_snapshot"] = summarize_snapshot(snapshot)
        finalize_triage_if_needed(state=state, incidents_path=incidents_path, snapshot=snapshot)

        if snapshot_all_phases_completed(snapshot):
            append_event(state, "remote all phases completed; monitor exiting")
            append_incident(
                incidents_path,
                build_incident(snapshot, "complete", "Remote training completed all phases and exited cleanly."),
            )
            save_state(state_path, state)
            print_event({"event": "complete", **summarize_snapshot(snapshot)})
            return

        launch_if_needed(
            args=args,
            state=state,
            incidents_path=incidents_path,
            bundle_dir=bundle_dir,
            snapshot=snapshot,
        )

        triage = state.get("triage")
        triage_alive = bool(triage and is_process_alive(int(triage.get("pid", 0))))
        print_event(
            {
                "event": "heartbeat",
                **summarize_snapshot(snapshot),
                "triage_pid": triage.get("pid") if triage else None,
                "triage_alive": triage_alive,
            }
        )
        save_state(state_path, state)

        if args.once:
            return
        time.sleep(next_sleep_seconds(state=state, snapshot=snapshot, args=args))


if __name__ == "__main__":
    main()
