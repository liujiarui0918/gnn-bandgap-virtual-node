from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - watchdog only runs on Linux remotely.
    fcntl = None


@dataclass
class PhaseSpec:
    name: str
    completion_path: Path
    log_path: Path
    output_dir: Path | None = None
    alignn_family: bool = False


@dataclass
class ProcessInfo:
    pid: int
    ppid: int
    etimes: int
    cmd: str
    role: str


class TrainingWatchdog:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.project_root / "results"
        self.summary_path = self.results_dir / "comparison_summary.csv"
        self.dataset_dir = self.project_root / "data" / "official_dataset"
        self.alignn_network_cache = self.dataset_dir / "alignn_cache"
        self.alignn_local_cache_root = Path(args.alignn_local_cache_root).resolve()
        self.alignn_local_cache_root.mkdir(parents=True, exist_ok=True)
        self.log_path = self.logs_dir / "watchdog.log"
        self.state_path = self.logs_dir / "watchdog_state.json"
        self.launch_log_path = self.logs_dir / "watchdog_supervisor.out"
        self.lock_path = self.logs_dir / "watchdog.lock"
        self.unknown_failure_path = self.logs_dir / "watchdog_unknown_failure.json"
        self.default_python = Path(args.default_python or sys.executable).resolve()
        self.alignn_python = Path(args.alignn_python or self.default_python).resolve()
        self.m3gnet_python = Path(args.m3gnet_python or self.default_python).resolve()
        self.state = self._load_state()
        self.lock_handle = None
        self.phases = self._build_phases()

    def _build_phases(self) -> list[PhaseSpec]:
        return [
            PhaseSpec(
                name="preprocess",
                completion_path=self.dataset_dir / "metadata.json",
                log_path=self.logs_dir / "preprocess.log",
            ),
            PhaseSpec(
                name="E1_CGCNN_baseline",
                completion_path=self.results_dir / "E1_CGCNN_baseline" / "metrics.json",
                log_path=self.logs_dir / "E1_CGCNN_baseline.log",
                output_dir=self.results_dir / "E1_CGCNN_baseline",
            ),
            PhaseSpec(
                name="E2_ALIGNN_baseline",
                completion_path=self.results_dir / "E2_ALIGNN_baseline" / "metrics.json",
                log_path=self.logs_dir / "E2_ALIGNN_baseline.log",
                output_dir=self.results_dir / "E2_ALIGNN_baseline",
                alignn_family=True,
            ),
            PhaseSpec(
                name="E3_M3GNet_baseline",
                completion_path=self.results_dir / "E3_M3GNet_baseline" / "metrics.json",
                log_path=self.logs_dir / "E3_M3GNet_baseline.log",
                output_dir=self.results_dir / "E3_M3GNet_baseline",
            ),
            PhaseSpec(
                name="E4_ALIGNN_virtualnode",
                completion_path=self.results_dir / "E4_ALIGNN_virtualnode" / "metrics.json",
                log_path=self.logs_dir / "E4_ALIGNN_virtualnode.log",
                output_dir=self.results_dir / "E4_ALIGNN_virtualnode",
                alignn_family=True,
            ),
            PhaseSpec(
                name="E5_ALIGNN_transformer",
                completion_path=self.results_dir / "E5_ALIGNN_transformer" / "metrics.json",
                log_path=self.logs_dir / "E5_ALIGNN_transformer.log",
                output_dir=self.results_dir / "E5_ALIGNN_transformer",
                alignn_family=True,
            ),
            PhaseSpec(
                name="E6_ALIGNN_physconstraint",
                completion_path=self.results_dir / "E6_ALIGNN_physconstraint" / "metrics.json",
                log_path=self.logs_dir / "E6_ALIGNN_physconstraint.log",
                output_dir=self.results_dir / "E6_ALIGNN_physconstraint",
                alignn_family=True,
            ),
            PhaseSpec(
                name="E7_ALIGNN_attentive_virtualnode",
                completion_path=self.results_dir / "E7_ALIGNN_attentive_virtualnode" / "metrics.json",
                log_path=self.logs_dir / "E7_ALIGNN_attentive_virtualnode.log",
                output_dir=self.results_dir / "E7_ALIGNN_attentive_virtualnode",
                alignn_family=True,
            ),
            PhaseSpec(
                name="E8_ALIGNN_multi_virtualnode",
                completion_path=self.results_dir / "E8_ALIGNN_multi_virtualnode" / "metrics.json",
                log_path=self.logs_dir / "E8_ALIGNN_multi_virtualnode.log",
                output_dir=self.results_dir / "E8_ALIGNN_multi_virtualnode",
                alignn_family=True,
            ),
            PhaseSpec(
                name="summarize",
                completion_path=self.summary_path,
                log_path=self.logs_dir / "summarize.log",
            ),
        ]

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "checks": 0,
            "launches": 0,
            "events": [],
        }

    def _save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, ensure_ascii=False)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {message}"
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        self.state["last_check_at"] = timestamp
        self.state.setdefault("events", []).append(line)
        self.state["events"] = self.state["events"][-200:]
        self._save_state()

    def _acquire_lock(self) -> None:
        if fcntl is None:
            return
        self.lock_handle = self.lock_path.open("w", encoding="utf-8")
        try:
            fcntl.flock(self.lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("watchdog already running")
        self.lock_handle.write(str(os.getpid()))
        self.lock_handle.flush()

    def _tail_text(self, path: Path, line_count: int = 120) -> str:
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-line_count:])

    def _last_textual_line(self, text: str) -> str:
        cleaned = text.replace("\r", "\n")
        for line in reversed(cleaned.splitlines()):
            stripped = line.strip()
            if any(ch.isalpha() for ch in stripped):
                return stripped
        return ""

    def _clear_unknown_failure(self, reason: str) -> bool:
        if not self.unknown_failure_path.exists():
            return False
        self.unknown_failure_path.unlink(missing_ok=True)
        self._log(f"cleared pending unknown failure bundle: {reason}")
        return True

    def _write_unknown_failure_bundle(
        self,
        phase: PhaseSpec | None,
        completed: list[str],
        processes: list[ProcessInfo],
        error_excerpt: str,
    ) -> str:
        phase_name = phase.name if phase is not None else "unknown"
        phase_tail = self._tail_text(phase.log_path, line_count=200) if phase is not None else ""
        supervisor_tail = self._tail_text(self.logs_dir / "supervisor.log", line_count=200)
        payload = {
            "status": "pending_codex_debug",
            "detected_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phase": phase_name,
            "completed_phases": completed,
            "error_excerpt": error_excerpt,
            "active_processes": [
                {
                    "pid": proc.pid,
                    "ppid": proc.ppid,
                    "etimes": proc.etimes,
                    "role": proc.role,
                    "cmd": proc.cmd,
                }
                for proc in processes
            ],
            "phase_log_path": str(phase.log_path) if phase is not None else None,
            "supervisor_log_path": str(self.logs_dir / "supervisor.log"),
            "phase_log_tail": phase_tail[-12000:],
            "supervisor_log_tail": supervisor_tail[-12000:],
        }
        with self.unknown_failure_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return phase_name

    def _completed_phase_names(self) -> list[str]:
        return [phase.name for phase in self.phases if phase.completion_path.exists()]

    def _current_phase(self) -> PhaseSpec | None:
        for phase in self.phases:
            if not phase.completion_path.exists():
                return phase
        return None

    def _all_phases_completed(self) -> bool:
        return self._current_phase() is None

    def _phase_local_cache_dir(self, phase: PhaseSpec) -> Path:
        return self.alignn_local_cache_root / phase.name

    def _remove_tree(self, path: Path | None, actions: list[str]) -> None:
        if path is None or not path.exists():
            return
        shutil.rmtree(path, ignore_errors=True)
        actions.append(f"removed {path}")

    def _cleanup_completed_alignn_caches(self) -> list[str]:
        actions: list[str] = []
        for phase in self.phases:
            if not phase.alignn_family or not phase.completion_path.exists():
                continue
            self._remove_tree(self._phase_local_cache_dir(phase), actions)
        return actions

    def _free_gib(self, path: Path) -> float:
        probe = path if path.exists() else path.parent
        while not probe.exists():
            probe = probe.parent
        return shutil.disk_usage(probe).free / (1024**3)

    def _project_processes(self) -> list[ProcessInfo]:
        process = subprocess.run(
            ["ps", "-eo", "pid=,ppid=,etimes=,cmd="],
            capture_output=True,
            text=True,
            check=True,
        )
        rows: list[ProcessInfo] = []
        needles = ("supervise_training.py", "run_faithful_experiment.py", "export_official_dataset.py", "summarize.py")
        for line in process.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parts = stripped.split(None, 3)
            if len(parts) != 4:
                continue
            pid, ppid, etimes, cmd = parts
            if "training_watchdog.py" in cmd:
                continue
            if str(self.project_root) not in cmd or not any(needle in cmd for needle in needles):
                continue
            role = "supervisor" if "supervise_training.py" in cmd else "runner"
            rows.append(
                ProcessInfo(
                    pid=int(pid),
                    ppid=int(ppid),
                    etimes=int(etimes),
                    cmd=cmd,
                    role=role,
                )
            )
        return rows

    def _kill(self, pid: int, sig: int) -> bool:
        try:
            os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False

    def _cleanup_duplicate_supervisors(self, processes: list[ProcessInfo]) -> list[str]:
        supervisors = [proc for proc in processes if proc.role == "supervisor"]
        if len(supervisors) <= 1:
            return []
        keep_pid = min(proc.pid for proc in supervisors)
        runner_parent_ids = {proc.ppid for proc in processes if proc.role == "runner"}
        matched_parent_ids = sorted(runner_parent_ids.intersection({proc.pid for proc in supervisors}))
        if len(matched_parent_ids) == 1:
            keep_pid = matched_parent_ids[0]
        actions: list[str] = []
        for proc in supervisors:
            if proc.pid == keep_pid:
                continue
            if self._kill(proc.pid, signal.SIGTERM):
                actions.append(f"terminated duplicate supervisor pid {proc.pid}")
        return actions

    def _terminate_project_processes(self, processes: list[ProcessInfo]) -> list[str]:
        actions: list[str] = []
        for proc in sorted(processes, key=lambda item: (item.role, item.pid), reverse=True):
            if self._kill(proc.pid, signal.SIGTERM):
                actions.append(f"sent SIGTERM to {proc.role} pid {proc.pid}")
        time.sleep(3)
        for proc in sorted(processes, key=lambda item: (item.role, item.pid), reverse=True):
            if self._kill(proc.pid, 0):
                if self._kill(proc.pid, signal.SIGKILL):
                    actions.append(f"sent SIGKILL to stuck {proc.role} pid {proc.pid}")
        return actions

    def _current_error_excerpt(self, phase: PhaseSpec | None) -> str:
        if phase is not None and not phase.log_path.exists():
            return ""
        candidates = [self.logs_dir / "supervisor.log"]
        if phase is not None and phase.log_path.exists():
            candidates.insert(0, phase.log_path)
        for path in candidates:
            tail = self._tail_text(path)
            cleaned = tail.replace("\r", "\n")
            lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
            excerpt_lines = [line for line in lines if any(token in line.lower() for token in ("traceback", "error", "exception", "failed"))]
            if excerpt_lines:
                return " | ".join(excerpt_lines[-3:])[:400]
        return self._last_textual_line(self._tail_text(candidates[0]))[:400]

    def _has_failure_signal(self, phase: PhaseSpec | None) -> bool:
        if phase is not None and not phase.log_path.exists():
            return False
        candidates = [self._tail_text(self.logs_dir / "supervisor.log", line_count=200)]
        if phase is not None:
            candidates.insert(0, self._tail_text(phase.log_path, line_count=200))
        haystack = "\n".join(candidates).lower()
        markers = [
            "traceback",
            "exception",
            "error:",
            "runtimeerror",
            "failed permanently",
            "failed with code",
            "lmdb.error",
            "filenotfounderror",
            "segmentation fault",
            "killed",
        ]
        return any(marker in haystack for marker in markers)

    def _apply_known_remediations(self, phase: PhaseSpec | None) -> list[str]:
        if phase is None:
            return []
        if not phase.log_path.exists():
            return []
        error_excerpt = self._current_error_excerpt(phase).lower()
        combined = (
            self._tail_text(phase.log_path, line_count=400)
            + "\n"
            + self._tail_text(self.logs_dir / "supervisor.log", line_count=400)
            + "\n"
            + error_excerpt
        ).lower()
        actions = self._cleanup_completed_alignn_caches()
        if phase.alignn_family and any(marker in combined for marker in ("no space left on device", "mdb_txn_commit", "input/output error")):
            for path in (self._phase_local_cache_dir(phase), self.alignn_network_cache, phase.output_dir):
                self._remove_tree(path, actions)
            self.alignn_local_cache_root.mkdir(parents=True, exist_ok=True)
            self.alignn_network_cache.mkdir(parents=True, exist_ok=True)
        if phase.alignn_family and "alignn_faithful_train_data" in combined and (
            "lmdb.error" in combined or ".nfs" in combined or "no such file or directory" in combined
        ):
            for path in (self._phase_local_cache_dir(phase), self.alignn_network_cache, phase.output_dir):
                self._remove_tree(path, actions)
            self.alignn_local_cache_root.mkdir(parents=True, exist_ok=True)
            self.alignn_network_cache.mkdir(parents=True, exist_ok=True)
        if actions:
            actions.append(f"free space {self._free_gib(self.alignn_local_cache_root):.2f} GiB at {self.alignn_local_cache_root}")
        return actions

    def _start_supervisor(self) -> int:
        env = os.environ.copy()
        env["MATGL_BACKEND"] = "DGL"
        env["ALIGNN_LOCAL_CACHE_ROOT"] = str(self.alignn_local_cache_root)
        env["PYTHONUNBUFFERED"] = "1"
        command = [
            str(self.default_python),
            str(self.project_root / "supervise_training.py"),
            "--project-root",
            str(self.project_root),
            "--default-python",
            str(self.default_python),
            "--alignn-python",
            str(self.alignn_python),
            "--m3gnet-python",
            str(self.m3gnet_python),
        ]
        with self.launch_log_path.open("ab") as handle:
            process = subprocess.Popen(
                command,
                cwd=self.project_root,
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=True,
            )
        self.state["launches"] = int(self.state.get("launches", 0)) + 1
        self._save_state()
        return process.pid

    def _sleep_seconds(self, has_active_runner: bool, has_active_supervisor: bool) -> int:
        if has_active_runner:
            return self.args.healthy_check_seconds
        if has_active_supervisor:
            return self.args.recovery_check_seconds
        return self.args.recovery_check_seconds

    def run_once(self) -> int:
        self.state["checks"] = int(self.state.get("checks", 0)) + 1
        completed = self._completed_phase_names()
        phase = self._current_phase()
        cache_cleanup_actions = self._cleanup_completed_alignn_caches()
        if cache_cleanup_actions:
            cache_cleanup_actions.append(
                f"free space {self._free_gib(self.alignn_local_cache_root):.2f} GiB at {self.alignn_local_cache_root}"
            )
            self._log("cleaned completed ALIGNN cache artifacts | " + " | ".join(cache_cleanup_actions))
        if self._all_phases_completed():
            self._clear_unknown_failure("all phases completed")
            self._log(f"all phases completed ({len(completed)}/{len(self.phases)}); watchdog exiting")
            return 0

        processes = self._project_processes()
        duplicate_actions = self._cleanup_duplicate_supervisors(processes)
        if duplicate_actions:
            self._log(" | ".join(duplicate_actions))
            processes = self._project_processes()

        supervisors = [proc for proc in processes if proc.role == "supervisor"]
        runners = [proc for proc in processes if proc.role == "runner"]
        excerpt = self._last_textual_line(self._tail_text(phase.log_path if phase is not None else self.logs_dir / "supervisor.log"))

        if runners or supervisors:
            self._clear_unknown_failure("training stack is active again")
            if phase is not None and runners and phase.log_path.exists():
                lag_seconds = time.time() - phase.log_path.stat().st_mtime
                if lag_seconds > self.args.stall_seconds:
                    actions = self._terminate_project_processes(processes)
                    self._log(
                        f"detected stalled phase {phase.name} after {int(lag_seconds)}s without log updates; "
                        + " | ".join(actions)
                    )
                    return self.args.recovery_check_seconds

            self._log(
                f"heartbeat | completed {len(completed)}/{len(self.phases)} | current {phase.name if phase else 'done'} "
                f"| supervisors={len(supervisors)} runners={len(runners)}"
                + (f" | last='{excerpt[:180]}'" if excerpt else "")
            )
            return self._sleep_seconds(bool(runners), bool(supervisors))

        remediation_actions = self._apply_known_remediations(phase)
        error_excerpt = self._current_error_excerpt(phase)
        if remediation_actions:
            self._clear_unknown_failure("known remediation will relaunch supervision")
            pid = self._start_supervisor()
            message = (
                f"launched supervisor pid {pid} | completed {len(completed)}/{len(self.phases)} "
                f"| current {phase.name if phase else 'unknown'}"
            )
            message += " | remediation=" + " ; ".join(remediation_actions)
            if error_excerpt:
                message += f" | error='{error_excerpt}'"
            self._log(message)
            return self.args.recovery_check_seconds

        if self._has_failure_signal(phase):
            phase_name = self._write_unknown_failure_bundle(
                phase=phase,
                completed=completed,
                processes=processes,
                error_excerpt=error_excerpt,
            )
            self._log(
                f"unknown failure captured for {phase_name}; awaiting Codex debug"
                + (f" | error='{error_excerpt}'" if error_excerpt else "")
            )
            return self.args.recovery_check_seconds

        self._clear_unknown_failure("no failure signal and supervisor will be relaunched")
        pid = self._start_supervisor()
        message = (
            f"launched supervisor pid {pid} | completed {len(completed)}/{len(self.phases)} "
            f"| current {phase.name if phase else 'unknown'}"
        )
        if error_excerpt:
            message += f" | error='{error_excerpt}'"
        self._log(message)
        return self.args.recovery_check_seconds

    def run(self) -> None:
        self._acquire_lock()
        self._log("watchdog started")
        while True:
            sleep_seconds = self.run_once()
            if self._all_phases_completed():
                return
            if self.args.once:
                return
            time.sleep(sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Background watchdog for faithful experiment training.")
    parser.add_argument("--project-root", default=".", help="Project root to supervise.")
    parser.add_argument("--default-python", default=None, help="Python executable for export and summarize phases.")
    parser.add_argument("--alignn-python", default=None, help="Python executable for CGCNN and ALIGNN phases.")
    parser.add_argument("--m3gnet-python", default=None, help="Python executable for M3GNet phases.")
    parser.add_argument("--alignn-local-cache-root", default="/tmp/gnn_alignn_cache", help="Local scratch root for ALIGNN caches.")
    parser.add_argument("--healthy-check-seconds", type=int, default=1800, help="Check interval while training is healthy.")
    parser.add_argument("--recovery-check-seconds", type=int, default=60, help="Check interval while recovering or waiting to restart.")
    parser.add_argument("--stall-seconds", type=int, default=5400, help="Restart if active training stops updating logs for this long.")
    parser.add_argument("--once", action="store_true", help="Run one check and exit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    watchdog = TrainingWatchdog(args)
    watchdog.run()


if __name__ == "__main__":
    main()
