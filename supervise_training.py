from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Phase:
    name: str
    kind: str
    command_base: list[str]
    completion_path: Path
    config_path: Path | None = None
    output_dir: Path | None = None


class Supervisor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.project_root = Path(args.project_root).resolve()
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.logs_dir / "supervisor_state.json"
        self.supervisor_log = self.logs_dir / "supervisor.log"
        self.results_dir = self.project_root / "results"
        self.dataset_dir = self.project_root / "data" / "official_dataset"
        self.raw_path = self.project_root / "data" / "raw" / "crystal_band_100000.jsonl"
        self.summary_path = self.results_dir / "comparison_summary.csv"
        self.default_python_executable = Path(args.default_python or sys.executable)
        self.alignn_python_executable = Path(args.alignn_python or self.default_python_executable)
        self.m3gnet_python_executable = Path(args.m3gnet_python or self.default_python_executable)
        self.attach_pid = args.attach_pid
        self.external_pid_released = args.attach_pid is None
        self.state = self._load_state()
        self.phase_defaults = self._load_phase_defaults()
        self.phases = self._build_phases()

    def _load_state(self) -> dict[str, Any]:
        if self.state_path.exists():
            with self.state_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        return {
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "phases": {},
            "overrides": {},
            "events": [],
        }

    def _save_state(self) -> None:
        with self.state_path.open("w", encoding="utf-8") as handle:
            json.dump(self.state, handle, indent=2, ensure_ascii=False)

    def _log(self, message: str) -> None:
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"{timestamp} | {message}"
        with self.supervisor_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
        print(line, flush=True)
        self.state.setdefault("events", []).append(line)
        self.state["events"] = self.state["events"][-200:]
        self._save_state()

    def _load_phase_defaults(self) -> dict[str, dict[str, Any]]:
        defaults: dict[str, dict[str, Any]] = {}
        config_map = {
            "E1_CGCNN_baseline": self.project_root / "configs" / "E1_cgcnn_faithful.yaml",
            "E2_ALIGNN_baseline": self.project_root / "configs" / "E2_alignn_faithful.yaml",
            "E3_M3GNet_baseline": self.project_root / "configs" / "E3_m3gnet_faithful.yaml",
            "E4_ALIGNN_virtualnode": self.project_root / "configs" / "E4_alignn_virtualnode.yaml",
            "E5_ALIGNN_transformer": self.project_root / "configs" / "E5_alignn_transformer_faithful.yaml",
            "E6_ALIGNN_physconstraint": self.project_root / "configs" / "E6_alignn_physconstraint.yaml",
            "E7_ALIGNN_attentive_virtualnode": self.project_root / "configs" / "E7_alignn_attentive_virtualnode.yaml",
            "E8_ALIGNN_multi_virtualnode": self.project_root / "configs" / "E8_alignn_multi_virtualnode.yaml",
        }
        for phase_name, config_path in config_map.items():
            with config_path.open("r", encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            defaults[phase_name] = {
                "batch_size": int(config["training"]["batch_size"]),
                "device": config["training"].get("device", "auto"),
                "num_workers": int(config["training"].get("num_workers", 0)),
                "lr": float(config["training"]["lr"]),
            }
        return defaults

    def _build_phases(self) -> list[Phase]:
        return [
            Phase(
                name="preprocess",
                kind="preprocess",
                command_base=[
                    str(self.default_python_executable),
                    str(self.project_root / "export_official_dataset.py"),
                    "--data_path",
                    str(self.raw_path),
                    "--output_dir",
                    str(self.dataset_dir),
                    "--seed",
                    "42",
                ],
                completion_path=self.dataset_dir / "metadata.json",
            ),
            Phase(
                name="E1_CGCNN_baseline",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E1_cgcnn_faithful.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E1_CGCNN_baseline"),
                ],
                completion_path=self.results_dir / "E1_CGCNN_baseline" / "metrics.json",
                config_path=self.project_root / "configs" / "E1_cgcnn_faithful.yaml",
                output_dir=self.results_dir / "E1_CGCNN_baseline",
            ),
            Phase(
                name="E2_ALIGNN_baseline",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E2_alignn_faithful.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E2_ALIGNN_baseline"),
                ],
                completion_path=self.results_dir / "E2_ALIGNN_baseline" / "metrics.json",
                config_path=self.project_root / "configs" / "E2_alignn_faithful.yaml",
                output_dir=self.results_dir / "E2_ALIGNN_baseline",
            ),
            Phase(
                name="E3_M3GNet_baseline",
                kind="train",
                command_base=[
                    str(self.m3gnet_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E3_m3gnet_faithful.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E3_M3GNet_baseline"),
                ],
                completion_path=self.results_dir / "E3_M3GNet_baseline" / "metrics.json",
                config_path=self.project_root / "configs" / "E3_m3gnet_faithful.yaml",
                output_dir=self.results_dir / "E3_M3GNet_baseline",
            ),
            Phase(
                name="E4_ALIGNN_virtualnode",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E4_alignn_virtualnode.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E4_ALIGNN_virtualnode"),
                ],
                completion_path=self.results_dir / "E4_ALIGNN_virtualnode" / "metrics.json",
                config_path=self.project_root / "configs" / "E4_alignn_virtualnode.yaml",
                output_dir=self.results_dir / "E4_ALIGNN_virtualnode",
            ),
            Phase(
                name="E5_ALIGNN_transformer",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E5_alignn_transformer_faithful.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E5_ALIGNN_transformer"),
                ],
                completion_path=self.results_dir / "E5_ALIGNN_transformer" / "metrics.json",
                config_path=self.project_root / "configs" / "E5_alignn_transformer_faithful.yaml",
                output_dir=self.results_dir / "E5_ALIGNN_transformer",
            ),
            Phase(
                name="E6_ALIGNN_physconstraint",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E6_alignn_physconstraint.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E6_ALIGNN_physconstraint"),
                ],
                completion_path=self.results_dir / "E6_ALIGNN_physconstraint" / "metrics.json",
                config_path=self.project_root / "configs" / "E6_alignn_physconstraint.yaml",
                output_dir=self.results_dir / "E6_ALIGNN_physconstraint",
            ),
            Phase(
                name="E7_ALIGNN_attentive_virtualnode",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E7_alignn_attentive_virtualnode.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E7_ALIGNN_attentive_virtualnode"),
                ],
                completion_path=self.results_dir / "E7_ALIGNN_attentive_virtualnode" / "metrics.json",
                config_path=self.project_root / "configs" / "E7_alignn_attentive_virtualnode.yaml",
                output_dir=self.results_dir / "E7_ALIGNN_attentive_virtualnode",
            ),
            Phase(
                name="E8_ALIGNN_multi_virtualnode",
                kind="train",
                command_base=[
                    str(self.alignn_python_executable),
                    str(self.project_root / "run_faithful_experiment.py"),
                    "--config",
                    str(self.project_root / "configs" / "E8_alignn_multi_virtualnode.yaml"),
                    "--output_dir",
                    str(self.results_dir / "E8_ALIGNN_multi_virtualnode"),
                ],
                completion_path=self.results_dir / "E8_ALIGNN_multi_virtualnode" / "metrics.json",
                config_path=self.project_root / "configs" / "E8_alignn_multi_virtualnode.yaml",
                output_dir=self.results_dir / "E8_ALIGNN_multi_virtualnode",
            ),
            Phase(
                name="summarize",
                kind="summarize",
                command_base=[
                    str(self.default_python_executable),
                    str(self.project_root / "summarize.py"),
                    "--results_dir",
                    str(self.results_dir),
                    "--output",
                    str(self.summary_path),
                ],
                completion_path=self.summary_path,
            ),
        ]

    def _phase_state(self, phase_name: str) -> dict[str, Any]:
        phases = self.state.setdefault("phases", {})
        phase_state = phases.setdefault(phase_name, {"attempts": 0, "completed": False})
        return phase_state

    def _phase_overrides(self, phase_name: str) -> dict[str, Any]:
        overrides = self.state.setdefault("overrides", {})
        return overrides.setdefault(phase_name, {})

    def _all_phases_completed(self) -> bool:
        return all(phase.completion_path.exists() for phase in self.phases)

    def _build_command(self, phase: Phase) -> list[str]:
        command = list(phase.command_base)
        overrides = self._phase_overrides(phase.name)
        if phase.kind == "train":
            if "batch_size" in overrides:
                command.extend(["--batch_size", str(overrides["batch_size"])])
            if "device" in overrides:
                command.extend(["--device", str(overrides["device"])])
            if "num_workers" in overrides:
                command.extend(["--num_workers", str(overrides["num_workers"])])
            if "lr" in overrides:
                command.extend(["--lr", str(overrides["lr"])])
        return command

    def _active_project_pids(self) -> list[int]:
        process = subprocess.run(
            ["ps", "-eo", "pid=,cmd="],
            capture_output=True,
            text=True,
            check=True,
        )
        pids: list[int] = []
        needles = ["export_official_dataset.py", "run_faithful_experiment.py", "summarize.py", "run_all_experiments.sh"]
        for line in process.stdout.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            pid_text, _, cmd = stripped.partition(" ")
            try:
                pid = int(pid_text)
            except ValueError:
                continue
            if pid == os.getpid():
                continue
            if str(self.project_root) in cmd and any(needle in cmd for needle in needles):
                pids.append(pid)
        return pids

    def _pid_alive(self, pid: int) -> bool:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def _wait_for_external_run(self) -> None:
        while True:
            if self._all_phases_completed():
                self._log("all phase completion files already exist while observing external run; supervisor exiting cleanly")
                raise SystemExit(0)
            if self.attach_pid is not None and self._pid_alive(self.attach_pid):
                self._log(f"observing external pid {self.attach_pid}; waiting for it to finish")
                time.sleep(self.args.poll_seconds)
                continue
            active_pids = self._active_project_pids()
            if active_pids:
                self._log(f"external pid ended but found active project pids {active_pids}; still waiting")
                time.sleep(self.args.poll_seconds)
                continue
            self._log("external run is no longer active; supervisor is taking over")
            self.external_pid_released = True
            return

    def _tail_text(self, path: Path, line_count: int = 120) -> str:
        if not path.exists():
            return ""
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            lines = handle.readlines()
        return "".join(lines[-line_count:])

    def _install_missing_module(self, module_name: str, python_executable: Path) -> bool:
        pip_name_map = {
            "yaml": "PyYAML",
            "sklearn": "scikit-learn",
            "matplotlib": "matplotlib",
            "pymatgen": "pymatgen",
            "pandas": "pandas",
            "scipy": "scipy",
            "jarvis": "jarvis-tools",
            "lightning": "lightning",
        }
        package_name = pip_name_map.get(module_name, module_name)
        self._log(f"attempting to install missing module {module_name} via package {package_name}")
        install = subprocess.run(
            [str(python_executable), "-m", "pip", "install", package_name],
            cwd=self.project_root,
            capture_output=True,
            text=True,
        )
        success = install.returncode == 0
        if success:
            self._log(f"module install succeeded for {package_name}")
        else:
            self._log(f"module install failed for {package_name}: {install.stderr[-400:]}")
        return success

    def _analyze_failure(self, phase: Phase, return_code: int, log_path: Path) -> tuple[bool, str]:
        tail = self._tail_text(log_path).lower()
        overrides = self._phase_overrides(phase.name)
        phase_python = Path(phase.command_base[0])

        module_match = re.search(r"no module named ['\"]([^'\"]+)['\"]", tail)
        if module_match:
            module_name = module_match.group(1).split(".")[0]
            if self._install_missing_module(module_name, phase_python):
                return True, f"installed missing module {module_name}"

        oom_markers = [
            "cuda out of memory",
            "cublas_status_alloc_failed",
            "outofmemoryerror",
            "out of memory",
        ]
        if any(marker in tail for marker in oom_markers) or return_code in {137, -9}:
            default_batch = self.phase_defaults.get(phase.name, {}).get("batch_size", 1)
            current_batch = int(overrides.get("batch_size", default_batch))
            if current_batch > 1:
                next_batch = max(1, current_batch // 2)
                overrides["batch_size"] = next_batch
                overrides.setdefault("device", "cuda")
                overrides.setdefault("num_workers", 0)
                self._save_state()
                return True, f"reduced batch size from {current_batch} to {next_batch}"

        has_nonfinite_signal = any(
            marker in tail
            for marker in [
                " loss nan",
                " mae nan",
                " rmse nan",
                " r2 nan",
                "metal_f1 nan",
                "nonfinite",
                "non-finite",
            ]
        ) or re.search(r"(?<![a-z])(nan|inf)(?![a-z])", tail)
        if phase.kind == "train" and has_nonfinite_signal:
            default_lr = float(self.phase_defaults.get(phase.name, {}).get("lr", 1e-3))
            current_lr = float(overrides.get("lr", default_lr))
            next_lr = round(max(1e-5, current_lr * 0.5), 8)
            if next_lr < current_lr:
                overrides["lr"] = next_lr
                self._save_state()
                return True, f"reduced learning rate from {current_lr} to {next_lr} after non-finite training signal"

        if (
            phase.kind != "preprocess"
            and any(marker in tail for marker in ["manifest.jsonl", "metadata.json", "splits.json"])
            and any(marker in tail for marker in ["no such file or directory", "filenotfounderror"])
        ):
            preprocess_state = self._phase_state("preprocess")
            preprocess_state["completed"] = False
            self._save_state()
            return True, "marked preprocess as incomplete and will rerun it"

        generic_attempts = self._phase_state(phase.name)["attempts"]
        if generic_attempts < self.args.generic_retry_limit:
            return True, f"generic retry {generic_attempts}/{self.args.generic_retry_limit}"

        return False, "no automated remediation available"

    def _run_phase(self, phase: Phase) -> None:
        phase_state = self._phase_state(phase.name)
        if phase.completion_path.exists():
            phase_state["completed"] = True
            self._save_state()
            self._log(f"skipping {phase.name}; completion file already exists")
            return

        log_path = self.logs_dir / f"{phase.name}.log"
        while True:
            phase_state["attempts"] += 1
            phase_state["last_started_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_state()

            command = self._build_command(phase)
            self._log(f"starting {phase.name} attempt {phase_state['attempts']}: {' '.join(command)}")
            with log_path.open("a", encoding="utf-8") as log_handle:
                log_handle.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} | START {' '.join(command)} ===\n")
                process = subprocess.run(
                    command,
                    cwd=self.project_root,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    env=os.environ.copy(),
                )

            phase_state["last_exit_code"] = process.returncode
            self._save_state()
            if process.returncode == 0 and phase.completion_path.exists():
                phase_state["completed"] = True
                phase_state["completed_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
                self._save_state()
                self._log(f"phase {phase.name} completed successfully")
                return

            should_retry, reason = self._analyze_failure(phase, process.returncode, log_path)
            self._log(f"phase {phase.name} failed with code {process.returncode}; decision: {reason}")
            if should_retry and phase_state["attempts"] < self.args.max_attempts_per_phase:
                time.sleep(self.args.retry_sleep_seconds)
                continue
            raise RuntimeError(f"phase {phase.name} failed permanently: {reason}")

    def run(self) -> None:
        self._log("supervisor started")
        if not self.external_pid_released:
            self._wait_for_external_run()

        while True:
            if self._all_phases_completed():
                self._log("all phase completion files detected; supervision complete")
                return

            active_pids = self._active_project_pids()
            if active_pids:
                self._log(f"detected active project process(es) {active_pids}; waiting before takeover")
                time.sleep(self.args.poll_seconds)
                continue

            for phase in self.phases:
                if self._all_phases_completed():
                    self._log("all phase completion files detected mid-loop; supervision complete")
                    return
                self._run_phase(phase)
            time.sleep(self.args.poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor training and resume on failures.")
    parser.add_argument("--project-root", default=".", help="Project root to supervise.")
    parser.add_argument("--attach-pid", type=int, default=None, help="Existing external PID to observe before takeover.")
    parser.add_argument("--default-python", default=None, help="Python executable for export/summarize phases.")
    parser.add_argument("--alignn-python", default=None, help="Python executable for CGCNN and ALIGNN-family phases.")
    parser.add_argument("--m3gnet-python", default=None, help="Python executable for M3GNet phase.")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--retry-sleep-seconds", type=int, default=10)
    parser.add_argument("--max-attempts-per-phase", type=int, default=8)
    parser.add_argument("--generic-retry-limit", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    supervisor = Supervisor(args)
    supervisor.run()


if __name__ == "__main__":
    main()
