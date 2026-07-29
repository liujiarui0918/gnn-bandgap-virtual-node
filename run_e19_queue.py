from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from crystal_gnn.utils import load_yaml


EXPERIMENTS = [
    {
        "key": "official",
        "config": "configs/E19_alignn_metalaware_virtualnode.yaml",
        "results_dir": "results",
    },
    {
        "key": "mp",
        "config": "configs/mp/MP_E19_alignn_metalaware_virtualnode.yaml",
        "results_dir": "results/mp_bandgap_standard",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the E19 metal-aware ALIGNN virtual-node experiments.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--datasets", nargs="*", default=["official", "mp"], help="Subset: official mp")
    parser.add_argument("--no_resume", action="store_true")
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    selected = {item.lower() for item in args.datasets}

    for experiment in EXPERIMENTS:
        if experiment["key"] not in selected:
            continue
        config_path = project_root / experiment["config"]
        config = load_yaml(config_path)
        output_dir = project_root / experiment["results_dir"] / config["experiment_name"]
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not args.no_resume:
            print(f"[skip] metrics already exist for {config['experiment_name']}: {metrics_path}", flush=True)
            continue
        run_command(
            [
                args.python,
                "run_faithful_experiment.py",
                "--config",
                str(config_path),
                "--output_dir",
                str(output_dir),
            ],
            cwd=project_root,
        )


if __name__ == "__main__":
    main()
