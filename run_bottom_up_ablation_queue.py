from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from crystal_gnn.utils import load_yaml


OFFICIAL_EXPERIMENTS = [
    ("E14", "configs/E14_alignn_physbranch_virtualnode.yaml"),
    ("E15", "configs/E15_alignn_gainsupervised_virtualnode.yaml"),
    ("E16", "configs/E16_alignn_orthogonal_dualslot.yaml"),
    ("E17", "configs/E17_alignn_complexitysupervised_virtualnode.yaml"),
    ("E18", "configs/E18_alignn_regimemoe_virtualnode.yaml"),
]

MP_EXPERIMENTS = [
    ("E14", "configs/mp/MP_E14_alignn_physbranch_virtualnode.yaml"),
    ("E15", "configs/mp/MP_E15_alignn_gainsupervised_virtualnode.yaml"),
    ("E16", "configs/mp/MP_E16_alignn_orthogonal_dualslot.yaml"),
    ("E17", "configs/mp/MP_E17_alignn_complexitysupervised_virtualnode.yaml"),
    ("E18", "configs/mp/MP_E18_alignn_regimemoe_virtualnode.yaml"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the bottom-up ALIGNN ablation queue on Hongzhiwei and MP.")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--results_dir_official", default="results/bottom_up_ablation")
    parser.add_argument("--results_dir_mp", default="results/mp_bottom_up_ablation")
    parser.add_argument("--datasets", nargs="*", default=["official", "mp"], help="Subset of datasets: official mp")
    parser.add_argument("--no_resume", action="store_true", help="Re-run experiments even when metrics.json already exists.")
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--teacher_temperature", type=float, default=0.15)
    return parser.parse_args()


def run_command(command: list[str], cwd: Path) -> None:
    print(f"[run] {' '.join(command)}", flush=True)
    subprocess.run(command, cwd=str(cwd), check=True)


def ensure_teacher_signals(
    python_executable: str,
    project_root: Path,
    output_path: str,
    teacher_a_config: str,
    teacher_a_dir: str,
    teacher_b_config: str,
    teacher_b_dir: str,
    temperature: float,
) -> None:
    teacher_signal_path = (project_root / output_path).resolve()
    if teacher_signal_path.exists():
        print(f"[skip] teacher signals already exist: {teacher_signal_path}", flush=True)
        return
    run_command(
        [
            python_executable,
            "prepare_teacher_signals.py",
            "--teacher_a_config",
            teacher_a_config,
            "--teacher_a_dir",
            teacher_a_dir,
            "--teacher_b_config",
            teacher_b_config,
            "--teacher_b_dir",
            teacher_b_dir,
            "--output",
            output_path,
            "--temperature",
            str(temperature),
        ],
        cwd=project_root,
    )


def run_suite(
    python_executable: str,
    project_root: Path,
    experiments: list[tuple[str, str]],
    results_dir: str,
    no_resume: bool,
    skip_summary: bool,
) -> None:
    results_root = (project_root / results_dir).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    for _key, config_path in experiments:
        config = load_yaml(project_root / config_path)
        output_dir = results_root / config["experiment_name"]
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not no_resume:
            print(f"[skip] metrics already exist for {config['experiment_name']}: {metrics_path}", flush=True)
            continue
        run_command(
            [
                python_executable,
                "run_faithful_experiment.py",
                "--config",
                config_path,
                "--output_dir",
                str(output_dir),
            ],
            cwd=project_root,
        )

    if not skip_summary:
        run_command(
            [
                python_executable,
                "summarize.py",
                "--results_dir",
                str(results_root),
                "--output",
                str(results_root / "comparison_summary.csv"),
            ],
            cwd=project_root,
        )


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent
    datasets = {dataset.lower() for dataset in args.datasets}

    if "official" in datasets:
        ensure_teacher_signals(
            python_executable=args.python,
            project_root=project_root,
            output_path="results/teacher_signals/hongzhiwei_e4_e8_gain_targets.csv",
            teacher_a_config="configs/E4_alignn_virtualnode.yaml",
            teacher_a_dir="results/E4_ALIGNN_virtualnode",
            teacher_b_config="configs/E8_alignn_multi_virtualnode.yaml",
            teacher_b_dir="results/E8_ALIGNN_multi_virtualnode",
            temperature=args.teacher_temperature,
        )
        run_suite(
            python_executable=args.python,
            project_root=project_root,
            experiments=OFFICIAL_EXPERIMENTS,
            results_dir=args.results_dir_official,
            no_resume=args.no_resume,
            skip_summary=args.skip_summary,
        )

    if "mp" in datasets:
        ensure_teacher_signals(
            python_executable=args.python,
            project_root=project_root,
            output_path="results/mp_bandgap_standard/teacher_signals/mp_e4_e8_gain_targets.csv",
            teacher_a_config="configs/mp/MP_E4_alignn_virtualnode.yaml",
            teacher_a_dir="results/mp_bandgap_standard/MP_E4_ALIGNN_virtualnode",
            teacher_b_config="configs/mp/MP_E8_alignn_multi_virtualnode.yaml",
            teacher_b_dir="results/mp_bandgap_standard/MP_E8_ALIGNN_multi_virtualnode",
            temperature=args.teacher_temperature,
        )
        run_suite(
            python_executable=args.python,
            project_root=project_root,
            experiments=MP_EXPERIMENTS,
            results_dir=args.results_dir_mp,
            no_resume=args.no_resume,
            skip_summary=args.skip_summary,
        )


if __name__ == "__main__":
    main()
