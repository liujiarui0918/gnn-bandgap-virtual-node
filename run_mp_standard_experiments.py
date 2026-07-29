from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from crystal_gnn.utils import load_yaml


EXPERIMENTS = [
    {
        "key": "E1",
        "name": "MP_E1_CGCNN_baseline",
        "config": "configs/mp/MP_E1_cgcnn_faithful.yaml",
        "python_role": "default",
    },
    {
        "key": "E2",
        "name": "MP_E2_ALIGNN_baseline",
        "config": "configs/mp/MP_E2_alignn_faithful.yaml",
        "python_role": "alignn",
    },
    {
        "key": "E3",
        "name": "MP_E3_M3GNet_baseline",
        "config": "configs/mp/MP_E3_m3gnet_faithful.yaml",
        "python_role": "m3gnet",
    },
    {
        "key": "E4",
        "name": "MP_E4_ALIGNN_virtualnode",
        "config": "configs/mp/MP_E4_alignn_virtualnode.yaml",
        "python_role": "alignn",
    },
    {
        "key": "E7",
        "name": "MP_E7_ALIGNN_attentive_virtualnode",
        "config": "configs/mp/MP_E7_alignn_attentive_virtualnode.yaml",
        "python_role": "alignn",
    },
    {
        "key": "E8",
        "name": "MP_E8_ALIGNN_multi_virtualnode",
        "config": "configs/mp/MP_E8_alignn_multi_virtualnode.yaml",
        "python_role": "alignn",
    },
    {
        "key": "E12",
        "name": "MP_E12_ALIGNN_sizesupervised_virtualnode",
        "config": "configs/mp/MP_E12_alignn_sizesupervised_virtualnode.yaml",
        "python_role": "alignn",
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Materials Project standard 7-experiment band-gap suite.")
    parser.add_argument("--api_key", default=os.environ.get("MP_API_KEY"), help="Materials Project API key.")
    parser.add_argument("--dataset_dir", default="data/mp_bandgap_official")
    parser.add_argument("--results_dir", default="results/mp_bandgap_standard")
    parser.add_argument("--default_python", default=sys.executable)
    parser.add_argument("--alignn_python", default=None)
    parser.add_argument("--m3gnet_python", default=None)
    parser.add_argument("--force_export", action="store_true")
    parser.add_argument("--export_limit", type=int, default=None, help="Accepted-sample limit for smoke tests.")
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=None,
        help="Optional subset of experiment keys, e.g. E1 E2 E4. Default runs the full MP standard suite.",
    )
    parser.add_argument("--skip_summary", action="store_true")
    parser.add_argument("--no_resume", action="store_true", help="Re-run experiments even when metrics.json already exists.")
    parser.add_argument(
        "--fail_fast",
        action="store_true",
        help="Stop immediately when one experiment fails. Default keeps running the remaining experiments.",
    )
    return parser.parse_args()


def run_command(command: list[str], cwd: Path, env: dict[str, str]) -> None:
    print(f"[run] {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd), env=env, check=True)


def select_experiments(keys: list[str] | None) -> list[dict[str, str]]:
    if not keys:
        return EXPERIMENTS
    normalized = {key.upper() for key in keys}
    selected = [item for item in EXPERIMENTS if item["key"].upper() in normalized]
    missing = sorted(normalized.difference({item["key"].upper() for item in selected}))
    if missing:
        raise ValueError(f"unknown experiment keys: {', '.join(missing)}")
    return selected


def main() -> None:
    args = parse_args()
    if not args.api_key:
        raise ValueError("Materials Project API key is required. Pass --api_key or set MP_API_KEY.")

    project_root = Path(__file__).resolve().parent
    dataset_dir = (project_root / args.dataset_dir).resolve()
    results_dir = (project_root / args.results_dir).resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["MP_API_KEY"] = args.api_key

    alignn_python = args.alignn_python or args.default_python
    m3gnet_python = args.m3gnet_python or args.default_python
    default_python = args.default_python

    manifest_path = dataset_dir / "manifest.jsonl"
    metadata_path = dataset_dir / "metadata.json"
    if args.force_export or not manifest_path.exists() or not metadata_path.exists():
        export_command = [
            default_python,
            "export_mp_dataset.py",
            "--output_dir",
            str(dataset_dir),
            "--seed",
            "42",
        ]
        if args.export_limit is not None:
            export_command.extend(["--limit", str(args.export_limit)])
        run_command(export_command, cwd=project_root, env=env)
    else:
        print(f"[skip] dataset already exists at {dataset_dir}")

    selected = select_experiments(args.experiments)
    failures: list[dict[str, str | int]] = []
    for experiment in selected:
        config_path = project_root / experiment["config"]
        config = load_yaml(config_path)
        output_dir = results_dir / config["experiment_name"]
        metrics_path = output_dir / "metrics.json"
        if metrics_path.exists() and not args.no_resume:
            print(f"[skip] metrics already exist for {experiment['name']}: {metrics_path}")
            continue

        python_executable = default_python
        if experiment["python_role"] == "alignn":
            python_executable = alignn_python
        elif experiment["python_role"] == "m3gnet":
            python_executable = m3gnet_python

        command = [
            python_executable,
            "run_faithful_experiment.py",
            "--config",
            str(config_path),
            "--output_dir",
            str(output_dir),
        ]
        try:
            run_command(command, cwd=project_root, env=env)
        except subprocess.CalledProcessError as exc:
            failure = {
                "experiment_key": experiment["key"],
                "experiment_name": experiment["name"],
                "config": str(config_path),
                "returncode": int(exc.returncode),
                "command": " ".join(command),
            }
            failures.append(failure)
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "failure.json").open("w", encoding="utf-8") as handle:
                json.dump(failure, handle, indent=2, ensure_ascii=False)
            print(f"[fail] {experiment['name']} exited with return code {exc.returncode}")
            if args.fail_fast:
                raise

    if not args.skip_summary:
        run_command(
            [
                default_python,
                "summarize.py",
                "--results_dir",
                str(results_dir),
                "--output",
                str(results_dir / "comparison_summary.csv"),
            ],
            cwd=project_root,
            env=env,
        )

    if failures:
        with (results_dir / "failures.json").open("w", encoding="utf-8") as handle:
            json.dump(failures, handle, indent=2, ensure_ascii=False)
        print(f"[warn] {len(failures)} experiment(s) failed; details saved to {results_dir / 'failures.json'}")


if __name__ == "__main__":
    main()
