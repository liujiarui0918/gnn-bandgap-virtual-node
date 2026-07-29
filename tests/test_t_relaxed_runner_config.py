from __future__ import annotations

from pathlib import Path

import yaml

from run_t_relaxed_matched_experiments import (
    DEFAULT_EXPERIMENTS,
    T_EXPERIMENTS,
    apply_config_overrides,
    generate_config,
    parse_override_value,
)


def test_default_t_series_includes_topk_huber_and_two_single_factor_ablations() -> None:
    assert DEFAULT_EXPERIMENTS == ["T1", "T2", "T3", "T4", "T6", "T7", "T8", "T9", "T10"]
    assert T_EXPERIMENTS["T6"]["name"] == "T6_ALIGNN_topk_huber_virtualnode"
    assert T_EXPERIMENTS["T6"]["cache_kind"] == "alignn"
    assert T_EXPERIMENTS["T7"]["name"] == "T7_ALIGNN_huber_virtualnode"
    assert T_EXPERIMENTS["T8"]["name"] == "T8_ALIGNN_topk_mse_virtualnode"
    assert T_EXPERIMENTS["T9"]["name"] == "T9_ALIGNN_warmup_topk_huber_virtualnode"
    assert T_EXPERIMENTS["T10"]["name"] == "T10_ALIGNN_decay_huber_virtualnode"


def test_t6_generated_config_uses_topk_virtualnode_and_huber_loss(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T6"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "epochs": 50,
                    "final_gap_loss": "huber",
                    "final_gap_huber_beta": 0.5,
                },
                "model": {
                    "runner_name": "ALIGNN_TopKVirtualNode",
                    "variant": "topkvirtualnode",
                    "topk_fraction": 0.2,
                    "topk_min_nodes": 4,
                    "topk_max_nodes": 32,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T6",
        experiment=T_EXPERIMENTS["T6"],
        epochs=None,
        performance_mode="safe",
        num_workers_override=None,
        mixed_precision_override="auto",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "T6_ALIGNN_topk_huber_virtualnode"
    assert config["data"]["dataset_root"] == str(tmp_path / "dataset")
    assert config["training"]["final_gap_loss"] == "huber"
    assert config["training"]["final_gap_huber_beta"] == 0.5
    assert config["model"]["variant"] == "topkvirtualnode"
    assert config["model"]["topk_fraction"] == 0.2
    assert "topk_enable_min_atoms" not in config["model"]
    assert "topk_enable_max_atoms" not in config["model"]


def test_t7_generated_config_is_huber_only_virtualnode(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T7"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "final_gap_loss": "huber",
                    "final_gap_huber_beta": 0.5,
                },
                "model": {
                    "runner_name": "ALIGNN_VirtualNode",
                    "variant": "virtualnode",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T7",
        experiment=T_EXPERIMENTS["T7"],
        epochs=None,
        performance_mode="safe",
        num_workers_override=None,
        mixed_precision_override="auto",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "T7_ALIGNN_huber_virtualnode"
    assert config["training"]["final_gap_loss"] == "huber"
    assert config["model"]["variant"] == "virtualnode"
    assert "topk_fraction" not in config["model"]


def test_t8_generated_config_is_topk_only_with_default_mse_loss(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T8"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "epochs": 50,
                },
                "model": {
                    "runner_name": "ALIGNN_TopKVirtualNode",
                    "variant": "topkvirtualnode",
                    "topk_fraction": 0.2,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T8",
        experiment=T_EXPERIMENTS["T8"],
        epochs=None,
        performance_mode="safe",
        num_workers_override=None,
        mixed_precision_override="auto",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "T8_ALIGNN_topk_mse_virtualnode"
    assert config["model"]["variant"] == "topkvirtualnode"
    assert config["model"]["topk_fraction"] == 0.2
    assert "final_gap_loss" not in config["training"]
    assert "topk_enable_min_atoms" not in config["model"]
    assert "topk_enable_max_atoms" not in config["model"]


def test_t9_generated_config_is_warmup_topk_huber(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T9"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "final_gap_loss": "huber",
                    "final_gap_huber_beta": 0.5,
                },
                "model": {
                    "runner_name": "ALIGNN_WarmupTopKVirtualNode",
                    "variant": "warmuptopkvirtualnode",
                    "topk_warmup_epochs": 10,
                    "topk_ramp_epochs": 10,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T9",
        experiment=T_EXPERIMENTS["T9"],
        epochs=None,
        performance_mode="safe",
        num_workers_override=None,
        mixed_precision_override="auto",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "T9_ALIGNN_warmup_topk_huber_virtualnode"
    assert config["training"]["final_gap_loss"] == "huber"
    assert config["model"]["variant"] == "warmuptopkvirtualnode"
    assert config["model"]["topk_warmup_epochs"] == 10
    assert config["model"]["topk_ramp_epochs"] == 10


def test_t10_generated_config_is_decay_huber_virtualnode(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T10"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "final_gap_loss": "huber",
                    "final_gap_huber_beta": 0.5,
                },
                "model": {
                    "runner_name": "ALIGNN_DecayVirtualNode",
                    "variant": "decayvirtualnode",
                    "broadcast_decay_start_epoch": 10,
                    "broadcast_decay_end_epoch": 50,
                    "broadcast_min_scale": 0.25,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T10",
        experiment=T_EXPERIMENTS["T10"],
        epochs=None,
        performance_mode="safe",
        num_workers_override=None,
        mixed_precision_override="auto",
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["experiment_name"] == "T10_ALIGNN_decay_huber_virtualnode"
    assert config["training"]["final_gap_loss"] == "huber"
    assert config["model"]["variant"] == "decayvirtualnode"
    assert config["model"]["broadcast_min_scale"] == 0.25


def test_parse_override_value_infers_types() -> None:
    assert parse_override_value("256") == 256
    assert isinstance(parse_override_value("256"), int)
    assert parse_override_value("0.0005") == 0.0005
    assert parse_override_value("true") is True
    assert parse_override_value("false") is False
    assert parse_override_value("huber") == "huber"


def test_apply_config_overrides_sets_nested_keys() -> None:
    config = {"training": {"lr": 0.001, "epochs": 50}, "model": {"units": 128}}
    apply_config_overrides(
        config,
        [
            "training.lr=0.0005",
            "training.epochs=100",
            "model.units=256",
            "model.broadcast_min_scale=0.5",
        ],
    )
    assert config["training"]["lr"] == 0.0005
    assert config["training"]["epochs"] == 100
    assert config["model"]["units"] == 256
    assert config["model"]["broadcast_min_scale"] == 0.5


def test_generate_config_applies_overrides_last(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    template_path = project_root / T_EXPERIMENTS["T7"]["template"]
    template_path.parent.mkdir(parents=True)
    template_path.write_text(
        yaml.safe_dump(
            {
                "experiment_name": "template",
                "runner": "alignn",
                "data": {},
                "training": {
                    "batch_size": 64,
                    "final_gap_loss": "huber",
                    "final_gap_huber_beta": 0.5,
                },
                "model": {
                    "runner_name": "ALIGNN_VirtualNode",
                    "variant": "virtualnode",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    config_path = generate_config(
        project_root=project_root,
        dataset_dir=tmp_path / "dataset",
        results_dir=tmp_path / "results",
        cache_root=tmp_path / "cache",
        key="T7",
        experiment=T_EXPERIMENTS["T7"],
        epochs=None,
        performance_mode="4090",
        num_workers_override=None,
        mixed_precision_override="auto",
        overrides=["training.final_gap_huber_beta=0.25", "training.batch_size=40"],
    )

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["training"]["final_gap_huber_beta"] == 0.25
    # 4090 performance profile sets batch_size=48; the override must win.
    assert config["training"]["batch_size"] == 40
