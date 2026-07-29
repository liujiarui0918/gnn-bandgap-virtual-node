from __future__ import annotations

import torch

from faithful_gnn.alignn_runner import compute_loss
from faithful_gnn.alignn_variants import build_alignn_model
from faithful_gnn.router_runner import route_rows_by_threshold, select_best_num_atoms_threshold


class FakeBatchGraph:
    def __init__(self, batch_sizes: list[int]) -> None:
        self._batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_sizes


def _prediction_row(sample_id: str, target: float, prediction: float, split: str = "val") -> dict[str, float | str]:
    return {
        "sample_id": sample_id,
        "target": target,
        "prediction": prediction,
        "split": split,
    }


def test_select_best_num_atoms_threshold_prefers_best_validation_router() -> None:
    val_e4 = [
        _prediction_row("s1", 1.0, 1.0),
        _prediction_row("s2", 2.0, 2.0),
        _prediction_row("s3", 3.0, 4.5),
        _prediction_row("s4", 4.0, 5.0),
    ]
    val_e8 = [
        _prediction_row("s1", 1.0, 1.8),
        _prediction_row("s2", 2.0, 2.6),
        _prediction_row("s3", 3.0, 3.0),
        _prediction_row("s4", 4.0, 4.0),
    ]
    num_atoms_by_id = {
        "s1": 12,
        "s2": 28,
        "s3": 96,
        "s4": 128,
    }

    best_threshold, best_metrics, sweep = select_best_num_atoms_threshold(val_e4, val_e8, num_atoms_by_id)

    assert best_threshold == 28
    assert best_metrics["mae"] == 0.0
    assert sweep[0]["threshold"] == 11
    assert sweep[-1]["threshold"] == 128


def test_route_rows_by_threshold_uses_e4_for_small_and_e8_for_large_structures() -> None:
    e4_rows = [
        _prediction_row("s1", 1.0, 1.1, split="test"),
        _prediction_row("s2", 2.0, 2.8, split="test"),
    ]
    e8_rows = [
        _prediction_row("s1", 1.0, 1.6, split="test"),
        _prediction_row("s2", 2.0, 2.1, split="test"),
    ]
    num_atoms_by_id = {"s1": 16, "s2": 144}

    routed = route_rows_by_threshold(e4_rows, e8_rows, num_atoms_by_id, threshold=80)

    assert routed[0]["sample_id"] == "s1"
    assert routed[0]["prediction"] == 1.1
    assert routed[0]["source_model"] == "E4"
    assert routed[1]["sample_id"] == "s2"
    assert routed[1]["prediction"] == 2.1
    assert routed[1]["source_model"] == "E8"


def test_size_supervised_variant_gate_is_not_directly_size_conditioned() -> None:
    model = build_alignn_model(
        {
            "variant": "sizesupervisedvirtualnode",
            "alignn_layers": 1,
            "gcn_layers": 1,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 16,
            "hidden_features": 32,
            "output_features": 1,
            "link": "identity",
            "attention_heads": 4,
            "dropout": 0.0,
            "use_size_features_in_gate": False,
            "use_size_prior_floor": False,
        }
    )

    for parameter in model.size_gate.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    graph = FakeBatchGraph([8, 128])
    node_features = torch.zeros(sum(graph.batch_num_nodes()).item(), 32, dtype=torch.float32)
    slot2_enable, fusion_strength = model._compute_size_gate(graph, node_features)

    assert slot2_enable.shape == (2, 1)
    assert fusion_strength.shape == (2, 1)
    assert float(slot2_enable[0].item()) == float(slot2_enable[1].item())
    assert float(fusion_strength[0].item()) == float(fusion_strength[1].item())


def test_compute_loss_adds_size_supervision_from_num_atoms() -> None:
    outputs = {
        "band_gap": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "slot2_gate_logit": torch.tensor([-2.0, 2.0], dtype=torch.float32),
    }
    labels = torch.tensor([1.0, 2.0], dtype=torch.float32)
    graph = FakeBatchGraph([10, 120])

    loss, components, band_gap_target = compute_loss(
        outputs=outputs,
        labels=labels,
        training_config={
            "lambda_size": 1.0,
            "size_loss_threshold": 80.0,
            "size_loss_temperature": 20.0,
        },
        physics_variant=False,
        graph=graph,
    )

    expected_targets = torch.sigmoid((torch.tensor([10.0, 120.0]) - 80.0) / 20.0)
    expected_size_loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["slot2_gate_logit"], expected_targets)

    assert torch.allclose(band_gap_target, labels)
    assert abs(components["main"]) < 1e-8
    assert abs(components["size"] - expected_size_loss.item()) < 1e-6
    assert abs(loss.item() - expected_size_loss.item()) < 1e-6


def test_size_supervised_phys_variant_exposes_physics_heads() -> None:
    model = build_alignn_model(
        {
            "variant": "sizesupervisedphysvirtualnode",
            "alignn_layers": 1,
            "gcn_layers": 1,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 16,
            "hidden_features": 32,
            "output_features": 1,
            "link": "identity",
            "attention_heads": 4,
            "dropout": 0.0,
            "use_size_features_in_gate": False,
            "use_size_prior_floor": False,
        }
    )

    assert hasattr(model, "cbm_head")
    assert hasattr(model, "vbm_head")


def test_compute_loss_combines_size_and_physics_terms() -> None:
    outputs = {
        "band_gap": torch.tensor([1.5, 0.1], dtype=torch.float32),
        "cbm": torch.tensor([2.0, 0.3], dtype=torch.float32),
        "vbm": torch.tensor([0.5, 0.2], dtype=torch.float32),
        "slot2_gate_logit": torch.tensor([-1.5, 1.5], dtype=torch.float32),
        "global_fusion_logit": torch.tensor([-1.0, 1.0], dtype=torch.float32),
    }
    labels = torch.tensor(
        [
            [1.0, 1.8, 0.8],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    graph = FakeBatchGraph([16, 144])

    loss, components, band_gap_target = compute_loss(
        outputs=outputs,
        labels=labels,
        training_config={
            "lambda_size": 0.15,
            "lambda_size_fusion": 0.05,
            "size_loss_threshold": 80.0,
            "size_loss_temperature": 20.0,
            "valid_edge_epsilon": 1e-8,
            "lambda_aux": 0.3,
            "lambda_phys": 0.5,
            "lambda_nonneg": 0.1,
        },
        physics_variant=True,
        graph=graph,
    )

    assert torch.allclose(band_gap_target, labels[:, 0])
    assert "main" in components
    assert "aux" in components
    assert "phys" in components
    assert "size" in components
    assert "size_fusion" in components
    assert components["edge_valid_ratio"] == 0.5
    assert loss.item() > 0.0
