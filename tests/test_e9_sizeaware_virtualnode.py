from __future__ import annotations

import torch

from faithful_gnn.alignn_runner import run_epoch
from faithful_gnn.alignn_variants import build_alignn_model


class FakeBatchGraph:
    def __init__(self, batch_sizes: list[int]) -> None:
        self._batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_sizes


def test_build_alignn_model_supports_sizeaware_virtualnode_variant() -> None:
    model = build_alignn_model(
        {
            "variant": "sizeawarevirtualnode",
            "alignn_layers": 4,
            "gcn_layers": 4,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 64,
            "hidden_features": 256,
            "output_features": 1,
            "link": "identity",
        }
    )

    assert isinstance(model, torch.nn.Module)


def test_sizeaware_gate_keeps_larger_graphs_more_active_even_with_neutral_learned_logits() -> None:
    model = build_alignn_model(
        {
            "variant": "sizeawarevirtualnode",
            "alignn_layers": 1,
            "gcn_layers": 1,
            "atom_input_features": 92,
            "edge_input_features": 80,
            "triplet_input_features": 40,
            "embedding_features": 16,
            "hidden_features": 32,
            "output_features": 1,
            "link": "identity",
            "slot2_min_scale": 0.02,
            "slot2_max_scale": 0.35,
            "fusion_min_scale": 0.15,
            "fusion_max_scale": 0.55,
            "size_prior_threshold": 48.0,
            "size_prior_temperature": 18.0,
        }
    )

    for parameter in model.size_gate.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    graph = FakeBatchGraph([8, 128])
    node_features = torch.zeros(sum(graph.batch_num_nodes()).item(), 32, dtype=torch.float32)
    slot2_enable, fusion_strength = model._compute_size_gate(graph, node_features)

    assert slot2_enable.shape == (2, 1)
    assert fusion_strength.shape == (2, 1)
    assert float(slot2_enable[1].item()) > float(slot2_enable[0].item())
    assert float(fusion_strength[1].item()) > float(fusion_strength[0].item())


def test_run_epoch_reports_gate_diagnostics_when_model_emits_gate_outputs() -> None:
    class DummyGateModel(torch.nn.Module):
        def forward(self, graph_inputs: list[torch.Tensor]) -> dict[str, torch.Tensor]:
            return {
                "band_gap": torch.tensor([0.5, 1.5], dtype=torch.float32),
                "slot2_enable": torch.tensor([0.2, 0.6], dtype=torch.float32),
                "global_fusion_strength": torch.tensor([0.4, 0.8], dtype=torch.float32),
            }

    loader = [
        (
            torch.zeros(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.float32),
            torch.zeros(2, 1, dtype=torch.float32),
            torch.tensor([0.5, 1.5], dtype=torch.float32),
        )
    ]

    metrics, _components, _targets, _predictions = run_epoch(
        model=DummyGateModel(),
        loader=loader,
        device=torch.device("cpu"),
        training_config={"mixed_precision": False, "grad_clip_norm": 0.0},
        physics_variant=False,
    )

    assert metrics["slot2_enable_mean"] == 0.4
    assert metrics["global_fusion_strength_mean"] == 0.6
