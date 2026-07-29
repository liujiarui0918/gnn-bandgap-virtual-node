from __future__ import annotations

import sys

import torch

from faithful_gnn.alignn_variants import build_alignn_model


class FakeBatchGraph:
    def __init__(self, batch_sizes: list[int], atom_features: torch.Tensor | None = None) -> None:
        self._batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)
        self.batch_size = len(batch_sizes)
        self.ndata: dict[str, torch.Tensor] = {}
        if atom_features is not None:
            self.ndata["atom_features"] = atom_features
        self.edata: dict[str, torch.Tensor] = {"r": torch.zeros(0, 3, dtype=torch.float32)}

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_sizes

    def local_var(self) -> "FakeBatchGraph":
        return self


class FakeLineGraph:
    def __init__(self) -> None:
        self.edata: dict[str, torch.Tensor] = {"h": torch.zeros(0, dtype=torch.float32)}

    def local_var(self) -> "FakeLineGraph":
        return self


def _small_t7_config() -> dict[str, int | float | str]:
    return {
        "variant": "sizegatedplaindualvirtualnode",
        "alignn_layers": 0,
        "gcn_layers": 0,
        "atom_input_features": 32,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 16,
        "hidden_features": 32,
        "output_features": 1,
        "link": "identity",
        "dropout": 0.0,
        "gate_bias_init": 0.0,
        "size_prior_threshold": 48.0,
        "size_prior_temperature": 18.0,
        "slot2_min_scale": 0.0,
        "slot2_max_scale": 1.0,
    }


def test_t7_gate_uses_size_prior_to_enable_slot2_differently_for_small_and_large_graphs() -> None:
    model = build_alignn_model(_small_t7_config())
    assert model.size_gate[0].in_features == 34

    for parameter in model.size_gate.parameters():
        torch.nn.init.constant_(parameter, 0.0)

    graph = FakeBatchGraph([8, 128])
    node_features = torch.zeros(sum(graph.batch_num_nodes()).item(), 32, dtype=torch.float32)
    slot2_enable, slot2_gate_logit = model._compute_gate_controls(graph, node_features)

    assert slot2_enable.shape == (2, 1)
    assert slot2_gate_logit.shape == (2, 1)
    assert float(slot2_enable[1].item()) > float(slot2_enable[0].item())
    assert float(slot2_gate_logit[1].item()) > float(slot2_gate_logit[0].item())


def test_t7_forward_emits_gate_outputs_and_lambda_size_loss_works(monkeypatch) -> None:
    if "pymatgen.core" in sys.modules and not hasattr(sys.modules["pymatgen.core"], "Element"):
        sys.modules["pymatgen.core"].Element = type("Element", (), {})
    from faithful_gnn.alignn_runner import compute_loss

    model = build_alignn_model(_small_t7_config())
    model.encoder.atom_embedding = torch.nn.Identity()
    model.encoder.edge_embedding = torch.nn.Identity()
    model.encoder.angle_embedding = torch.nn.Identity()

    def fake_pool(graph: FakeBatchGraph, node_features: torch.Tensor) -> torch.Tensor:
        outputs = []
        cursor = 0
        for count in graph.batch_num_nodes().tolist():
            outputs.append(node_features[cursor : cursor + count].mean(dim=0))
            cursor += count
        return torch.stack(outputs, dim=0)

    monkeypatch.setattr(model.encoder, "pool", fake_pool)

    graph = FakeBatchGraph([8, 128], atom_features=torch.zeros(136, 32, dtype=torch.float32))
    outputs = model([graph, FakeLineGraph(), None])

    assert set(["band_gap", "slot2_enable", "slot2_gate_logit"]).issubset(outputs)
    assert outputs["band_gap"].shape == (2,)
    assert outputs["slot2_enable"].shape == (2,)
    assert outputs["slot2_gate_logit"].shape == (2,)

    loss, components, band_gap_target = compute_loss(
        outputs=outputs,
        labels=torch.zeros(2, dtype=torch.float32),
        training_config={
            "lambda_size": 1.0,
            "size_loss_threshold": 48.0,
            "size_loss_temperature": 18.0,
        },
        physics_variant=False,
        graph=graph,
    )

    assert torch.allclose(band_gap_target, torch.zeros(2, dtype=torch.float32))
    assert "size" in components
    assert torch.isfinite(loss)
