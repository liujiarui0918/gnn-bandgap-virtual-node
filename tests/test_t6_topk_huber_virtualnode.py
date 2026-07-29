from __future__ import annotations

import sys

import torch

from faithful_gnn.alignn_variants import build_alignn_model


class FakeBatchGraph:
    def __init__(self, batch_sizes: list[int]) -> None:
        self._batch_sizes = torch.tensor(batch_sizes, dtype=torch.long)
        self.batch_size = len(batch_sizes)

    def batch_num_nodes(self) -> torch.Tensor:
        return self._batch_sizes


def _small_t6_config(**overrides: int | float | str) -> dict[str, int | float | str]:
    config: dict[str, int | float | str] = {
        "variant": "topkvirtualnode",
        "alignn_layers": 0,
        "gcn_layers": 0,
        "atom_input_features": 2,
        "edge_input_features": 80,
        "triplet_input_features": 40,
        "embedding_features": 4,
        "hidden_features": 2,
        "output_features": 1,
        "link": "identity",
        "dropout": 0.0,
        "topk_fraction": 0.4,
        "topk_min_nodes": 1,
        "topk_max_nodes": 2,
        "topk_enable_min_atoms": 0,
        "topk_enable_max_atoms": 100,
    }
    config.update(overrides)
    return config


def test_topk_virtual_node_pool_keeps_only_high_scoring_nodes() -> None:
    model = build_alignn_model(_small_t6_config())
    score = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        score.weight.copy_(torch.tensor([[1.0, 0.0]]))
    model.topk_score = score

    graph = FakeBatchGraph([5])
    node_features = torch.tensor(
        [
            [1.0, 0.0],
            [5.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    pooled = model._topk_pool_nodes(graph, node_features)

    top_scores = torch.tensor([5.0, 4.0], dtype=torch.float32)
    expected = (torch.softmax(top_scores, dim=0) * top_scores).sum()
    assert pooled.shape == (1, 2)
    assert torch.allclose(pooled[0, 0], expected)


def test_topk_virtual_node_falls_back_to_mean_outside_enabled_atom_interval() -> None:
    model = build_alignn_model(_small_t6_config(topk_enable_min_atoms=10))
    graph = FakeBatchGraph([5])
    node_features = torch.arange(10, dtype=torch.float32).view(5, 2)
    pooled = model._topk_pool_nodes(graph, node_features)
    assert torch.allclose(pooled, node_features.mean(dim=0, keepdim=True))


def test_compute_loss_supports_huber_final_gap_loss() -> None:
    if "pymatgen.core" in sys.modules and not hasattr(sys.modules["pymatgen.core"], "Element"):
        sys.modules["pymatgen.core"].Element = type("Element", (), {})
    from faithful_gnn.alignn_runner import compute_loss

    outputs = {"band_gap": torch.tensor([0.0, 2.0], dtype=torch.float32)}
    labels = torch.tensor([0.0, 0.0], dtype=torch.float32)
    loss, components, target = compute_loss(
        outputs=outputs,
        labels=labels,
        training_config={"final_gap_loss": "huber", "final_gap_huber_beta": 0.5},
        physics_variant=False,
    )

    expected = torch.nn.functional.smooth_l1_loss(outputs["band_gap"], labels, beta=0.5)
    assert torch.allclose(target, labels)
    assert torch.allclose(loss, expected)
    assert components["main"] == round(expected.item(), 6)


def test_warmup_topk_virtual_node_mixes_mean_and_topk_by_epoch() -> None:
    model = build_alignn_model(
        _small_t6_config(
            variant="warmuptopkvirtualnode",
            topk_warmup_epochs=2,
            topk_ramp_epochs=2,
        )
    )
    score = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        score.weight.copy_(torch.tensor([[1.0, 0.0]]))
    model.topk_score = score

    graph = FakeBatchGraph([5])
    node_features = torch.tensor(
        [
            [1.0, 0.0],
            [5.0, 0.0],
            [2.0, 0.0],
            [4.0, 0.0],
            [3.0, 0.0],
        ],
        dtype=torch.float32,
    )
    mean_pool = node_features.mean(dim=0, keepdim=True)

    model.set_training_epoch(1, 6)
    assert model.topk_mix_alpha == 0.0
    assert torch.allclose(model._topk_pool_nodes(graph, node_features), mean_pool)

    model.set_training_epoch(3, 6)
    half_pool = model._topk_pool_nodes(graph, node_features)
    assert model.topk_mix_alpha == 0.5
    assert not torch.allclose(half_pool, mean_pool)

    model.set_training_epoch(4, 6)
    full_pool = model._topk_pool_nodes(graph, node_features)
    assert model.topk_mix_alpha == 1.0
    assert not torch.allclose(full_pool, mean_pool)


def test_decay_virtual_node_broadcast_scale_schedule() -> None:
    model = build_alignn_model(
        _small_t6_config(
            variant="decayvirtualnode",
            broadcast_decay_start_epoch=2,
            broadcast_decay_end_epoch=6,
            broadcast_min_scale=0.25,
        )
    )

    model.set_training_epoch(1, 6)
    assert model.broadcast_scale == 1.0

    model.set_training_epoch(4, 6)
    assert model.broadcast_scale == 0.625

    model.set_training_epoch(6, 6)
    assert model.broadcast_scale == 0.25


def test_decay_virtual_node_scales_global_to_node_broadcast_before_pooling() -> None:
    model = build_alignn_model(
        _small_t6_config(
            variant="decayvirtualnode",
            broadcast_decay_start_epoch=0,
            broadcast_decay_end_epoch=2,
            broadcast_min_scale=0.5,
        )
    )
    model.set_training_epoch(2, 2)

    class IdentityUpdate(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    model.virtual_node_update = IdentityUpdate()
    graph = FakeBatchGraph([2])
    node_features = torch.zeros(2, 2, dtype=torch.float32)
    virtual_node = torch.tensor([[2.0, 4.0]], dtype=torch.float32)

    updated_nodes, updated_virtual_node = model._update_virtual_node(graph, node_features, virtual_node)

    expected_nodes = torch.tensor([[1.0, 2.0], [1.0, 2.0]], dtype=torch.float32)
    expected_virtual_node = torch.tensor([[3.0, 6.0]], dtype=torch.float32)
    assert torch.allclose(updated_nodes, expected_nodes)
    assert torch.allclose(updated_virtual_node, expected_virtual_node)
