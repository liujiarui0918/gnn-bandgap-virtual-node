from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .utils import scatter_mean, scatter_sum, sinusoidal_encoding


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        output_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        dims = [input_dim, *hidden_dims, output_dim]
        layers: list[nn.Module] = []
        for in_dim, out_dim in zip(dims[:-1], dims[1:]):
            layers.append(nn.Linear(in_dim, out_dim))
            if out_dim != output_dim:
                layers.append(nn.SiLU())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class RBFExpansion(nn.Module):
    def __init__(self, start: float, stop: float, num_basis: int) -> None:
        super().__init__()
        centers = torch.linspace(start, stop, num_basis)
        gamma = 1.0 / ((centers[1] - centers[0]).item() ** 2 + 1e-12) if num_basis > 1 else 1.0
        self.register_buffer("centers", centers)
        self.gamma = gamma

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        deltas = values.unsqueeze(-1) - self.centers
        return torch.exp(-self.gamma * deltas.square())


def _graph_pool(
    node_states: torch.Tensor,
    node_batch: torch.Tensor,
    graph_features: torch.Tensor,
) -> torch.Tensor:
    num_graphs = int(graph_features.size(0))
    pooled_mean = scatter_mean(node_states, node_batch, num_graphs)
    pooled_sum = scatter_sum(node_states, node_batch, num_graphs)
    return torch.cat([pooled_mean, pooled_sum, graph_features], dim=-1)


class CGCNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, dropout: float) -> None:
        super().__init__()
        self.message = MLP(hidden_dim * 2 + edge_dim, [hidden_dim * 2], hidden_dim * 2, dropout=dropout)
        self.update = nn.Linear(hidden_dim * 2, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, node_states: torch.Tensor, edge_index: torch.Tensor, edge_states: torch.Tensor) -> torch.Tensor:
        source, target = edge_index
        edge_inputs = torch.cat([node_states[source], node_states[target], edge_states], dim=-1)
        gated, candidate = self.message(edge_inputs).chunk(2, dim=-1)
        messages = torch.sigmoid(gated) * F.silu(candidate)
        aggregated = scatter_sum(messages, target, node_states.size(0))
        updated = self.update(torch.cat([node_states, aggregated], dim=-1))
        return F.silu(self.norm(node_states + self.dropout(updated)))


class CGCNNModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        atom_dim = int(config.get("atom_embedding_dim", hidden_dim))
        edge_dim = int(config.get("edge_basis_dim", 32))
        graph_feature_dim = int(config.get("graph_feature_dim", 8))
        self.atom_embedding = nn.Embedding(100, atom_dim)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.edge_rbf = RBFExpansion(0.0, float(config["cutoff"]), edge_dim)
        self.edge_projection = MLP(edge_dim, [hidden_dim], hidden_dim, dropout=float(config.get("dropout", 0.0)))
        self.layers = nn.ModuleList(
            [CGCNNBlock(hidden_dim, hidden_dim, float(config.get("dropout", 0.0))) for _ in range(int(config["num_layers"]))]
        )
        self.graph_projection = MLP(graph_feature_dim, [hidden_dim // 2], hidden_dim // 2)
        readout_input = hidden_dim * 2 + hidden_dim // 2
        self.readout = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=float(config.get("dropout", 0.0)))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_states = self.atom_projection(self.atom_embedding(batch["atomic_numbers"]))
        edge_states = self.edge_projection(self.edge_rbf(batch["edge_distances"]))
        for layer in self.layers:
            node_states = layer(node_states, batch["edge_index"], edge_states)
        pooled = _graph_pool(node_states, batch["node_batch"], self.graph_projection(batch["graph_features"]))
        return {"band_gap": self.readout(pooled).squeeze(-1)}


class CGCNNVirtualNodeModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        atom_dim = int(config.get("atom_embedding_dim", hidden_dim))
        edge_dim = int(config.get("edge_basis_dim", 32))
        graph_feature_dim = int(config.get("graph_feature_dim", 8))
        dropout = float(config.get("dropout", 0.0))
        self.atom_embedding = nn.Embedding(100, atom_dim)
        self.atom_projection = nn.Linear(atom_dim, hidden_dim)
        self.edge_rbf = RBFExpansion(0.0, float(config["cutoff"]), edge_dim)
        self.edge_projection = MLP(edge_dim, [hidden_dim], hidden_dim, dropout=dropout)
        self.layers = nn.ModuleList([CGCNNBlock(hidden_dim, hidden_dim, dropout) for _ in range(int(config["num_layers"]))])
        self.vn_init = MLP(graph_feature_dim, [hidden_dim], hidden_dim)
        self.vn_update = MLP(hidden_dim, [hidden_dim], hidden_dim, dropout=dropout)
        self.vn_broadcast = MLP(hidden_dim, [hidden_dim], hidden_dim, dropout=dropout)
        self.node_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(int(config["num_layers"]))])
        self.graph_projection = MLP(graph_feature_dim, [hidden_dim // 2], hidden_dim // 2)
        readout_input = hidden_dim * 2 + hidden_dim // 2
        self.readout = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        num_graphs = int(batch["graph_features"].size(0))
        node_states = self.atom_projection(self.atom_embedding(batch["atomic_numbers"]))
        edge_states = self.edge_projection(self.edge_rbf(batch["edge_distances"]))
        virtual_nodes = self.vn_init(batch["graph_features"])

        for layer, norm in zip(self.layers, self.node_norms):
            node_states = layer(node_states, batch["edge_index"], edge_states)
            vn_messages = scatter_mean(node_states, batch["node_batch"], num_graphs)
            virtual_nodes = virtual_nodes + self.vn_update(vn_messages)
            node_states = norm(node_states + self.vn_broadcast(virtual_nodes[batch["node_batch"]]))

        pooled = _graph_pool(node_states, batch["node_batch"], self.graph_projection(batch["graph_features"]))
        return {"band_gap": self.readout(pooled).squeeze(-1)}


class ALIGNNBlock(nn.Module):
    def __init__(self, hidden_dim: int, edge_dim: int, angle_dim: int, dropout: float) -> None:
        super().__init__()
        self.line_message = MLP(edge_dim * 2 + angle_dim, [edge_dim * 2], edge_dim, dropout=dropout)
        self.edge_update = MLP(hidden_dim * 2 + edge_dim * 2, [edge_dim * 2], edge_dim, dropout=dropout)
        self.node_message = MLP(hidden_dim + edge_dim, [hidden_dim * 2], hidden_dim, dropout=dropout)
        self.node_update = MLP(hidden_dim * 2, [hidden_dim * 2], hidden_dim, dropout=dropout)
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edge_index: torch.Tensor,
        line_edge_index: torch.Tensor,
        line_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if line_edge_index.numel() > 0:
            line_source, line_target = line_edge_index
            line_inputs = torch.cat([edge_states[line_source], edge_states[line_target], line_states], dim=-1)
            line_messages = self.line_message(line_inputs)
            line_aggregated = scatter_mean(line_messages, line_target, edge_states.size(0))
        else:
            line_aggregated = edge_states.new_zeros(edge_states.size(0), edge_states.size(1))

        source, target = edge_index
        edge_inputs = torch.cat([node_states[source], node_states[target], edge_states, line_aggregated], dim=-1)
        edge_states = self.edge_norm(edge_states + self.dropout(self.edge_update(edge_inputs)))

        node_inputs = torch.cat([node_states[source], edge_states], dim=-1)
        node_messages = self.node_message(node_inputs)
        node_aggregated = scatter_sum(node_messages, target, node_states.size(0))
        node_states = self.node_norm(
            node_states + self.dropout(self.node_update(torch.cat([node_states, node_aggregated], dim=-1)))
        )
        return node_states, edge_states


class ALIGNNLikeModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        edge_dim = int(config.get("edge_hidden_dim", hidden_dim))
        edge_basis_dim = int(config.get("edge_basis_dim", 41))
        angle_basis_dim = int(config.get("angle_basis_dim", 24))
        graph_feature_dim = int(config.get("graph_feature_dim", 8))
        dropout = float(config.get("dropout", 0.0))

        self.atom_embedding = nn.Embedding(100, hidden_dim)
        self.edge_rbf = RBFExpansion(0.0, float(config["cutoff"]), edge_basis_dim)
        self.angle_rbf = RBFExpansion(0.0, math.pi, angle_basis_dim)
        self.edge_projection = MLP(edge_basis_dim, [edge_dim], edge_dim, dropout=dropout)
        self.angle_projection = MLP(angle_basis_dim, [edge_dim], edge_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [ALIGNNBlock(hidden_dim, edge_dim, edge_dim, dropout) for _ in range(int(config["num_layers"]))]
        )

        self.transformer: nn.TransformerEncoder | None = None
        transformer_layers = int(config.get("transformer_layers", 0))
        self.position_frequencies = int(config.get("position_frequencies", 8))
        if transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=int(config.get("attention_heads", 4)),
                dim_feedforward=int(config.get("transformer_ffn_dim", hidden_dim * 2)),
                dropout=dropout,
                batch_first=True,
                norm_first=True,
                activation="gelu",
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
            self.position_projection = nn.Linear(3 * self.position_frequencies * 2, hidden_dim)
            self.graph_to_node = nn.Linear(graph_feature_dim, hidden_dim)

        self.graph_projection = MLP(graph_feature_dim, [hidden_dim // 2], hidden_dim // 2)
        self.edge_readout = MLP(edge_dim, [hidden_dim // 2], hidden_dim // 2, dropout=dropout)
        readout_input = hidden_dim * 2 + hidden_dim // 2 + hidden_dim // 2
        self.readout = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=dropout)

    def _apply_transformer(
        self,
        node_states: torch.Tensor,
        node_batch: torch.Tensor,
        frac_coords: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> torch.Tensor:
        if self.transformer is None:
            return node_states

        counts = torch.bincount(node_batch, minlength=graph_features.size(0)).tolist()
        max_nodes = max(counts) if counts else 0
        hidden_dim = node_states.size(-1)
        padded_states = node_states.new_zeros(graph_features.size(0), max_nodes, hidden_dim)
        mask = torch.ones(graph_features.size(0), max_nodes, dtype=torch.bool, device=node_states.device)
        position_states = self.position_projection(sinusoidal_encoding(frac_coords, self.position_frequencies))
        start = 0
        for graph_id, count in enumerate(counts):
            if count == 0:
                continue
            graph_slice = slice(start, start + count)
            padded_states[graph_id, :count] = (
                node_states[graph_slice]
                + position_states[graph_slice]
                + self.graph_to_node(graph_features[graph_id]).unsqueeze(0)
            )
            mask[graph_id, :count] = False
            start += count
        transformed = self.transformer(padded_states, src_key_padding_mask=mask)
        restored = node_states.new_zeros(node_states.size())
        start = 0
        for graph_id, count in enumerate(counts):
            if count == 0:
                continue
            restored[start : start + count] = transformed[graph_id, :count]
            start += count
        return restored

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_states = self.atom_embedding(batch["atomic_numbers"])
        edge_states = self.edge_projection(self.edge_rbf(batch["edge_distances"]))
        line_states = self.angle_projection(self.angle_rbf(batch["line_angles"]))

        for layer in self.layers:
            node_states, edge_states = layer(
                node_states,
                edge_states,
                batch["edge_index"],
                batch["line_edge_index"],
                line_states,
            )

        node_states = self._apply_transformer(
            node_states,
            batch["node_batch"],
            batch["frac_coords"],
            batch["graph_features"],
        )

        pooled_nodes = _graph_pool(node_states, batch["node_batch"], self.graph_projection(batch["graph_features"]))
        pooled_edges = scatter_mean(edge_states, batch["edge_batch"], batch["num_graphs"])
        pooled_edges = self.edge_readout(pooled_edges)
        readout_input = torch.cat([pooled_nodes, pooled_edges], dim=-1)
        return {"band_gap": self.readout(readout_input).squeeze(-1)}


class M3GNetBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        state_dim: int,
        angle_dim: int,
        graph_feature_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.triplet_message = MLP(edge_dim * 2 + angle_dim + state_dim, [edge_dim * 2], edge_dim, dropout=dropout)
        self.edge_update = MLP(hidden_dim * 2 + edge_dim * 2 + state_dim, [edge_dim * 2], edge_dim, dropout=dropout)
        self.node_message = MLP(hidden_dim + edge_dim + state_dim, [hidden_dim * 2], hidden_dim, dropout=dropout)
        self.node_update = MLP(hidden_dim * 2 + state_dim, [hidden_dim * 2], hidden_dim, dropout=dropout)
        self.state_update = MLP(
            hidden_dim + edge_dim + state_dim + graph_feature_dim,
            [state_dim * 2],
            state_dim,
            dropout=dropout,
        )
        self.edge_norm = nn.LayerNorm(edge_dim)
        self.node_norm = nn.LayerNorm(hidden_dim)
        self.state_norm = nn.LayerNorm(state_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        state: torch.Tensor,
        edge_index: torch.Tensor,
        edge_batch: torch.Tensor,
        node_batch: torch.Tensor,
        line_edge_index: torch.Tensor,
        line_states: torch.Tensor,
        graph_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if line_edge_index.numel() > 0:
            line_source, line_target = line_edge_index
            state_for_line = state[edge_batch[line_target]]
            line_inputs = torch.cat([edge_states[line_source], edge_states[line_target], line_states, state_for_line], dim=-1)
            line_messages = self.triplet_message(line_inputs)
            line_aggregated = scatter_mean(line_messages, line_target, edge_states.size(0))
        else:
            line_aggregated = edge_states.new_zeros(edge_states.size(0), edge_states.size(1))

        source, target = edge_index
        edge_inputs = torch.cat(
            [node_states[source], node_states[target], edge_states, line_aggregated, state[edge_batch]],
            dim=-1,
        )
        edge_states = self.edge_norm(edge_states + self.dropout(self.edge_update(edge_inputs)))

        node_messages = self.node_message(torch.cat([node_states[source], edge_states, state[edge_batch]], dim=-1))
        node_aggregated = scatter_sum(node_messages, target, node_states.size(0))
        node_states = self.node_norm(
            node_states
            + self.dropout(self.node_update(torch.cat([node_states, node_aggregated, state[node_batch]], dim=-1)))
        )

        pooled_nodes = scatter_mean(node_states, node_batch, state.size(0))
        pooled_edges = scatter_mean(edge_states, edge_batch, state.size(0))
        state_inputs = torch.cat([pooled_nodes, pooled_edges, state, graph_features], dim=-1)
        state = self.state_norm(state + self.dropout(self.state_update(state_inputs)))
        return node_states, edge_states, state


class M3GNetLikeModel(nn.Module):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        hidden_dim = int(config["hidden_dim"])
        edge_dim = int(config.get("edge_hidden_dim", hidden_dim))
        state_dim = int(config.get("state_hidden_dim", hidden_dim))
        edge_basis_dim = int(config.get("edge_basis_dim", 24))
        angle_basis_dim = int(config.get("angle_basis_dim", 16))
        graph_feature_dim = int(config.get("graph_feature_dim", 8))
        dropout = float(config.get("dropout", 0.0))
        self.multi_task = bool(config.get("multi_task", False))

        self.atom_embedding = nn.Embedding(100, hidden_dim)
        self.edge_rbf = RBFExpansion(0.0, float(config["cutoff"]), edge_basis_dim)
        self.angle_rbf = RBFExpansion(0.0, math.pi, angle_basis_dim)
        self.edge_projection = MLP(edge_basis_dim, [edge_dim], edge_dim, dropout=dropout)
        self.angle_projection = MLP(angle_basis_dim, [edge_dim], edge_dim, dropout=dropout)
        self.state_projection = MLP(graph_feature_dim, [state_dim], state_dim, dropout=dropout)
        self.layers = nn.ModuleList(
            [
                M3GNetBlock(hidden_dim, edge_dim, state_dim, edge_dim, graph_feature_dim, dropout)
                for _ in range(int(config["num_layers"]))
            ]
        )
        readout_input = hidden_dim * 2 + edge_dim + state_dim
        self.band_gap_head = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=dropout)
        if self.multi_task:
            self.cbm_head = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=dropout)
            self.vbm_head = MLP(readout_input, [hidden_dim, hidden_dim // 2], 1, dropout=dropout)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        node_states = self.atom_embedding(batch["atomic_numbers"])
        edge_states = self.edge_projection(self.edge_rbf(batch["edge_distances"]))
        line_states = self.angle_projection(self.angle_rbf(batch["line_angles"]))
        state = self.state_projection(batch["graph_features"])

        for layer in self.layers:
            node_states, edge_states, state = layer(
                node_states,
                edge_states,
                state,
                batch["edge_index"],
                batch["edge_batch"],
                batch["node_batch"],
                batch["line_edge_index"],
                line_states,
                batch["graph_features"],
            )

        pooled_nodes = _graph_pool(node_states, batch["node_batch"], state)
        pooled_edges = scatter_mean(edge_states, batch["edge_batch"], batch["num_graphs"])
        readout_input = torch.cat([pooled_nodes, pooled_edges], dim=-1)
        outputs = {"band_gap": self.band_gap_head(readout_input).squeeze(-1)}
        if self.multi_task:
            outputs["cbm"] = self.cbm_head(readout_input).squeeze(-1)
            outputs["vbm"] = self.vbm_head(readout_input).squeeze(-1)
        return outputs


def build_model(model_config: dict[str, Any]) -> nn.Module:
    model_name = model_config["name"]
    if model_name == "cgcnn":
        return CGCNNModel(model_config)
    if model_name == "cgcnn_virtualnode":
        return CGCNNVirtualNodeModel(model_config)
    if model_name in {"alignn", "alignn_transformer"}:
        return ALIGNNLikeModel(model_config)
    if model_name in {"m3gnet", "m3gnet_phys"}:
        return M3GNetLikeModel(model_config)
    raise ValueError(f"unsupported model name: {model_name}")
