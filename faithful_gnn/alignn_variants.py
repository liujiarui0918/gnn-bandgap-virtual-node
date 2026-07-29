from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from crystal_gnn.utils import scatter_mean, sinusoidal_encoding
from faithful_gnn.vendor import configure_vendor_paths

configure_vendor_paths()

from dgl.nn import AvgPooling
from alignn.models.alignn import ALIGNN as OfficialALIGNN  # noqa: E402
from alignn.models.alignn import ALIGNNConfig, ALIGNNConv, EdgeGatedGraphConv, MLPLayer  # noqa: E402
from alignn.models.utils import RBFExpansion  # noqa: E402


def build_alignn_config(model_config: dict[str, Any]) -> ALIGNNConfig:
    return ALIGNNConfig(
        name="alignn",
        alignn_layers=int(model_config.get("alignn_layers", 4)),
        gcn_layers=int(model_config.get("gcn_layers", 4)),
        atom_input_features=int(model_config.get("atom_input_features", 92)),
        edge_input_features=int(model_config.get("edge_input_features", 80)),
        triplet_input_features=int(model_config.get("triplet_input_features", 40)),
        embedding_features=int(model_config.get("embedding_features", 64)),
        hidden_features=int(model_config.get("hidden_features", 256)),
        output_features=int(model_config.get("output_features", 1)),
        link=model_config.get("link", "identity"),
    )


def graph_node_batch(graph, device: torch.device) -> torch.Tensor:
    batch_sizes = graph.batch_num_nodes().to(device=device, dtype=torch.long)
    return torch.repeat_interleave(torch.arange(len(batch_sizes), device=device), batch_sizes)


def pad_node_tensor(node_features: torch.Tensor, graph) -> tuple[torch.Tensor, torch.Tensor]:
    batch_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=torch.long)
    batch_size = len(batch_sizes)
    max_nodes = int(batch_sizes.max().item()) if batch_size else 0
    padded = node_features.new_zeros((batch_size, max_nodes, node_features.size(-1)))
    padding_mask = torch.ones((batch_size, max_nodes), device=node_features.device, dtype=torch.bool)
    cursor = 0
    for graph_index, count in enumerate(batch_sizes.tolist()):
        padded[graph_index, :count] = node_features[cursor : cursor + count]
        padding_mask[graph_index, :count] = False
        cursor += count
    return padded, padding_mask


def flatten_padded_tokens(padded_tokens: torch.Tensor, graph) -> torch.Tensor:
    batch_sizes = graph.batch_num_nodes().tolist()
    flat_chunks = [padded_tokens[index, :count] for index, count in enumerate(batch_sizes)]
    return torch.cat(flat_chunks, dim=0) if flat_chunks else padded_tokens.new_zeros((0, padded_tokens.size(-1)))


def masked_mean_padded_tokens(padded_tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
    valid_mask = (~padding_mask).unsqueeze(-1).to(dtype=padded_tokens.dtype)
    counts = valid_mask.sum(dim=1).clamp_min(1.0)
    return (padded_tokens * valid_mask).sum(dim=1) / counts


def graph_extra_features(graph, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    batch_size = int(graph.batch_size)
    if "extra_features" not in graph.ndata:
        return torch.zeros((batch_size, 0), device=device, dtype=dtype)
    extra_features = graph.ndata["extra_features"].to(device=device, dtype=dtype)
    node_batch = graph_node_batch(graph, device)
    return scatter_mean(extra_features, node_batch, batch_size)


class GraphTransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        attention_heads: int,
        ffn_hidden_features: int,
        dropout: float,
        layerscale_init: float,
    ) -> None:
        super().__init__()
        self.attn_norm = nn.LayerNorm(hidden_features)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_features,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.LayerNorm(hidden_features)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_features, ffn_hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden_features, hidden_features),
            nn.Dropout(dropout),
        )
        self.attn_scale = nn.Parameter(torch.full((hidden_features,), layerscale_init))
        self.ffn_scale = nn.Parameter(torch.full((hidden_features,), layerscale_init))

    def forward(self, tokens: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        attn_input = self.attn_norm(tokens)
        attn_output, _ = self.attention(
            attn_input,
            attn_input,
            attn_input,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = tokens + attn_output * self.attn_scale.view(1, 1, -1)
        ffn_output = self.ffn(self.ffn_norm(tokens))
        return tokens + ffn_output * self.ffn_scale.view(1, 1, -1)


class AttentiveVirtualNodeBlock(nn.Module):
    def __init__(
        self,
        hidden_features: int,
        num_virtual_nodes: int,
        attention_heads: int,
        dropout: float,
        node_residual_init: float,
        virtual_residual_init: float,
    ) -> None:
        super().__init__()
        self.num_virtual_nodes = num_virtual_nodes
        self.node_norm = nn.LayerNorm(hidden_features)
        self.virtual_norm = nn.LayerNorm(hidden_features)
        self.virtual_to_node_query = nn.Linear(hidden_features, hidden_features, bias=False)
        self.virtual_to_node_key = nn.Linear(hidden_features, hidden_features, bias=False)
        self.virtual_to_node_value = nn.Linear(hidden_features, hidden_features, bias=False)
        self.node_context = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
        )
        self.node_gate = nn.Sequential(
            nn.Linear(hidden_features * 2, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Sigmoid(),
        )
        self.virtual_attention = nn.MultiheadAttention(
            hidden_features,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.virtual_update = nn.Sequential(
            nn.Linear(hidden_features * 3, hidden_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, hidden_features),
        )
        self.virtual_gate = nn.Sequential(
            nn.Linear(hidden_features * 3, hidden_features),
            nn.SiLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.Sigmoid(),
        )
        self.virtual_self_norm = nn.LayerNorm(hidden_features)
        self.virtual_self_attention = nn.MultiheadAttention(
            hidden_features,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.node_residual_scale = nn.Parameter(torch.full((hidden_features,), node_residual_init))
        self.virtual_residual_scale = nn.Parameter(torch.full((hidden_features,), virtual_residual_init))
        self.virtual_self_scale = nn.Parameter(torch.full((hidden_features,), virtual_residual_init))

    def inject_to_nodes(self, graph, node_features: torch.Tensor, virtual_nodes: torch.Tensor) -> torch.Tensor:
        node_batch = graph_node_batch(graph, node_features.device)
        node_virtual_nodes = virtual_nodes[node_batch]
        if self.num_virtual_nodes == 1:
            context = node_virtual_nodes[:, 0]
        else:
            query = self.virtual_to_node_query(self.node_norm(node_features)).unsqueeze(1)
            keys = self.virtual_to_node_key(self.virtual_norm(node_virtual_nodes))
            values = self.virtual_to_node_value(self.virtual_norm(node_virtual_nodes))
            attention_scores = (query * keys).sum(dim=-1) * (keys.size(-1) ** -0.5)
            attention_weights = torch.softmax(attention_scores, dim=-1)
            context = torch.sum(attention_weights.unsqueeze(-1) * values, dim=1)
        update_input = torch.cat([node_features, context], dim=-1)
        node_update = self.node_context(update_input)
        node_gate = self.node_gate(update_input)
        return node_features + node_gate * self.node_residual_scale.view(1, -1) * node_update

    def update_virtual_nodes(self, graph, node_features: torch.Tensor, virtual_nodes: torch.Tensor) -> torch.Tensor:
        padded_nodes, padding_mask = pad_node_tensor(self.node_norm(node_features), graph)
        virtual_query = self.virtual_norm(virtual_nodes)
        attended_nodes, _ = self.virtual_attention(
            virtual_query,
            padded_nodes,
            padded_nodes,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        node_summary = masked_mean_padded_tokens(padded_nodes, padding_mask).unsqueeze(1)
        node_summary = node_summary.expand(-1, virtual_nodes.size(1), -1)
        update_input = torch.cat([virtual_nodes, attended_nodes, node_summary], dim=-1)
        virtual_update = self.virtual_update(update_input)
        virtual_gate = self.virtual_gate(update_input)
        virtual_nodes = virtual_nodes + virtual_gate * self.virtual_residual_scale.view(1, 1, -1) * virtual_update
        if self.num_virtual_nodes > 1:
            self_attention_input = self.virtual_self_norm(virtual_nodes)
            self_attention_output, _ = self.virtual_self_attention(
                self_attention_input,
                self_attention_input,
                self_attention_input,
                need_weights=False,
            )
            virtual_nodes = virtual_nodes + self.virtual_self_scale.view(1, 1, -1) * self_attention_output
        return virtual_nodes


class ALIGNNEncoder(nn.Module):
    def __init__(self, config: ALIGNNConfig) -> None:
        super().__init__()
        self.config = config
        if config.extra_features != 0:
            raise ValueError("extra_features is not supported in the faithful experiment runner.")

        self.atom_embedding = MLPLayer(config.atom_input_features, config.hidden_features)
        self.edge_embedding = nn.Sequential(
            RBFExpansion(vmin=0, vmax=8.0, bins=config.edge_input_features),
            MLPLayer(config.edge_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.angle_embedding = nn.Sequential(
            RBFExpansion(vmin=-1, vmax=1.0, bins=config.triplet_input_features),
            MLPLayer(config.triplet_input_features, config.embedding_features),
            MLPLayer(config.embedding_features, config.hidden_features),
        )
        self.alignn_layers = nn.ModuleList(
            [ALIGNNConv(config.hidden_features, config.hidden_features) for _ in range(config.alignn_layers)]
        )
        self.gcn_layers = nn.ModuleList(
            [EdgeGatedGraphConv(config.hidden_features, config.hidden_features) for _ in range(config.gcn_layers)]
        )
        self.readout = AvgPooling()

    def encode(self, graph_inputs: list[Any] | tuple[Any, ...]) -> tuple[Any, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()

        angle_features = self.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.edge_embedding(bond_lengths)

        for alignn_layer in self.alignn_layers:
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.gcn_layers:
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)
        return graph, node_features

    def pool(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        return self.readout(graph, node_features)


class FaithfulALIGNNRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig) -> None:
        super().__init__()
        self.model = OfficialALIGNN(config)

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        return {"band_gap": self.model(graph_inputs).view(-1)}


class ALIGNNVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, config.output_features)
        self.dropout = nn.Dropout(float(model_config.get("dropout", 0.0)))

    def _update_virtual_node(self, graph, node_features: torch.Tensor, virtual_node: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        broadcast = virtual_node[node_batch]
        node_features = node_features + broadcast
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node = self.virtual_node_embedding.expand(batch_size, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        fused = self.fusion(torch.cat([pooled, virtual_node], dim=-1))
        fused = self.dropout(fused)
        return {"band_gap": self.output(fused).view(-1)}


class ALIGNNTopKVirtualNodeRegressor(ALIGNNVirtualNodeRegressor):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__(config, model_config)
        hidden = config.hidden_features
        self.topk_fraction = max(float(model_config.get("topk_fraction", 0.2)), 0.0)
        self.topk_min_nodes = max(int(model_config.get("topk_min_nodes", 4)), 1)
        self.topk_max_nodes = max(int(model_config.get("topk_max_nodes", 32)), self.topk_min_nodes)
        self.topk_enable_min_atoms = max(int(model_config.get("topk_enable_min_atoms", 0)), 0)
        self.topk_enable_max_atoms = max(
            int(model_config.get("topk_enable_max_atoms", 1_000_000_000)),
            self.topk_enable_min_atoms,
        )
        self.topk_score = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, 1),
        )
        self.topk_mix_alpha = 1.0

    def set_training_epoch(self, epoch: int, total_epochs: int) -> None:
        self.topk_mix_alpha = 1.0

    def _topk_pool_nodes(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        batch_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=torch.long)
        if len(batch_sizes) == 0:
            return node_features.new_zeros((0, node_features.size(-1)))

        padded_nodes, padding_mask = pad_node_tensor(node_features, graph)
        mean_pool = masked_mean_padded_tokens(padded_nodes, padding_mask)
        max_nodes = padded_nodes.size(1)
        if max_nodes == 0:
            return mean_pool

        raw_k = torch.ceil(batch_sizes.to(dtype=torch.float32) * self.topk_fraction).to(dtype=torch.long)
        k_per_graph = raw_k.clamp(min=self.topk_min_nodes, max=self.topk_max_nodes)
        k_per_graph = torch.minimum(k_per_graph, batch_sizes.clamp_min(1))
        enabled = (
            (batch_sizes >= self.topk_enable_min_atoms)
            & (batch_sizes <= self.topk_enable_max_atoms)
            & (k_per_graph < batch_sizes)
        )
        if not bool(enabled.any().item()):
            return mean_pool

        max_k = int(k_per_graph[enabled].max().item())
        scores = self.topk_score(padded_nodes).squeeze(-1).masked_fill(padding_mask, torch.finfo(node_features.dtype).min)
        top_scores, top_indices = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=False)
        gather_index = top_indices.unsqueeze(-1).expand(-1, -1, padded_nodes.size(-1))
        selected_nodes = torch.gather(padded_nodes, dim=1, index=gather_index)

        rank = torch.arange(max_k, device=node_features.device).unsqueeze(0)
        selected_mask = enabled.unsqueeze(1) & (rank < k_per_graph.unsqueeze(1))
        masked_scores = top_scores.masked_fill(~selected_mask, torch.finfo(node_features.dtype).min)
        weights = torch.softmax(masked_scores, dim=1).masked_fill(~selected_mask, 0.0)
        topk_pool = (selected_nodes * weights.unsqueeze(-1)).sum(dim=1)
        mixed_pool = (1.0 - self.topk_mix_alpha) * mean_pool + self.topk_mix_alpha * topk_pool
        return torch.where(enabled.unsqueeze(-1), mixed_pool, mean_pool)

    def _update_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        broadcast = virtual_node[node_batch]
        node_features = node_features + broadcast
        pooled = self._topk_pool_nodes(graph, node_features)
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node


class ALIGNNWarmupTopKVirtualNodeRegressor(ALIGNNTopKVirtualNodeRegressor):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__(config, model_config)
        self.topk_warmup_epochs = max(int(model_config.get("topk_warmup_epochs", 10)), 0)
        self.topk_ramp_epochs = max(int(model_config.get("topk_ramp_epochs", 10)), 0)
        self.topk_mix_alpha = 0.0

    def set_training_epoch(self, epoch: int, total_epochs: int) -> None:
        if epoch <= self.topk_warmup_epochs:
            self.topk_mix_alpha = 0.0
        elif self.topk_ramp_epochs <= 0:
            self.topk_mix_alpha = 1.0
        else:
            ramp_position = (float(epoch) - float(self.topk_warmup_epochs)) / float(self.topk_ramp_epochs)
            self.topk_mix_alpha = float(max(0.0, min(1.0, ramp_position)))


class ALIGNNDecayVirtualNodeRegressor(ALIGNNVirtualNodeRegressor):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__(config, model_config)
        self.broadcast_decay_start_epoch = max(int(model_config.get("broadcast_decay_start_epoch", 10)), 0)
        self.broadcast_decay_end_epoch = max(
            int(model_config.get("broadcast_decay_end_epoch", 50)),
            self.broadcast_decay_start_epoch,
        )
        self.broadcast_min_scale = float(model_config.get("broadcast_min_scale", 0.25))
        self.broadcast_scale = 1.0

    def set_training_epoch(self, epoch: int, total_epochs: int) -> None:
        if epoch <= self.broadcast_decay_start_epoch:
            self.broadcast_scale = 1.0
            return
        if self.broadcast_decay_end_epoch <= self.broadcast_decay_start_epoch:
            self.broadcast_scale = self.broadcast_min_scale
            return
        progress = (float(epoch) - float(self.broadcast_decay_start_epoch)) / (
            float(self.broadcast_decay_end_epoch) - float(self.broadcast_decay_start_epoch)
        )
        progress = max(0.0, min(1.0, progress))
        self.broadcast_scale = float(1.0 + progress * (self.broadcast_min_scale - 1.0))

    def _update_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        broadcast = virtual_node[node_batch]
        node_features = node_features + self.broadcast_scale * broadcast
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node


class ALIGNNPlainDualVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        self.virtual_node_1 = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_2 = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_1_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.virtual_node_2_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden, config.output_features)
        self.dropout = nn.Dropout(float(model_config.get("dropout", 0.0)))

    def _update_virtual_nodes(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node_1: torch.Tensor,
        virtual_node_2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        virtual_context = 0.5 * (virtual_node_1 + virtual_node_2)
        node_features = node_features + virtual_context[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node_1.size(0))
        virtual_node_1 = virtual_node_1 + self.virtual_node_1_update(pooled)
        virtual_node_2 = virtual_node_2 + self.virtual_node_2_update(pooled)
        return node_features, virtual_node_1, virtual_node_2

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node_1 = self.virtual_node_1.expand(batch_size, -1)
        virtual_node_2 = self.virtual_node_2.expand(batch_size, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node_1, virtual_node_2 = self._update_virtual_nodes(
                graph, node_features, virtual_node_1, virtual_node_2
            )
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node_1, virtual_node_2 = self._update_virtual_nodes(
                graph, node_features, virtual_node_1, virtual_node_2
            )
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        fused = self.fusion(torch.cat([pooled, virtual_node_1, virtual_node_2], dim=-1))
        fused = self.dropout(fused)
        return {"band_gap": self.output(fused).view(-1)}


class ALIGNNSizeGatedPlainDualVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        dropout = float(model_config.get("dropout", 0.0))
        gate_bias_init = float(model_config.get("gate_bias_init", -2.0))
        self.use_size_features_in_gate = bool(model_config.get("use_size_features_in_gate", True))
        self.use_size_prior = bool(model_config.get("use_size_prior", True))
        self.size_prior_threshold = float(model_config.get("size_prior_threshold", 80.0))
        self.size_prior_temperature = max(float(model_config.get("size_prior_temperature", 20.0)), 1e-6)
        self.slot2_min_scale = float(model_config.get("slot2_min_scale", 0.0))
        self.slot2_max_scale = float(model_config.get("slot2_max_scale", 1.0))
        if self.slot2_max_scale < self.slot2_min_scale:
            raise ValueError("slot2_max_scale must be >= slot2_min_scale")

        self.virtual_node_1 = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_2 = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_1_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.virtual_node_2_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        gate_input_features = hidden + 2 if self.use_size_features_in_gate else hidden
        gate_hidden = max(hidden // 2, 1)
        self.size_gate = nn.Sequential(
            nn.Linear(gate_input_features, gate_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, 1),
        )
        final_gate_layer = self.size_gate[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)

        self.single_path = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.dual_path = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
        )
        self.dual_readout_scale = nn.Parameter(
            torch.full((hidden,), float(model_config.get("dual_readout_init", 0.15)))
        )
        self.output = nn.Linear(hidden, config.output_features)
        self.dropout = nn.Dropout(dropout)
        nn.init.normal_(self.virtual_node_2, mean=0.0, std=0.02)

    def _size_features(
        self,
        batch_sizes: torch.Tensor,
    ) -> torch.Tensor:
        num_atoms = batch_sizes.clamp_min(1.0)
        threshold = max(self.size_prior_threshold, 1.0)
        log_threshold = torch.log1p(num_atoms.new_tensor(threshold))
        log_num_atoms = torch.log1p(num_atoms) / log_threshold.clamp_min(1e-6)
        normalized_num_atoms = num_atoms / threshold
        return torch.cat([normalized_num_atoms, log_num_atoms], dim=-1)

    def _compute_gate_controls(
        self,
        graph,
        node_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=node_features.dtype).unsqueeze(-1)
        batch_size = int(batch_sizes.size(0))
        node_batch = graph_node_batch(graph, node_features.device)
        graph_mean = scatter_mean(node_features, node_batch, batch_size)
        gate_input = graph_mean
        if self.use_size_features_in_gate:
            gate_input = torch.cat([graph_mean, self._size_features(batch_sizes)], dim=-1)
        learned_gate_logit = self.size_gate(gate_input)
        if self.use_size_prior:
            size_prior_logit = (batch_sizes - self.size_prior_threshold) / self.size_prior_temperature
            slot2_gate_logit = learned_gate_logit + size_prior_logit
        else:
            slot2_gate_logit = learned_gate_logit
        slot2_probability = torch.sigmoid(slot2_gate_logit)
        slot2_enable = self.slot2_min_scale + slot2_probability * (self.slot2_max_scale - self.slot2_min_scale)
        return slot2_enable, slot2_gate_logit

    def _compute_size_gate(self, graph, node_features: torch.Tensor) -> torch.Tensor:
        slot2_enable, _slot2_gate_logit = self._compute_gate_controls(graph, node_features)
        return slot2_enable

    def _update_virtual_nodes(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node_1: torch.Tensor,
        virtual_node_2: torch.Tensor,
        slot2_enable: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        scaled_virtual_node_2 = virtual_node_2 * slot2_enable
        virtual_context = 0.5 * (virtual_node_1 + scaled_virtual_node_2)
        node_features = node_features + virtual_context[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node_1.size(0))
        virtual_node_1 = virtual_node_1 + self.virtual_node_1_update(pooled)
        virtual_node_2 = virtual_node_2 + slot2_enable * self.virtual_node_2_update(pooled)
        return node_features, virtual_node_1, virtual_node_2

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node_1 = self.virtual_node_1.expand(batch_size, -1)
        virtual_node_2 = self.virtual_node_2.expand(batch_size, -1)
        slot2_enable, slot2_gate_logit = self._compute_gate_controls(graph, node_features)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node_1, virtual_node_2 = self._update_virtual_nodes(
                graph, node_features, virtual_node_1, virtual_node_2, slot2_enable
            )
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node_1, virtual_node_2 = self._update_virtual_nodes(
                graph, node_features, virtual_node_1, virtual_node_2, slot2_enable
            )
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        scaled_virtual_node_2 = virtual_node_2 * slot2_enable
        single_features = self.single_path(torch.cat([pooled, virtual_node_1], dim=-1))
        dual_features = self.dual_path(torch.cat([pooled, virtual_node_1, scaled_virtual_node_2], dim=-1))
        fused = single_features + slot2_enable * self.dual_readout_scale.view(1, -1) * dual_features
        fused = self.dropout(fused)
        return {
            "band_gap": self.output(fused).view(-1),
            "slot2_enable": slot2_enable.view(-1),
            "slot2_gate_logit": slot2_gate_logit.view(-1),
            "primary_slot_state": virtual_node_1,
            "secondary_slot_state": virtual_node_2,
        }


class ALIGNNMetalAwareVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        dropout = float(model_config.get("dropout", 0.1))
        self.max_log_gap = float(model_config.get("max_log_gap", 4.0))
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.shared_readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.metal_head = nn.Linear(hidden, 1)
        self.log_gap_head = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def _update_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        node_features = node_features + virtual_node[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node = self.virtual_node_embedding.expand(batch_size, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        fused = self.dropout(self.fusion(torch.cat([pooled, virtual_node], dim=-1)))
        shared = self.shared_readout(fused)
        metal_logit = self.metal_head(shared).view(-1)
        metal_probability = torch.sigmoid(metal_logit)
        positive_log_gap = F.softplus(self.log_gap_head(shared).view(-1)).clamp(max=self.max_log_gap)
        positive_gap = torch.expm1(positive_log_gap)
        band_gap = (1.0 - metal_probability) * positive_gap
        return {
            "band_gap": band_gap.view(-1),
            "metal_logit": metal_logit,
            "metal_probability": metal_probability,
            "positive_log_gap": positive_log_gap,
            "positive_gap": positive_gap,
        }


class ALIGNNPhysBranchVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        dropout = float(model_config.get("dropout", 0.1))
        phys_residual_init = float(model_config.get("phys_residual_init", 0.2))
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.base_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.phys_projection = nn.Sequential(
            nn.LazyLinear(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )
        self.phys_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.phys_scale = nn.Parameter(torch.full((hidden,), phys_residual_init))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, config.output_features)

    def _update_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        node_features = node_features + virtual_node[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node = self.virtual_node_embedding.expand(batch_size, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        base_features = self.dropout(self.base_fusion(torch.cat([pooled, virtual_node], dim=-1)))
        extra_features = graph_extra_features(graph, node_features.device, node_features.dtype)
        if extra_features.size(-1) == 0:
            phys_features = torch.zeros_like(base_features)
        else:
            phys_features = self.phys_projection(extra_features)
        phys_gate = self.phys_gate(torch.cat([base_features, phys_features], dim=-1))
        fused = base_features + phys_gate * self.phys_scale.view(1, -1) * phys_features
        return {"band_gap": self.output(fused).view(-1)}


class ALIGNNAttentiveVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        self.num_virtual_nodes = int(model_config.get("num_virtual_nodes", 1))
        attention_heads = int(model_config.get("attention_heads", 4))
        dropout = float(model_config.get("dropout", 0.1))
        node_residual_init = float(model_config.get("node_residual_init", 0.2))
        virtual_residual_init = float(model_config.get("virtual_residual_init", 0.2))

        self.virtual_nodes = nn.Parameter(torch.zeros(1, self.num_virtual_nodes, hidden))
        self.virtual_block = AttentiveVirtualNodeBlock(
            hidden_features=hidden,
            num_virtual_nodes=self.num_virtual_nodes,
            attention_heads=attention_heads,
            dropout=dropout,
            node_residual_init=node_residual_init,
            virtual_residual_init=virtual_residual_init,
        )
        self.local_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.slot_projection = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.slot_query = nn.Linear(hidden, hidden, bias=False)
        self.slot_key = nn.Linear(hidden, hidden, bias=False)
        self.slot_value = nn.Linear(hidden, hidden, bias=False)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.fusion_scale = nn.Parameter(torch.full((hidden,), float(model_config.get("fusion_residual_init", 0.2))))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, config.output_features),
        )
        nn.init.normal_(self.virtual_nodes, mean=0.0, std=0.02)

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_nodes = self.virtual_nodes.expand(batch_size, -1, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features = self.virtual_block.inject_to_nodes(graph, node_features, virtual_nodes)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
            virtual_nodes = self.virtual_block.update_virtual_nodes(graph, node_features, virtual_nodes)
        for gcn_layer in self.encoder.gcn_layers:
            node_features = self.virtual_block.inject_to_nodes(graph, node_features, virtual_nodes)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)
            virtual_nodes = self.virtual_block.update_virtual_nodes(graph, node_features, virtual_nodes)

        pooled = self.encoder.pool(graph, node_features)
        local_features = pooled + self.local_projection(pooled)
        slot_query = self.slot_query(local_features).unsqueeze(1)
        slot_keys = self.slot_key(virtual_nodes)
        slot_values = self.slot_value(virtual_nodes)
        slot_scores = (slot_query * slot_keys).sum(dim=-1) * (slot_keys.size(-1) ** -0.5)
        slot_weights = torch.softmax(slot_scores, dim=-1)
        slot_context = torch.sum(slot_weights.unsqueeze(-1) * slot_values, dim=1)
        slot_mean = virtual_nodes.mean(dim=1)
        global_features = self.slot_projection(torch.cat([slot_context, slot_mean], dim=-1))
        fusion_gate = self.fusion_gate(torch.cat([local_features, global_features], dim=-1))
        fused = local_features + fusion_gate * self.fusion_scale.view(1, -1) * global_features
        return {"band_gap": self.output(fused).view(-1)}


class ALIGNNSizeAwareVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        self.physics_heads = bool(model_config.get("physics_heads", False))
        attention_heads = int(model_config.get("attention_heads", 4))
        dropout = float(model_config.get("dropout", 0.1))
        slot2_residual_init = float(model_config.get("slot2_residual_init", 0.15))
        fusion_residual_init = float(model_config.get("fusion_residual_init", 0.15))
        gate_bias_init = float(model_config.get("gate_bias_init", -2.0))
        self.slot2_min_scale = float(model_config.get("slot2_min_scale", 0.0))
        self.slot2_max_scale = float(model_config.get("slot2_max_scale", self.slot2_min_scale))
        self.fusion_min_scale = float(model_config.get("fusion_min_scale", 0.0))
        self.fusion_max_scale = float(model_config.get("fusion_max_scale", self.fusion_min_scale))
        self.size_prior_threshold = float(model_config.get("size_prior_threshold", 48.0))
        self.size_prior_temperature = float(model_config.get("size_prior_temperature", 18.0))
        self.use_size_features_in_gate = bool(model_config.get("use_size_features_in_gate", True))
        self.use_size_prior_floor = bool(model_config.get("use_size_prior_floor", True))

        self.primary_virtual_node = nn.Parameter(torch.zeros(1, hidden))
        self.primary_virtual_node_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )

        self.secondary_virtual_node = nn.Parameter(torch.zeros(1, hidden))
        self.node_norm = nn.LayerNorm(hidden)
        self.secondary_norm = nn.LayerNorm(hidden)
        self.secondary_attention = nn.MultiheadAttention(
            hidden,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.secondary_node_context = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.secondary_node_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.secondary_update = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.secondary_update_gate = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        gate_input_features = hidden + 2 if self.use_size_features_in_gate else hidden
        self.size_gate = nn.Sequential(
            nn.Linear(gate_input_features, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 2),
        )
        final_gate_layer = self.size_gate[-1]
        if isinstance(final_gate_layer, nn.Linear):
            nn.init.constant_(final_gate_layer.bias, gate_bias_init)

        self.primary_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.secondary_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.final_fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.slot2_residual_scale = nn.Parameter(torch.full((hidden,), slot2_residual_init))
        self.fusion_residual_scale = nn.Parameter(torch.full((hidden,), fusion_residual_init))
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(hidden, config.output_features)
        if self.physics_heads:
            self.cbm_head = nn.Linear(hidden, 1)
            self.vbm_head = nn.Linear(hidden, 1)
        nn.init.normal_(self.secondary_virtual_node, mean=0.0, std=0.02)

    def _compute_gate_controls(
        self,
        graph,
        node_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=node_features.dtype).unsqueeze(-1)
        batch_size = int(batch_sizes.size(0))
        node_batch = graph_node_batch(graph, node_features.device)
        graph_mean = scatter_mean(node_features, node_batch, batch_size)
        gate_input = graph_mean
        if self.use_size_features_in_gate:
            size_features = torch.cat(
                [
                    torch.log1p(batch_sizes) / 5.0,
                    batch_sizes.clamp(max=200.0) / 200.0,
                ],
                dim=-1,
            )
            gate_input = torch.cat([graph_mean, size_features], dim=-1)
        raw_gate_logits = self.size_gate(gate_input)
        raw_gates = torch.sigmoid(raw_gate_logits)
        if self.use_size_prior_floor:
            size_prior = torch.sigmoid((batch_sizes - self.size_prior_threshold) / self.size_prior_temperature)
            slot2_floor = self.slot2_min_scale + size_prior * (self.slot2_max_scale - self.slot2_min_scale)
            fusion_floor = self.fusion_min_scale + size_prior * (self.fusion_max_scale - self.fusion_min_scale)
        else:
            slot2_floor = torch.zeros_like(batch_sizes)
            fusion_floor = torch.zeros_like(batch_sizes)
        slot2_enable = slot2_floor + (1.0 - slot2_floor) * raw_gates[:, 0:1]
        global_fusion_strength = fusion_floor + (1.0 - fusion_floor) * raw_gates[:, 1:2]
        return slot2_enable, global_fusion_strength, raw_gate_logits

    def _compute_size_gate(self, graph, node_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        slot2_enable, global_fusion_strength, _raw_gate_logits = self._compute_gate_controls(graph, node_features)
        return slot2_enable, global_fusion_strength

    def _update_primary_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        node_features = node_features + virtual_node[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.primary_virtual_node_update(pooled)
        return node_features, virtual_node

    def _inject_secondary_slot(
        self,
        graph,
        node_features: torch.Tensor,
        secondary_slot: torch.Tensor,
        slot2_enable: torch.Tensor,
    ) -> torch.Tensor:
        node_batch = graph_node_batch(graph, node_features.device)
        secondary_context = secondary_slot[node_batch]
        update_input = torch.cat([node_features, secondary_context], dim=-1)
        node_update = self.secondary_node_context(update_input)
        node_gate = self.secondary_node_gate(update_input)
        graph_gate = slot2_enable[node_batch]
        return node_features + graph_gate * node_gate * self.slot2_residual_scale.view(1, -1) * node_update

    def _update_secondary_slot(
        self,
        graph,
        node_features: torch.Tensor,
        secondary_slot: torch.Tensor,
        slot2_enable: torch.Tensor,
    ) -> torch.Tensor:
        padded_nodes, padding_mask = pad_node_tensor(self.node_norm(node_features), graph)
        query = self.secondary_norm(secondary_slot).unsqueeze(1)
        attended_nodes, _ = self.secondary_attention(
            query,
            padded_nodes,
            padded_nodes,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        node_summary = masked_mean_padded_tokens(padded_nodes, padding_mask)
        update_input = torch.cat([secondary_slot, attended_nodes.squeeze(1), node_summary], dim=-1)
        secondary_update = self.secondary_update(update_input)
        secondary_gate = self.secondary_update_gate(update_input)
        return (
            secondary_slot
            + slot2_enable
            * secondary_gate
            * self.slot2_residual_scale.view(1, -1)
            * secondary_update
        )

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        primary_slot = self.primary_virtual_node.expand(batch_size, -1)
        secondary_slot = self.secondary_virtual_node.expand(batch_size, -1)
        slot2_enable, global_fusion_strength, raw_gate_logits = self._compute_gate_controls(graph, node_features)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, primary_slot = self._update_primary_virtual_node(graph, node_features, primary_slot)
            node_features = self._inject_secondary_slot(graph, node_features, secondary_slot, slot2_enable)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
            secondary_slot = self._update_secondary_slot(graph, node_features, secondary_slot, slot2_enable)
        for gcn_layer in self.encoder.gcn_layers:
            node_features, primary_slot = self._update_primary_virtual_node(graph, node_features, primary_slot)
            node_features = self._inject_secondary_slot(graph, node_features, secondary_slot, slot2_enable)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)
            secondary_slot = self._update_secondary_slot(graph, node_features, secondary_slot, slot2_enable)

        pooled = self.encoder.pool(graph, node_features)
        base_features = self.primary_fusion(torch.cat([pooled, primary_slot], dim=-1))
        base_features = self.dropout(base_features)
        scaled_secondary = secondary_slot * slot2_enable
        secondary_features = self.secondary_projection(torch.cat([pooled, primary_slot, scaled_secondary], dim=-1))
        fusion_gate = self.final_fusion_gate(torch.cat([base_features, secondary_features], dim=-1))
        fused = (
            base_features
            + global_fusion_strength
            * fusion_gate
            * self.fusion_residual_scale.view(1, -1)
            * secondary_features
        )
        outputs = {
            "band_gap": self.output(fused).view(-1),
            "slot2_enable": slot2_enable.view(-1),
            "global_fusion_strength": global_fusion_strength.view(-1),
            "slot2_gate_logit": raw_gate_logits[:, 0].view(-1),
            "global_fusion_logit": raw_gate_logits[:, 1].view(-1),
            "primary_slot_state": primary_slot,
            "secondary_slot_state": secondary_slot,
        }
        if self.physics_heads:
            outputs["cbm"] = self.cbm_head(fused).view(-1)
            outputs["vbm"] = self.vbm_head(fused).view(-1)
        return outputs


class ALIGNNTransformerRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        position_frequencies = int(model_config.get("position_frequencies", 8))
        radial_frequencies = int(model_config.get("radial_frequencies", max(4, position_frequencies // 2)))
        heads = int(model_config.get("attention_heads", 4))
        transformer_layers = int(model_config.get("transformer_layers", 2))
        ffn_hidden = int(model_config.get("ffn_hidden_features", hidden * 2))
        dropout = float(model_config.get("dropout", 0.1))
        layerscale_init = float(model_config.get("transformer_layerscale_init", 1e-3))
        global_residual_init = float(model_config.get("global_residual_init", 0.15))

        self.position_frequencies = position_frequencies
        self.radial_frequencies = radial_frequencies
        position_dim = (3 * position_frequencies * 2 * 2) + (radial_frequencies * 2)
        self.position_projection = nn.Sequential(
            nn.Linear(position_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.degree_projection = nn.Sequential(
            nn.Linear(1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.graph_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.graph_token_init = nn.Sequential(
            nn.Linear(hidden * 2 + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.pre_transformer_norm = nn.LayerNorm(hidden)
        self.transformer_blocks = nn.ModuleList(
            [
                GraphTransformerBlock(
                    hidden_features=hidden,
                    attention_heads=heads,
                    ffn_hidden_features=ffn_hidden,
                    dropout=dropout,
                    layerscale_init=layerscale_init,
                )
                for _ in range(transformer_layers)
            ]
        )
        self.post_transformer_norm = nn.LayerNorm(hidden)
        self.local_projection = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.global_projection = nn.Sequential(
            nn.Linear(hidden * 3, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.readout_query = nn.Linear(hidden, hidden, bias=False)
        self.fusion_gate = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.Sigmoid(),
        )
        self.global_residual_scale = nn.Parameter(torch.full((hidden,), global_residual_init))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, config.output_features),
        )
        nn.init.normal_(self.graph_token, mean=0.0, std=0.02)

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, node_features = self.encoder.encode(graph_inputs)
        frac_coords = graph.ndata["frac_coords"]
        batch_size = int(graph.batch_size)
        node_batch = graph_node_batch(graph, node_features.device)
        graph_centers = scatter_mean(frac_coords, node_batch, batch_size)
        centered_coords = frac_coords - graph_centers[node_batch]
        radial_distance = centered_coords.norm(dim=-1, keepdim=True)

        padded_nodes, padding_mask = pad_node_tensor(node_features, graph)
        position_features = torch.cat(
            [
                sinusoidal_encoding(frac_coords, self.position_frequencies),
                sinusoidal_encoding(centered_coords, self.position_frequencies),
                sinusoidal_encoding(radial_distance, self.radial_frequencies),
            ],
            dim=-1,
        )
        position_tokens, _ = pad_node_tensor(self.position_projection(position_features), graph)
        node_degrees = graph.in_degrees().to(device=node_features.device, dtype=node_features.dtype).unsqueeze(-1)
        degree_tokens, _ = pad_node_tensor(self.degree_projection(torch.log1p(node_degrees)), graph)

        local_pooled = self.encoder.pool(graph, node_features)
        geometry_summary = masked_mean_padded_tokens(position_tokens, padding_mask)
        graph_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=node_features.dtype).unsqueeze(-1)
        graph_token = self.graph_token.expand(batch_size, -1, -1) + self.graph_token_init(
            torch.cat([local_pooled, geometry_summary, torch.log1p(graph_sizes)], dim=-1)
        ).unsqueeze(1)

        node_tokens = padded_nodes + position_tokens + degree_tokens
        transformer_input = torch.cat([graph_token, node_tokens], dim=1)
        transformer_padding_mask = torch.cat(
            [torch.zeros((batch_size, 1), device=node_features.device, dtype=torch.bool), padding_mask],
            dim=1,
        )
        transformer_input = self.pre_transformer_norm(transformer_input)
        for transformer_block in self.transformer_blocks:
            transformer_input = transformer_block(transformer_input, transformer_padding_mask)
        transformer_output = self.post_transformer_norm(transformer_input)

        graph_token_output = transformer_output[:, 0]
        node_token_output = transformer_output[:, 1:]
        global_mean = masked_mean_padded_tokens(node_token_output, padding_mask)
        readout_query = self.readout_query(graph_token_output).unsqueeze(1)
        attention_scores = (node_token_output * readout_query).sum(dim=-1) * (node_token_output.size(-1) ** -0.5)
        attention_scores = attention_scores.masked_fill(padding_mask, -1e9)
        attention_weights = torch.softmax(attention_scores, dim=-1).masked_fill(padding_mask, 0.0)
        attentive_pool = torch.sum(attention_weights.unsqueeze(-1) * node_token_output, dim=1)

        local_features = local_pooled + self.local_projection(local_pooled)
        global_features = self.global_projection(torch.cat([graph_token_output, attentive_pool, global_mean], dim=-1))
        fusion_gate = self.fusion_gate(torch.cat([local_features, global_features], dim=-1))
        fused = local_features + fusion_gate * (self.global_residual_scale.view(1, -1) * global_features)
        return {"band_gap": self.output(fused).view(-1)}


class ALIGNNRegimeMoEVirtualNodeRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig, model_config: dict[str, Any]) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        dropout = float(model_config.get("dropout", 0.1))
        regime_hidden = int(model_config.get("regime_hidden_features", hidden))
        self.virtual_node_embedding = nn.Parameter(torch.zeros(1, hidden))
        self.virtual_node_update = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
        )
        self.base_fusion = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.SiLU(),
        )
        self.shared_readout = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, regime_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.base_head = nn.Linear(regime_hidden, config.output_features)
        self.regime_logits = nn.Linear(regime_hidden, 3)
        self.regime_heads = nn.ModuleList([nn.Linear(regime_hidden, config.output_features) for _ in range(3)])
        self.dropout = nn.Dropout(dropout)

    def _update_virtual_node(
        self,
        graph,
        node_features: torch.Tensor,
        virtual_node: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_batch = graph_node_batch(graph, node_features.device)
        node_features = node_features + virtual_node[node_batch]
        pooled = scatter_mean(node_features, node_batch, virtual_node.size(0))
        virtual_node = virtual_node + self.virtual_node_update(pooled)
        return node_features, virtual_node

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, line_graph, _lattice = graph_inputs
        graph = graph.local_var()
        line_graph = line_graph.local_var()
        angle_features = self.encoder.angle_embedding(line_graph.edata.pop("h"))
        node_features = self.encoder.atom_embedding(graph.ndata.pop("atom_features"))
        bond_lengths = torch.norm(graph.edata.pop("r"), dim=1)
        edge_features = self.encoder.edge_embedding(bond_lengths)
        batch_size = int(graph.batch_size)
        virtual_node = self.virtual_node_embedding.expand(batch_size, -1)

        for alignn_layer in self.encoder.alignn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features, angle_features = alignn_layer(
                graph, line_graph, node_features, edge_features, angle_features
            )
        for gcn_layer in self.encoder.gcn_layers:
            node_features, virtual_node = self._update_virtual_node(graph, node_features, virtual_node)
            node_features, edge_features = gcn_layer(graph, node_features, edge_features)

        pooled = self.encoder.pool(graph, node_features)
        base_features = self.dropout(self.base_fusion(torch.cat([pooled, virtual_node], dim=-1)))
        shared_features = self.shared_readout(base_features)
        regime_logits = self.regime_logits(shared_features)
        regime_weights = torch.softmax(regime_logits, dim=-1)
        base_band_gap = self.base_head(shared_features)
        regime_residuals = torch.cat([head(shared_features) for head in self.regime_heads], dim=-1)
        regime_band_gaps = base_band_gap + regime_residuals
        band_gap = torch.sum(regime_weights * regime_band_gaps, dim=-1, keepdim=True)
        return {
            "band_gap": band_gap.view(-1),
            "regime_logits": regime_logits,
            "regime_band_gaps": regime_band_gaps,
        }


class ALIGNNPhysicsRegressor(nn.Module):
    def __init__(self, config: ALIGNNConfig) -> None:
        super().__init__()
        self.encoder = ALIGNNEncoder(config)
        hidden = config.hidden_features
        self.band_gap_head = nn.Linear(hidden, 1)
        self.cbm_head = nn.Linear(hidden, 1)
        self.vbm_head = nn.Linear(hidden, 1)

    def forward(self, graph_inputs: list[Any] | tuple[Any, ...]) -> dict[str, torch.Tensor]:
        graph, node_features = self.encoder.encode(graph_inputs)
        pooled = self.encoder.pool(graph, node_features)
        return {
            "band_gap": self.band_gap_head(pooled).view(-1),
            "cbm": self.cbm_head(pooled).view(-1),
            "vbm": self.vbm_head(pooled).view(-1),
        }


def build_alignn_model(model_config: dict[str, Any]) -> nn.Module:
    config = build_alignn_config(model_config)
    variant = model_config.get("variant", "baseline").lower()
    if variant == "baseline":
        return FaithfulALIGNNRegressor(config)
    if variant == "virtualnode":
        return ALIGNNVirtualNodeRegressor(config, model_config)
    if variant == "topkvirtualnode":
        return ALIGNNTopKVirtualNodeRegressor(config, model_config)
    if variant == "warmuptopkvirtualnode":
        return ALIGNNWarmupTopKVirtualNodeRegressor(config, model_config)
    if variant == "decayvirtualnode":
        return ALIGNNDecayVirtualNodeRegressor(config, model_config)
    if variant == "plaindualvirtualnode":
        return ALIGNNPlainDualVirtualNodeRegressor(config, model_config)
    if variant in {"sizegatedplaindualvirtualnode", "adaptiveplaindualvirtualnode"}:
        return ALIGNNSizeGatedPlainDualVirtualNodeRegressor(config, model_config)
    if variant == "metalawarevirtualnode":
        return ALIGNNMetalAwareVirtualNodeRegressor(config, model_config)
    if variant == "physbranchvirtualnode":
        return ALIGNNPhysBranchVirtualNodeRegressor(config, model_config)
    if variant == "attentivevirtualnode":
        return ALIGNNAttentiveVirtualNodeRegressor(config, model_config)
    if variant == "multivirtualnode":
        return ALIGNNAttentiveVirtualNodeRegressor(config, model_config)
    if variant == "sizeawarevirtualnode":
        return ALIGNNSizeAwareVirtualNodeRegressor(config, model_config)
    if variant in {"gainsupervisedvirtualnode", "complexitysupervisedvirtualnode", "orthogonaldualslot"}:
        adaptive_config = dict(model_config)
        adaptive_config.setdefault("use_size_features_in_gate", False)
        adaptive_config.setdefault("use_size_prior_floor", False)
        return ALIGNNSizeAwareVirtualNodeRegressor(config, adaptive_config)
    if variant == "sizesupervisedvirtualnode":
        size_supervised_config = dict(model_config)
        size_supervised_config.setdefault("use_size_features_in_gate", False)
        size_supervised_config.setdefault("use_size_prior_floor", False)
        return ALIGNNSizeAwareVirtualNodeRegressor(config, size_supervised_config)
    if variant == "sizesupervisedphysvirtualnode":
        size_supervised_phys_config = dict(model_config)
        size_supervised_phys_config.setdefault("use_size_features_in_gate", False)
        size_supervised_phys_config.setdefault("use_size_prior_floor", False)
        size_supervised_phys_config["physics_heads"] = True
        return ALIGNNSizeAwareVirtualNodeRegressor(config, size_supervised_phys_config)
    if variant == "transformer":
        return ALIGNNTransformerRegressor(config, model_config)
    if variant == "regimemoevirtualnode":
        return ALIGNNRegimeMoEVirtualNodeRegressor(config, model_config)
    if variant == "physconstraint":
        return ALIGNNPhysicsRegressor(config)
    raise ValueError(f"unsupported ALIGNN variant: {variant}")
