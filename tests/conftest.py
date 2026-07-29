from __future__ import annotations

import sys
import types
from pathlib import Path

import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT_STR = str(PROJECT_ROOT)
if PROJECT_ROOT_STR not in sys.path:
    sys.path.insert(0, PROJECT_ROOT_STR)


def _install_fake_dgl() -> None:
    if "dgl" in sys.modules:
        return

    dgl_module = types.ModuleType("dgl")
    dgl_nn_module = types.ModuleType("dgl.nn")
    dgl_dataloading_module = types.ModuleType("dgl.dataloading")

    class AvgPooling(nn.Module):
        def forward(self, graph, node_features: torch.Tensor) -> torch.Tensor:
            if hasattr(graph, "batch_num_nodes"):
                batch_sizes = graph.batch_num_nodes().to(device=node_features.device, dtype=torch.long)
                outputs = []
                cursor = 0
                for count in batch_sizes.tolist():
                    outputs.append(node_features[cursor : cursor + count].mean(dim=0))
                    cursor += count
                return torch.stack(outputs, dim=0)
            return node_features.mean(dim=0, keepdim=True)

    class GraphDataLoader(list):
        pass

    dgl_nn_module.AvgPooling = AvgPooling
    dgl_dataloading_module.GraphDataLoader = GraphDataLoader
    dgl_module.nn = dgl_nn_module
    dgl_module.dataloading = dgl_dataloading_module
    sys.modules["dgl"] = dgl_module
    sys.modules["dgl.nn"] = dgl_nn_module
    sys.modules["dgl.dataloading"] = dgl_dataloading_module


def _install_fake_alignn() -> None:
    if "alignn" in sys.modules:
        return

    alignn_module = types.ModuleType("alignn")
    alignn_models_module = types.ModuleType("alignn.models")
    alignn_models_alignn_module = types.ModuleType("alignn.models.alignn")
    alignn_models_utils_module = types.ModuleType("alignn.models.utils")
    alignn_data_module = types.ModuleType("alignn.data")

    class ALIGNNConfig:
        def __init__(self, **kwargs):
            self.extra_features = kwargs.pop("extra_features", 0)
            for key, value in kwargs.items():
                setattr(self, key, value)

    class ALIGNN(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.output_features = getattr(config, "output_features", 1)

        def forward(self, graph_inputs):
            graph = graph_inputs[0]
            batch_size = int(getattr(graph, "batch_size", 1))
            return torch.zeros(batch_size, self.output_features)

    class ALIGNNConv(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

        def forward(self, graph, line_graph, node_features, edge_features, angle_features):
            return node_features, edge_features, angle_features

    class EdgeGatedGraphConv(nn.Module):
        def __init__(self, *_args, **_kwargs):
            super().__init__()

        def forward(self, graph, node_features, edge_features):
            return node_features, edge_features

    class MLPLayer(nn.Module):
        def __init__(self, in_features: int, out_features: int):
            super().__init__()
            self.linear = nn.Linear(in_features, out_features)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            return self.linear(inputs)

    class RBFExpansion(nn.Module):
        def __init__(self, vmin=0.0, vmax=1.0, bins=40):
            super().__init__()
            self.bins = bins

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            values = inputs.view(-1, 1)
            return values.repeat(1, self.bins)

    def get_train_val_loaders(**_kwargs):
        loader = []
        return loader, loader, loader, None

    alignn_models_alignn_module.ALIGNN = ALIGNN
    alignn_models_alignn_module.ALIGNNConfig = ALIGNNConfig
    alignn_models_alignn_module.ALIGNNConv = ALIGNNConv
    alignn_models_alignn_module.EdgeGatedGraphConv = EdgeGatedGraphConv
    alignn_models_alignn_module.MLPLayer = MLPLayer
    alignn_models_utils_module.RBFExpansion = RBFExpansion
    alignn_data_module.get_train_val_loaders = get_train_val_loaders

    alignn_module.models = alignn_models_module
    sys.modules["alignn"] = alignn_module
    sys.modules["alignn.models"] = alignn_models_module
    sys.modules["alignn.models.alignn"] = alignn_models_alignn_module
    sys.modules["alignn.models.utils"] = alignn_models_utils_module
    sys.modules["alignn.data"] = alignn_data_module


def _install_fake_jarvis_and_pymatgen() -> None:
    if "jarvis" not in sys.modules:
        jarvis_module = types.ModuleType("jarvis")
        jarvis_core_module = types.ModuleType("jarvis.core")
        jarvis_atoms_module = types.ModuleType("jarvis.core.atoms")

        class _Atoms:
            def to_dict(self):
                return {}

        def pmg_to_atoms(_structure):
            return _Atoms()

        jarvis_atoms_module.pmg_to_atoms = pmg_to_atoms
        jarvis_module.core = jarvis_core_module
        sys.modules["jarvis"] = jarvis_module
        sys.modules["jarvis.core"] = jarvis_core_module
        sys.modules["jarvis.core.atoms"] = jarvis_atoms_module

    if "pymatgen" not in sys.modules:
        pymatgen_module = types.ModuleType("pymatgen")
        pymatgen_core_module = types.ModuleType("pymatgen.core")

        class Structure:
            @classmethod
            def from_dict(cls, _payload):
                return cls()

        pymatgen_core_module.Structure = Structure
        pymatgen_module.core = pymatgen_core_module
        sys.modules["pymatgen"] = pymatgen_module
        sys.modules["pymatgen.core"] = pymatgen_core_module


_install_fake_dgl()
_install_fake_alignn()
_install_fake_jarvis_and_pymatgen()
