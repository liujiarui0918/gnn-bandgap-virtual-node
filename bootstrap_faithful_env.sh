#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ROOT="${CONDA_ROOT:-/root/miniconda}"
ENV_NAME="${ENV_NAME:-gnn-faithful}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON_BIN="$CONDA_ROOT/envs/$ENV_NAME/bin/python"
CONDA_BIN="$CONDA_ROOT/bin/conda"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DGL_WHEEL_URL="${DGL_WHEEL_URL:-https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html}"

if [ ! -x "$CONDA_BIN" ]; then
  echo "conda not found at $CONDA_BIN" >&2
  exit 1
fi

for required_path in \
  "$PROJECT_DIR/.vendor/cgcnn/cgcnn" \
  "$PROJECT_DIR/.vendor/alignn/alignn" \
  "$PROJECT_DIR/.vendor/matgl/src/matgl"
do
  if [ ! -e "$required_path" ]; then
    echo "required vendor path missing: $required_path" >&2
    exit 1
  fi
done

if [ ! -d "$CONDA_ROOT/envs/$ENV_NAME" ]; then
  "$CONDA_BIN" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

"$PYTHON_BIN" -m pip install --upgrade pip setuptools wheel
"$PYTHON_BIN" -m pip install --index-url "$TORCH_INDEX_URL" "torch==2.2.1"
"$PYTHON_BIN" -m pip install "torchdata==0.7.1"
"$PYTHON_BIN" -m pip install "dgl==2.2.1" -f "$DGL_WHEEL_URL"
"$PYTHON_BIN" -m pip install \
  "numpy<2" \
  "scipy>=1.11" \
  "PyYAML>=6.0" \
  "matplotlib>=3.8" \
  "tqdm>=4.66" \
  "pandas>=2.2" \
  "scikit-learn>=1.5" \
  "pydantic>=2.7" \
  "pydantic-settings>=2.2" \
  "ase>=3.23" \
  "lmdb>=1.5" \
  "jarvis-tools>=2024.10.30" \
  "lightning<=2.6.1" \
  "pymatgen>=2025.1.9"

"$PYTHON_BIN" - <<'PY'
import sys
import os
os.environ.setdefault("MATGL_BACKEND", "DGL")
import torch
import dgl
import pymatgen
import yaml
import lightning
import jarvis
from pathlib import Path

project_dir = Path.cwd()
sys.path.insert(0, str(project_dir / ".vendor" / "alignn"))
sys.path.insert(0, str(project_dir / ".vendor" / "cgcnn"))
sys.path.insert(0, str(project_dir / ".vendor" / "matgl" / "src"))

from alignn.models.alignn import ALIGNNConfig, ALIGNN
from matgl.models import M3GNet
import matgl

alignn_model = ALIGNN(ALIGNNConfig(name="alignn"))
m3gnet_model = M3GNet(element_types=("H",), is_intensive=True)

print("python", sys.version)
print("torch", torch.__version__)
print("dgl", dgl.__version__)
print("pymatgen", getattr(pymatgen, "__version__", "unknown"))
print("lightning", lightning.__version__)
print("matgl", getattr(matgl, "__version__", "unknown"))
print("jarvis", getattr(jarvis, "__version__", "unknown"))
print("alignn_params", sum(p.numel() for p in alignn_model.parameters()))
print("m3gnet_params", sum(p.numel() for p in m3gnet_model.parameters()))
print("cuda_available", torch.cuda.is_available())
print("python_bin", sys.executable)
PY
