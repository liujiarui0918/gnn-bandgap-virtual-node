#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gnn-faithful}"
PYTHON_BASE="${PYTHON_BASE:-python3}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DGL_WHEEL_URL="${DGL_WHEEL_URL:-https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html}"

mkdir -p "$WORK_ROOT"/{data,cache,results,logs,run,envs}
mkdir -p "$WORK_ROOT/tmp" "$WORK_ROOT/pip_cache"
export TMPDIR="$WORK_ROOT/tmp"
export TEMP="$WORK_ROOT/tmp"
export TMP="$WORK_ROOT/tmp"
export PIP_CACHE_DIR="$WORK_ROOT/pip_cache"

if [ ! -d "$ENV_PREFIX" ]; then
  "$PYTHON_BASE" -m venv "$ENV_PREFIX"
fi

PYTHON_BIN="$ENV_PREFIX/bin/python"
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
  "pymatgen>=2025.1.9" \
  "requests>=2.32"

cd "$PROJECT_ROOT"
MATGL_BACKEND=DGL "$PYTHON_BIN" - <<'PY'
import os
import sys
from pathlib import Path

os.environ.setdefault("MATGL_BACKEND", "DGL")
project_dir = Path.cwd()
sys.path.insert(0, str(project_dir / ".vendor" / "alignn"))
sys.path.insert(0, str(project_dir / ".vendor" / "cgcnn"))
sys.path.insert(0, str(project_dir / ".vendor" / "matgl" / "src"))

import torch
import dgl
import pymatgen
import yaml
import lightning
import jarvis
import matgl
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from matgl.models import M3GNet

alignn_model = ALIGNN(ALIGNNConfig(name="alignn"))
m3gnet_model = M3GNet(element_types=("H",), is_intensive=True)

print("python", sys.version)
print("python_bin", sys.executable)
print("torch", torch.__version__, "cuda_available_on_this_node", torch.cuda.is_available())
print("dgl", dgl.__version__)
print("pymatgen", getattr(pymatgen, "__version__", "unknown"))
print("lightning", lightning.__version__)
print("matgl", getattr(matgl, "__version__", "unknown"))
print("jarvis", getattr(jarvis, "__version__", "unknown"))
print("alignn_params", sum(p.numel() for p in alignn_model.parameters()))
print("m3gnet_params", sum(p.numel() for p in m3gnet_model.parameters()))
PY
