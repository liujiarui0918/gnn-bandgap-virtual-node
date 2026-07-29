#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
MAMBA_BIN="${MAMBA_BIN:-$WORK_ROOT/miniforge3/micromamba}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gnn-faithful}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
DGL_WHEEL_URL="${DGL_WHEEL_URL:-https://data.dgl.ai/wheels/torch-2.2/cu121/repo.html}"

mkdir -p "$WORK_ROOT"/{installers,logs,run,envs,pkgs}
mkdir -p "$WORK_ROOT/tmp" "$WORK_ROOT/pip_cache"
export TMPDIR="$WORK_ROOT/tmp"
export TEMP="$WORK_ROOT/tmp"
export TMP="$WORK_ROOT/tmp"
export PIP_CACHE_DIR="$WORK_ROOT/pip_cache"
export PYTHONNOUSERSITE=1

if [ ! -x "$MAMBA_BIN" ]; then
  cd "$WORK_ROOT/installers"
  wget -q -O Miniforge3-Linux-x86_64.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$WORK_ROOT/miniforge3"
fi

if [ ! -x "$ENV_PREFIX/bin/python" ]; then
  "$MAMBA_BIN" create -y -p "$ENV_PREFIX" -c conda-forge python=3.10 pip
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
import pymatgen
import yaml
import lightning
import jarvis
import matgl
from alignn.models.alignn import ALIGNN, ALIGNNConfig
from matgl.models import M3GNet

try:
    import dgl
    dgl_version = dgl.__version__
except Exception as exc:
    dgl_version = f"import_failed_on_this_node:{type(exc).__name__}:{exc}"

alignn_model = ALIGNN(ALIGNNConfig(name="alignn"))
m3gnet_model = M3GNet(element_types=("H",), is_intensive=True)

print("python", sys.version)
print("python_bin", sys.executable)
print("torch", torch.__version__, "cuda_available_on_this_node", torch.cuda.is_available())
print("dgl", dgl_version)
print("pymatgen", getattr(pymatgen, "__version__", "unknown"))
print("lightning", lightning.__version__)
print("matgl", getattr(matgl, "__version__", "unknown"))
print("jarvis", getattr(jarvis, "__version__", "unknown"))
print("alignn_params", sum(p.numel() for p in alignn_model.parameters()))
print("m3gnet_params", sum(p.numel() for p in m3gnet_model.parameters()))
PY
