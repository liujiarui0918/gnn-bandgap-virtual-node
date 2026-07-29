#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
MINIFORGE_DIR="${MINIFORGE_DIR:-$WORK_ROOT/miniforge3}"
ENV_PREFIX="${ENV_PREFIX:-$WORK_ROOT/envs/gnn-faithful}"

mkdir -p "$WORK_ROOT"/{installers,logs,run,envs}
mkdir -p "$WORK_ROOT/tmp" "$WORK_ROOT/pip_cache"
export TMPDIR="$WORK_ROOT/tmp"
export TEMP="$WORK_ROOT/tmp"
export TMP="$WORK_ROOT/tmp"
export PIP_CACHE_DIR="$WORK_ROOT/pip_cache"

if [ ! -x "$MINIFORGE_DIR/bin/conda" ]; then
  cd "$WORK_ROOT/installers"
  wget -q -O Miniforge3-Linux-x86_64.sh \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash Miniforge3-Linux-x86_64.sh -b -p "$MINIFORGE_DIR"
fi

cd "$PROJECT_ROOT"
CONDA_BIN="$MINIFORGE_DIR/bin/conda" ENV_PREFIX="$ENV_PREFIX" bash bootstrap_remote_env.sh
