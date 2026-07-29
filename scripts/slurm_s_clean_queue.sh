#!/usr/bin/env bash
#SBATCH -J gnn_s_clean
#SBATCH -p tx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -t 7-00:00:00
#SBATCH -o /data/home/tg100025/gnn_work/logs/slurm-%x-%j.out

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
PYTHON_BIN="${PYTHON_BIN:-$WORK_ROOT/envs/gnn-faithful/bin/python}"
HZW_RAW_PATH="${HZW_RAW_PATH:-$WORK_ROOT/data/raw/hzw/crystal_band_100000.jsonl.gz.tmp}"

mkdir -p "$WORK_ROOT"/{data,cache,results,logs,run}

if [ -z "${MP_API_KEY:-}" ] && [ -f "$HOME/.mp_api_key" ]; then
  MP_API_KEY="$(tr -d '\r\n' < "$HOME/.mp_api_key")"
  export MP_API_KEY
fi

if [ -z "${MP_API_KEY:-}" ]; then
  echo "MP_API_KEY is required. Export it before sbatch or write it to \$HOME/.mp_api_key with chmod 600." >&2
  exit 2
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env missing: $PYTHON_BIN" >&2
  exit 3
fi

cd "$PROJECT_ROOT"

export MATGL_BACKEND="${MATGL_BACKEND:-DGL}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export GNN_ENABLE_TF32="${GNN_ENABLE_TF32:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
export TEMP="${TEMP:-$WORK_ROOT/tmp}"
export TMP="${TMP:-$WORK_ROOT/tmp}"
mkdir -p "$TMPDIR"

"$PYTHON_BIN" - <<'PY'
import os
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available inside the Slurm job.")
PY

"$PYTHON_BIN" run_s_clean_matched_experiments.py \
  --hzw_raw_path "$HZW_RAW_PATH" \
  --audit_dir "$WORK_ROOT/results/data_quality" \
  --dataset_dir "$WORK_ROOT/data/mp_hzw_clean_matched_s" \
  --results_dir "$WORK_ROOT/results/mp_hzw_clean_matched_s" \
  --cache_root "$WORK_ROOT/cache/mp_hzw_clean_s" \
  --cgcnn_root "$WORK_ROOT/data/mp_hzw_clean_matched_s_cgcnn" \
  --experiments S1 S2 S3 S4 S5 S7 S8 S10 S11 S12 \
  --performance_mode 4090
