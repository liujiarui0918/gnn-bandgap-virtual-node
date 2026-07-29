#!/usr/bin/env bash
#SBATCH -J gnn_s_equal_mp
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
DATASET_DIR="${DATASET_DIR:-$WORK_ROOT/data/mp_hzw_final_clean_equal_atoms_mp}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_final_clean_equal_atoms_mp_s}"
CACHE_ROOT="${CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_final_clean_equal_atoms_mp_s}"
CGCNN_ROOT="${CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_final_clean_equal_atoms_mp_cgcnn}"

mkdir -p "$WORK_ROOT"/{cache,results,logs,run,tmp}

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python env missing: $PYTHON_BIN" >&2
  exit 3
fi
if [ ! -f "$DATASET_DIR/manifest.jsonl" ]; then
  echo "Final dataset missing: $DATASET_DIR/manifest.jsonl" >&2
  exit 4
fi

cd "$PROJECT_ROOT"

export MATGL_BACKEND="${MATGL_BACKEND:-DGL}"
export DGLBACKEND="${DGLBACKEND:-pytorch}"
export GNN_ENABLE_TF32="${GNN_ENABLE_TF32:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-10}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-10}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128,garbage_collection_threshold:0.8}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export TMPDIR="${TMPDIR:-$WORK_ROOT/tmp}"
export TEMP="${TEMP:-$WORK_ROOT/tmp}"
export TMP="${TMP:-$WORK_ROOT/tmp}"

"$PYTHON_BIN" - <<'PY'
import sys
import torch

print("python", sys.executable)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
else:
    raise SystemExit("CUDA is not available inside the Slurm job.")
try:
    import sklearn
    from jarvis.core.atoms import pmg_to_atoms  # noqa: F401
    print("sklearn", sklearn.__version__)
    print("jarvis_import_ok")
except Exception as exc:
    raise SystemExit(f"dependency check failed: {type(exc).__name__}: {exc}") from exc
PY

"$PYTHON_BIN" run_s_clean_matched_experiments.py \
  --hzw_raw_path "$HZW_RAW_PATH" \
  --audit_dir "$WORK_ROOT/results/data_quality" \
  --dataset_dir "$DATASET_DIR" \
  --results_dir "$RESULTS_DIR" \
  --cache_root "$CACHE_ROOT" \
  --cgcnn_root "$CGCNN_ROOT" \
  --experiments S1 S2 S3 S4 S5 S7 S8 S10 S11 S12 S13 \
  --performance_mode 4090 \
  --skip_audit \
  --skip_export
