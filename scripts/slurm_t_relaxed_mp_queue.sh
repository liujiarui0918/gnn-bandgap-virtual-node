#!/usr/bin/env bash
#SBATCH -J gnn_t_relaxed_dyn
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
MP_DATASET_DIR="${MP_DATASET_DIR:-${DATASET_DIR:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_mp}}"
MP_RESULTS_DIR="${MP_RESULTS_DIR:-${RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_relaxed_clean_equal_atoms_mp_t}}"
MP_CACHE_ROOT="${MP_CACHE_ROOT:-${CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_relaxed_clean_equal_atoms_mp_t}}"
MP_CGCNN_ROOT="${MP_CGCNN_ROOT:-${CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_mp_cgcnn}}"
HZW_DATASET_DIR="${HZW_DATASET_DIR:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_hzw}"
HZW_RESULTS_DIR="${HZW_RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t}"
HZW_CACHE_ROOT="${HZW_CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_relaxed_clean_equal_atoms_hzw_t}"
HZW_CGCNN_ROOT="${HZW_CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_hzw_cgcnn}"
T_EXPERIMENTS="${T_EXPERIMENTS:-T1 T2 T3 T4 T6 T7 T8 T9 T10}"
FORCE_EXPORT="${FORCE_EXPORT:-0}"

mkdir -p "$WORK_ROOT"/{cache,results,logs,run,tmp}
export PROJECT_ROOT
export WORK_ROOT

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

run_t_series() {
  local label_source="$1"
  local dataset_dir="$2"
  local results_dir="$3"
  local cache_root="$4"
  local cgcnn_root="$5"
  shift 5

  local export_args=()
  if [ "$FORCE_EXPORT" = "1" ]; then
    export_args+=(--force_export)
  fi

  echo "===== Running T-series label_source=$label_source ====="
  echo "dataset_dir=$dataset_dir"
  echo "results_dir=$results_dir"

# shellcheck disable=SC2086
  "$PYTHON_BIN" run_t_relaxed_matched_experiments.py \
    --hzw_raw_path "$HZW_RAW_PATH" \
    --audit_dir "$WORK_ROOT/results/data_quality" \
    --dataset_dir "$dataset_dir" \
    --results_dir "$results_dir" \
    --cache_root "$cache_root" \
    --cgcnn_root "$cgcnn_root" \
    --label_source "$label_source" \
    --experiments $T_EXPERIMENTS \
    --performance_mode 4090 \
    --skip_audit \
    "${export_args[@]}"
}

run_t_series mp "$MP_DATASET_DIR" "$MP_RESULTS_DIR" "$MP_CACHE_ROOT" "$MP_CGCNN_ROOT"
run_t_series hzw "$HZW_DATASET_DIR" "$HZW_RESULTS_DIR" "$HZW_CACHE_ROOT" "$HZW_CGCNN_ROOT"
