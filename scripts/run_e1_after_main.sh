#!/usr/bin/env bash
set -euo pipefail

MAIN_PID="${1:-}"
PROJECT_ROOT="${PROJECT_ROOT:-/root/lanyun-fs/gnn}"
RESULTS_DIR="${RESULTS_DIR:-results/mp_hzw_matched_clean_e1_e12}"
DATASET_DIR="${DATASET_DIR:-data/mp_hzw_matched_clean_no_mp_metal_mislabel}"
CACHE_ROOT="${CACHE_ROOT:-/root/lanyun-tmp/gnn_mp_hzw_matched_clean_no_mp_metal_mislabel_cache}"
CGCNN_ROOT="${CGCNN_ROOT:-/root/lanyun-tmp/gnn_mp_hzw_matched_clean_no_mp_metal_mislabel_cgcnn}"
PYTHON_BIN="${PYTHON_BIN:-/root/miniconda/bin/python}"

cd "$PROJECT_ROOT"

echo "[wait] E1 follow-up created at $(date '+%F %T'), waiting for main pid ${MAIN_PID:-none}"
if [[ -n "$MAIN_PID" ]]; then
  while kill -0 "$MAIN_PID" 2>/dev/null; do
    sleep 300
  done
fi

echo "[start] main queue finished at $(date '+%F %T'), starting E1 follow-up"

if [[ ! -f "$RESULTS_DIR/MPHZW_E1_CGCNN_baseline/metrics.json" ]]; then
  archive="$RESULTS_DIR/archive_failed_e1_$(date +%Y%m%d_%H%M%S)"
  mkdir -p "$archive"
  if [[ -e "$RESULTS_DIR/MPHZW_E1_CGCNN_baseline" ]]; then
    mv "$RESULTS_DIR/MPHZW_E1_CGCNN_baseline" "$archive/"
  fi
  shopt -s nullglob
  for log_file in "$RESULTS_DIR"/queue_logs/E1_*.log; do
    mv "$log_file" "$archive/"
  done
  shopt -u nullglob
  rm -f "$RESULTS_DIR"/generated_configs/MPHZW_E1_CGCNN_baseline_batchdiv*.yaml
fi

ulimit -n 65535
export MATGL_BACKEND="${MATGL_BACKEND:-DGL}"
export GNN_ENABLE_TF32="${GNN_ENABLE_TF32:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True,max_split_size_mb:256}"

"$PYTHON_BIN" run_mp_hzw_matched_experiments.py \
  --api_key "$MP_API_KEY" \
  --dataset_dir "$DATASET_DIR" \
  --results_dir "$RESULTS_DIR" \
  --cache_root "$CACHE_ROOT" \
  --cgcnn_root "$CGCNN_ROOT" \
  --default_python "$PYTHON_BIN" \
  --alignn_python "$PYTHON_BIN" \
  --m3gnet_python "$PYTHON_BIN" \
  --export_chunk_size 100 \
  --experiments E1 \
  --performance_mode 4090 \
  --max_batch_retries 4

echo "[done] E1 follow-up finished at $(date '+%F %T')"
