#!/usr/bin/env bash
#SBATCH -J gnn_rev_p1p2
#SBATCH -p tx4090
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH -t 7-00:00:00
#SBATCH -o /data/home/tg100025/gnn_work/logs/slurm-%x-%j.out

# Reviewer-response follow-up queue (comments 2, 3, 5, 6, 10), ordered cheap
# to expensive so partial progress is maximally useful:
#   Phase 1  Over-smoothing hook diagnostics on the FULL test set (comment 3),
#            mp + hzw, 8455 samples each.
#   Phase 2  S14 strict-clean + Huber + virtual node (comment 6): isolates the
#            Huber contribution on the strict subset where S4 (MSE+VN) already
#            exists. Runs via the original S-series runner, seed 42, 50 epochs.
#   Phase 3  M3GNet tuning probes on MP labels (comment 2): longer budget,
#            lower LR, wider units. One factor per run.
#   Phase 4  Hyperparameter sensitivity (comments 5/10): Huber beta
#            {0.25, 1.0} on T7 and broadcast min_scale {0.5, 0.75} on T10,
#            both label sources. Seed 42 so each run is directly comparable
#            to the mainline seed-42 T7/T10.
# All training runs use early-stopping patience 8, the same rule as the
# multi-seed queue (provably would not have changed any seed-42 result).
# Every phase is resume-safe: finished work is detected and skipped, so the
# job can be requeued after a crash without redoing anything.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
PYTHON_BIN="${PYTHON_BIN:-$WORK_ROOT/envs/gnn-faithful/bin/python}"
HZW_RAW_PATH="${HZW_RAW_PATH:-$WORK_ROOT/data/raw/hzw/crystal_band_100000.jsonl.gz.tmp}"

MP_DATASET_DIR="${MP_DATASET_DIR:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_mp}"
MP_RESULTS_DIR="${MP_RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_relaxed_clean_equal_atoms_mp_t}"
MP_CACHE_ROOT="${MP_CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_relaxed_clean_equal_atoms_mp_t}"
MP_CGCNN_ROOT="${MP_CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_mp_cgcnn}"
HZW_DATASET_DIR="${HZW_DATASET_DIR:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_hzw}"
HZW_RESULTS_DIR="${HZW_RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t}"
HZW_CACHE_ROOT="${HZW_CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_relaxed_clean_equal_atoms_hzw_t}"
HZW_CGCNN_ROOT="${HZW_CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_relaxed_clean_equal_atoms_hzw_cgcnn}"

STRICT_DATASET_DIR="${STRICT_DATASET_DIR:-$WORK_ROOT/data/mp_hzw_final_clean_equal_atoms_mp}"
STRICT_RESULTS_DIR="${STRICT_RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_final_clean_equal_atoms_mp_s}"
STRICT_CACHE_ROOT="${STRICT_CACHE_ROOT:-$WORK_ROOT/cache/mp_hzw_final_clean_equal_atoms_mp_s}"
STRICT_CGCNN_ROOT="${STRICT_CGCNN_ROOT:-$WORK_ROOT/data/mp_hzw_final_clean_equal_atoms_mp_cgcnn}"

EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-8}"
OVERSMOOTH_MAX_SAMPLES="${OVERSMOOTH_MAX_SAMPLES:-8455}"

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

# ---------------------------------------------------------------------------
# Phase 1: over-smoothing diagnostics on the full test set (comment 3)
# ---------------------------------------------------------------------------
run_oversmoothing() {
  local src="$1"
  local results_root="$2"
  local out_dir="$WORK_ROOT/results/analysis/oversmoothing_hooks_${src}_full"
  if [ -f "$out_dir/oversmoothing_aggregate_metrics.csv" ]; then
    echo "[skip] oversmoothing ${src} full-test diagnostics already at $out_dir"
    return 0
  fi
  echo "===== Phase 1: oversmoothing hooks ${src} max_samples=$OVERSMOOTH_MAX_SAMPLES ====="
  "$PYTHON_BIN" scripts/analyze_oversmoothing_hooks.py \
    --results-root "$results_root" \
    --output-dir "$out_dir" \
    --experiments T2 T4 T7 T10 \
    --max-samples "$OVERSMOOTH_MAX_SAMPLES" \
    --batch-size 32 \
    --device cuda
}

# A missing checkpoint here must not kill phases 2-4, so failures degrade to a
# warning; rerun this phase alone after fixing the cause.
run_oversmoothing mp "$MP_RESULTS_DIR" || echo "[warn] oversmoothing mp failed; continuing"
run_oversmoothing hzw "$HZW_RESULTS_DIR" || echo "[warn] oversmoothing hzw failed; continuing"

# ---------------------------------------------------------------------------
# Phase 2: S14 strict-clean + Huber + virtual node (comment 6)
# ---------------------------------------------------------------------------
echo "===== Phase 2: S14 strict-clean Huber virtual node ====="
if [ ! -f "$STRICT_DATASET_DIR/manifest.jsonl" ]; then
  echo "Strict dataset missing: $STRICT_DATASET_DIR/manifest.jsonl" >&2
  exit 4
fi
"$PYTHON_BIN" run_s_clean_matched_experiments.py \
  --hzw_raw_path "$HZW_RAW_PATH" \
  --audit_dir "$WORK_ROOT/results/data_quality" \
  --dataset_dir "$STRICT_DATASET_DIR" \
  --results_dir "$STRICT_RESULTS_DIR" \
  --cache_root "$STRICT_CACHE_ROOT" \
  --cgcnn_root "$STRICT_CGCNN_ROOT" \
  --experiments S14 \
  --performance_mode 4090 \
  --skip_audit \
  --skip_export \
  --skip_summary

# ---------------------------------------------------------------------------
# Shared helper for T-series variant runs (phases 3 and 4)
# ---------------------------------------------------------------------------
run_t_variant() {
  local label_source="$1"
  local dataset_dir="$2"
  local results_dir="$3"
  local cache_root="$4"
  local cgcnn_root="$5"
  local suffix="$6"
  local experiment="$7"
  shift 7
  # Remaining args are --override specs.
  local overrides=()
  local spec
  for spec in "$@"; do
    overrides+=(--override "$spec")
  done
  echo "===== T-variant ${experiment} label_source=${label_source} suffix=${suffix} overrides: $* ====="
  "$PYTHON_BIN" run_t_relaxed_matched_experiments.py \
    --hzw_raw_path "$HZW_RAW_PATH" \
    --audit_dir "$WORK_ROOT/results/data_quality" \
    --dataset_dir "$dataset_dir" \
    --results_dir "$results_dir" \
    --cache_root "$cache_root" \
    --cgcnn_root "$cgcnn_root" \
    --label_source "$label_source" \
    --experiments "$experiment" \
    --performance_mode 4090 \
    --results-suffix "$suffix" \
    --early-stopping-patience "$EARLY_STOP_PATIENCE" \
    --skip_audit \
    --skip_summary \
    "${overrides[@]}"
}

# ---------------------------------------------------------------------------
# Phase 3: M3GNet tuning probes, MP labels (comment 2) — one factor per run
# ---------------------------------------------------------------------------
echo "===== Phase 3: M3GNet tuning probes ====="
run_t_variant mp "$MP_DATASET_DIR" "$MP_RESULTS_DIR" "$MP_CACHE_ROOT" "$MP_CGCNN_ROOT" \
  _m3ep100 T3 training.epochs=100
run_t_variant mp "$MP_DATASET_DIR" "$MP_RESULTS_DIR" "$MP_CACHE_ROOT" "$MP_CGCNN_ROOT" \
  _m3lr5e4 T3 training.lr=0.0005
run_t_variant mp "$MP_DATASET_DIR" "$MP_RESULTS_DIR" "$MP_CACHE_ROOT" "$MP_CGCNN_ROOT" \
  _m3u256 T3 model.units=256

# ---------------------------------------------------------------------------
# Phase 4: hyperparameter sensitivity (comments 5/10), both label sources
# ---------------------------------------------------------------------------
echo "===== Phase 4: sensitivity sweeps ====="
for SRC in mp hzw; do
  if [ "$SRC" = "mp" ]; then
    DATASET_DIR="$MP_DATASET_DIR"; RESULTS_DIR="$MP_RESULTS_DIR"
    CACHE_ROOT="$MP_CACHE_ROOT"; CGCNN_ROOT="$MP_CGCNN_ROOT"
  else
    DATASET_DIR="$HZW_DATASET_DIR"; RESULTS_DIR="$HZW_RESULTS_DIR"
    CACHE_ROOT="$HZW_CACHE_ROOT"; CGCNN_ROOT="$HZW_CGCNN_ROOT"
  fi
  run_t_variant "$SRC" "$DATASET_DIR" "$RESULTS_DIR" "$CACHE_ROOT" "$CGCNN_ROOT" \
    _hb025 T7 training.final_gap_huber_beta=0.25
  run_t_variant "$SRC" "$DATASET_DIR" "$RESULTS_DIR" "$CACHE_ROOT" "$CGCNN_ROOT" \
    _hb100 T7 training.final_gap_huber_beta=1.0
  run_t_variant "$SRC" "$DATASET_DIR" "$RESULTS_DIR" "$CACHE_ROOT" "$CGCNN_ROOT" \
    _ms050 T10 model.broadcast_min_scale=0.5
  run_t_variant "$SRC" "$DATASET_DIR" "$RESULTS_DIR" "$CACHE_ROOT" "$CGCNN_ROOT" \
    _ms075 T10 model.broadcast_min_scale=0.75
done

echo "ALL_REVIEWER_P1P2_DONE"
