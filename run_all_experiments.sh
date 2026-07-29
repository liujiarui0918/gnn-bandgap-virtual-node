#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

RAW_PATH="${RAW_PATH:-data/raw/crystal_band_100000.jsonl}"
DATASET_DIR="${DATASET_DIR:-data/official_dataset}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DEFAULT_PYTHON="${DEFAULT_PYTHON:-python}"
ALIGNN_PYTHON="${ALIGNN_PYTHON:-$DEFAULT_PYTHON}"
M3GNET_PYTHON="${M3GNET_PYTHON:-$DEFAULT_PYTHON}"

mkdir -p "$RESULTS_DIR"

if [ ! -f "$DATASET_DIR/metadata.json" ]; then
  "$DEFAULT_PYTHON" export_official_dataset.py \
    --data_path "$RAW_PATH" \
    --output_dir "$DATASET_DIR" \
    --seed 42
fi

"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E1_cgcnn_faithful.yaml --output_dir "$RESULTS_DIR/E1_CGCNN_baseline"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E2_alignn_faithful.yaml --output_dir "$RESULTS_DIR/E2_ALIGNN_baseline"
"$M3GNET_PYTHON" run_faithful_experiment.py --config configs/E3_m3gnet_faithful.yaml --output_dir "$RESULTS_DIR/E3_M3GNet_baseline"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E4_alignn_virtualnode.yaml --output_dir "$RESULTS_DIR/E4_ALIGNN_virtualnode"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E5_alignn_transformer_faithful.yaml --output_dir "$RESULTS_DIR/E5_ALIGNN_transformer"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E6_alignn_physconstraint.yaml --output_dir "$RESULTS_DIR/E6_ALIGNN_physconstraint"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E7_alignn_attentive_virtualnode.yaml --output_dir "$RESULTS_DIR/E7_ALIGNN_attentive_virtualnode"
"$ALIGNN_PYTHON" run_faithful_experiment.py --config configs/E8_alignn_multi_virtualnode.yaml --output_dir "$RESULTS_DIR/E8_ALIGNN_multi_virtualnode"
"$DEFAULT_PYTHON" summarize.py --results_dir "$RESULTS_DIR" --output "$RESULTS_DIR/comparison_summary.csv"
