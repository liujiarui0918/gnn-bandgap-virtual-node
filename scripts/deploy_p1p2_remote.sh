#!/usr/bin/env bash
# Runs ON THE CLUSTER after the incoming tarball is extracted to
# ~/gnn_incoming_p1p2. Backs up the one overwritten repo file, installs the
# updated runner + queue script, syntax-checks both, then submits the queue.
set -euo pipefail

REPO=/data/home/tg100025/gnn
IN="$HOME/gnn_incoming_p1p2"
BK="$HOME/gnn_backup_pre_p1p2_20260720"
PY=/data/home/tg100025/gnn_work/envs/gnn-faithful/bin/python

mkdir -p "$BK"
cp -f "$REPO/run_t_relaxed_matched_experiments.py" "$BK/" 2>/dev/null || true

cp -f "$IN/run_t_relaxed_matched_experiments.py" "$REPO/"
cp -f "$IN/tests/test_t_relaxed_runner_config.py" "$REPO/tests/"
cp -f "$IN/scripts/slurm_reviewer_p1p2_queue.sh" "$REPO/scripts/"

cd "$REPO"
"$PY" -m py_compile run_t_relaxed_matched_experiments.py && echo PY_SYNTAX_OK
bash -n scripts/slurm_reviewer_p1p2_queue.sh && echo SH_SYNTAX_OK

echo "== strict dataset check =="
ls -la /data/home/tg100025/gnn_work/data/mp_hzw_final_clean_equal_atoms_mp/manifest.jsonl

echo "== queue before =="
squeue -u tg100025 || true
sbatch scripts/slurm_reviewer_p1p2_queue.sh
echo "== queue after =="
squeue -u tg100025
echo DEPLOY_DONE
