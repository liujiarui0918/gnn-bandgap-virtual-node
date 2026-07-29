#!/usr/bin/env bash
# Fix pass for the p1p2 queue: the remote repo predates S14 (and possibly the
# oversmoothing hook script). Installs the missing files, shows why phase 1
# produced no output, then resubmits the queue (resume-safe).
set -euo pipefail

REPO=/data/home/tg100025/gnn
IN="$HOME/gnn_incoming_p1p2"
BK="$HOME/gnn_backup_pre_p1p2_20260720"
PY=/data/home/tg100025/gnn_work/envs/gnn-faithful/bin/python

mkdir -p "$BK"
cp -f "$REPO/run_s_clean_matched_experiments.py" "$BK/" 2>/dev/null || true
cp -f "$REPO/scripts/analyze_oversmoothing_hooks.py" "$BK/" 2>/dev/null || true

cp -f "$IN/run_s_clean_matched_experiments.py" "$REPO/"
cp -f "$IN/configs/mp/MP_S14_alignn_huber_virtualnode.yaml" "$REPO/configs/mp/"
cp -f "$IN/scripts/analyze_oversmoothing_hooks.py" "$REPO/scripts/"

cd "$REPO"
"$PY" -m py_compile run_s_clean_matched_experiments.py scripts/analyze_oversmoothing_hooks.py && echo PY_SYNTAX_OK
grep -c "S14" run_s_clean_matched_experiments.py
ls -la configs/mp/MP_S14_alignn_huber_virtualnode.yaml

echo "== failed log: phase-1 diagnosis =="
L=$(ls -t /data/home/tg100025/gnn_work/logs/slurm-gnn_rev_p1p2-*.out 2>/dev/null | head -1)
echo "log: $L"
grep -n "Phase 1\|oversmoothing\|warn\|Error\|Traceback" "$L" | head -30 || true

echo "== resubmit =="
squeue -u tg100025 || true
sbatch scripts/slurm_reviewer_p1p2_queue.sh
squeue -u tg100025
echo FIX_DEPLOY_DONE
