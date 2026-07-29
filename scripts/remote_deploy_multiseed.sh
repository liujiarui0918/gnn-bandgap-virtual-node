#!/bin/sh
# One-shot deploy for the multi-seed reviewer-response queue (P0-a).
# Runs ON THE REMOTE HOST. Assumes ~/gnn_incoming_multiseed/ holds fresh files.
set -e

PROJECT=/data/home/tg100025/gnn
WORK=/data/home/tg100025/gnn_work
STAGE=$HOME/gnn_incoming_multiseed
BK=$HOME/gnn_backup_pre_multiseed_20260715

echo "== queue before =="
/opt/slurm-22.05.2/bin/squeue -u tg100025 -h 2>/dev/null | wc -l

mkdir -p "$BK/faithful_gnn" "$BK/scripts"
cd "$PROJECT"
cp -a run_t_relaxed_matched_experiments.py "$BK/" 2>/dev/null || true
for f in common.py alignn_runner.py cgcnn_runner.py m3gnet_runner.py; do
  cp -a "faithful_gnn/$f" "$BK/faithful_gnn/" 2>/dev/null || true
done
cp -a scripts/aggregate_multiseed.py "$BK/scripts/" 2>/dev/null || true
echo BACKUP_DONE

cp -a "$STAGE/run_t_relaxed_matched_experiments.py" .
cp -a "$STAGE"/faithful_gnn/*.py faithful_gnn/
cp -a "$STAGE"/scripts/slurm_t_multiseed_queue.sh scripts/
cp -a "$STAGE"/scripts/aggregate_multiseed.py scripts/
echo DEPLOY_DONE

"$WORK/envs/gnn-faithful/bin/python" - <<'PY'
import ast
files = [
    "run_t_relaxed_matched_experiments.py",
    "faithful_gnn/common.py",
    "faithful_gnn/alignn_runner.py",
    "faithful_gnn/cgcnn_runner.py",
    "faithful_gnn/m3gnet_runner.py",
]
for p in files:
    ast.parse(open(p).read())
print("REMOTE_SYNTAX_OK")
PY

mkdir -p "$WORK/logs"
SUBMIT=$(/opt/slurm-22.05.2/bin/sbatch scripts/slurm_t_multiseed_queue.sh)
echo "$SUBMIT"
JOB=$(echo "$SUBMIT" | grep -o '[0-9][0-9]*')
echo "SUBMITTED_JOB=$JOB"
sleep 6
echo "== queue after =="
/opt/slurm-22.05.2/bin/squeue -u tg100025 2>/dev/null | head -5
echo "== first log lines (if started) =="
head -25 "$WORK/logs/slurm-gnn_t_multiseed-$JOB.out" 2>/dev/null || echo "log not started yet"
echo ALL_DONE
