#!/usr/bin/env bash
# LOCAL monitor for the reviewer p1p2 queue (job gnn_rev_p1p2). Polls the
# cluster every 2 h, counts finished artifacts, tails the slurm log. Exits
# when all 14 markers are present or the job is gone and no progress remains.
set -u
cd "$(dirname "$0")/.."

W=/data/home/tg100025/gnn_work
MARKERS=(
  "$W/results/analysis/oversmoothing_hooks_mp_full/oversmoothing_aggregate_metrics.csv"
  "$W/results/analysis/oversmoothing_hooks_hzw_full/oversmoothing_aggregate_metrics.csv"
  "$W/results/mp_hzw_final_clean_equal_atoms_mp_s/S14_ALIGNN_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_m3ep100/T3_M3GNet_baseline/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_m3lr5e4/T3_M3GNet_baseline/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_m3u256/T3_M3GNet_baseline/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_hb025/T7_ALIGNN_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_hb100/T7_ALIGNN_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_ms050/T10_ALIGNN_decay_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_mp_t_ms075/T10_ALIGNN_decay_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t_hb025/T7_ALIGNN_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t_hb100/T7_ALIGNN_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t_ms050/T10_ALIGNN_decay_huber_virtualnode/metrics.json"
  "$W/results/mp_hzw_relaxed_clean_equal_atoms_hzw_t_ms075/T10_ALIGNN_decay_huber_virtualnode/metrics.json"
)
REMOTE_CHECK='D=0; MISSING=""
for f in FILE_LIST; do
  if [ -f "$f" ]; then D=$((D+1)); else MISSING="$MISSING ${f##*/results/}"; fi
done
echo "P1P2_DONE=$D/14"
squeue -u tg100025 -h -n gnn_rev_p1p2 -o "%i %T %M" | head -2
L=$(ls -t /data/home/tg100025/gnn_work/logs/slurm-gnn_rev_p1p2-*.out 2>/dev/null | head -1)
if [ -n "$L" ]; then echo "== log tail =="; tail -4 "$L"; fi'
REMOTE_CHECK="${REMOTE_CHECK/FILE_LIST/${MARKERS[*]}}"

for POLL in $(seq 1 40); do
  OUT=""
  for ATTEMPT in 1 2 3; do
    OUT=$(ssh -i tg100025.pem -o ConnectTimeout=25 -o StrictHostKeyChecking=accept-new \
      tg100025@124.220.81.245 "bash -lc '$REMOTE_CHECK'" 2>&1) && break
    echo "poll $POLL attempt $ATTEMPT failed, backoff 6 min"
    sleep 360
  done
  echo "[$(date +%m-%d_%H:%M)] $OUT"
  if echo "$OUT" | grep -q "P1P2_DONE=14/14"; then
    echo "ALL_P1P2_DONE"
    exit 0
  fi
  sleep 7200
done
echo "MONITOR_POLL_BUDGET_EXHAUSTED"
