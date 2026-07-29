#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
PYTHON_BIN="${PYTHON_BIN:-$WORK_ROOT/envs/gnn-faithful/bin/python}"
RESULTS_DIR="${RESULTS_DIR:-$WORK_ROOT/results/mp_hzw_final_clean_equal_atoms_mp_s}"
DATASET_DIR="${DATASET_DIR:-$WORK_ROOT/data/mp_hzw_final_clean_equal_atoms_mp}"
AUDIT_DIR="${AUDIT_DIR:-$WORK_ROOT/results/data_quality}"

mkdir -p "$WORK_ROOT"/{cache,results,logs,run,tmp}
cd "$PROJECT_ROOT"

if [ -f "$WORK_ROOT/run/s_final_equal_mp_queue_monitor.pid" ]; then
  old_monitor_pid="$(cat "$WORK_ROOT/run/s_final_equal_mp_queue_monitor.pid" || true)"
  if [ -n "$old_monitor_pid" ]; then
    kill "$old_monitor_pid" 2>/dev/null || true
  fi
fi

echo "[pre-submit] squeue -a"
/opt/slurm-22.05.2/bin/squeue -a -o "%.18i %.10P %.28j %.14u %.2t %.12M %.6D %R" || true

job_id="$(sbatch --parsable --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",WORK_ROOT="$WORK_ROOT",PYTHON_BIN="$PYTHON_BIN" scripts/slurm_s_final_mp_queue.sh)"
printf '%s\n' "$job_id" > "$WORK_ROOT/run/s_final_equal_mp_queue.jobid"
echo "submitted $job_id"

nohup "$PYTHON_BIN" monitor_s_clean_queue.py \
  --project_root "$PROJECT_ROOT" \
  --results_dir "$RESULTS_DIR" \
  --dataset_dir "$DATASET_DIR" \
  --audit_dir "$AUDIT_DIR" \
  --job_file "$WORK_ROOT/run/s_final_equal_mp_queue.jobid" \
  --log_path "$WORK_ROOT/logs/s_final_equal_mp_queue_monitor.log" \
  --state_path "$WORK_ROOT/logs/s_final_equal_mp_queue_monitor_state.json" \
  --error_report_path "$WORK_ROOT/logs/s_final_equal_mp_last_error_report.md" \
  --interval_seconds 1800 \
  > "$WORK_ROOT/logs/s_final_equal_mp_queue_monitor.nohup.log" 2>&1 &
printf '%s\n' "$!" > "$WORK_ROOT/run/s_final_equal_mp_queue_monitor.pid"
echo "monitor_pid $(cat "$WORK_ROOT/run/s_final_equal_mp_queue_monitor.pid")"
