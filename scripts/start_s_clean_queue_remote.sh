#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/tg100025/gnn}"
WORK_ROOT="${WORK_ROOT:-/data/home/tg100025/gnn_work}"
PYTHON_BIN="${PYTHON_BIN:-$WORK_ROOT/envs/gnn-faithful/bin/python}"

mkdir -p "$WORK_ROOT"/{data,cache,results,logs,run,envs}
cd "$PROJECT_ROOT"

if [ -z "${MP_API_KEY:-}" ] && [ -f "$HOME/.mp_api_key" ]; then
  MP_API_KEY="$(tr -d '\r\n' < "$HOME/.mp_api_key")"
  export MP_API_KEY
fi

if [ -z "${MP_API_KEY:-}" ]; then
  echo "MP_API_KEY is required. Export it or write it to \$HOME/.mp_api_key with chmod 600." >&2
  exit 2
fi

if [ -f "$WORK_ROOT/run/s_clean_queue_monitor.pid" ]; then
  old_monitor_pid="$(cat "$WORK_ROOT/run/s_clean_queue_monitor.pid" || true)"
  if [ -n "$old_monitor_pid" ]; then
    kill "$old_monitor_pid" 2>/dev/null || true
  fi
fi

job_id="$(sbatch --parsable --export=ALL,PROJECT_ROOT="$PROJECT_ROOT",WORK_ROOT="$WORK_ROOT",PYTHON_BIN="$PYTHON_BIN" scripts/slurm_s_clean_queue.sh)"
printf '%s\n' "$job_id" > "$WORK_ROOT/run/s_clean_queue.jobid"
echo "submitted $job_id"

nohup "$PYTHON_BIN" monitor_s_clean_queue.py \
  --project_root "$PROJECT_ROOT" \
  --results_dir "$WORK_ROOT/results/mp_hzw_clean_matched_s" \
  --dataset_dir "$WORK_ROOT/data/mp_hzw_clean_matched_s" \
  --audit_dir "$WORK_ROOT/results/data_quality" \
  --job_file "$WORK_ROOT/run/s_clean_queue.jobid" \
  --log_path "$WORK_ROOT/logs/s_clean_queue_monitor.log" \
  --state_path "$WORK_ROOT/logs/s_clean_queue_monitor_state.json" \
  --error_report_path "$WORK_ROOT/logs/last_error_report.md" \
  --interval_seconds 1800 \
  > "$WORK_ROOT/logs/s_clean_queue_monitor.nohup.log" 2>&1 &
printf '%s\n' "$!" > "$WORK_ROOT/run/s_clean_queue_monitor.pid"
echo "monitor_pid $(cat "$WORK_ROOT/run/s_clean_queue_monitor.pid")"
