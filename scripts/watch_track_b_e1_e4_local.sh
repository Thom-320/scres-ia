#!/usr/bin/env bash
set +e

ROOT="/Users/thom/Projects/research/scres-ia"
LOG="$ROOT/outputs/experiments/q1_local_e1_e4_watcher_2026-07-02.log"
E1_DIR="$ROOT/outputs/experiments/track_b_e1_confirmatory_2026-07-02"
E4_DIR="$ROOT/outputs/experiments/track_b_ablation_8d_final_2026-07-01"
E1_PID=82256
E4_PID=81973
E1_MERGED=0
E4_DONE_REPORTED=0

mkdir -p "$(dirname "$LOG")"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start e1_pid=$E1_PID e4_pid=$E4_PID" >> "$LOG"

while true; do
  E1_RUNNING=0
  E4_RUNNING=0
  ps -p "$E1_PID" >/dev/null 2>&1 && E1_RUNNING=1
  ps -p "$E4_PID" >/dev/null 2>&1 && E4_RUNNING=1
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] e1_running=$E1_RUNNING e4_running=$E4_RUNNING" >> "$LOG"

  if [ "$E1_RUNNING" -eq 0 ] && [ "$E1_MERGED" -eq 0 ]; then
    if [ -f "$E1_DIR/episode_metrics.csv" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E1 episode_metrics.csv found; running go/no-go merge" >> "$LOG"
      cd "$ROOT" && .venv/bin/python scripts/build_track_b_e1_go_no_go.py >> "$LOG" 2>&1
      E1_MERGED=1
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E1 stopped but episode_metrics.csv missing" >> "$LOG"
      E1_MERGED=1
    fi
  fi

  if [ "$E4_RUNNING" -eq 0 ] && [ "$E4_DONE_REPORTED" -eq 0 ]; then
    if [ -f "$E4_DIR/shift_only/summary.json" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E4 terminal with shift_only summary present" >> "$LOG"
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] E4 stopped but shift_only summary missing" >> "$LOG"
    fi
    E4_DONE_REPORTED=1
  fi

  if [ "$E1_MERGED" -eq 1 ] && [ "$E4_DONE_REPORTED" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher complete" >> "$LOG"
    exit 0
  fi

  sleep 180
done
