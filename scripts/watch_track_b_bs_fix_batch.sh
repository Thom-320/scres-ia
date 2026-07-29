#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

HOST="ovh-agent-lab"
LOCAL_PIDS=(98704 98759)
REMOTE_JOBS=(
  "track_b_belief_encoder_ppo_3seed_30k_2026-07-04_v3"
  "track_b_belief_encoder_real_kan_3seed_30k_2026-07-04_v4"
)
LOG="outputs/experiments/watch_track_b_bs_fix_batch.log"
mkdir -p outputs/experiments

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start" >> "$LOG"

while true; do
  ALL_DONE=1
  for pid in "${LOCAL_PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      ALL_DONE=0
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] local pid $pid still running" >> "$LOG"
    fi
  done
  for job in "${REMOTE_JOBS[@]}"; do
    RUNNING=$(ssh -o ConnectTimeout=15 "$HOST" "pgrep -f run_track_b_belief_encoder_sidecar.py.*${job} | wc -l" 2>>"$LOG")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $job processes_running=$RUNNING" >> "$LOG"
    if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
      ALL_DONE=0
    else
      mkdir -p "outputs/experiments/${job}"
      rsync -az "$HOST:~/scres-ia/outputs/experiments/${job}/" "outputs/experiments/${job}/" >> "$LOG" 2>&1
    fi
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all batch-size-fixed jobs done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
