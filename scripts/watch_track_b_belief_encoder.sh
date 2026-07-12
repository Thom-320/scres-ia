#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

HOST="ovh-agent-lab"
LOCAL_PID=51098
REMOTE_JOB="track_b_belief_encoder_real_kan_3seed_30k_2026-07-04"
LOG="outputs/experiments/watch_track_b_belief_encoder.log"
mkdir -p outputs/experiments

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start" >> "$LOG"

while true; do
  LOCAL_DONE=0
  if ! kill -0 "$LOCAL_PID" 2>/dev/null; then
    LOCAL_DONE=1
  fi
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] local_ppo_done=$LOCAL_DONE" >> "$LOG"

  REMOTE_RUNNING=$(ssh -o ConnectTimeout=15 "$HOST" "pgrep -f run_track_b_belief_encoder_sidecar.py.*${REMOTE_JOB} | wc -l" 2>>"$LOG")
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] remote_kan_processes_running=$REMOTE_RUNNING" >> "$LOG"

  if [ "$REMOTE_RUNNING" -eq 0 ] 2>/dev/null; then
    mkdir -p "outputs/experiments/${REMOTE_JOB}"
    rsync -az "$HOST:~/scres-ia/outputs/experiments/${REMOTE_JOB}/" "outputs/experiments/${REMOTE_JOB}/" >> "$LOG" 2>&1
  fi

  if [ "$LOCAL_DONE" -eq 1 ] && [ "$REMOTE_RUNNING" -eq 0 ] 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] both belief-encoder jobs done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
