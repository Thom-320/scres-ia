#!/usr/bin/env bash
set +e
cd /Users/thom/Projects/research/scres-ia || exit 2
LOG="outputs/experiments/watch_track_b_belief_cond_v2_sweep.log"
mkdir -p outputs/experiments
PIDS=(85201 85656 85710 85762)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start (kappa/rho sweep v2, PIDs ${PIDS[*]})" >> "$LOG"
while true; do
  ALL_DONE=1
  for pid in "${PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && ALL_DONE=0
  done
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all_done=$ALL_DONE" >> "$LOG"
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] sweep done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
