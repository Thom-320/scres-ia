#!/usr/bin/env bash
set +e
cd /Users/thom/Projects/research/scres-ia || exit 2
LOG="outputs/experiments/watch_track_b_belief_conditioned.log"
mkdir -p outputs/experiments
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start (belief-conditioned combined arm, PID 51117)" >> "$LOG"
while true; do
  DONE=1
  kill -0 51117 2>/dev/null && DONE=0
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] done=$DONE" >> "$LOG"
  if [ "$DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] job done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
