#!/usr/bin/env bash
set +e
cd /Users/thom/Projects/research/scres-ia || exit 2
LOG="outputs/experiments/watch_track_b_arm1_arm2.log"
mkdir -p outputs/experiments
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start (arm1=reward-shaping PID 7418, arm2=R22-belief PID 5911)" >> "$LOG"
while true; do
  A1=1; A2=1
  kill -0 7418 2>/dev/null && A1=0
  kill -0 5911 2>/dev/null && A2=0
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] arm1_done=$A1 arm2_done=$A2" >> "$LOG"
  if [ "$A1" -eq 1 ] && [ "$A2" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] both arms done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
