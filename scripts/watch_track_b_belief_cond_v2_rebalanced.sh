#!/usr/bin/env bash
set +e
cd /Users/thom/Projects/research/scres-ia || exit 2
LOG="outputs/experiments/watch_track_b_belief_cond_v2_rebalanced.log"
mkdir -p outputs/experiments
LOCAL_PIDS=(16426 18051 18298 19651)
VPS_JOBS=(bc_k010_r000 bc_k010_r002 bc_k010_r010 bc_k020_r000 bc_k020_r002 bc_k040_r000 bc_k040_r002 bc_k040_r010)
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start" >> "$LOG"
while true; do
  ALL_DONE=1
  for pid in "${LOCAL_PIDS[@]}"; do
    kill -0 "$pid" 2>/dev/null && ALL_DONE=0
  done
  REMAINING=$(ssh -o ConnectTimeout=15 ovh-agent-lab "tmux ls 2>/dev/null | wc -l")
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] local_all_done=$ALL_DONE vps_sessions_remaining=$REMAINING" >> "$LOG"
  if [ "$REMAINING" -gt 0 ] 2>/dev/null; then
    ALL_DONE=0
  fi
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all done" >> "$LOG"
    exit 0
  fi
  sleep 120
done
