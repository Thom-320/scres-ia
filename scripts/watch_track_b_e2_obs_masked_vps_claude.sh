#!/usr/bin/env bash
set +e

ROOT="/Users/thom/Projects/research/scres-ia"
REMOTE="ovh-agent-lab"
REMOTE_DIR="/home/ubuntu/scres-ia"
RUN_DIR="outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02"
LOCAL_DIR="$ROOT/$RUN_DIR"
LOG="$LOCAL_DIR/vps_watcher.log"

mkdir -p "$LOCAL_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start remote=$REMOTE run_dir=$RUN_DIR" >> "$LOG"

while true; do
  STATUS=$(ssh "$REMOTE" "ps -eo pid,etime,%cpu,%mem,command | grep 'scripts/run_track_b_observation_ablation.py --obs-configs v7_no_regime_forecast --output-dir $RUN_DIR' | grep -v grep || true" 2>&1)
  if [ -n "$STATUS" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] RUNNING $STATUS" >> "$LOG"
  else
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STOPPED no matching remote process" >> "$LOG"
  fi

  rsync -az --delete \
    --exclude 'vps_watcher.log' \
    --exclude 'watch_stdout.log' \
    --exclude 'watch_local.pid' \
    "$REMOTE:$REMOTE_DIR/$RUN_DIR/" "$LOCAL_DIR/" >> "$LOG" 2>&1

  if [ -z "$STATUS" ]; then
    if [ -f "$LOCAL_DIR/v7_no_regime_forecast/summary.json" ] || [ -f "$LOCAL_DIR/summary.json" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] terminal with summary artifact present" >> "$LOG"
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] terminal without summary artifact" >> "$LOG"
    fi
    exit 0
  fi

  sleep 180
done
