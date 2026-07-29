#!/usr/bin/env bash
set +e

ROOT="/Users/thom/Projects/research/scres-ia"
REMOTE="ovh-agent-lab"
REMOTE_DIR="/home/ubuntu/scres-ia-e2-20260702-min"
RUN_DIR="outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02_ovh"
LOCAL_DIR="$ROOT/$RUN_DIR"
LOG="$LOCAL_DIR/ovh_watcher.log"

mkdir -p "$LOCAL_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start remote=$REMOTE run_dir=$RUN_DIR" >> "$LOG"

while true; do
  STATUS=$(ssh "$REMOTE" "cd '$REMOTE_DIR' && if [ -f '$RUN_DIR/run_ovh.pid' ]; then pid=\\$(cat '$RUN_DIR/run_ovh.pid'); if ps -p \\$pid >/dev/null 2>&1; then echo RUNNING pid=\\$pid; else echo STOPPED pid=\\$pid; fi; else echo MISSING_PID; fi" 2>&1)
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $STATUS" >> "$LOG"

  rsync -az --delete "$REMOTE:$REMOTE_DIR/$RUN_DIR/" "$LOCAL_DIR/" >> "$LOG" 2>&1

  if printf '%s' "$STATUS" | grep -Eq 'STOPPED|MISSING_PID'; then
    if [ -f "$LOCAL_DIR/v7_no_regime_forecast/summary.json" ] || [ -f "$LOCAL_DIR/summary.json" ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] terminal with summary artifact present" >> "$LOG"
    else
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] terminal without summary artifact" >> "$LOG"
    fi
    exit 0
  fi

  sleep 180
done
