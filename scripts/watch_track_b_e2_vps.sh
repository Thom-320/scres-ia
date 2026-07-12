#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

HOST="ovh-agent-lab"
REMOTE_DIR="~/scres-ia/outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02"
LOCAL_DIR="outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02"
LOG="$LOCAL_DIR/watcher.log"

mkdir -p "$LOCAL_DIR"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start, host=$HOST" >> "$LOG"

while true; do
  RUNNING=$(ssh -o ConnectTimeout=15 "$HOST" "pgrep -f run_track_b_observation_ablation.py | wc -l" 2>>"$LOG")
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] processes_running=$RUNNING" >> "$LOG"

  if [ "$RUNNING" -eq 0 ] 2>/dev/null; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] process gone, checking for completion artifact" >> "$LOG"
    HAS_SUMMARY=$(ssh -o ConnectTimeout=15 "$HOST" "test -f $REMOTE_DIR/v7_no_regime_forecast/summary.json && echo yes || echo no" 2>>"$LOG")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] has_summary=$HAS_SUMMARY" >> "$LOG"
    rsync -az "$HOST:$REMOTE_DIR/" "$LOCAL_DIR/" >> "$LOG" 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] fetched to $LOCAL_DIR" >> "$LOG"
    exit 0
  fi

  sleep 120
done
