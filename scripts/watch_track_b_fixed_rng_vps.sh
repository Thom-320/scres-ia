#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

HOST="ovh-agent-lab"
JOBS=(
  "run_track_b_smoke.py|track_b_fixed_rng_confirm_5seed_60k_2026-07-03"
  "run_track_b_real_kan_sidecar.py|track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03"
)
LOG="outputs/experiments/watch_track_b_fixed_rng_vps.log"
mkdir -p outputs/experiments

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start, host=$HOST" >> "$LOG"

while true; do
  ALL_DONE=1
  for job in "${JOBS[@]}"; do
    PATTERN="${job%%|*}"
    REMOTE_DIR="~/scres-ia/outputs/experiments/${job##*|}"
    LOCAL_DIR="outputs/experiments/${job##*|}"
    RUNNING=$(ssh -o ConnectTimeout=15 "$HOST" "pgrep -f $PATTERN | wc -l" 2>>"$LOG")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $PATTERN processes_running=$RUNNING" >> "$LOG"
    if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
      ALL_DONE=0
    else
      mkdir -p "$LOCAL_DIR"
      rsync -az "$HOST:$REMOTE_DIR/" "$LOCAL_DIR/" >> "$LOG" 2>&1
    fi
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] all jobs done, fetched both" >> "$LOG"
    exit 0
  fi
  sleep 180
done
