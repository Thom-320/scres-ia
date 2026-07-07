#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

HOST="ovh-agent-lab"
JOBS=(
  "track_b_risk_belief_ppo_unweighted_3seed_30k_2026-07-04"
  "track_b_risk_belief_real_kan_unweighted_3seed_30k_2026-07-04"
)
LOG="outputs/experiments/watch_track_b_belief_unweighted_vps.log"
mkdir -p outputs/experiments

echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher start, host=$HOST" >> "$LOG"

while true; do
  ALL_DONE=1
  for job in "${JOBS[@]}"; do
    REMOTE_DIR="~/scres-ia/outputs/experiments/${job}"
    LOCAL_DIR="outputs/experiments/${job}"
    RUNNING=$(ssh -o ConnectTimeout=15 "$HOST" "pgrep -f run_track_b_risk_belief_sidecar.py.*${job} | wc -l" 2>>"$LOG")
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $job processes_running=$RUNNING" >> "$LOG"
    if [ "$RUNNING" -gt 0 ] 2>/dev/null; then
      ALL_DONE=0
    else
      mkdir -p "$LOCAL_DIR"
      rsync -az "$HOST:$REMOTE_DIR/" "$LOCAL_DIR/" >> "$LOG" 2>&1
      HAS_SUMMARY=$(test -f "$LOCAL_DIR/summary.json" && echo yes || echo no)
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] $job has_summary=$HAS_SUMMARY" >> "$LOG"
      if [ "$HAS_SUMMARY" != "yes" ]; then
        ALL_DONE=0
      fi
    fi
  done
  if [ "$ALL_DONE" -eq 1 ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] both belief-unweighted jobs done, fetched" >> "$LOG"
    exit 0
  fi
  sleep 120
done
