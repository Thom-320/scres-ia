#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

SLUG="thomaschisica/scresia-track-b-e2-obs-masked-confirm"
OUT="outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02"
FETCH="$OUT/fetched"
LOG="$OUT/watcher.log"
KAGGLE="/Users/thom/.local/bin/kaggle"

mkdir -p "$FETCH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher script start slug=$SLUG" >> "$LOG"

BAD_COUNT=0
while true; do
  STATUS=$("$KAGGLE" kernels status "$SLUG" 2>&1)
  LOW=$(printf '%s' "$STATUS" | tr '[:upper:]' '[:lower:]')
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STATUS $STATUS" >> "$LOG"

  if printf '%s' "$LOW" | grep -Eq 'authentication required|permission .*denied|cannot access kernel|forbidden|not found'; then
    BAD_COUNT=$((BAD_COUNT + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ACCESS_PROBLEM count=$BAD_COUNT $STATUS" >> "$LOG"
    if [ "$BAD_COUNT" -ge 5 ]; then
      echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] giving up after repeated access problems" >> "$LOG"
      exit 3
    fi
  else
    BAD_COUNT=0
  fi

  if printf '%s' "$LOW" | grep -Eq 'complete|error|failed|cancel'; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TERMINAL $STATUS" >> "$LOG"
    "$KAGGLE" kernels output "$SLUG" -p "$FETCH" -q >> "$LOG" 2>&1
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] fetched to $FETCH" >> "$LOG"
    exit 0
  fi

  sleep 120
done
