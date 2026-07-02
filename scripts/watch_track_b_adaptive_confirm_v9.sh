#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

SLUG="thomaschisica/scresia-track-b-adaptive-confirm-v9"
OUT="outputs/experiments/track_b_adaptive_confirm_v9_2026-07-01"
FETCH="$OUT/fetched_v4"
LOG="$OUT/watcher_v4.log"
KAGGLE="/Users/thom/.local/bin/kaggle"

mkdir -p "$FETCH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] watcher script start slug=$SLUG" >> "$LOG"

BAD_COUNT=0
while true; do
  TOKEN=$($KAGGLE auth print-access-token 2>/dev/null)
  STATUS=$(KAGGLE_API_TOKEN="$TOKEN" "$KAGGLE" kernels status "$SLUG" 2>&1)
  LOW=$(printf '%s' "$STATUS" | tr '[:upper:]' '[:lower:]')
  echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STATUS $STATUS" >> "$LOG"

  if printf '%s' "$LOW" | grep -Eq 'authentication required|permission .*denied|cannot access kernel|forbidden|not found'; then
    BAD_COUNT=$((BAD_COUNT + 1))
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ACCESS_PROBLEM count=$BAD_COUNT $STATUS" >> "$LOG"
    if [ "$BAD_COUNT" -ge 3 ]; then
      osascript -e 'display notification "Check watcher_v4.log" with title "Kaggle confirm ACCESS ERROR" sound name "Glass"' >/dev/null 2>&1
      exit 3
    fi
  else
    BAD_COUNT=0
  fi

  if printf '%s' "$LOW" | grep -Eq 'complete|error|failed|cancel'; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TERMINAL $STATUS" >> "$LOG"
    TOKEN=$($KAGGLE auth print-access-token 2>/dev/null)
    KAGGLE_API_TOKEN="$TOKEN" "$KAGGLE" kernels output "$SLUG" -p "$FETCH" -q >> "$LOG" 2>&1
    osascript -e 'display notification "Track B confirm v9 fetched" with title "SCRES-IA" sound name "Glass"' >/dev/null 2>&1
    exit 0
  fi

  sleep 120
done
