#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

SLUG="thomaschisica/scresia-track-b-adaptive-sweep"
OUT="outputs/experiments/track_b_adaptive_sweep_kaggle_2026-07-01_v5"
FETCH="$OUT/fetched"
LOG="$OUT/live_launch_watcher.log"
KAGGLE="/Users/thom/.local/bin/kaggle"

mkdir -p "$FETCH"
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] launch watcher start slug=$SLUG" >> "$LOG"

LAST=""
while true; do
  STATUS=$("$KAGGLE" kernels status "$SLUG" 2>&1)
  if [ "$STATUS" != "$LAST" ]; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] STATUS $STATUS" >> "$LOG"
    LAST="$STATUS"
  fi

  LOW=$(printf '%s' "$STATUS" | tr '[:upper:]' '[:lower:]')
  if printf '%s' "$LOW" | grep -Eq 'complete|error|failed|cancel'; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TERMINAL $STATUS" >> "$LOG"
    "$KAGGLE" kernels output "$SLUG" -p "$FETCH" -q >> "$LOG" 2>&1

    if printf '%s' "$LOW" | grep -q 'complete'; then
      .venv/bin/python scripts/run_track_b_adaptive_sweep.py \
        --reward-modes control_v1,ReT_excel_plus_cvar,ReT_tail_v2,ReT_garrido2024_train \
        --observation-versions v7,v8,v9 \
        --cvar-alphas 0.05,0.1,0.2 \
        --output-dir "$FETCH/track_b_adaptive_sweep" \
        --summarize-only >> "$LOG" 2>&1
      osascript -e 'display notification "Downloaded and postprocessed" with title "Kaggle sweep DONE" sound name "Glass"' >/dev/null 2>&1
    else
      osascript -e 'display notification "Check live_launch_watcher.log" with title "Kaggle sweep ERROR" sound name "Glass"' >/dev/null 2>&1
    fi
    exit 0
  fi

  sleep 60
done
