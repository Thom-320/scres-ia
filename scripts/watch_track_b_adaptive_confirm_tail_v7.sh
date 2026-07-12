#!/usr/bin/env bash
set +e

cd /Users/thom/Projects/research/scres-ia || exit 2

SLUG="thomaschisica/scresia-track-b-adaptive-confirm-tail-v7"
OUT="outputs/experiments/track_b_adaptive_confirm_tail_v7_2026-07-02"
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
    if [ "$BAD_COUNT" -ge 3 ]; then
      osascript -e 'display notification "Check watcher.log" with title "Kaggle tail-v7 ACCESS ERROR" sound name "Glass"' >/dev/null 2>&1
      exit 3
    fi
  else
    BAD_COUNT=0
  fi

  if printf '%s' "$LOW" | grep -Eq 'complete|error|failed|cancel'; then
    echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] TERMINAL $STATUS" >> "$LOG"
    "$KAGGLE" kernels output "$SLUG" -p "$FETCH" -q >> "$LOG" 2>&1
    if [ -f "$FETCH/track_b_adaptive_confirm_tail_v7/decision.json" ]; then
      /usr/bin/python3 - <<'PY' >> "$LOG" 2>&1
import json
from pathlib import Path
p = Path("outputs/experiments/track_b_adaptive_confirm_tail_v7_2026-07-02/fetched/track_b_adaptive_confirm_tail_v7/decision.json")
d = json.loads(p.read_text())
c = d.get("candidate", {})
print("DECISION", {
    "status": c.get("status"),
    "raw_ret_win": c.get("raw_ret_win"),
    "tail_win": c.get("tail_win"),
    "cost_nonworse": c.get("cost_nonworse"),
    "delta": c.get("order_ret_excel_delta"),
    "cvar_delta": c.get("order_ret_excel_cvar05_delta"),
    "cost_delta": c.get("cost_index_delta"),
})
PY
    fi
    osascript -e 'display notification "Track B tail-v7 fetched" with title "SCRES-IA" sound name "Glass"' >/dev/null 2>&1
    exit 0
  fi

  sleep 120
done
