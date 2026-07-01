#!/usr/bin/env bash
# Watcher for Kaggle kernel: thomaschisica/scresia-track-b-campaign-optimized
# Polls every 3 minutes. Downloads output when done and runs analysis.
# Usage: bash scripts/watch_kaggle_tb.sh &
# To stop: kill %1

KERNEL="thomaschisica/scresia-track-b-campaign-optimized"
WATCH_DIR="/tmp/kaggle_tb_watch"
OUTPUT_DIR="$HOME/Projects/research/scres-ia/outputs/experiments/track_b_kaggle_v1"
KAGGLE_BIN="$HOME/.local/bin/kaggle"
MAX_WAIT_SEC=$((3 * 3600))  # 3h timeout
POLL_SEC=180                  # 3 min

mkdir -p "$WATCH_DIR"

echo "=== KAGGLE WATCHER: $KERNEL ==="
echo "Started at $(date)"
echo "Polling every ${POLL_SEC}s, timeout ${MAX_WAIT_SEC}s"
echo ""

START_TS=$(date +%s)
LAST_STATUS=""

while true; do
    NOW_TS=$(date +%s)
    ELAPSED=$((NOW_TS - START_TS))

    if [ $ELAPSED -gt $MAX_WAIT_SEC ]; then
        echo "[$(date +%H:%M:%S)] ⏰ TIMEOUT after ${MAX_WAIT_SEC}s"
        break
    fi

    STATUS=$($KAGGLE_BIN kernels status "$KERNEL" 2>&1)
    STATUS_SHORT=$(echo "$STATUS" | grep "has status" | sed 's/.*has status //')

    if [ "$STATUS_SHORT" != "$LAST_STATUS" ]; then
        echo "[$(date +%H:%M:%S)] Status: $STATUS_SHORT  (${ELAPSED}s elapsed)"
        LAST_STATUS="$STATUS_SHORT"
    else
        echo "[$(date +%H:%M:%S)] Status: $STATUS_SHORT  (${ELAPSED}s elapsed, no change)"
    fi

    case "$STATUS_SHORT" in
        *COMPLETE*|*complete*|*Complete*)
            echo ""
            echo "=== KERNEL COMPLETED ==="
            echo "Downloading output..."
            cd "$OUTPUT_DIR/.." 2>/dev/null || mkdir -p "$OUTPUT_DIR"
            $KAGGLE_BIN kernels output "$KERNEL" -p "$OUTPUT_DIR" 2>&1

            # Show result
            cd "$HOME/Projects/research/scres-ia" 2>/dev/null
            if [ -f "$OUTPUT_DIR/summary.json" ]; then
                echo ""
                echo "=== RESULTS ==="
                python3 -c "
import json
s=json.load(open('$OUTPUT_DIR/summary.json'))
v=s['verdict']
d=s['deltas']
print(f\"  PPO ReT: {s['ppo']['ret_excel_mean']:.6f}\")
print(f\"  Best static: {s['best_static']['ret_excel_mean']:.6f} ({s['best_static']['policy']})\")
print(f\"  Δ ReT: {d['ret_excel']:+.6f}\")
print(f\"  CI95 Δ: [{d.get('ret_excel_ci95_low',0):+.6f}]\")
print(f\"  Cohen's d: {v['cohens_d']:+.4f}\")
print(f\"  raw_ret_win: {v['raw_ret_win']}\")
print(f\"  pareto_ret_cost: {v['pareto_ret_cost']}\")
print(f\"  CV: static={v['h3_static_cv']:.4f}  PPO={v['h3_ppo_cv']:.4f}\")
print(f\"  same_reward_win: {v['same_reward_win']}\")
"
            else
                echo "No summary.json found. Checking for errors..."
                ls -la "$OUTPUT_DIR/" 2>/dev/null
            fi
            echo ""
            echo "=== DONE at $(date) ==="

            # macOS notification
            osascript -e "display notification \"Track B Kaggle kernel completed\" with title \"SCRES-IA Watcher\"" 2>/dev/null || true

            # Also print a prominent message
            printf '\n\033[1;33m🔔 KAGGLE KERNEL DONE — CHECK RESULTS 🔔\033[0m\n\n'
            exit 0
            ;;
        *ERROR*|*error*|*Error*|*FAILED*|*Failed*|*failed*)
            echo ""
            echo "=== KERNEL FAILED ==="
            echo "Downloading output for error inspection..."
            mkdir -p "$OUTPUT_DIR"
            $KAGGLE_BIN kernels output "$KERNEL" -p "$OUTPUT_DIR" 2>&1
            ls -la "$OUTPUT_DIR/" 2>/dev/null
            osascript -e "display notification \"Track B Kaggle kernel FAILED\" with title \"SCRES-IA Error\"" 2>/dev/null || true
            printf '\n\033[1;31m❌ KAGGLE KERNEL FAILED ❌\033[0m\n\n'
            exit 1
            ;;
        *RUNNING*|*running*)
            ;;
        *)
            echo "  Unknown status: $STATUS_SHORT"
            ;;
    esac

    sleep $POLL_SEC
done

echo "[$(date +%H:%M:%S)] Watcher exiting."
