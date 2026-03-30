#!/usr/bin/env bash
# =============================================================================
# Final overnight comparison: ReT_seq_v1 vs control_v1 (both with priming fix)
# 500k × 5 seeds each, under increased + stochastic_pt
#
# This settles the reward function question once and for all.
# =============================================================================
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "=== FINAL OVERNIGHT: ReT_seq_v1 vs control_v1 ==="
echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Phase 1: ReT_seq_v1 (500k × 5 seeds)
echo ""
echo "[Phase 1] ReT_seq_v1 κ=0.20 (500k × 5 seeds)"
python scripts/benchmark_control_reward.py \
  --reward-mode ReT_seq_v1 --ret-seq-kappa 0.20 \
  --seeds 11 22 33 44 55 \
  --train-timesteps 500000 \
  --eval-episodes 20 \
  --step-size-hours 168 --max-steps 260 \
  --risk-level increased --stochastic-pt \
  --eval-risk-levels increased severe \
  --output-dir outputs/benchmarks/final_ret_seq_v1_500k
echo "[Phase 1] Done"

# Phase 2: control_v1 (500k × 5 seeds)
echo ""
echo "[Phase 2] control_v1 w_bo=4.0 w_cost=0.02 (500k × 5 seeds)"
python scripts/benchmark_control_reward.py \
  --reward-mode control_v1 \
  --w-bo 4.0 --w-cost 0.02 --w-disr 0.0 \
  --seeds 11 22 33 44 55 \
  --train-timesteps 500000 \
  --eval-episodes 20 \
  --step-size-hours 168 --max-steps 260 \
  --risk-level increased --stochastic-pt \
  --eval-risk-levels increased severe \
  --output-dir outputs/benchmarks/final_control_v1_500k
echo "[Phase 2] Done"

echo ""
echo "=== COMPLETE ==="
echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""
echo "Results:"
echo "  outputs/benchmarks/final_ret_seq_v1_500k/"
echo "  outputs/benchmarks/final_control_v1_500k/"
