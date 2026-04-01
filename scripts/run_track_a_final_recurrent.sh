#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
[ -f .venv/bin/activate ] && source .venv/bin/activate

echo "=== TRACK A FINAL: RecurrentPPO + control_v1 + v4 + 168h ==="
echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"

python scripts/benchmark_control_reward.py \
  --algo recurrent_ppo \
  --reward-mode control_v1 \
  --observation-version v4 \
  --w-bo 4.0 --w-cost 0.02 --w-disr 0.0 \
  --seeds 11 22 33 44 55 \
  --train-timesteps 500000 \
  --eval-episodes 20 \
  --step-size-hours 168 --max-steps 260 \
  --risk-level increased --stochastic-pt \
  --eval-risk-levels current increased severe \
  --output-dir outputs/benchmarks/track_a_recurrent_control_v1_500k

echo ""
echo "=== COMPLETE ==="
echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Results: outputs/benchmarks/track_a_recurrent_control_v1_500k/"
