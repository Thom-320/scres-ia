#!/usr/bin/env bash
# =============================================================================
# Overnight production run: ReT_seq_v1 κ=0.20 × 500k × 10 seeds
#
# This produces the paper-facing artifact bundle with:
#   - training_trace.csv (learning curves)
#   - proof_trajectories.csv (per-step shift/disruption data)
#   - episode_metrics.csv, policy_summary.csv, comparison_table.csv
#   - Proof-of-learning plots (learning curve, shift timeline, cross-scenario)
#
# Estimated runtime: 4-8 hours depending on machine.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# Activate venv if present
if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== OVERNIGHT PRODUCTION RUN ==="
echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Reward: ReT_seq_v1 κ=0.20"
echo "Seeds: 10 (11-100)"
echo "Timesteps: 500,000"
echo "Risk: increased + stochastic_pt"
echo "Cross-eval: current, increased, severe"
echo ""

python scripts/run_paper_benchmark.py \
  --label paper_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20 \
  --seeds 11 22 33 44 55 66 77 88 99 100 \
  --train-timesteps 500000 \
  --eval-episodes 50 \
  --export-artifact-bundle

echo ""
echo "=== COMPLETED ==="
echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "Output: outputs/paper_benchmarks/paper_ret_seq_k020_500k/"
echo ""
echo "Next steps:"
echo "  1. Check outputs/paper_benchmarks/paper_ret_seq_k020_500k/summary.json"
echo "  2. Review proof-of-learning plots in proof_of_learning/ subdirectory"
echo "  3. Run: python scripts/analyze_paper_benchmark_trio.py (if available)"
