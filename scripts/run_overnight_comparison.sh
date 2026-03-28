#!/usr/bin/env bash
# =============================================================================
# Overnight comparison: ReT_seq_v1 vs ReT_cd (Cobb-Douglas)
#
# Phase 1: Short comparison (100k × 5 seeds each) to determine winner
# Phase 2: Production run (500k × 10 seeds) with the winner
#
# The user is sleeping — this runs autonomously.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

echo "=== OVERNIGHT COMPARISON: ReT_seq_v1 vs ReT_cd ==="
echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# ─────────────────────────────────────────────────────────
# Phase 1a: ReT_seq_v1 (current primary)
# ─────────────────────────────────────────────────────────
echo "=== Phase 1a: ReT_seq_v1 κ=0.20 (100k × 5 seeds) ==="
python scripts/run_paper_benchmark.py \
  --label comparison_ret_seq_v1_100k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20 \
  --seeds 11 22 33 44 55 \
  --train-timesteps 100000 \
  --eval-episodes 20

echo ""
echo "=== Phase 1a complete ==="
echo ""

# ─────────────────────────────────────────────────────────
# Phase 1b: ReT_cd (Cobb-Douglas, Garrido 2024)
# ─────────────────────────────────────────────────────────
echo "=== Phase 1b: ReT_cd (100k × 5 seeds) ==="
python scripts/run_paper_benchmark.py \
  --label comparison_ret_cd_100k \
  --reward-mode ReT_cd \
  --seeds 11 22 33 44 55 \
  --train-timesteps 100000 \
  --eval-episodes 20

echo ""
echo "=== Phase 1b complete ==="
echo ""

# ─────────────────────────────────────────────────────────
# Phase 2: Production run with ReT_seq_v1 (500k × 10 seeds)
# (Always run this — it's the frozen primary regardless)
# ─────────────────────────────────────────────────────────
echo "=== Phase 2: Production ReT_seq_v1 (500k × 10 seeds) ==="
python scripts/run_paper_benchmark.py \
  --label paper_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20 \
  --seeds 11 22 33 44 55 66 77 88 99 100 \
  --train-timesteps 500000 \
  --eval-episodes 50 \
  --export-artifact-bundle

echo ""
echo "=== Phase 2 complete ==="
echo ""

echo "=== ALL PHASES COMPLETE ==="
echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""
echo "Results:"
echo "  Phase 1a: outputs/paper_benchmarks/comparison_ret_seq_v1_100k/"
echo "  Phase 1b: outputs/paper_benchmarks/comparison_ret_cd_100k/"
echo "  Phase 2:  outputs/paper_benchmarks/paper_ret_seq_k020_500k/"
echo ""
echo "Compare Phase 1a vs 1b to verify ReT_seq_v1 >= ReT_cd."
echo "Phase 2 is the paper-facing production artifact."
