#!/usr/bin/env bash
# =============================================================================
# Overnight comparison: ReT_seq_v1 vs ReT_cd_v1 vs ReT_cd_sigmoid
# Then production run with ReT_seq_v1 (the primary).
#
# Phase 1: Short comparison (100k × 3 seeds) — ~45 min each
# Phase 2: Production (500k × 10 seeds) — ~4 hours
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

if [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi

COMMON="--seeds 3 --train-timesteps 100000 --eval-episodes 10 --step-size-hours 168 --max-steps 260 --risk-level increased --stochastic-pt --n-envs 1"
OUTDIR="outputs/benchmarks"

echo "=== OVERNIGHT COMPARISON ==="
echo "Start: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo ""

# ─── Phase 1a: ReT_seq_v1 ───
echo "[Phase 1a] ReT_seq_v1 κ=0.20 (100k × 3 seeds)"
python scripts/benchmark_control_reward.py \
  --reward-mode ReT_seq_v1 --ret-seq-kappa 0.20 \
  $COMMON \
  --output-dir "$OUTDIR/compare_ret_seq_v1_100k"
echo "[Phase 1a] Done: $OUTDIR/compare_ret_seq_v1_100k"
echo ""

# ─── Phase 1b: ReT_cd_v1 ───
echo "[Phase 1b] ReT_cd_v1 (100k × 3 seeds)"
python scripts/benchmark_control_reward.py \
  --reward-mode ReT_cd_v1 \
  $COMMON \
  --output-dir "$OUTDIR/compare_ret_cd_v1_100k"
echo "[Phase 1b] Done: $OUTDIR/compare_ret_cd_v1_100k"
echo ""

# ─── Phase 1c: ReT_cd_sigmoid ───
echo "[Phase 1c] ReT_cd_sigmoid (100k × 3 seeds)"
python scripts/benchmark_control_reward.py \
  --reward-mode ReT_cd_sigmoid \
  $COMMON \
  --output-dir "$OUTDIR/compare_ret_cd_sigmoid_100k"
echo "[Phase 1c] Done: $OUTDIR/compare_ret_cd_sigmoid_100k"
echo ""

echo "=== Phase 1 Complete. Compare results in $OUTDIR/compare_* ==="
echo ""

# ─── Phase 2: Production run with ReT_seq_v1 ───
echo "[Phase 2] Production: ReT_seq_v1 κ=0.20 (500k × 10 seeds)"
python scripts/run_paper_benchmark.py \
  --label paper_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20 \
  --seeds 11 22 33 44 55 66 77 88 99 100 \
  --train-timesteps 500000 \
  --eval-episodes 50 \
  --export-artifact-bundle
echo "[Phase 2] Done: outputs/paper_benchmarks/paper_ret_seq_k020_500k/"
echo ""

echo "=== ALL PHASES COMPLETE ==="
echo "End: $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
