#!/usr/bin/env bash
set -euo pipefail

# PBRS experiment runner for SCRES-IA Q1/Q2 journal submission.
# Runs PBRS ablation experiments after the main publication suite completes.
#
# Usage:
#   bash scripts/run_pbrs_experiments.sh           # full suite
#   bash scripts/run_pbrs_experiments.sh --smoke    # quick smoke test

SEEDS="11 22 33 44 55 66 77 88 99 111"
TIMESTEPS=500000
EVAL_EPISODES=10
RISK_LEVEL="increased"
EVAL_LEVEL_ARGS="--eval-risk-levels current increased severe"
WEIGHT_ARGS="--w-bo 2.0 --w-cost 0.06 --w-disr 0.0"
COMMON_ARGS="--step-size-hours 168 --max-steps 260 --stochastic-pt"
MODE_ARGS=""

PBRS1_ARGS="--output-dir outputs/benchmarks/pbrs_cumulative --artifact-label pbrs_cumulative"
PBRS2_ARGS="--output-dir outputs/benchmarks/pbrs_step_level --artifact-label pbrs_step_level"
PBRS3A_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_0.1 --artifact-label pbrs_alpha_0.1"
PBRS3B_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_0.5 --artifact-label pbrs_alpha_0.5"
PBRS3C_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_2.0 --artifact-label pbrs_alpha_2.0"

MODE="${1:-}"
case "$MODE" in
    "")
        ;;
    "--smoke")
        echo "=== SMOKE TEST MODE ==="
        SEEDS="11"
        TIMESTEPS=256
        EVAL_EPISODES=2
        COMMON_ARGS="--step-size-hours 24 --max-steps 8 --stochastic-pt"
        PBRS1_ARGS="--output-dir outputs/benchmarks/pbrs_cumulative_smoke --artifact-label pbrs_cumulative_smoke"
        PBRS2_ARGS="--output-dir outputs/benchmarks/pbrs_step_level_smoke --artifact-label pbrs_step_level_smoke"
        PBRS3A_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_0.1_smoke --artifact-label pbrs_alpha_0.1_smoke"
        PBRS3B_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_0.5_smoke --artifact-label pbrs_alpha_0.5_smoke"
        PBRS3C_ARGS="--output-dir outputs/benchmarks/pbrs_alpha_2.0_smoke --artifact-label pbrs_alpha_2.0_smoke"
        MODE_ARGS="--skip-artifact-export"
        ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        exit 1
        ;;
esac

echo "Seeds: $SEEDS"
echo "Timesteps: $TIMESTEPS"
echo "Eval episodes: $EVAL_EPISODES"

# Phase PBRS-1: PPO + PBRS cumulative (alpha=1.0, tau=0.95) — main PBRS result
echo ""
echo "=== PBRS-1: PPO + cumulative PBRS (alpha=1.0) ==="
python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    $EVAL_LEVEL_ARGS \
    $WEIGHT_ARGS \
    --algo ppo \
    --reward-mode control_v1_pbrs \
    --pbrs-alpha 1.0 \
    --pbrs-tau 0.95 \
    --pbrs-gamma 0.99 \
    --pbrs-variant cumulative \
    $PBRS1_ARGS \
    $MODE_ARGS \
    $COMMON_ARGS

# Phase PBRS-2: PPO + PBRS step-level (v2 obs) — temporal responsiveness ablation
echo ""
echo "=== PBRS-2: PPO + step-level PBRS (alpha=1.0, v2) ==="
python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    $EVAL_LEVEL_ARGS \
    $WEIGHT_ARGS \
    --algo ppo \
    --reward-mode control_v1_pbrs \
    --pbrs-alpha 1.0 \
    --pbrs-tau 0.95 \
    --pbrs-gamma 0.99 \
    --pbrs-variant step_level \
    --observation-version v2 \
    $PBRS2_ARGS \
    $MODE_ARGS \
    $COMMON_ARGS

# Phase PBRS-3: Alpha sweep (cumulative variant) — sensitivity analysis
for ALPHA_VAL in 0.1 0.5 2.0; do
    echo ""
    echo "=== PBRS-3: Alpha sweep (alpha=$ALPHA_VAL) ==="
    case "$ALPHA_VAL" in
        "0.1") SWEEP_ARGS="$PBRS3A_ARGS" ;;
        "0.5") SWEEP_ARGS="$PBRS3B_ARGS" ;;
        "2.0") SWEEP_ARGS="$PBRS3C_ARGS" ;;
    esac
    python scripts/benchmark_control_reward.py \
        --seeds $SEEDS \
        --train-timesteps "$TIMESTEPS" \
        --eval-episodes "$EVAL_EPISODES" \
        --risk-level "$RISK_LEVEL" \
        $EVAL_LEVEL_ARGS \
        $WEIGHT_ARGS \
        --algo ppo \
        --reward-mode control_v1_pbrs \
        --pbrs-alpha "$ALPHA_VAL" \
        --pbrs-tau 0.95 \
        --pbrs-gamma 0.99 \
        --pbrs-variant cumulative \
        $SWEEP_ARGS \
        $MODE_ARGS \
        $COMMON_ARGS
done

echo ""
echo "=== All PBRS phases complete ==="
echo "Check outputs/benchmarks/pbrs_* for results."
