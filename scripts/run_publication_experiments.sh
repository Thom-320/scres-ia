#!/usr/bin/env bash
set -euo pipefail

# Publication experiment runner for SCRES-IA Q1/Q2 journal submission.
# Runs the full benchmark suite: heuristic tuning, PPO/SAC/frame-stack
# training at 500k timesteps x 10 seeds, cross-scenario evaluation.
#
# Usage:
#   bash scripts/run_publication_experiments.sh           # full suite
#   bash scripts/run_publication_experiments.sh --smoke    # quick smoke test

SEEDS="11 22 33 44 55 66 77 88 99 100"
TIMESTEPS=500000
EVAL_EPISODES=10
RISK_LEVEL="increased"
COMMON_ARGS="--step-size-hours 168 --max-steps 260 --stochastic-pt"
PHASE1_ARGS="--output-dir outputs/benchmarks/control_reward_tuning --artifact-label control_reward_tuning"
PHASE2_ARGS="--output-dir outputs/benchmarks/control_reward --artifact-label control_reward"
PHASE3_ARGS="--output-dir outputs/benchmarks/control_reward_sac_fs1 --artifact-label control_reward_sac_fs1"
PHASE4_ARGS="--output-dir outputs/benchmarks/control_reward_ppo_fs4 --artifact-label control_reward_ppo_fs4"
PHASE5_ARGS="--output-dir outputs/benchmarks/control_reward_recurrent_ppo_fs1 --artifact-label control_reward_recurrent_ppo_fs1"
SMOKE_ARGS=""

if [[ "${1:-}" == "--smoke" ]]; then
    echo "=== SMOKE TEST MODE ==="
    SEEDS="11"
    TIMESTEPS=256
    EVAL_EPISODES=2
    COMMON_ARGS="--step-size-hours 24 --max-steps 8 --stochastic-pt"
    PHASE1_ARGS="--output-dir outputs/benchmarks/control_reward_tuning_smoke --artifact-label control_reward_tuning_smoke"
    PHASE2_ARGS="--output-dir outputs/benchmarks/control_reward_stopt_smoke --artifact-label control_reward_stopt_smoke"
    PHASE3_ARGS="--output-dir outputs/benchmarks/control_reward_sac_smoke --artifact-label control_reward_sac_smoke"
    PHASE4_ARGS="--output-dir outputs/benchmarks/control_reward_ppo_fs4_smoke --artifact-label control_reward_ppo_fs4_smoke"
    PHASE5_ARGS="--output-dir outputs/benchmarks/control_reward_recurrent_ppo_smoke --artifact-label control_reward_recurrent_ppo_smoke"
    SMOKE_ARGS="--skip-artifact-export"
fi

echo "Seeds: $SEEDS"
echo "Timesteps: $TIMESTEPS"
echo "Eval episodes: $EVAL_EPISODES"

# Phase 1: Tune heuristic parameters on training seeds
echo ""
echo "=== Phase 1: Heuristic tuning ==="
python scripts/benchmark_control_reward.py \
    --tune-heuristic \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    --w-bo 1.0 2.0 4.0 \
    --w-cost 0.02 0.06 0.10 \
    --w-disr 0.0 \
    --algo ppo \
    $PHASE1_ARGS \
    $SMOKE_ARGS \
    $COMMON_ARGS

# Phase 2: PPO 500k x 10 seeds, trained on increased, cross-eval on all
echo ""
echo "=== Phase 2: PPO baseline ==="
python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    --eval-risk-levels current increased severe \
    --w-bo 1.0 2.0 4.0 \
    --w-cost 0.02 0.06 0.10 \
    --w-disr 0.0 \
    --algo ppo \
    $PHASE2_ARGS \
    $SMOKE_ARGS \
    $COMMON_ARGS

# Phase 3: SAC 500k x 10 seeds
echo ""
echo "=== Phase 3: SAC ==="
python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    --eval-risk-levels current increased severe \
    --w-bo 1.0 2.0 4.0 \
    --w-cost 0.02 0.06 0.10 \
    --w-disr 0.0 \
    --algo sac \
    $PHASE3_ARGS \
    $SMOKE_ARGS \
    $COMMON_ARGS

# Phase 4: PPO + frame-stacking (fs=4, obs v2)
echo ""
echo "=== Phase 4: PPO + frame-stack 4 ==="
python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    --eval-risk-levels current increased severe \
    --w-bo 1.0 2.0 4.0 \
    --w-cost 0.02 0.06 0.10 \
    --w-disr 0.0 \
    --algo ppo \
    --frame-stack 4 \
    --observation-version v2 \
    $PHASE4_ARGS \
    $SMOKE_ARGS \
    $COMMON_ARGS

# Phase 5: RecurrentPPO (optional — uncomment if frame-stack shows improvement)
# echo ""
# echo "=== Phase 5: RecurrentPPO ==="
# python scripts/benchmark_control_reward.py \
#     --seeds $SEEDS \
#     --train-timesteps "$TIMESTEPS" \
#     --eval-episodes "$EVAL_EPISODES" \
#     --risk-level "$RISK_LEVEL" \
#     --eval-risk-levels current increased severe \
#     --w-bo 1.0 2.0 4.0 \
#     --w-cost 0.02 0.06 0.10 \
#     --w-disr 0.0 \
#     --algo recurrent_ppo \
#     --observation-version v2 \
#     $PHASE5_ARGS \
#     $SMOKE_ARGS \
#     $COMMON_ARGS

echo ""
echo "=== All phases complete ==="
echo "Check outputs/benchmarks/ for results."
