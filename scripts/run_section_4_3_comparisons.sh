#!/usr/bin/env bash
set -euo pipefail

# Matched-budget algorithm comparison for manuscript Section 4.3.
# Train on the frozen `increased + stochastic_pt` benchmark and evaluate on
# current / increased / severe via the built-in cross-evaluation path.
# This keeps Section 4.2 (500k backbone) separate from Section 4.3.
#
# Variants:
#   - PPO + MLP, v1, frame_stack=1
#   - PPO + MLP, v2, frame_stack=1
#   - PPO + MLP, v1, frame_stack=4
#   - RecurrentPPO + LSTM, v2, frame_stack=1
#
# Usage:
#   bash scripts/run_section_4_3_comparisons.sh
#   bash scripts/run_section_4_3_comparisons.sh --smoke

MODE="${1:-}"
SEEDS="11 22 33 44 55 66 77 88 99 111"
TIMESTEPS=50000
EVAL_EPISODES=10
COMMON_ARGS="--step-size-hours 168 --max-steps 260 --stochastic-pt"
BASE_ARGS="--risk-level increased --eval-risk-levels current increased severe --w-bo 4.0 --w-cost 0.02 --w-disr 0.0"
ARTIFACT_ROOT="docs/artifacts/control_reward"
OUTPUT_ROOT="outputs/benchmarks"

if [[ "$MODE" == "--smoke" ]]; then
  SEEDS="11"
  TIMESTEPS=512
  EVAL_EPISODES=2
  COMMON_ARGS="--step-size-hours 24 --max-steps 8 --stochastic-pt"
fi

run_variant () {
  local algo="$1"
  local obs="$2"
  local frame_stack="$3"
  local label="$4"

  python scripts/benchmark_control_reward.py \
    --seeds ${SEEDS} \
    --train-timesteps ${TIMESTEPS} \
    --eval-episodes ${EVAL_EPISODES} \
    ${BASE_ARGS} \
    --learned-only \
    --algo ${algo} \
    --observation-version ${obs} \
    --frame-stack ${frame_stack} \
    --output-dir ${OUTPUT_ROOT}/${label} \
    --artifact-root ${ARTIFACT_ROOT} \
    --artifact-label ${label} \
    ${COMMON_ARGS}
}

run_variant ppo v1 1 section4_3_ppo_v1_fs1
run_variant ppo v2 1 section4_3_ppo_v2_fs1
run_variant ppo v1 4 section4_3_ppo_v1_fs4
run_variant recurrent_ppo v2 1 section4_3_recurrent_v2_fs1

python scripts/section_4_3_analysis.py \
  --baseline-run-dir outputs/benchmarks/control_reward \
  --run-dirs \
  ${OUTPUT_ROOT}/section4_3_ppo_v1_fs1 \
  ${OUTPUT_ROOT}/section4_3_ppo_v2_fs1 \
  ${OUTPUT_ROOT}/section4_3_ppo_v1_fs4 \
  ${OUTPUT_ROOT}/section4_3_recurrent_v2_fs1 \
  --output-dir ${ARTIFACT_ROOT}/section4_3_algorithm_comparison

echo ""
echo "Section 4.3 comparison complete."
