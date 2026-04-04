#!/usr/bin/env bash
# reproduce_paper.sh — Full reproduction of paper results
# Estimated wall-clock: ~48h on M3 Pro (all experiments)
# Quick smoke: ~30 min (with --smoke flag)
set -euo pipefail

SEEDS="11 22 33 44 55"
TIMESTEPS=500000
EVAL_EPISODES=20
MAX_STEPS=260
OUTPUT_ROOT="outputs/reproduce_$(date +%Y%m%d_%H%M%S)"

if [[ "${1:-}" == "--smoke" ]]; then
    echo "🔬 SMOKE MODE: reduced timesteps and seeds"
    SEEDS="42"
    TIMESTEPS=20000
    EVAL_EPISODES=5
    OUTPUT_ROOT="outputs/reproduce_smoke_$(date +%Y%m%d_%H%M%S)"
fi

PYTHON="${PYTHON:-$(which python3 || which python)}"
echo "Using Python: $PYTHON"
echo "Output root: $OUTPUT_ROOT"
mkdir -p "$OUTPUT_ROOT"

echo ""
echo "============================================"
echo "Step 0: Validate DES against thesis baselines"
echo "============================================"
$PYTHON run_static.py --year-basis thesis 2>&1 | tee "$OUTPUT_ROOT/step0_static_baselines.log"
$PYTHON validation_report.py --official-basis thesis 2>&1 | tee "$OUTPUT_ROOT/step0_validation.log"

echo ""
echo "============================================"
echo "Step 1: Track A — PPO vs S2 (negative result)"
echo "============================================"
$PYTHON scripts/run_paper_benchmark.py \
    --reward-mode control_v1 \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --seeds $SEEDS \
    --output-dir "$OUTPUT_ROOT/track_a_control_v1" \
    2>&1 | tee "$OUTPUT_ROOT/step1_track_a.log"

echo ""
echo "============================================"
echo "Step 2: Track B — PPO benchmark (positive result)"
echo "============================================"
$PYTHON scripts/run_track_b_benchmark.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --output-dir "$OUTPUT_ROOT/track_b_benchmark" \
    2>&1 | tee "$OUTPUT_ROOT/step2_track_b.log"

echo ""
echo "============================================"
echo "Step 3: Track B — Causal ablation"
echo "============================================"
$PYTHON scripts/run_track_b_ablation.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --ablation-configs joint shift_only downstream_only \
    --output-dir "$OUTPUT_ROOT/track_b_ablation" \
    2>&1 | tee "$OUTPUT_ROOT/step3_ablation.log"

echo ""
echo "============================================"
echo "Step 4: Track B — Reward sweep (7 modes)"
echo "============================================"
$PYTHON scripts/run_track_b_reward_sweep.py \
    --algo ppo \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --max-steps "$MAX_STEPS" \
    --output-dir "$OUTPUT_ROOT/track_b_reward_sweep" \
    2>&1 | tee "$OUTPUT_ROOT/step4_reward_sweep.log"

echo ""
echo "============================================"
echo "Step 5: Cross-scenario evaluation"
echo "============================================"
# Use the model from step 2
MODEL_DIR=$(find "$OUTPUT_ROOT/track_b_benchmark" -name "models" -type d | head -1)
if [ -n "$MODEL_DIR" ]; then
    $PYTHON scripts/eval_track_b_cross_scenario.py \
        --model-dir "$(dirname "$MODEL_DIR")" \
        --eval-risk-levels current increased severe severe_extended \
        --eval-episodes "$EVAL_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --output-dir "$OUTPUT_ROOT/track_b_cross_scenario" \
        2>&1 | tee "$OUTPUT_ROOT/step5_cross_scenario.log"
else
    echo "WARNING: No model found from step 2, skipping cross-scenario"
fi

echo ""
echo "============================================"
echo "Step 6: Forecast sensitivity"
echo "============================================"
if [ -n "$MODEL_DIR" ]; then
    $PYTHON scripts/eval_track_b_forecast_sensitivity.py \
        --model-dir "$(dirname "$MODEL_DIR")" \
        --seeds $SEEDS \
        --eval-episodes "$EVAL_EPISODES" \
        --max-steps "$MAX_STEPS" \
        --output-dir "$OUTPUT_ROOT/track_b_forecast_sensitivity" \
        2>&1 | tee "$OUTPUT_ROOT/step6_forecast_sensitivity.log"
else
    echo "WARNING: No model found from step 2, skipping forecast sensitivity"
fi

echo ""
echo "============================================"
echo "Step 7: Tests"
echo "============================================"
$PYTHON -m pytest tests/ -q --tb=short 2>&1 | tee "$OUTPUT_ROOT/step7_tests.log"

echo ""
echo "============================================"
echo "✅ COMPLETE — Results in: $OUTPUT_ROOT"
echo "============================================"
echo ""
echo "Next: compare results against paper_results_package/README.md tables"
