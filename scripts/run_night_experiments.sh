#!/usr/bin/env bash
set -euo pipefail

# Night run script for SCRES-IA Q1/Q2 journal submission.
# Runs ONLY the remaining comparator phases (SAC, PPO+FS4, RecurrentPPO)
# since PPO Phase 2 is already complete.
#
# Usage:
#   bash scripts/run_night_experiments.sh           # full night run
#   bash scripts/run_night_experiments.sh --smoke    # quick smoke test

# ---------- Fixed best-weight combo from Phase 1 tuning ----------
WEIGHT_ARGS="--w-bo 4.0 --w-cost 0.02 --w-disr 0.0 --max-survivors 1"

# ---------- Defaults (full run) ----------
SEEDS="11 22 33 44 55"
TIMESTEPS=500000
EVAL_EPISODES=10
RISK_LEVEL="increased"
EVAL_LEVEL_ARGS="--eval-risk-levels current increased severe"
COMMON_ARGS="--step-size-hours 168 --max-steps 260 --stochastic-pt"
MODE_ARGS=""

SAC_DIR="outputs/benchmarks/control_reward_sac_fs1"
SAC_LABEL="control_reward_sac_fs1"
PPO_FS4_DIR="outputs/benchmarks/control_reward_ppo_fs4"
PPO_FS4_LABEL="control_reward_ppo_fs4"
RECURRENT_DIR="outputs/benchmarks/control_reward_recurrent_ppo_fs1"
RECURRENT_LABEL="control_reward_recurrent_ppo_fs1"

# Include the existing PPO run in the final analysis
PPO_BASELINE_DIR="outputs/benchmarks/control_reward"

# ---------- Mode handling ----------
MODE="${1:-}"
case "$MODE" in
    "")
        echo "=== FULL NIGHT RUN ==="
        ;;
    "--smoke")
        echo "=== SMOKE TEST MODE ==="
        SEEDS="11"
        TIMESTEPS=256
        EVAL_EPISODES=2
        COMMON_ARGS="--step-size-hours 24 --max-steps 8 --stochastic-pt"
        MODE_ARGS="--skip-artifact-export"
        SAC_DIR="outputs/benchmarks/control_reward_sac_smoke"
        SAC_LABEL="control_reward_sac_smoke"
        PPO_FS4_DIR="outputs/benchmarks/control_reward_ppo_fs4_smoke"
        PPO_FS4_LABEL="control_reward_ppo_fs4_smoke"
        RECURRENT_DIR="outputs/benchmarks/control_reward_recurrent_ppo_smoke"
        RECURRENT_LABEL="control_reward_recurrent_ppo_smoke"
        PPO_BASELINE_DIR=""
        ;;
    *)
        echo "Unsupported mode: $MODE" >&2
        echo "Usage: $0 [--smoke]" >&2
        exit 1
        ;;
esac

# ---------- Banner ----------
NIGHT_START="$(date +%s)"
echo ""
echo "================================================================"
echo "  Night experiment runner (comparator algorithms only)"
echo "  Started: $(date)"
echo "================================================================"
echo ""
echo "  Weight combo : $WEIGHT_ARGS"
echo "  Seeds        : $SEEDS"
echo "  Timesteps    : $TIMESTEPS"
echo "  Eval episodes: $EVAL_EPISODES"
echo "  Risk level   : $RISK_LEVEL"
echo "  Cross-eval   : $EVAL_LEVEL_ARGS"
echo ""

# ---------- Helper: elapsed time ----------
phase_timer() {
    local label="$1"
    local start="$2"
    local end="$(date +%s)"
    local elapsed=$(( end - start ))
    local mins=$(( elapsed / 60 ))
    local secs=$(( elapsed % 60 ))
    echo ""
    echo "  >> $label completed in ${mins}m ${secs}s"
    echo ""
}

# ==========================================================
# Phase A: SAC x 5 seeds x 500k
# ==========================================================
echo "=== Phase A: SAC ==="
echo "  Output: $SAC_DIR"
PHASE_START="$(date +%s)"

python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    $EVAL_LEVEL_ARGS \
    $WEIGHT_ARGS \
    --algo sac \
    --output-dir "$SAC_DIR" \
    --artifact-label "$SAC_LABEL" \
    $MODE_ARGS \
    $COMMON_ARGS

phase_timer "Phase A (SAC)" "$PHASE_START"

# ==========================================================
# Phase B: PPO + frame-stack 4 + v2 obs x 5 seeds x 500k
# ==========================================================
echo "=== Phase B: PPO + frame-stack 4 ==="
echo "  Output: $PPO_FS4_DIR"
PHASE_START="$(date +%s)"

python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    $EVAL_LEVEL_ARGS \
    $WEIGHT_ARGS \
    --algo ppo \
    --frame-stack 4 \
    --observation-version v2 \
    --output-dir "$PPO_FS4_DIR" \
    --artifact-label "$PPO_FS4_LABEL" \
    $MODE_ARGS \
    $COMMON_ARGS

phase_timer "Phase B (PPO+FS4)" "$PHASE_START"

# ==========================================================
# Phase C: RecurrentPPO + v2 obs x 5 seeds x 500k
# ==========================================================
echo "=== Phase C: RecurrentPPO ==="
echo "  Output: $RECURRENT_DIR"
PHASE_START="$(date +%s)"

python scripts/benchmark_control_reward.py \
    --seeds $SEEDS \
    --train-timesteps "$TIMESTEPS" \
    --eval-episodes "$EVAL_EPISODES" \
    --risk-level "$RISK_LEVEL" \
    $EVAL_LEVEL_ARGS \
    $WEIGHT_ARGS \
    --algo recurrent_ppo \
    --observation-version v2 \
    --output-dir "$RECURRENT_DIR" \
    --artifact-label "$RECURRENT_LABEL" \
    $MODE_ARGS \
    $COMMON_ARGS

phase_timer "Phase C (RecurrentPPO)" "$PHASE_START"

# ==========================================================
# Phase D: Publication analysis across all algorithms
# ==========================================================
if [[ "$MODE" != "--smoke" ]]; then
    echo "=== Phase D: Publication analysis ==="

    ANALYSIS_DIRS=()
    if [[ -n "$PPO_BASELINE_DIR" && -d "$PPO_BASELINE_DIR" ]]; then
        ANALYSIS_DIRS+=("$PPO_BASELINE_DIR")
    fi
    ANALYSIS_DIRS+=("$SAC_DIR" "$PPO_FS4_DIR" "$RECURRENT_DIR")

    echo "  Analyzing: ${ANALYSIS_DIRS[*]}"
    PHASE_START="$(date +%s)"

    python scripts/publication_run_analysis.py \
        --run-dirs "${ANALYSIS_DIRS[@]}" \
        --output-dir docs/artifacts/control_reward/publication_run_analysis

    phase_timer "Phase D (Analysis)" "$PHASE_START"
else
    echo ""
    echo "=== Skipping publication analysis in smoke mode ==="
fi

# ---------- Final summary ----------
NIGHT_END="$(date +%s)"
TOTAL_ELAPSED=$(( NIGHT_END - NIGHT_START ))
TOTAL_MINS=$(( TOTAL_ELAPSED / 60 ))
TOTAL_SECS=$(( TOTAL_ELAPSED % 60 ))
TOTAL_HRS=$(( TOTAL_MINS / 60 ))
REM_MINS=$(( TOTAL_MINS % 60 ))

echo ""
echo "================================================================"
echo "  Night run complete!"
echo "  Finished: $(date)"
echo "  Total elapsed: ${TOTAL_HRS}h ${REM_MINS}m ${TOTAL_SECS}s"
echo "================================================================"
echo ""
echo "Results:"
echo "  SAC           -> $SAC_DIR"
echo "  PPO+FS4       -> $PPO_FS4_DIR"
echo "  RecurrentPPO  -> $RECURRENT_DIR"
if [[ "$MODE" != "--smoke" ]]; then
    echo "  Analysis      -> docs/artifacts/control_reward/publication_run_analysis"
fi
echo ""
echo "Check outputs/benchmarks/ for detailed results."
