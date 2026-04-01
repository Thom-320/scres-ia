#!/usr/bin/env bash
# run_ret_cd_v1_benchmark.sh
# Overnight PPO training with ReT_cd_v1 reward.
# 500k timesteps × 5 seeds × 2 risk levels (increased + severe)
# n_envs=4, stochastic_pt, observation_version=v1
#
# Usage:
#   bash scripts/run_ret_cd_v1_benchmark.sh           # full run
#   bash scripts/run_ret_cd_v1_benchmark.sh --smoke   # quick test

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON="${REPO_ROOT}/.venv/bin/python"
TRAIN="${REPO_ROOT}/train_agent.py"

# ---------- Defaults ----------
SEEDS="11 22 33 44 55"
TIMESTEPS=500000
N_ENVS=4
STEP_SIZE=168
MAX_STEPS=260

# ---------- Smoke mode ----------
MODE="${1:-}"
if [[ "$MODE" == "--smoke" ]]; then
    echo "=== SMOKE TEST MODE ==="
    SEEDS="11"
    TIMESTEPS=2048
    MAX_STEPS=16
    N_ENVS=2
fi

# ---------- Directories ----------
INCREASED_DIR="${REPO_ROOT}/outputs/ret_cd_v1_500k/increased"
SEVERE_DIR="${REPO_ROOT}/outputs/ret_cd_v1_500k/severe"
mkdir -p "$INCREASED_DIR" "$SEVERE_DIR"

echo "============================================================"
echo " ReT_cd_v1 Overnight Benchmark"
echo " seeds: ${SEEDS}"
echo " timesteps: ${TIMESTEPS}"
echo " n_envs: ${N_ENVS}"
echo "============================================================"

# ---------- increased + stochastic_pt ----------
echo ""
echo "[1/2] risk=increased + stochastic_pt"
for SEED in $SEEDS; do
    echo "  Seed ${SEED}..."
    "$PYTHON" "$TRAIN" \
        --env-variant shift_control \
        --reward-mode ReT_cd_v1 \
        --risk-level increased \
        --stochastic-pt \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --seed "$SEED" \
        --step-size-hours "$STEP_SIZE" \
        --max-steps-per-episode "$MAX_STEPS" \
        --observation-version v1 \
        --output-dir "${INCREASED_DIR}/seed_${SEED}" \
        2>&1 | tee "${INCREASED_DIR}/seed_${SEED}_train.log"
    echo "  ✓ seed ${SEED} done"
done
echo "[1/2] increased DONE"

# ---------- severe + stochastic_pt ----------
echo ""
echo "[2/2] risk=severe + stochastic_pt"
for SEED in $SEEDS; do
    echo "  Seed ${SEED}..."
    "$PYTHON" "$TRAIN" \
        --env-variant shift_control \
        --reward-mode ReT_cd_v1 \
        --risk-level severe \
        --stochastic-pt \
        --timesteps "$TIMESTEPS" \
        --n-envs "$N_ENVS" \
        --seed "$SEED" \
        --step-size-hours "$STEP_SIZE" \
        --max-steps-per-episode "$MAX_STEPS" \
        --observation-version v1 \
        --output-dir "${SEVERE_DIR}/seed_${SEED}" \
        2>&1 | tee "${SEVERE_DIR}/seed_${SEED}_train.log"
    echo "  ✓ seed ${SEED} done"
done
echo "[2/2] severe DONE"

echo ""
echo "============================================================"
echo " All runs complete."
echo " Artifacts: outputs/ret_cd_v1_500k/"
echo "============================================================"
