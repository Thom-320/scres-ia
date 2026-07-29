#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/scres-ia-e2-20260702-min
source .venv/bin/activate

python -u scripts/run_track_b_observation_ablation.py \
  --output-dir outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02_ovh \
  --obs-configs v7_no_regime_forecast \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --seeds 1 2 3 4 5 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --max-steps 104 \
  --n-envs 4 \
  --learning-rate 0.0003 \
  --n-steps 1024 \
  --batch-size 256 \
  --n-epochs 10 \
  --export-order-ledger
