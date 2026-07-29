#!/usr/bin/env bash
# Launch the clean-joint replication eval on ovh-agent-lab.
# Run from inside a tmux session on the VPS:
#   tmux new -s clean_replication
#   bash ~/scres-ia/scripts/launch_track_b_clean_replication_vps.sh
set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

PY=${PY:-.venv/bin/python}
OUT="outputs/experiments/track_b_clean_replication_2026-07-10"
STATIC="outputs/experiments/track_b_static_contract_search_2026-07-10/frozen_static_policy.json"
JOINT="outputs/experiments/track_b_factorial_joint_2026-07-09"

echo "[$(date -u +%FT%TZ)] clean-joint replication eval starting" | tee "$OUT.log"

$PY scripts/evaluate_track_b_clean_replication.py \
  --output-dir "$OUT" \
  --frozen-static "$STATIC" \
  --joint-root "$JOINT" \
  --workers 5 \
  --test-seed-base 500061 \
  --test-tapes 60 \
  2>&1 | tee -a "$OUT.log"

echo "[$(date -u +%FT%TZ)] clean-joint replication eval done" | tee -a "$OUT.log"
touch "$OUT.done"
