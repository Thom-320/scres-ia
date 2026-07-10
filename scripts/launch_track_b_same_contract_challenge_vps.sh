#!/usr/bin/env bash
# Same-contract static challenge (frozen protocol:
# docs/TRACK_B_SAME_CONTRACT_CHALLENGE_PROTOCOL_2026-07-10.md).
#   search   — calibration-only Sobol + refinement static full-contract search
#              (tapes 300001-300024 only).
#   anchored — train upstream_shift_best_dispatch arm (Op10=2.0x/Op12=1.5x
#              frozen), 5 seeds x 60k, corrected factorial settings.
#   evaluate — held-out verdict on virgin tapes 400001-400060 (runs only after
#              search + anchored are both done).
#   all      — search and anchored in parallel, then evaluate.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/bin/python}
STAMP=2026-07-10
OUT=outputs/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

SEARCH_DIR="$OUT/track_b_static_contract_search_${STAMP}"
ANCHOR_DIR="$OUT/track_b_factorial_upstream_shift_best_dispatch_${STAMP}"
EVAL_DIR="$OUT/track_b_same_contract_challenge_${STAMP}"

case "${1:-all}" in
  search)
    $PY scripts/run_track_b_static_contract_search.py \
      --output-dir "$SEARCH_DIR" --workers "${2:-3}" \
      > "$SEARCH_DIR.log" 2>&1
    touch "$SEARCH_DIR.done"
    ;;
  anchored)
    $PY scripts/run_track_b_contract_factorial.py \
      --output-dir "$ANCHOR_DIR" \
      --arms upstream_shift_best_dispatch --skip-static \
      > "$ANCHOR_DIR.log" 2>&1
    touch "$ANCHOR_DIR.done"
    ;;
  evaluate)
    $PY scripts/evaluate_track_b_same_contract_challenge.py \
      --output-dir "$EVAL_DIR" \
      --frozen-static "$SEARCH_DIR/frozen_static_policy.json" \
      --canonical-root-a "$OUT/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104" \
      --canonical-root-b "$OUT/track_b_seed_expansion_2026-07-02/track_b_seed_expansion_6_10_claude" \
      --factorial-joint-root "$OUT/track_b_factorial_joint_2026-07-09" \
      --anchored-root "$ANCHOR_DIR" \
      --workers "${2:-5}" \
      > "$EVAL_DIR.log" 2>&1
    touch "$EVAL_DIR.done"
    ;;
  all)
    mkdir -p "$SEARCH_DIR" "$ANCHOR_DIR"
    "$0" search 3 &
    SEARCH_PID=$!
    "$0" anchored &
    ANCHOR_PID=$!
    wait "$SEARCH_PID" "$ANCHOR_PID"
    "$0" evaluate 5
    touch "$OUT/track_b_same_contract_challenge_${STAMP}.all_done"
    ;;
  *)
    echo "usage: $0 {search|anchored|evaluate|all}" >&2
    exit 1
    ;;
esac
