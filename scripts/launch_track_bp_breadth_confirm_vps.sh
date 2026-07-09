#!/usr/bin/env bash
# Track B-P follow-up battery (2026-07-09):
#  breadth  — R21 freq x impact grid, Gate-1 oracle (always/never/calendar), maps how far
#             below freq x8 / impact x4 the static preventive channel survives.
#  confirm  — 5-seed x 60k Gate-2 confirm in Cell A, both contracts (11D + 8D ablation).
#  reattr   — Gate-0 R11 rerun with buffers-only forced posture (re-attribution).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
STAMP=2026-07-09
OUT=outputs/experiments
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

case "${1:-}" in
  breadth)
    for f in 1 2 4 8; do
      for i in 1 2 3 4; do
        out="$OUT/track_bp_breadth_r21_f${f}_i${i}_${STAMP}"
        $PY scripts/run_track_bp_gate1_oracle.py \
          --output-dir "$out" \
          --episodes 24 --max-steps 104 \
          --enabled-risks R21 \
          --risk-frequency-by-id "R21=${f}" \
          --risk-impact-by-id "R21=${i}" \
          --replenishment-lead-time 168 \
          > "$out.log" 2>&1
        echo "breadth f${f} i${i} done"
      done
    done
    touch "$OUT/track_bp_breadth_${STAMP}.done"
    ;;
  confirm11)
    out="$OUT/track_bp_confirm_cellA_11d_5seed_60k_${STAMP}"
    $PY scripts/run_track_bp_gate2_screen.py \
      --output-dir "$out" \
      --seeds 1 2 3 4 5 --train-timesteps 60000 --eval-episodes 24 \
      --enabled-risks R21 --risk-frequency-by-id R21=8 --risk-impact-by-id R21=4 \
      --contract track_bp \
      > "$out.log" 2>&1
    touch "$out.done"
    ;;
  confirm8)
    out="$OUT/track_bp_confirm_cellA_8d_5seed_60k_${STAMP}"
    $PY scripts/run_track_bp_gate2_screen.py \
      --output-dir "$out" \
      --seeds 1 2 3 4 5 --train-timesteps 60000 --eval-episodes 24 \
      --enabled-risks R21 --risk-frequency-by-id R21=8 --risk-impact-by-id R21=4 \
      --contract track_b \
      > "$out.log" 2>&1
    touch "$out.done"
    ;;
  reattr)
    out="$OUT/track_bp_g0_r11_buffers_only_${STAMP}"
    $PY scripts/audit_prevention_headroom_sweep.py \
      --output-dir "$out" \
      --env-factory track_bp --reference-policy constant \
      --posture-dims buffers_only \
      --obs-config v7_no_forecast \
      --seeds 1 --eval-episodes 16 --max-steps 104 \
      --enabled-risks R11 --target-risk R11 --isolation-risks R11 \
      --risk-frequency-by-id R11=0.125 --risk-impact-by-id R11=8 \
      --replenishment-lead-time 168 \
      > "$out.log" 2>&1
    touch "$out.done"
    ;;
  *) echo "usage: $0 breadth|confirm11|confirm8|reattr"; exit 1 ;;
esac
