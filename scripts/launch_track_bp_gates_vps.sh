#!/usr/bin/env bash
# Track B-P Gate 0 (forced-prep ceiling, 6 tiers) + Gate 1 (clock-policy oracle, 3 configs).
# Pre-registered in docs/TRACK_BP_PREREGISTRATION_2026-07-08.md. Run on ovh-agent-lab.
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PY:-python3}
STAMP=2026-07-08
OUT_ROOT=outputs/experiments
mkdir -p "$OUT_ROOT"

export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

g0() { # name, extra args...
  local name=$1; shift
  local out="$OUT_ROOT/track_bp_g0_${name}_${STAMP}"
  echo "[launch] G0 $name -> $out"
  $PY scripts/audit_prevention_headroom_sweep.py \
    --output-dir "$out" \
    --env-factory track_bp \
    --reference-policy constant \
    --obs-config v7_no_forecast \
    --seeds 1 \
    --eval-episodes 16 \
    --max-steps 104 \
    "$@" \
    > "$out.log" 2>&1
  touch "$out.done"
}

g1() { # name, extra args...
  local name=$1; shift
  local out="$OUT_ROOT/track_bp_g1_${name}_${STAMP}"
  echo "[launch] G1 $name -> $out"
  $PY scripts/run_track_bp_gate1_oracle.py \
    --output-dir "$out" \
    --episodes 24 \
    --max-steps 104 \
    "$@" \
    > "$out.log" 2>&1
  touch "$out.done"
}

case "${1:-}" in
  g0_r21_l168)  g0 r21_l168 --enabled-risks R21 --target-risk R21 --isolation-risks R21 \
                  --risk-frequency-by-id '' --risk-impact-by-id R21=3 --replenishment-lead-time 168 ;;
  g0_r21_l336)  g0 r21_l336 --enabled-risks R21 --target-risk R21 --isolation-risks R21 \
                  --risk-frequency-by-id '' --risk-impact-by-id R21=3 --replenishment-lead-time 336 ;;
  g0_r23_l168)  g0 r23_l168 --enabled-risks R23 --target-risk R23 --isolation-risks R23 \
                  --risk-frequency-by-id '' --risk-impact-by-id R23=3 --replenishment-lead-time 168 ;;
  g0_r24_l168)  g0 r24_l168 --enabled-risks R24 --target-risk R24 --isolation-risks R24 \
                  --risk-frequency-by-id '' --risk-impact-by-id '' --replenishment-lead-time 168 ;;
  g0_r11_l168)  g0 r11_l168 --enabled-risks R11 --target-risk R11 --isolation-risks R11 \
                  --risk-frequency-by-id R11=0.125 --risk-impact-by-id R11=8 --replenishment-lead-time 168 ;;
  g0_r22_l168)  g0 r22_l168 --enabled-risks R22 --target-risk R22 --isolation-risks R22 \
                  --risk-frequency-by-id '' --risk-impact-by-id R22=3 --replenishment-lead-time 168 ;;
  g1_r21_starv) g1 r21_starv --enabled-risks R21 --risk-frequency-by-id R21=8 \
                  --risk-impact-by-id R21=4 --replenishment-lead-time 168 ;;
  g1_r24)       g1 r24 --enabled-risks R24 --risk-frequency-by-id R24=3 \
                  --replenishment-lead-time 168 ;;
  g1_r13_cal)   g1 r13_cal --enabled-risks R13 --risk-impact-by-id R13=8 \
                  --calendar-cycle-weeks 4 --replenishment-lead-time 168 ;;
  all_flag)     touch /tmp/track_bp_gates_all_launched.flag ;;
  *) echo "usage: $0 <tier>"; exit 1 ;;
esac
