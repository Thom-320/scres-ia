#!/usr/bin/env bash
set -euo pipefail

cd /home/ubuntu/scres-ia-e2-20260702-min

run_dir="outputs/experiments/track_b_e2_obs_masked_confirm_2026-07-02_ovh"
mkdir -p "$run_dir"

nohup scripts/run_track_b_e2_obs_masked_ovh.sh > "$run_dir/run_ovh.log" 2>&1 &
pid=$!
echo "$pid" > "$run_dir/run_ovh.pid"
echo "$pid"
ps -p "$pid" -o pid,etime,%cpu,%mem,command
