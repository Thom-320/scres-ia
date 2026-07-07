# Track B fixed-RNG relaunch status — 2026-07-03

## What changed

The Track B environment now constructs `MFSCSimulation` with `strict_exogenous_crn=True`.
In the current working tree, this activates `seed_stream_mode="split"` and separates:

- `rng`
- `demand_rng`
- `risk_rng`
- `regime_rng`

The adaptive regime controller and forecast noise already use `regime_rng`.

## Verification

Local and VPS checks both passed:

- `env.unwrapped.sim.seed_stream_mode == "split"`
- `env.unwrapped.sim.strict_exogenous_crn == True`
- with the same eval seed but very different static actions (`s1_d1.00` vs `s3_d2.00`),
  the exogenous event calendars for `R11`, `R13`, `R24`, `R22`, `R23`, `R12`, `R21`, and `R3`
  are identical.
- `R14` counts still differ because R14 defects depend on actual production; this is expected
  and should not be treated as an exogenous calendar mismatch.

## Local smoke

Command:

```text
.venv/bin/python scripts/run_track_b_smoke.py \
  --output-dir outputs/experiments/track_b_fixed_rng_smoke_2026-07-03 \
  --seeds 1 --train-timesteps 512 --eval-episodes 2 \
  --reward-mode control_v1 --risk-level adaptive_benchmark_v2 \
  --observation-version v7 --max-steps 104 \
  --n-steps 128 --batch-size 64 --n-envs 1
```

Result: completed successfully. PPO smoke beat the best static on the raw ReT metric in this tiny
wiring test. This is not evidence for the paper; it only verifies that training/evaluation still run.

## VPS confirmatory run

Launched in tmux:

```text
track_b_fixed_rng_5seed
```

Output directory on VPS:

```text
outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/
```

Command:

```text
.venv/bin/python scripts/run_track_b_smoke.py \
  --output-dir outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03 \
  --seeds 1 2 3 4 5 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --max-steps 104 \
  --n-steps 1024 \
  --batch-size 64 \
  --n-envs 1
```

Watcher:

```text
watch-track-b-fixed-rng-confirm
```

## Claim boundary

This is a new evidence lane. It should not silently replace the existing Track B package.
If the fixed-RNG run is positive, the next decision is whether to rerun the full Track B evidence
stack under fixed RNG for consistency.
