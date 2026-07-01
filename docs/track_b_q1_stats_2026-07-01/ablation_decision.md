# Track B 8D Ablation Decision

Date: 2026-07-01

## Decision

Spend compute on current-contract Track B ablations next, but do not launch H4 yet.

The canonical Track B result is strong enough to justify the reviewer-defense
ablation:

- Excel ReT delta vs best dense static: +0.000426
- CI95: [+0.000389, +0.000463]
- Paired Cohen's d: 2.87
- PPO is non-dominated against the dense static frontier on Excel ReT, cost,
  CTj p99 tail, and flow fill.

## Why Ablations Now

The biggest reviewer attack is not statistical strength anymore. It is causal:
"Track B just gives the agent more power." The correct defense is to show which
part of the 8D action contract carries the win:

- joint: full 8D Track B control
- downstream_only: freezes shift, keeps Op10/Op12 dispatch control
- shift_only: freezes Op10/Op12, keeps upstream/shift controls

If downstream_only keeps most of the win and shift_only loses, the action-space
alignment claim is strong. If only joint wins, the claim becomes "dispatch plus
capacity coordination" rather than downstream sufficiency.

## Why Not H4 Yet

H4 retained-vs-reset is a theoretical add-on. It should wait until H1 and the
mechanism/ablation package are locked. Running H4 before causal ablations would
answer a less urgent reviewer question.

## Suggested Compute Plan

Screen first:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_track_b_ablation.py \
  --output-dir outputs/experiments/track_b_ablation_8d_screen_2026-07-01 \
  --ablation-configs joint downstream_only shift_only \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --seeds 1 2 \
  --train-timesteps 30000 \
  --eval-episodes 8 \
  --max-steps 104 \
  --n-envs 4 \
  --learning-rate 0.0001 \
  --export-order-ledger
```

Promote only if the screen is clean:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_track_b_ablation.py \
  --output-dir outputs/experiments/track_b_ablation_8d_final_2026-07-01 \
  --ablation-configs joint downstream_only shift_only \
  --reward-mode control_v1 \
  --risk-level adaptive_benchmark_v2 \
  --observation-version v7 \
  --seeds 1 2 3 4 5 \
  --train-timesteps 60000 \
  --eval-episodes 12 \
  --max-steps 104 \
  --n-envs 4 \
  --learning-rate 0.0001 \
  --export-order-ledger
```

Use the same Q1 stats script after the final ablation if it emits comparable
episode metrics.
