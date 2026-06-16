# Track A Continuous I_t,S Plan - 2026-06-15

## Thesis Basis

Garrido-Rios' decision variables are the buffering strategy `(I_t,S, S)`:

- `I_t,S`: on-hand inventory buffers at Op3, Op5, and Op9.
- `S`: short-term manufacturing capacity, represented by work shifts.

The thesis discretizes `I_t,S` into five Table 6.16 levels:
`I168,1`, `I336,1`, `I504,1`, `I672,1`, and `I1344,1`. These are best
understood as coverage-hours levels, not as a naturally discrete operational
variable. The thesis also notes that military supply chains can switch buffering
strategies on/off quickly, and that the cost factor and demand stationarity are
important limitations/future-work areas.

## Why This Extension

Strict Track A (`thesis_factorized`) is faithful but coarse:

```text
MultiDiscrete([6, 3])
buffer level in {0, I168, I336, I504, I672, I1344}
S in {1, 2, 3}
```

The continuous extension keeps the same two conceptual decisions but removes the
DOE discretization:

```text
continuous_it_s: Box([0, -1], [1, 1])
action[0] = fraction of I1344,1 applied commonly at Op3, Op5, and Op9
action[1] = shift signal mapped to S1/S2/S3
```

This is not Track B. It does not expose ROP control, per-node independent
buffers, or downstream Op10/Op12 dispatch. It is a continuous relaxation of
Garrido's buffer-size variable.

## Fair Comparison Chain

The comparison must separate three effects:

1. Garrido static discrete baseline.
2. Best static continuous `I_t,S` policy.
3. PPO continuous `I_t,S` policy.

Only the third vs second contrast can support an adaptivity claim. If PPO beats
Garrido discrete but not the best static continuous policy, the gain is from
de-discretizing the DOE, not from adaptive learning.

## Other Thesis-Aligned Ideas

These are more defensible than Track B because they stay close to Garrido's own
limitations and future-work agenda:

- Cost-aware resilience: add a light holding/shift cost term, following the
  thesis limitation that cost was excluded for war-mode assumptions. This is
  useful if static policies saturate fill but overuse buffers.
- Demand non-stationarity: vary the regular/contingent demand process because
  Garrido explicitly identifies demand stationarity as a strong assumption that
  limits buffering effectiveness.
- Synergy of `I_t,S` and `S`: the thesis lists this as future work. Continuous
  buffer sizing plus shifts is the direct route to test that interaction.
- Lead-time factor: Garrido mentions supply-chain lead-time as an additional
  factor for buffer-related strategies. This is more invasive and should come
  after continuous `I_t,S`.

## First Gate

Run a cheap local smoke before any Kaggle job:

```bash
python scripts/run_thesis_decision_ppo_smoke.py \
  --label track_a_continuous_it_s_30k_probe \
  --train-timesteps 30000 \
  --eval-episodes 3 \
  --seed 4242 \
  --reward-mode ReT_thesis \
  --risk-level severe_extended \
  --risk-occurrence-mode thesis_periodic \
  --raw-material-flow-mode kit_equivalent_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --action-space-mode continuous_it_s \
  --inventory-period-mode thesis_strict \
  --max-steps 80 \
  --include-static-grid \
  --no-eval-ai-on-garrido-cfis
```

If this only beats the discrete static grid, add a static continuous optimizer
before making any claim. If it beats a best static continuous policy on the same
panel, then the adaptivity story becomes credible.

## First Local Probe Result

Completed locally with `30k` PPO timesteps, `3` eval episodes, `80` weekly
steps, `risk_level=severe_extended`, `risk_occurrence_mode=thesis_periodic`,
and `raw_material_flow_mode=kit_equivalent_order_up_to`:

| policy | fill | ReT | reward |
|---|---:|---:|---:|
| PPO `continuous_it_s` | 0.5162 | 0.2384 | 64.43 |
| best discrete static grid, `I672,S3` | 0.6097 | 0.2922 | 61.40 |
| delta PPO - static | -0.0935 | -0.0537 | +3.04 |

Artifact:
`outputs/benchmarks/thesis_decision_ppo_smoke/track_a_continuous_it_s_30k_probe/`

Interpretation: the first cheap screen does not yet support a positive Track A+
claim. The continuous action surface is valid and runnable, but PPO at 30k does
not beat the discrete static grid on fill/ReT. Before any larger run, either
improve the reward/training protocol or add a best-static-continuous search to
measure whether continuous buffer sizing itself has value.
