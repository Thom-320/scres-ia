**Shift-Control Audit Note**

This note freezes the current scientific contract of the weekly shift-control lane.

## What reward is optimized

The main training reward is `control_v1`, implemented in `supply_chain/env_experimental_shifts.py`.

Formal definition:

`reward_total = -(w_bo * service_loss_step + w_cost * shift_cost_step + w_disr * disruption_fraction_step)`

with:

- `service_loss_step = new_backorder_qty / max(new_demanded, 1.0)`
- `shift_cost_step = shifts - 1`
- `disruption_fraction_step = min(1.0, step_disruption_hours / (step_size_hours * 13))`

`ReT_thesis` remains implemented, but it is not the preferred training objective for the current benchmark lane. `ret_thesis_corrected` is exposed as a reporting metric during `control_v1` runs.

## What the agent controls

The action is 5-dimensional.

The environment maps it to the following control knobs:

- `op3_q`
- `op9_q_min`
- `op9_q_max`
- `op3_rop`
- `op9_rop`
- `assembly_shifts` in `{S1, S2, S3}`

The first four dimensions are converted into multiplicative inventory-policy updates. The fifth dimension selects the assembly shift regime through three threshold bands.

## State-action-transition contract

The environment should be described as a partially observed control problem.

- `v1` observation: 15-dimensional operational snapshot.
- `v2` observation: `v1` plus previous-step demand, backorder, and disruption diagnostics.
- The underlying DES evolves with richer latent state than the observed vector exposes.

Recommended paper language:

> The weekly controller operates on an augmented observed state rather than a fully sufficient Markov state; the resulting formulation is best interpreted as a POMDP-style sequential control problem.

## Baselines

The benchmark compares the learned policy against:

- `static_s1`, `static_s2`, `static_s3` — fixed-shift policies (no adaptation)
- `random` — uniform random actions
- `heuristic_hysteresis` — deadband shift control on backorder_rate with hysteresis bands
- `heuristic_disruption` — reactive shift + inventory boost on disruption/low fill_rate
- `heuristic_tuned` — combined hysteresis + disruption-aware with grid-searched parameters

The main reproducible benchmark script is `scripts/benchmark_control_reward.py`, which exports static, heuristic, random, and learned-policy summaries under a shared configuration.

## Reward shaping (PBRS)

`control_v1_pbrs` adds a Potential-Based Reward Shaping bonus to the `control_v1` reward:

`F(s, s') = γ × Φ(s') - Φ(s)`

Two variants:

- **Cumulative (main):** `Φ(s) = -α × max(0, τ - FR_cumulative) / τ`
  Uses obs[6] (cumulative fill rate). Policy-invariant (Ng et al. 1999).
- **Step-level (ablation):** `Φ(s) = -α × prev_step_backorder_qty_norm`
  Uses obs[16] from v2 observation. Responds to recent service failures.
  Requires `observation_version="v2"`. Also policy-invariant.

Hyperparameters: `pbrs_alpha` (scale, default 1.0), `pbrs_tau` (target fill rate, default 0.95), `pbrs_gamma` (discount, must match SB3 gamma).

PBRS experiment runner: `scripts/run_pbrs_experiments.sh`.
