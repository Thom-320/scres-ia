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

- `static_s1`
- `static_s2`
- `static_s3`
- `random`

The main reproducible benchmark script is `scripts/benchmark_control_reward.py`, which exports static, random, and learned-policy summaries under a shared configuration.
