# Excel Reward and Initial Prepositioning Audit

Date: 2026-06-27

## Question

Why do dynamic policies often fail to beat static policies, even when the
action space can represent every Garrido Track-A static choice?

## Short Answer

The agent is learning, but the useful signal is mostly in initial
prepositioning and static-style buffer selection, not in week-by-week switching.
The current `Discrete(18)` comparison runner can fix an initial Track-A policy
before warmup, but it does not learn that initial decision.

## Initial Prepositioning Mechanism

`ThesisFactorizedTrackAEnv.reset()` accepts `initial_action` and applies it
before the DES warmup by passing:

- `initial_buffers`
- `initial_shifts`
- `inventory_replenishment_period`

to the underlying DES reset.

`Discrete18TrackAEnv` also supports an integer initial action through
`initial_discrete_action`/`initial_action`, and the comparison runner exposes
this as `--ppo-initial-static-policy`.

This means `static_S1_I1344` prepositions real stock before the weekly policy
starts acting. If the weekly policy later selects `I0`, the replenishment target
is removed, but the already prepositioned stock is not flushed.

Important limitation: `learn_initial_decision` exists in the DKANA thesis
decision wrapper, but `make_discrete18_track_a_env()` currently ignores it.
Therefore, current `compare_garrido_dynamic_vs_static.py` runs can use a fixed
initial preparation policy, but do not learn the preparation decision.

## 4k vs 20k Action Trace

Runs compared on common eval seeds `12001,12002,12003`:

- `excel_signal_phi2_train_h26_i1344init_2seed_2026-06-27`
- `excel_signal_phi2_train_h26_i1344init_20k_2seed_2026-06-27`

The apparent "longer training destroys the Excel signal" result does not hold
on common eval seeds:

- 4k mean Excel ReT on common seeds: about `0.000695`
- 20k mean Excel ReT on common seeds: about `0.000797`

What changes is the learned action distribution:

- 4k uses mostly `S1_I504`, `S1_I0`, and `S2_I336`.
- 20k bifurcates by seed: one policy stays near `S2_I336/S2_I168`, while the
  other alternates `S1_I0` with occasional `S3_I1344/S3_I336`.

So the issue is not simple training decay. It is seed instability plus
evaluation-seed variance.

## ReT Excel Delta Reward Test

Run:

`excel_delta_phi2_h26_i1344init_2seed_2026-06-27`

Configuration:

- regime: `severe`
- `risk_frequency_multiplier=2.0`
- `risk_impact_multiplier=1.0`
- horizon: `26` weekly steps
- reward: `ReT_excel_delta`
- initial policy: `static_S1_I1344`
- seeds: `9781,9782`

Result:

- dynamic Excel ReT: `0.000784`
- best static Excel ReT: `static_S1_I1344 = 0.000857`
- Excel delta vs best static: `-0.000073`
- dynamic C-D: `0.601713`
- best static C-D: `static_S1_I168 = 0.612539`
- C-D delta vs best static: `-0.010826`

The learned policies became constants:

- seed `9781`: always `S3_I1344`
- seed `9782`: always `S1_I168`

This reward does not yet produce useful dynamic switching. It behaves like a
static policy selector.

## Interpretation

There is a real preparation lever, but current successful signals rely on a
researcher-fixed initial prepositioning policy. The next high-value change is
not "more seeds" or "more PPO timesteps"; it is to make initial prepositioning
itself part of the learned policy in the same consistent comparison runner.

## Recommended Next Experiment

Add a two-phase `Discrete(18)` runner:

1. Initial decision phase: choose the pre-warmup buffer/shift action.
2. Weekly decision phase: choose Track-A actions during the episode.

Then rerun the fast 2-seed Excel front:

- primary eval: Excel ReT
- secondary: flow fill, lost orders, service-loss CVaR, C-D components
- rewards to compare:
  - `ReT_excel_delta`
  - `ReT_garrido2024_train`
  - `control_v1`

Use common eval seeds for candidate comparisons before scaling.

## Follow-Up: Forecast Visibility and Learned Initial Decision

After this audit, `compare_garrido_dynamic_vs_static.py` was extended to support
`--learn-initial-decision` on the `Discrete(18)` surface. The runner now lets PPO
choose the pre-warmup Track-A action as its first action, then skips that
zero-reward initial phase when aggregating weekly episode metrics.

The same patch fixed a provenance bug: the runner used `--observation-version`
for training/evaluation, but did not persist it in `summary.json`. This caused
post-hoc tracing tools to reconstruct v7 models with a v4 observation shape.

Fast v7-visible pilots show:

- `forecast_v7_control_v1_phi2_h26_i1344init_2seed_2026-06-27`:
  one seed became genuinely adaptive (`S3_I0/S3_I672` switching), while the
  other collapsed to constant `S1_I672`. It nearly matched Excel ReT but did
  not beat the best static.
- `forecast_v7_excel_delta_phi2_h26_i1344init_2seed_2026-06-27`:
  both seeds became constants (`S1_I1344` or `S3_I0`). This improved C-D
  relative to many statics but did not beat the best Excel static.
- `learned_init_v7_control_v1_phi2_h26_2seed_2026-06-27`:
  failed. PPO often chose no-buffer initial actions (`S1_I0`/`S2_I0`), causing
  one-step collapse.
- `learned_init_v7_excel_delta_phi2_h26_2seed_2026-06-27`:
  mixed. One seed learned a plausible preparation path (`S1_I672` initially,
  then mostly `S1_I1344`) and matched the best S1-buffer static on common eval
  seeds. The other seed collapsed to `S2_I0`.

Interpretation: forecast visibility alone is not enough, and learned initial
prepositioning is possible but unstable under vanilla PPO. The next narrow test
is not another environment sweep; it is a higher-budget run of
`ReT_excel_delta + v7 + learned-initial-decision` to see whether the first-action
credit assignment stabilizes.

That 20k follow-up did **not** stabilize it:

- `learned_init_v7_excel_delta_phi2_h26_20k_2seed_2026-06-27`
- dynamic Excel ReT: `0.000376`
- best static Excel ReT: `static_S2_I168 = 0.000787`
- dynamic C-D: `0.498`
- best static C-D: `static_S1_I1344 = 0.606`

Action traces explain the failure:

- seed `9841`: initial `S3_I0`, then `S3_I1344`, collapsing to Excel `0`.
- seed `9842`: initial and weekly `S2_I1344`, nonzero Excel but far below the
  best static and with very high unattended orders.

Conclusion: more vanilla PPO timesteps are not the fix. The agent can discover
a plausible initial-preposition path in some seeds, but the first action has a
long delayed-credit problem. The next high-probability route is a two-stage
training protocol: choose or pretrain the initial prepositioning decision on
the static surface, then train the weekly adaptive policy.
