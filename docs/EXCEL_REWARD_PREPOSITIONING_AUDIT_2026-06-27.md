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
