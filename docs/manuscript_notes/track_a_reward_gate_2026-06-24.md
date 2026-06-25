# Track A Reward Gate Result - 2026-06-24

## Scope

This note records the full no-training reward-surface gate required before any
Track A PPO/DQN retained-vs-reset smoke. The gate used common seeds and the
same static `6 x 3` thesis-factorized action grid under `current`, `increased`,
and `severe` risk levels.

## Runs

- Figure 6.2 primary lane:
  `outputs/benchmarks/thesis_reward_surface/track_a_full_figure_6_2_20260624`
- Table 6.20 robustness lane:
  `outputs/benchmarks/thesis_reward_surface/track_a_full_table_6_20_20260624`
- Downstream-Q comparison:
  `outputs/benchmarks/downstream_q_sensitivity/track_a_full_downstream_q_20260624`

Common run settings:

```text
seed = 550000
replications = 3
max_steps = 260
risk_levels = current, increased, severe
stochastic_pt = false
```

## Decision

No reward mode passed the downstream-Q stability gate. Therefore no PPO/DQN
retained-vs-reset smoke was launched under the frozen Track A stop rule.

The Figure 6.2 lane alone shortlisted:

```text
ReT_ladder_v1
ReT_cd_v1
ReT_cd_sigmoid
rt_v0
ReT_cd
ReT_garrido2024_train
```

The Table 6.20 lane changed either the selected best static policy, the
shortlist status, or the rank stability for every candidate:

- `ReT_ladder_v1`: best policy changed.
- `ReT_cd_v1`: best policy changed.
- `ReT_cd_sigmoid`: best policy changed.
- `rt_v0`: best policy changed.
- `ReT_cd`: best policy matched, but rank was unstable.
- `ReT_garrido2024_train`: not shortlisted in both lanes.

## Interpretation

This is a useful negative gate, not a failed training run. It says the reward
surface is sensitive to the downstream-Q interpretation, so a retained/reset
learning smoke would be premature if framed as paper evidence.

The least unstable candidate for a future exploratory-only smoke is `ReT_cd`,
because it selected `L1a_uniform_I336_S3` under both downstream-Q sources.
However, its rank moved from 7 under Figure 6.2 to 1 under Table 6.20, so it
does not satisfy the preregistered stability rule.

## Next Methodological Choice

To proceed without fishing, choose one of these explicitly:

1. Keep the strict downstream-Q stability gate. Result: no reward is eligible
   yet; improve or pre-register a new reward using training/calibration tapes.
2. Declare Figure 6.2 as the sole paper-facing training source and move Table
   6.20 to robustness-only reporting. Result: Figure-shortlisted rewards can
   enter exploratory smokes, but the manuscript must state that robustness to
   Table 6.20 failed.
3. Develop `control_v2` or another reward only after documenting why all current
   rewards failed the frozen gate.

