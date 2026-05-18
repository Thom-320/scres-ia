# Track B To Thesis-Faithful Transfer Audit

## Question

What can be reused from Track B to help the thesis-faithful Garrido decision lane beat static thesis policies, without changing the action variables away from Garrido's decision surface?

## What Made Track B Work

Track B worked because it gave the agent control over the active bottleneck:

- Track B adds Op10 and Op12 downstream dispatch actions.
- The downstream-only ablation reaches fill ~= 1.000 with shifts frozen at S2.
- The shift-only ablation loses because downstream dispatch is frozen.
- Joint control improves cost efficiency by letting PPO use fewer assembly hours while managing downstream dispatch.

The key lesson is not "bigger neural net" or "better reward." It is action-space alignment with the binding operational constraint.

## What Cannot Transfer

The following should not be moved into the thesis-faithful lane:

- Op10/Op12 dispatch action variables.
- Track B `track_b_v1` 7D action contract as a replacement for Garrido's decision variables.
- Any claim that Track B is thesis-exact.

Those changes are useful for an extended-action-space paper lane, but they violate the current goal: keep the action variables tied to Garrido's thesis decisions.

## What Can Transfer

The following are valid transfers because they change information, training, or evaluation without changing the decision variables:

- `observation_version=v7` downstream visibility, tested as a 100k x 5 thesis-decision ablation.
- `env_sdm_history_reward`, which adds compact Table 6.25-style history.
- Cf curriculum training across Garrido scenarios.
- Paired Garrido Cf evaluation.
- Reward audit discipline: avoid `ReT_thesis` as a training reward; keep it as an audit/reporting metric.
- Policy telemetry from Track B: shift mix, assembly hours, downstream/disruption context, and family-specific effects.

## What The Transfer Tests Showed

Current 100k x 5 paired results:

| Setup | Paired wins | Mean fill-rate delta | Mean ReT delta |
|---|---:|---:|---:|
| v5, `control_v1`, Cf31-Cf90 curriculum | 90 / 300 (30.0%) | +0.01154 | +0.07129 |
| v7, `control_v1`, Cf31-Cf90 curriculum | 87 / 300 (29.0%) | +0.01160 | +0.07360 |
| v5, `ReT_seq_v1`, Cf31-Cf90 curriculum | 86 / 300 (28.7%) | +0.01049 | +0.07169 |
| v5, `control_v1`, large MLP | 86 / 300 (28.7%) | +0.01098 | +0.07419 |

Interpretation:

- v7 improves mean delta slightly but loses paired wins.
- `ReT_seq_v1` does not beat `control_v1` as a training objective in this thesis-decision lane.
- Larger MLP does not beat medium MLP.
- The current best 100k setting remains v5 + `control_v1` + PPO medium + broad Cf curriculum.

## Family-Specific Result

The most important finding is that learning gains are family-specific:

| Family | Paired wins | Mean fill-rate delta | Mean ReT delta |
|---|---:|---:|---:|
| Inventory Cf31-Cf60 | 25 / 150 (16.7%) | +0.00179 | -0.00115 |
| Capacity Cf61-Cf90 | 65 / 150 (43.3%) | +0.02130 | +0.14374 |

The agent is not meaningfully beating Garrido by inventory-period control. It is learning adaptive capacity behavior.

## Recommendation

Keep as primary thesis-faithful lane:

- action contract: `thesis_faithful_dkana_v1`
- observation: `v5 + env_sdm_history_reward`
- reward: `control_v1`
- model: PPO+MLP medium
- normalization: VecNormalize
- training curriculum: Cf31-Cf90
- evaluation: paired Garrido Cf31-Cf90, reported separately for inventory and capacity families

Do not switch the main lane to Track B, v7, large MLP, RecurrentPPO, SAC, or `ReT_seq_v1` training unless a later production run beats this evidence.

The production next step is the current 500k x 5 run of the best 100k lane.

Update after 250k/500k:

| Training length | Paired wins | Mean fill-rate delta | Capacity wins |
|---|---:|---:|---:|
| 100k | 90 / 300 (30.0%) | +0.01154 | 65 / 150 (43.3%) |
| 250k | 85 / 300 (28.3%) | +0.01131 | 64 / 150 (42.7%) |
| 500k | 83 / 300 (27.7%) | +0.01038 | 58 / 150 (38.7%) |

The 100k gate remains the current best thesis-faithful configuration.
