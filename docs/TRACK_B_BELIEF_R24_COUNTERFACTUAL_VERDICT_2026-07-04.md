# Track B belief sidecar R24 counterfactual verdict — 2026-07-04

## Question

Does the frozen R24 belief sidecar produce preventive value before frequent R24
risk events?

This is not a training result. It is a fixed-policy audit using the corrected
fixed-RNG Track B environment, so the discrete risk calendar is exogenous to the
agent actions. The metric is Garrido Excel ReT (`order_ret_excel`), measured as:

```text
R_full - R_reset(pre-R24)
```

where `R_reset(pre-R24)` replaces the learned actions in the four weekly
decision steps before sampled R24 onsets with that policy's own calm baseline,
then rolls the episode forward under the same fixed-RNG exogenous process.

## Artifacts

```text
outputs/experiments/track_b_belief_counterfactual_r24_balanced_2026-07-04/
outputs/experiments/track_b_belief_counterfactual_r24_unweighted_2026-07-04/
```

Both runs use:

- policies: `ppo_mlp_belief`, `real_kan_belief`
- observation: `v10` plus the two frozen R24 belief probabilities
- target risk: `R24`
- seeds: 1, 2, 3
- evaluation episodes: 8 per seed
- max steps: 104
- reset window: pre-risk weeks `[-4, -1]`
- primary metric: `order_ret_excel`

The balanced run uses the original logistic heads with
`class_weight="balanced"`. The unweighted run uses calibrated heads with
`class_weight=None`.

## Results

| Belief head | Policy | Mean R full | Mean R reset | Mean delta ReT Excel | Positive pairs | Positive pair rate |
|---|---|---:|---:|---:|---:|---:|
| balanced | PPO+MLP belief | 0.005743 | 0.005745 | -0.00000200 | 23/158 | 14.6% |
| balanced | Real-KAN belief | 0.005935 | 0.005935 | -0.00000059 | 4/158 | 2.5% |
| unweighted | PPO+MLP belief | 0.005651 | 0.005656 | -0.00000471 | 28/158 | 17.7% |
| unweighted | Real-KAN belief | 0.005937 | 0.005937 | -0.00000003 | 10/158 | 6.3% |

Coverage was the same in all rows: about 3.68% of episode decision steps were
reset per sampled anchor, with roughly 39 R24 events per episode on average.

## Interpretation

The result does **not** support a causal prevention claim for the frozen belief
sidecar.

For both calibration choices, replacing pre-R24 learned actions with the
policy's calm baseline does not reduce Garrido Excel ReT. In fact, the mean
delta is slightly negative or essentially zero, and the positive-pair rates are
far below the bar needed to call the pre-risk behavior beneficial.

This matters because the earlier scalar belief sidecar did show a small ReT
improvement for Real-KAN in ordinary evaluation. This audit says that the gain
is not explained by clearly valuable pre-R24 action timing. It may instead come
from general robustness, a different resource posture, or nonlinear use of the
belief features outside the narrow pre-event window.

## Verdict

Do not claim that the frozen R24 belief sidecar is preventive.

The correct statement is:

> The scalar R24 belief sidecar is performance-promising for Real-KAN at smoke
> scale, but the fixed-RNG `R_full - R_reset(pre-R24)` audit does not show that
> its gain comes from preventive action before R24 events.

This strengthens the case for testing Ruta A: a pretrained shared belief
encoder may shape the policy representation more deeply than two appended
probabilities. But Ruta A still needs its own training result and, if positive,
the same counterfactual audit on its checkpoints.

