# Track A Stochastic PT Response Plan - 2026-06-15

## Purpose

Exhaust the strict thesis-factorized Track A contract before moving the main
positive-RL narrative to Track B.

The question is not whether PPO beats Garrido under the nominal thesis lane. The
current evidence says it does not. The remaining Track A question is narrower:

> Does adaptive control become valuable when process-time uncertainty increases
> while risk timing and inventory semantics remain fixed?

## Fixed Contract

Use the repaired thesis-facing environment:

- `action_space_mode=thesis_factorized`
- `inventory_period_mode=thesis_strict`
- `risk_occurrence_mode=thesis_periodic`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`
- fixed `risk_level` per run
- `stochastic_pt=True`
- `stochastic_pt_mean_preserving=True`

Only vary `stochastic_pt_spread`.

## Variability Axis

For this Track A question, use the mean-preserving PT mode. It scales a
symmetric triangular processing-time distribution around the thesis PT:

- low: `(1 - 0.25 * spread) * PT`, clipped at zero
- mode: `PT`
- high: `(1 + 0.25 * spread) * PT`

With `--stochastic-pt-mean-preserving`, changing spread changes exogenous
process-time variance without changing expected PT. It is not an additional
agent action.

The historical repo PT mode remains available as a reference:
`Tri(0.75 * PT, PT, 1.5 * PT)`. Do not use that as the main variance-isolation
axis because its expected PT increases with spread.

Recommended probe levels:

| label | `stochastic_pt_spread` | interpretation |
|---|---:|---|
| deterministic | 0.0 | no PT variance |
| historical | 1.0 | historical repo stochastic PT |
| high | 2.0 | stronger but still structured PT variance |
| extreme probe | 3.0 | stress probe, report only if full axis is shown |

## Comparison

At each spread:

- train PPO MLP under `[6,3]`;
- evaluate the trained PPO on the same panel;
- evaluate static grid policies under the same panel;
- compare PPO against the best static grid by fill and ReT.

The useful result is a dose-response curve:

- If PPO advantage grows monotonically with `stochastic_pt_spread`, adaptive
  control has a defensible Track A stress-extension story.
- If PPO still loses or only wins at one isolated point, the Track A conclusion is
  parsimonious robustness, and positive-RL claims should move to Track B or
  another declared extension.

## First Local Probe

Use a cheap local smoke before any Kaggle budget:

```bash
python scripts/run_thesis_decision_ppo_smoke.py \
  --label track_a_pt_spread_<spread>_smoke \
  --train-timesteps 30000 \
  --eval-episodes 3 \
  --seed 4242 \
  --reward-mode ReT_thesis \
  --risk-level severe_extended \
  --risk-occurrence-mode thesis_periodic \
  --raw-material-flow-mode kit_equivalent_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --action-space-mode thesis_factorized \
  --inventory-period-mode thesis_strict \
  --stochastic-pt \
  --stochastic-pt-spread <spread> \
  --stochastic-pt-mean-preserving \
  --max-steps 80 \
  --include-static-grid \
  --no-eval-ai-on-garrido-cfis
```

Run the smoke for `spread in {0.0, 1.0, 2.0}` first. Escalate to Kaggle only if
the axis shows a coherent signal.

## First Local Probe Result

Completed locally with `30k` PPO timesteps, `3` eval episodes, `80` weekly
steps, `risk_level=severe_extended`, and the mean-preserving PT mode above:

| spread | PPO fill | best static | static fill | delta fill | PPO ReT | static ReT | delta ReT |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0.0 | 0.5717 | `static_grid_I672_S3` | 0.6097 | -0.0380 | 0.2675 | 0.2922 | -0.0247 |
| 1.0 | 0.6025 | `static_grid_I504_S2` | 0.6071 | -0.0046 | 0.1582 | 0.1582 | +0.0000 |
| 2.0 | 0.5429 | `static_grid_I1344_S1` | 0.6051 | -0.0622 | 0.1444 | 0.1623 | -0.0179 |

Artifact:
`outputs/benchmarks/thesis_decision_ppo_smoke/track_a_pt_mean_preserving_30k_probe_summary.csv`

Interpretation: this cheap probe does not support a monotonic Track A
positive-RL claim. `spread=1.0` nearly ties the best static grid, but the
advantage does not grow with PT variance and `spread=2.0` loses clearly. Treat
this as screening evidence only; it argues against launching a large Kaggle
dose-response run unless a stronger model/training protocol is first justified.
