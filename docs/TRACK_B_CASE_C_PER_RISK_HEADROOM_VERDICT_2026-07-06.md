# Track B Case C per-risk headroom verdict - 2026-07-06

## Status

Completed. This is the per-risk Case C screen promised in
`docs/TRACK_B_CASE_C_PER_RISK_HEADROOM_PREREG_2026-07-06.md`.

Artifacts:

- `outputs/experiments/track_b_case_c_per_risk_headroom_grid_2026-07-06/per_risk_headroom_results.csv`
- `outputs/experiments/track_b_case_c_per_risk_headroom_grid_2026-07-06/per_risk_cvar_panel.csv`

Protocol verified from summaries/logs:

- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Enabled risks: `R22,R23,R24`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3`
- Train timesteps: `30000`
- Eval episodes: `8`
- Horizon: `104`
- PPO batch size: `64`

Important scope note: this is downstream-only, so absolute ReT values are not
comparable to all-risk Track B. Compare cells by within-scenario static margin,
branch mix, cost, and CVaR.

## Result

| Rank | Cell | Per-risk knobs | ReT Excel | Delta vs best static | Relative delta | Cost | Fill-rate branch | Recovery branch |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | `r24_freq3_r22r23_impact1p5` | `R24 freq=3`, `R22/R23 impact=1.5` | 0.483718 | +0.043485 | +9.88% | 0.727 | 55.85% | 29.90% |
| 2 | `r24_freq3_impact1p0` | `R24 freq=3` | 0.490573 | +0.042814 | +9.56% | 0.817 | 56.75% | 29.18% |
| 3 | `r24_freq2_impact1p25` | `R24 freq=2`, `R24 impact=1.25` | 0.588783 | +0.034197 | +6.17% | 0.623 | 67.88% | 27.40% |
| 4 | `r24_freq2_r22r23_impact1p5` | `R24 freq=2`, `R22/R23 impact=1.5` | 0.545566 | +0.021355 | +4.07% | 0.593 | 67.52% | 21.83% |
| 5 | `r22r23_impact1p5` | `R22/R23 impact=1.5` | 0.599839 | +0.014393 | +2.46% | 0.581 | 79.31% | 14.51% |
| 6 | `r22_impact1p5` | `R22 impact=1.5` | 0.604377 | +0.014432 | +2.45% | 0.495 | 79.95% | 13.76% |
| 7 | `r23_impact1p5` | `R23 impact=1.5` | 0.601573 | +0.014066 | +2.39% | 0.550 | 79.55% | 14.25% |
| 8 | `base` | none | 0.606336 | +0.014124 | +2.38% | 0.628 | 80.12% | 13.69% |

## CVaR

`order_ret_excel_cvar05_mean` is zero in every per-risk cell and in the matched
best static baselines. That means this screen does not create a useful
worst-5%-order tail separation under the current downstream-only branch mix.

So CVaR does **not** choose the Case C candidate. The meaningful signal here is
headroom over the best static baseline, plus the branch shift/exposure mechanism.

## Interpretation

The clean pattern is that **R24 frequency creates headroom**. Increasing only
R22/R23 severity barely moves the relative PPO-vs-static margin above the
downstream-only base: roughly +2.4% to +2.5%. By contrast, making R24 frequent
raises the margin to +6.2% at 2x and about +9.6% to +9.9% at 3x.

This fits the mechanism seen earlier: PPO is not showing causal prevention in
the strict pre-event counterfactual sense. It is buying adaptive resilience by
dispatching quickly enough to reduce risk exposure and improve recovery-branch
outcomes. R24 is the useful stressor because it is frequent enough for the agent
to learn around and discrete enough to create visible headroom.

The strongest screen cell is:

`r24_freq3_r22r23_impact1p5`

It has the largest relative margin (+9.88%) and lower cost than the simpler
`r24_freq3_impact1p0` cell (0.727 vs 0.817), but it also has lower absolute ReT
because the environment is harder.

The cleanest minimalist cell is:

`r24_freq3_impact1p0`

It isolates the headroom source to R24 frequency alone and gives almost the same
relative margin (+9.56%), but at a higher cost.

## Parallel 4x/5x Frequency Extension

The VPS global-frequency extension also completed for both all-risk and
downstream-only cases. It is not per-risk, but it is directionally consistent:

| Scenario | Frequency | ReT Excel | Delta vs best static |
|---|---:|---:|---:|
| all risks | 4x | 0.003754 | +0.000699 |
| all risks | 5x | 0.003310 | +0.000720 |
| downstream only | 4x | 0.381602 | +0.051080 |
| downstream only | 5x | 0.327894 | +0.067413 |

This reinforces the broad conclusion: more risk pressure increases adaptive
headroom, while absolute ReT falls because the scenario is harsher.

## Recommendation

Promote `r24_freq3_r22r23_impact1p5` as the Case C stress/adaptation candidate
for the heavier final confirmation, unless the paper needs the simplest possible
environmental story. If simplicity is preferred, use `r24_freq3_impact1p0`.

Do not present this as prevention yet. The current claim should be:

> PPO gains adaptive resilience under controlled downstream stress, mainly by
> reducing exposure and improving recovery, not by proven anticipatory prevention.

Next gate:

1. Wait for the horizon screen to finish.
2. Run the final A/B/C confirmation at the selected horizon.
3. If Case C remains strong at 5 seeds x 60k, run the fixed-RNG event-aligned
   prevention audit only as a mechanism check, not as a primary result.
