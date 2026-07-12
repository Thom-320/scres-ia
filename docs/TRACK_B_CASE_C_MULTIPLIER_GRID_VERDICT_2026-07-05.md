# Track B Case C multiplier grid verdict - 2026-07-05

## Scope

Case C tuned the risk frequency and impact multipliers around the selected Garrido-native downstream scenario.

Primary metric: Garrido Excel resilience, `order_ret_excel_mean`.

Artifact:

`outputs/experiments/track_b_case_c_multiplier_grid_2026-07-05/`

Parent selected by the runner:

- Scenario: `garrido_downstream_cherry`
- Risks enabled: `R22,R23,R24`
- Observation: `v7_no_forecast`
- Batch size: 64
- Reward: `control_v1`
- Seeds: 1, 2, 3
- Train timesteps: 30,000
- Eval episodes: 8
- Horizon: 104

This is a tuning screen, not a paper headline. The downstream-only risk set changes the scale of ReT, so these numbers should not be compared directly to the all-risk Track B headline.

## Results

| Frequency | Impact | ReT Excel | Cost | Delta vs static |
|---:|---:|---:|---:|---:|
| 0.75 | 0.75 | 0.627457 | 0.525 | +0.012688 |
| 0.75 | 1.00 | 0.624898 | 0.502 | +0.013906 |
| 0.75 | 1.25 | **0.641942** | 0.524 | +0.015639 |
| 0.75 | 1.50 | 0.636148 | 0.528 | +0.016382 |
| 1.00 | 0.75 | 0.608393 | 0.514 | +0.015540 |
| 1.00 | 1.00 | 0.606302 | 0.567 | +0.017504 |
| 1.00 | 1.25 | 0.625339 | 0.552 | +0.019201 |
| 1.00 | 1.50 | 0.620454 | 0.642 | +0.021222 |
| 1.25 | 0.75 | 0.590645 | **0.467** | +0.015674 |
| 1.25 | 1.00 | 0.587581 | 0.578 | +0.018884 |
| 1.25 | 1.25 | 0.607117 | 0.659 | +0.022265 |
| 1.25 | 1.50 | 0.603368 | 0.607 | +0.026959 |
| 1.50 | 0.75 | 0.566312 | 0.564 | +0.017385 |
| 1.50 | 1.00 | 0.561643 | 0.659 | +0.020675 |
| 1.50 | 1.25 | 0.583343 | 0.627 | +0.024235 |
| 1.50 | 1.50 | 0.578895 | 0.617 | **+0.034169** |

## Main pattern

The strongest ReT cell is:

- `frequency=0.75`, `impact=1.25`
- ReT Excel: `0.641942`
- Cost: `0.524`
- Delta vs static: `+0.015639`

The largest adaptive margin vs static is:

- `frequency=1.50`, `impact=1.50`
- ReT Excel: `0.578895`
- Cost: `0.617`
- Delta vs static: `+0.034169`

These are different answers. Lower frequency preserves high absolute resilience; higher frequency/impact creates more room for the learned policy to beat the static comparator, but the absolute ReT is lower because the scenario is harder.

## Interpretation

Case C gives a useful tuning signal:

1. Frequency is the main stress knob. Increasing frequency from `0.75` to `1.50` steadily lowers absolute ReT.
2. Impact increases the learned policy's advantage over static. The delta vs static grows as impact rises, especially at high frequency.
3. The best balanced candidate is `frequency=0.75`, `impact=1.25`: it has the highest ReT, moderate cost, and a positive margin over static.
4. The best "stress-test advantage" candidate is `frequency=1.50`, `impact=1.50`: it maximizes the dynamic-vs-static delta, but it is not the best resilience environment.

## Recommendation

Use Case C as a diagnostic, not as the main Track B headline.

For a confirmatory follow-up, run one of these two:

- **Balanced resilience confirmation:** `garrido_downstream_cherry`, `frequency=0.75`, `impact=1.25`, batch size 64, no forecast, 5-10 seeds at 60k.
- **Stress/adaptation confirmation:** `garrido_downstream_cherry`, `frequency=1.50`, `impact=1.50`, batch size 64, no forecast, 5-10 seeds at 60k.

If the paper needs one reviewer-facing Case C variant, prefer the balanced resilience cell. It is easier to defend because it maximizes ReT Excel rather than only maximizing the gap against a weakened static baseline.

Do not replace the all-risk Garrido evaluation with this downstream-only scenario. Keep it as a controlled stress/tuning lane for Track B's downstream authority.
