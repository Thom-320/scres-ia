# Track B Real-KAN no-forecast batch-size 512 A/B verdict - 2026-07-06

## Status

Completed. This is the corrective same-protocol no-forecast check requested after
the Real-KAN batch-size sweep turned out to be `adaptive_benchmark_v2` / `v7`
full, not reviewer-safe `v7_no_forecast`.

Artifacts:

- `outputs/experiments/track_b_realkan_no_forecast_bs512_ab_h104_confirm_2026-07-06/final_scenario_confirm_results.csv`
- `outputs/experiments/track_b_realkan_no_forecast_bs512_ab_h104_confirm_2026-07-06/final_scenario_confirm_results.md`

Protocol verified from logs and summaries:

- Architecture: Real-KAN only
- Scenarios: `case_a_all_risks`, `case_b_downstream`
- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3,4,5`
- Train timesteps: `60000`
- Eval episodes: `12`
- Horizon: `104` weeks
- Real-KAN batch size: `512`

Primary metric: Garrido Excel ReT, `order_ret_excel_mean`.

## Results

| Scenario | Real-KAN batch | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---|---:|---:|---:|---|---:|---:|---:|
| Case A: all Garrido risks | 512 | 0.005774 | 0.001463 | `s2_d2.00` | +0.000130 | +2.30% | 0.429 |
| Case B: R22/R23/R24 only | 512 | 0.603024 | 0.000000 | `s2_d2.00` | +0.012133 | +2.05% | 0.357 |

## Comparison Against No-Forecast Batch 256

The no-forecast final A/B confirmation used Real-KAN batch size 256. Batch 512
does **not** improve that same protocol.

| Scenario | Real-KAN bs256 ReT | Real-KAN bs512 ReT | bs512 minus bs256 | bs256 cost | bs512 cost |
|---|---:|---:|---:|---:|---:|
| Case A: all Garrido risks | 0.005854707 | 0.005774 | -0.000081 | 0.378 | 0.429 |
| Case B: R22/R23/R24 only | 0.604782218 | 0.603024 | -0.001758 | 0.359 | 0.357 |

For Case A, batch 512 is strictly worse: lower ReT, lower CVaR05, and higher
cost than batch 256. For Case B, batch 512 is slightly cheaper but loses ReT
relative to batch 256.

## Comparison Against PPO Spine

| Scenario | PPO+MLP ReT | Real-KAN bs512 ReT | Gap |
|---|---:|---:|---:|
| Case A: all Garrido risks | 0.005900714 | 0.005774 | -0.000127 |
| Case B: R22/R23/R24 only | 0.604785754 | 0.603024 | -0.001762 |

Batch 512 does not make Real-KAN competitive with the PPO+MLP no-forecast spine
in Case A, and it also weakens the near-tie previously observed in Case B.

## Branch / Exposure Check

Case B still shows the exposure-reduction mechanism:

- Best static `s2_d2.00`: 77.06% fill-rate branch.
- Real-KAN bs512: 80.13% fill-rate branch.

So Real-KAN bs512 still learns a useful adaptive dispatch pattern in the
downstream-only setting, but it does not outperform the cleaner bs256 Real-KAN
confirmation and does not change the paper recommendation.

## Verdict

Do **not** promote batch size 512 as the no-forecast Real-KAN setting for A/B.

The earlier batch-size sweep correctly showed that 512 was promising under
`adaptive_benchmark_v2` / `v7` full, but the same preference does not transfer
to reviewer-safe `v7_no_forecast`.

Recommendation:

- Keep Real-KAN batch size 256 for no-forecast A/B references.
- Keep PPO+MLP no-forecast as the Case A paper spine.
- Use Real-KAN as an interpretability / efficiency sidecar, not as the main
  all-risk no-forecast architecture.

