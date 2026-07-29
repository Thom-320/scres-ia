# Track B horizon screen verdict - 2026-07-06

## Status

Completed. This was a lightweight horizon screen to choose a defensible horizon
before heavier PPO/Real-KAN confirmations.

Artifacts:

- `outputs/experiments/track_b_horizon_screen_2026-07-06_rerun/horizon_screen_results.csv`
- `outputs/experiments/track_b_horizon_screen_2026-07-06_rerun/horizon_screen_results.md`

Protocol verified from summaries/logs:

- Scenarios: `case_a_all_risks`, `case_b_downstream`
- Horizons: `52`, `104`, `156`, `260` weeks
- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3`
- Train timesteps: `20000`
- Eval episodes: `6`
- PPO batch size: `64`

This is a screen, not a final confirmation. Its job is to choose the horizon for
the expensive final runs.

## Results

| Scenario | Horizon | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---|---:|---:|---:|---|---:|---:|---:|
| all risks | 52 | 0.005651970 | 0.001313831 | `s3_d2.00` | +0.000456958 | +8.80% | 0.693 |
| all risks | 104 | 0.005869449 | 0.001725086 | `s3_d2.00` | +0.000234159 | +4.16% | 0.654 |
| all risks | 156 | 0.005924869 | 0.002002685 | `s3_d2.00` | +0.000150869 | +2.61% | 0.610 |
| all risks | 260 | 0.005980843 | 0.002333901 | `s3_d2.00` | +0.000088220 | +1.50% | 0.578 |
| downstream only | 52 | 0.468482703 | 0.000000000 | `s2_d2.00` | +0.027788970 | +6.31% | 0.755 |
| downstream only | 104 | 0.606753988 | 0.000000000 | `s2_d1.50` | +0.014759988 | +2.49% | 0.594 |
| downstream only | 156 | 0.664868856 | 0.000000000 | `s2_d1.50` | +0.010111042 | +1.54% | 0.616 |
| downstream only | 260 | 0.729443161 | 0.000000000 | `s2_d1.50` | +0.005704632 | +0.79% | 0.627 |

## Reading

The pattern is monotone and useful:

- Longer horizons improve absolute ReT in both scenarios.
- Longer horizons reduce the relative PPO-vs-static margin.
- The 52-week horizon maximizes headroom, but it is less compelling as a
  reviewer-facing resilience horizon.
- The 260-week horizon gives the highest absolute ReT, but it makes the adaptive
  advantage small and therefore harder to defend as the main learning result.

CVaR behaves differently by scenario. In the all-risk case, CVaR05 improves with
horizon because longer episodes average through more operational recovery. In
the downstream-only case, CVaR05 remains zero and is not useful for selecting a
horizon under this branch mix.

## Recommendation

Keep `104` weeks as the main final-confirmation horizon.

It is the best compromise:

1. It is still a realistic two-year resilience horizon.
2. It preserves visible adaptive headroom: +4.16% in all-risk and +2.49% in
   downstream-only.
3. It avoids the one-year screen looking too short or opportunistic.
4. It avoids the five-year screen diluting the PPO-vs-static advantage into a
   small long-run average.

Use `52` weeks only as a diagnostic/high-headroom sensitivity result, not as the
main paper horizon. Do not move the final confirmation to `156` or `260` unless
the paper objective changes from showing adaptive headroom to reporting long-run
steady resilience.

## Next Step

Continue the already-running final A/B confirmation at `104` weeks. Once Case C
is promoted into the final runner, use `104` weeks there as well.
