# Track B Case C selected h104 confirmation verdict - 2026-07-06

## Status

Completed. This is the final 5-seed h104 confirmation for the selected Case C
stress/adaptation candidate from the per-risk headroom screen.

Selected Case C:

- Enabled risks: `R22,R23,R24`
- `R24` frequency multiplier: `3.0`
- `R22` impact multiplier: `1.5`
- `R23` impact multiplier: `1.5`
- Descriptor: `configs/track_b_case_c_r24_freq3_r22r23_impact1p5.json`

Artifacts:

- `outputs/experiments/track_b_case_c_selected_h104_confirm_2026-07-06/final_scenario_confirm_results.csv`
- `outputs/experiments/track_b_case_c_selected_h104_confirm_2026-07-06/final_scenario_confirm_results.md`

Protocol verified from logs and summaries:

- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3,4,5`
- Train timesteps: `60000`
- Eval episodes: `12`
- Horizon: `104` weeks
- PPO batch size: `64`
- Real-KAN batch size: `512`

Primary metric: Garrido Excel ReT, `order_ret_excel_mean`.

## Results

| Architecture | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---|---:|---:|---|---:|---:|---:|
| PPO+MLP | 0.481160 | 0.000000 | `s2_d2.00` | +0.046324 | +10.65% | 0.719 |
| Real-KAN | 0.470096 | 0.000000 | `s2_d2.00` | +0.035259 | +8.11% | 0.387 |

The selected Case C signal survives the heavier confirmation. The 3-seed screen
reported +9.88% relative PPO headroom; the final 5-seed run reports +10.65%.

## Policy Comparison

| Architecture | ReT Excel | Delta vs static | Share of PPO delta | Cost |
|---|---:|---:|---:|---:|
| PPO+MLP | 0.481160 | +0.046324 | 100.0% | 0.719 |
| Real-KAN | 0.470096 | +0.035259 | 76.1% | 0.387 |

PPO+MLP is the better resilience optimizer in selected Case C. Real-KAN captures
about three quarters of PPO's static margin while spending much less resource.

That gives a clean paper-facing division:

- PPO+MLP: stronger adaptive performance.
- Real-KAN: cheaper interpretable sidecar with a real, but smaller, gain.

## Branch / Exposure Mechanism

Best static baseline `s2_d2.00`:

- Fill-rate branch: 49.48%
- Recovery branch: 33.74%
- Risk-no-recovery branch: 14.39%
- Risk-conditional ReT: 0.121567

PPO+MLP:

- Fill-rate branch: 55.88%
- Recovery branch: 29.61%
- Risk-no-recovery branch: 14.21%
- Risk-conditional ReT: 0.138663

Real-KAN:

- Fill-rate branch: 54.66%
- Recovery branch: 29.64%
- Risk-no-recovery branch: 15.03%
- Risk-conditional ReT: 0.135817

The mechanism is the same one observed in Case B, now under stronger downstream
stress: the learned policies move more orders into the fill-rate branch by
reducing exposure windows. PPO also improves risk-conditional ReT relative to
the best static baseline.

This is a real adaptive-resilience mechanism. It is **not** proof of preventive
anticipation unless a separate event-aligned counterfactual shows stable
pre-event benefit.

## CVaR

`order_ret_excel_cvar05_mean` is zero for PPO, Real-KAN, and the static baselines
under this downstream-only branch mix. CVaR therefore does not discriminate
policies in selected Case C.

For this scenario, the informative metrics are:

- Delta vs best static on `order_ret_excel_mean`
- Relative delta vs best static
- Branch mix / exposure shift
- Risk-conditional ReT
- Resource cost

## Claim Boundary

Supported:

- Selected Case C creates strong adaptive headroom: PPO beats the best static by
  +10.65% on Garrido Excel ReT.
- Real-KAN also beats the best static by +8.11%, at much lower cost than PPO.
- The mechanism is exposure reduction and improved downstream-stress handling.

Not supported:

- A causal prevention claim.
- Comparing selected Case C absolute ReT directly against all-risk Case A
  absolute ReT. The Excel branch mix is different.
- Promoting Real-KAN as the stronger optimizer for selected Case C. It is
  cheaper, not better on ReT.

## Recommendation

Use selected Case C as the controlled stress/adaptation panel, not as the main
headline scenario.

Paper framing:

> Under a controlled downstream stress scenario designed to create adaptive
> headroom, no-forecast PPO improves Garrido Excel ReT by +10.65% over the best
> static baseline. Real-KAN also improves by +8.11% with substantially lower
> resource use, making it a useful interpretable efficiency sidecar.

Next optional gate:

Run the event-aligned `R_full - R_reset(pre-R24)` or `pre-R22/pre-R23` audit only
if we want to explicitly test whether any of the Case C gain is anticipatory.
Given the previous negative prevention audits, the default claim should remain
adaptive exposure reduction unless that audit changes the evidence.

