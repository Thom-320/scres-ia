# Track B final A/B h104 confirmation verdict - 2026-07-06

## Status

Completed. This is the h104 final confirmation for the two reviewer-facing
Garrido-native scenarios that do not depend on the Case C tuning result.

Artifacts:

- `outputs/experiments/track_b_final_ab_h104_confirm_2026-07-06/final_scenario_confirm_results.csv`
- `outputs/experiments/track_b_final_ab_h104_confirm_2026-07-06/final_scenario_confirm_results.md`
- Independent VPS PPO corroboration:
  `outputs/experiments/track_b_ppo_ab_h104_confirm_vps_2026-07-06/final_scenario_confirm_results.csv`

Protocol verified from summaries/logs:

- Scenarios: `case_a_all_risks`, `case_b_downstream`
- Architectures: PPO+MLP and Real-KAN
- Observation: `v7_no_forecast`
- Risk level: Garrido `current`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Seeds: `1,2,3,4,5`
- Train timesteps: `60000`
- Eval episodes: `12`
- Horizon: `104` weeks
- PPO batch size: `64`
- Real-KAN batch size: `256`

Primary metric: Garrido Excel ReT, `order_ret_excel_mean`.

## Results

| Scenario | Architecture | ReT Excel | CVaR05 | Best static | Delta vs static | Relative delta | Cost |
|---|---|---:|---:|---|---:|---:|---:|
| Case A: all Garrido risks | PPO+MLP | 0.005900714 | 0.001911092 | `s2_d2.00` | +0.000256755 | +4.55% | 0.488 |
| Case A: all Garrido risks | Real-KAN | 0.005854707 | 0.001787318 | `s2_d2.00` | +0.000210748 | +3.73% | 0.378 |
| Case B: R22/R23/R24 only | PPO+MLP | 0.604785754 | 0.000000000 | `s2_d2.00` | +0.013894863 | +2.35% | 0.431 |
| Case B: R22/R23/R24 only | Real-KAN | 0.604782218 | 0.000000000 | `s2_d2.00` | +0.013891327 | +2.35% | 0.359 |

## Per-Seed Stability

PPO+MLP is extremely stable in Case A:

| Seed | ReT Excel | CVaR05 | Cost |
|---:|---:|---:|---:|
| 1 | 0.005897 | 0.001964 | 0.491 |
| 2 | 0.005899 | 0.001934 | 0.472 |
| 3 | 0.005899 | 0.001890 | 0.483 |
| 4 | 0.005902 | 0.001898 | 0.515 |
| 5 | 0.005906 | 0.001869 | 0.479 |

Real-KAN is lower than PPO+MLP in Case A on every seed, but also cheaper:

| Seed | ReT Excel | CVaR05 | Cost |
|---:|---:|---:|---:|
| 1 | 0.005889 | 0.001930 | 0.379 |
| 2 | 0.005862 | 0.001852 | 0.377 |
| 3 | 0.005867 | 0.001856 | 0.381 |
| 4 | 0.005808 | 0.001604 | 0.374 |
| 5 | 0.005847 | 0.001695 | 0.376 |

In Case B, PPO+MLP and Real-KAN are effectively tied on ReT, while Real-KAN is
meaningfully cheaper:

| Seed | PPO ReT | Real-KAN ReT | PPO cost | Real-KAN cost |
|---:|---:|---:|---:|---:|
| 1 | 0.606204 | 0.606209 | 0.404 | 0.361 |
| 2 | 0.604879 | 0.604773 | 0.481 | 0.362 |
| 3 | 0.604830 | 0.604819 | 0.377 | 0.362 |
| 4 | 0.602945 | 0.603101 | 0.479 | 0.353 |
| 5 | 0.605071 | 0.605009 | 0.412 | 0.358 |

## Independent VPS PPO Corroboration

A separate PPO-only A/B confirmation was also run on the VPS with the same
intended protocol: `v7_no_forecast`, h104, seeds `1..5`, `60000` train
timesteps, `12` eval episodes, PPO batch size `64`, reward `control_v1`.

The VPS run landed after the local A/B/C package and was fetched to:

`outputs/experiments/track_b_ppo_ab_h104_confirm_vps_2026-07-06/`

| Scenario | Local PPO ReT | VPS PPO ReT | Difference |
|---|---:|---:|---:|
| Case A: all Garrido risks | 0.005900714 | 0.005903104 | +0.000002390 |
| Case B: R22/R23/R24 only | 0.604785754 | 0.604846242 | +0.000060489 |

This corroborates the local PPO result. The small differences are not material
to the claim boundary: PPO remains the stronger Case A spine, and Case B remains
a downstream exposure-reduction panel rather than the main headline.

## Interpretation

Case A is the reviewer-safe Garrido-native main scenario: all thesis risks are
active, no forecast variables are exposed, and the horizon is the selected
two-year h104 horizon. PPO+MLP is the stronger spine here: it beats the best
static by +4.55% on Excel ReT and has better CVaR05 than Real-KAN.

Real-KAN is not the Case A spine under this protocol, but it is not a failed
sidecar. It beats the best static by +3.73% while spending much less resource
than PPO+MLP. That is a useful interpretability/efficiency comparison, not the
headline.

Case B shows the branch/exposure mechanism cleanly. Both learned policies move
more orders into the fill-rate branch than the best static baseline:

- Best static `s2_d2.00`: 77.06% fill-rate branch.
- PPO+MLP: 80.32% fill-rate branch.
- Real-KAN: 80.34% fill-rate branch.

This is a real resilience mechanism: faster/adaptive dispatch shortens order
exposure windows and reduces overlap with downstream risk events. It should be
reported as **adaptive exposure reduction**, not as proven anticipatory
prevention.

## Claim Boundary

Supported:

- Under Garrido-native all-risk current conditions, no-forecast PPO+MLP improves
  Garrido Excel ReT over the best static baseline at h104.
- Real-KAN also improves over static, but below PPO+MLP in Case A.
- Under downstream-only Garrido risk conditions, PPO+MLP and Real-KAN are tied
  on ReT, and Real-KAN reaches the same resilience with lower resource cost.
- The downstream-only mechanism is branch/exposure reduction.

Not supported by this run:

- A prevention/anticipation claim.
- Promoting downstream-only absolute ReT values as comparable to all-risk ReT.
- Promoting Real-KAN as the main spine for the all-risk no-forecast result.

## Recommendation

Use Case A PPO+MLP h104 no-forecast as the main reviewer-safe Track B result.

Use Real-KAN as a novelty/interpretable efficiency sidecar:

- In Case A it is slightly weaker than PPO+MLP but still better than static and
  cheaper.
- In Case B it essentially matches PPO+MLP while using less resource.

Use Case B as a mechanism panel for exposure reduction and branch shift, not as
the primary headline because its Excel branch mix differs from the all-risk
Garrido scenario.
