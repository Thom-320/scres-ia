# Track B v10 risk-memory smoke verdict — 2026-07-04

## What ran

Experimental observation:

`v10 = v9 + observed historical memory for R11/R13/R24`

This is not a forecast oracle. It adds only observed history:

- weeks since last event;
- count in last 8 weeks;
- count in last 26 weeks;
- EWMA-like decayed event memory.

Runs:

- `PPO+MLP v10`: `outputs/experiments/track_b_v10_risk_memory_ppo_3seed_30k_2026-07-04/`
- `Real-KAN v10`: `outputs/experiments/track_b_v10_risk_memory_real_kan_3seed_30k_2026-07-04/`

Protocol:

- seeds 1-3
- 30,000 timesteps
- 8 eval episodes
- h104
- `reward_mode=control_v1`
- `risk_level=adaptive_benchmark_v2`
- `action_contract=track_b_v1`
- primary metric: `order_ret_excel_mean`

## Results

| Policy | Observation | ReT Excel | CI95 | Cost |
|---|---|---:|---:|---:|
| PPO+MLP | v10 | 0.00581070 | [0.00571953, 0.00590187] | 0.763 |
| Real-KAN | v10 | 0.00591475 | [0.00590589, 0.00592361] | 0.982 |
| Best static in v10 bundle | v10 | 0.00541452 | -- | 1.000 |
| PPO+MLP fixed-RNG reference | v7 | 0.00592064 | [0.00590559, 0.00593569] | 0.700 |
| Real-KAN fixed-RNG reference | v7 | 0.00594620 | [0.00594103, 0.00595137] | 1.000 |

## Comparisons

- PPO+MLP v10 vs best static: **+0.00039617** (**+7.32%**).
- Real-KAN v10 vs best static: **+0.00050023** (**+9.24%**).
- PPO+MLP v10 vs PPO+MLP v7 reference: **-0.00010994** (**-1.86%**).
- Real-KAN v10 vs Real-KAN v7 reference: **-0.00003145** (**-0.53%**).
- Real-KAN v10 vs PPO+MLP v10: **+0.00010405** (**+1.79%**), positive in 3/3 seeds.

Per-seed Real-KAN v10 minus PPO+MLP v10:

| Seed | Real-KAN v10 | PPO+MLP v10 | Delta |
|---:|---:|---:|---:|
| 1 | 0.00591400 | 0.00579666 | +0.00011734 |
| 2 | 0.00590732 | 0.00573807 | +0.00016925 |
| 3 | 0.00592293 | 0.00589736 | +0.00002557 |

## Interpretation

This smoke **does not prove preventive learning**.

It says:

1. Adding observed risk memory does not automatically improve PPO+MLP. In this short 3-seed/30k smoke, PPO+MLP v10 is worse than the v7 fixed-RNG reference and costs more.
2. Real-KAN is more robust to the expanded observation. It still beats statics and beats PPO+MLP v10 in 3/3 seeds, but it also remains high-cost.
3. The historical-memory signal found by the supervised predictor is real enough to test, but raw memory features alone are not the same as an auxiliary belief head.

## Verdict

Do **not** promote `v10` as the main paper observation contract.

Recommended next step if we continue the prevention lane:

- Keep `v7`/no-forecast as the efficient spine.
- Treat `v10` as an experimental prevention lane.
- The next real test should be **auxiliary prediction**, not just appending memory features:
  - `PPO+MLP-belief`
  - `PPO+RealKAN-belief`
  - targets focused first on R24 at 1-2 weeks, because the supervised smoke showed the cleanest learnable memory signal there.

Current claim boundary:

> Real-KAN learns and is more resilient, but current evidence still points to high-capacity robustness rather than clean prevention. Observed risk memory alone did not make PPO+MLP more preventive or more resilient in this smoke.
