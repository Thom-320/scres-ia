# Track B Ruta B Live-Auxiliary Case C Verdict - 2026-07-06

## Status

3-seed Ruta B screen completed. This is the first Track B prevention lane in
this session that clears the causal screen at non-trivial sample size.

Ruta B means:

```text
L = L_PPO + lambda * BCE(P(future risk) | shared_features)
```

The future-risk prediction task stays active during PPO training, so the shared
representation cannot simply forget the belief task during fine-tuning.

## Protocol

- Scenario: selected Case C
  - enabled risks: `R22,R23,R24`
  - `R24` frequency multiplier: `3.0`
  - `R22/R23` impact multiplier: `1.5`
- Observation: `v10_no_forecast`
  - v10 memory retained
  - explicit forecast fields masked
- Label channel: kept raw under `VecNormalize`
  - `label_channel_raw_under_vecnormalize = true`
- Policy: PPO+MLP with live auxiliary belief head
- Seeds: `1,2,3`
- Train timesteps: `30000`
- Eval episodes: `8`
- Horizon: `104` weeks
- Batch size: `64`
- n_steps: `1024`
- Reward: `control_v1`

Primary metric remains Garrido Excel ReT. Prevention is judged by the inlined
fixed-RNG `R_full - R_reset(pre-risk)` counterfactual.

## Results

| Variant | ReT Excel | Static Delta | Cost | Pairs | Positive Rate | Mean Delta |
|---|---:|---:|---:|---:|---:|---:|
| `R22`, lead 4w, lambda 0.25 | 0.484659 | +9.74% | 0.418 | 111 | 0.063 | -0.00001135 |
| `R22`, lead 4w, lambda 0.50 | 0.484615 | +9.73% | 0.409 | 111 | 0.036 | -0.00000452 |
| `R22+R24`, lead 2w, lambda 0.25 | 0.484735 | +9.76% | 0.416 | 171 | 0.749 | +0.00105249 |
| `R22+R24`, lead 4w, lambda 0.25 | 0.484897 | +9.79% | 0.427 | 171 | 0.673 | +0.00077000 |

Reference final Case C PPO baseline:

- PPO+MLP, v7 no-forecast, 5 seeds x 60k: ReT `0.481160397`, cost `0.719`

## Interpretation

The R22-only auxiliary objective does **not** produce prevention. It preserves
strong adaptive performance but counterfactual positive rates remain tiny
(3.6%-6.3%).

The joint `R22+R24` auxiliary objective changes the mechanism:

- lead 2w: 128/171 positive pairs, mean delta `+0.00105249`
- lead 4w: 115/171 positive pairs, mean delta `+0.00077000`

This is qualitatively different from the previous failed prevention lanes,
where positive-pair rates were usually below 10%-20% and mean deltas were near
zero or negative.

The best prevention screen is `R22+R24`, lead 2 weeks, lambda `0.25`. The 4-week
variant has slightly higher ReT but weaker causal prevention. Since the stated
objective is preventive learning, the lead-2w variant is the confirmatory
candidate.

## Claim Boundary

Supported at screen scale:

- Ruta B creates a live predictive representation that produces non-trivial
  pre-risk causal benefit in selected Case C.
- The signal requires the joint `R22+R24` objective; `R22` alone fails.
- The effect is not coming from explicit forecast fields (`v10_no_forecast`).

Not yet supported:

- Final manuscript-grade prevention claim.
- Generalization to all-risk Case A.
- A Real-KAN Ruta B claim.

## Next Step

Launch confirmatory Ruta B:

- selected Case C
- `v10_no_forecast`
- auxiliary risks `R22 R24`
- lead `2` weeks
- lambda `0.25`
- seeds `1..5`
- train timesteps `60000`
- eval episodes `12`

If that confirmatory run keeps positive-pair rate near or above 0.5 with
positive mean delta, Track B finally has a defensible preventive-learning
result, not just adaptive exposure reduction.

## Artifacts

- Grid output:
  `outputs/experiments/track_b_ruta_b_grid_case_c_3seed_30k_2026-07-06`
- Best screen:
  `outputs/experiments/track_b_ruta_b_grid_case_c_3seed_30k_2026-07-06/r22r24_l2_c025/summary.json`
