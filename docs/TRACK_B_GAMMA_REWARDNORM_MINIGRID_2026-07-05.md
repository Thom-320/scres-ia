# Track B gamma / GAE / reward-normalization mini-grid -- 2026-07-05

## Question

David asked whether the PPO discount factor `gamma` can be changed to prioritize longer-term rewards, and whether the reward should be normalized to mean 0 and standard deviation 1.

## Short answer

Yes. Track B PPO already exposes `--gamma` and `--gae-lambda`. This run adds an explicit `--norm-reward` flag to the canonical Track B smoke runner, using Stable-Baselines3 `VecNormalize(norm_reward=True)`. Evaluation metrics remain on the original Garrido/Excel scale; only the training signal is normalized.

## Why this is diagnostic, not confirmatory

Changing `gamma` and reward normalization can improve credit assignment and value-target scale. It does not by itself prove preventive learning. The primary metric remains `order_ret_excel_mean`; if a cell looks promising here, it must be escalated to the full fixed-RNG/no-forecast protocol before becoming a paper claim.

## Hyperparameters under review

Canonical Track B PPO settings:

- `learning_rate = 3e-4`
- `n_steps = 1024`
- `batch_size = 64` for PPO+MLP
- `n_epochs = 10`
- `gamma = 0.99`
- `gae_lambda = 0.95`
- `clip_range = 0.2`
- `norm_obs = True`
- `norm_reward = False` before this diagnostic

Mini-grid:

- `gamma in {0.99, 0.995, 0.999}`
- `gae_lambda in {0.95, 0.98}`
- `norm_reward in {False, True}`
- `clip_reward = 10.0`

## Protocol

- Runner: `scripts/run_track_b_gamma_rewardnorm_grid.py`
- Base runner: `scripts/run_track_b_observation_ablation.py`
- Observation config: `v7_no_forecast`
- Action contract: `track_b_v1`
- Reward mode: `control_v1`
- Risk level: `adaptive_benchmark_v2`
- Seeds: `1,2,3`
- Train timesteps: `20,000`
- Eval episodes: `6`
- Horizon: `104`
- PPO: `n_steps=1024`, `batch_size=64`, `n_epochs=10`

## Live artifacts

- Output: `outputs/experiments/track_b_gamma_rewardnorm_grid_2026-07-05/`
- Run log: `outputs/experiments/track_b_gamma_rewardnorm_grid_2026-07-05.run.log`
- tmux session: `track_b_gamma_rewardnorm_grid`

## Final result

The grid completed successfully with 12/12 cells. All cells used:

- observation config: `v7_no_forecast`
- reward mode: `control_v1`
- risk level: `adaptive_benchmark_v2`
- seeds: `1,2,3`
- train timesteps: `20,000`
- eval episodes: `6`
- horizon: `104`
- `n_steps=1024`, `batch_size=64`, `n_epochs=10`, `learning_rate=3e-4`

Final ranking by Garrido/Excel `order_ret_excel_mean`:

| rank | cell | gamma | gae | norm reward | ReT Excel | cost | CVaR05 | 4w min | S1/S2/S3 |
|---:|---|---:|---:|---|---:|---:|---:|---:|---:|
| 1 | `g0p990_gae0p95_raw` | 0.990 | 0.95 | no | 0.005824936 | 0.676 | 0.001657906 | 0.002243757 | 8.9/79.3/11.8 |
| 2 | `g0p990_gae0p95_norm` | 0.990 | 0.95 | yes | 0.005817493 | 0.682 | 0.001610214 | 0.002228712 | 10.4/74.6/15.0 |
| 3 | `g0p999_gae0p98_norm` | 0.999 | 0.98 | yes | 0.005788294 | 0.778 | 0.001411340 | 0.002129607 | 6.4/53.6/40.0 |
| 4 | `g0p995_gae0p95_raw` | 0.995 | 0.95 | no | 0.005777676 | 0.640 | 0.001376468 | 0.002086683 | 12.5/83.0/4.5 |
| 5 | `g0p999_gae0p95_norm` | 0.999 | 0.95 | yes | 0.005762179 | 0.645 | 0.001274967 | 0.002089197 | 14.0/78.4/7.5 |
| 6 | `g0p990_gae0p98_norm` | 0.990 | 0.98 | yes | 0.005761456 | 0.653 | 0.001318252 | 0.002056942 | 9.4/85.1/5.4 |
| 7 | `g0p999_gae0p95_raw` | 0.999 | 0.95 | no | 0.005755649 | 0.718 | 0.001260689 | 0.001976114 | 8.9/66.9/24.2 |
| 8 | `g0p995_gae0p98_raw` | 0.995 | 0.98 | no | 0.005752366 | 0.679 | 0.001206039 | 0.002024297 | 3.2/90.0/6.8 |
| 9 | `g0p990_gae0p98_raw` | 0.990 | 0.98 | no | 0.005748150 | 0.621 | 0.001334672 | 0.002045822 | 17.8/78.2/4.0 |
| 10 | `g0p995_gae0p95_norm` | 0.995 | 0.95 | yes | 0.005734298 | 0.632 | 0.001222702 | 0.001971453 | 21.3/68.0/10.7 |
| 11 | `g0p995_gae0p98_norm` | 0.995 | 0.98 | yes | 0.005727307 | 0.784 | 0.001095025 | 0.001953884 | 3.3/58.3/38.4 |
| 12 | `g0p999_gae0p98_raw` | 0.999 | 0.98 | no | 0.005685511 | 0.708 | 0.000892362 | 0.001908720 | 6.9/73.9/19.2 |

## Verdict

Do not promote any gamma/reward-normalization change from this screen.

The best cell is the canonical setting: `gamma=0.99`, `gae_lambda=0.95`,
`norm_reward=False`. Reward normalization did not improve the Garrido/Excel
metric in the top comparable cell, and higher gamma values tended to reduce
`order_ret_excel_mean` or buy similar performance with higher cost.

This answers David's question cleanly: yes, `gamma` and reward normalization are
available and now audited, but this diagnostic does not support changing them
for the next confirmatory Track B run.

## Decision rule

Promote only if a cell clearly improves Garrido/Excel `order_ret_excel_mean` over the same-budget canonical cell, without simply buying the result through much higher resource cost.

If a cell is promising, next step is a matched 5--10 seed, 60k fixed-RNG no-forecast confirmation.

No cell is promoted from this mini-grid.
