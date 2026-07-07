# Track B environment and batch-size tuning verdict - 2026-07-05

## Scope

This note summarizes the first environment/hyperparameter tuning pass requested before a longer confirmatory training run.

Primary metric throughout: Garrido Excel resilience, `order_ret_excel_mean` (not `order_level_ret_mean`).

Protocol for the main grid:

- Observation: `v7_no_forecast`
- Action contract: `track_b_v1`
- Reward: `control_v1`
- Seeds: 1, 2, 3
- Train timesteps: 30,000
- Eval episodes: 8
- Horizon: 104 steps
- PPO settings: `n_steps=1024`, `n_epochs=10`, `learning_rate=3e-4`, `gamma=0.99`, `gae_lambda=0.95`, `ent_coef=0.0`
- Batch sizes tested: 64, 128, 256

Artifact:

`outputs/experiments/track_b_env_batch_grid_3x3_30k_2026-07-05/`

## Main grid results

| Scenario | Batch | ReT Excel | Cost | Best static | Delta vs static |
|---|---:|---:|---:|---|---:|
| Garrido all risks, current | 64 | 0.005870 | 0.581 | s3_d2.00 | +0.000230 |
| Garrido all risks, current | 128 | 0.005869 | 0.671 | s3_d2.00 | +0.000229 |
| Garrido all risks, current | 256 | **0.005880** | 0.591 | s3_d2.00 | +0.000240 |
| Garrido downstream cherry, R22/R23/R24 | 64 | **0.606347** | 0.549 | s2_d2.00 | +0.017395 |
| Garrido downstream cherry, R22/R23/R24 | 128 | 0.606294 | 0.545 | s2_d2.00 | **+0.017487** |
| Garrido downstream cherry, R22/R23/R24 | 256 | 0.606333 | **0.543** | s2_d2.00 | +0.017330 |
| Adaptive benchmark v2 | 64 | **0.005840** | 0.629 | s3_d1.50 | **+0.000414** |
| Adaptive benchmark v2 | 128 | 0.005839 | 0.657 | s3_d1.50 | +0.000413 |
| Adaptive benchmark v2 | 256 | 0.005794 | 0.621 | s3_d1.50 | +0.000367 |

## Reading

For the Garrido-native all-risk setting, batch size 256 is the best of the three cells by ReT Excel, but the margin over 64 is small: about `+0.000010`. Batch size 64 is nearly tied and slightly cheaper.

For the downstream cherry setting, the absolute ReT scale is not comparable to the all-risk setting because only R22/R23/R24 are active. Within that scenario, batch size 64 has the highest learned-policy ReT, batch size 128 has the largest delta versus its static comparator, and batch size 256 is cheapest. The three are effectively close at this 3-seed smoke scale.

For adaptive benchmark v2 under no-forecast, batch size 64 is the best current cell. Batch size 128 is almost tied on ReT, but more expensive. Batch size 256 is clearly weaker in this 30k smoke run.

## Secondary batch-size sweep

There was also a separate opportunistic batch-size sweep:

`outputs/experiments/track_b_batch_size_sweep_{32,64,128,256,512}_3seed_30k_2026-07-05/`

This sweep is useful but not identical to the main grid: it uses adaptive benchmark v2 with full `v7`, not `v7_no_forecast`. Therefore it should not be collapsed into the no-forecast environment grid.

| Batch | PPO ReT Excel | Cost |
|---:|---:|---:|
| 32 | 0.005766 | 0.623 |
| 64 | 0.005823 | 0.763 |
| 128 | **0.005824** | 0.738 |
| 256 | 0.005780 | 0.647 |
| 512 | 0.005690 | 0.672 |

Secondary reading: under full-forecast adaptive v2, batch size 128 and 64 are tied on ReT at 30k, with 128 cheaper. Batch sizes 32, 256, and 512 are weaker.

## Decision for Case C

The Case C multiplier grid should proceed from the best parent among the Garrido-native candidates. The launched runner selected:

- Parent scenario: `garrido_downstream_cherry`
- Parent batch size: `64`
- Enabled risks: `R22,R23,R24`
- Multiplier grid: frequency x impact in `{0.75, 1.0, 1.25, 1.5} x {0.75, 1.0, 1.25, 1.5}`

That is appropriate for the diagnostic purpose: tune frequency/impact on the risks most directly touched by Track B's downstream authority. It is not a replacement for the all-risk Garrido headline; it is a targeted stress/tuning lane.

## Recommendation

For the reviewer-safe no-forecast Track B spine, keep two candidates alive:

1. `garrido_all_current`, batch size 256, if the goal is thesis-native all-risk evidence.
2. `adaptive_v2_current`, batch size 64, if the goal is continuity with the existing adaptive Track B benchmark.

For Case C, continue with the selected `garrido_downstream_cherry`, batch size 64 parent and wait for the multiplier grid before launching any confirmatory longer run.

Do not promote any 3-seed/30k cell as a final headline. This is a tuning screen.
