# Track B Environment + Hyperparameter Tuning Preregistration

Created: 2026-07-05

## Why this exists

We found that the manuscript headline bundle used `batch_size=256`, while the
new fixed-RNG canonical lane used `batch_size=64`. That means the fixed-RNG lane
changed two things at once: RNG streams and PPO minibatch size.

This screen separates those effects before launching another large confirmatory
run.

## Stage A: environment and batch-size screen

Runner: `scripts/run_track_b_env_hparam_grid.py`

Live VPS run:

```bash
tmux attach -t track_b_env_batch_grid_3x3_30k
```

Output:

```text
outputs/experiments/track_b_env_batch_grid_3x3_30k_2026-07-05/
```

Protocol:

- observation: `v7_no_forecast`
- action contract: `track_b_v1`
- reward: `control_v1`
- seeds: `1,2,3`
- training: `30k` PPO timesteps per seed
- evaluation: `8` episodes per seed, `h104`
- PPO: `n_steps=1024`, `n_epochs=10`, `learning_rate=3e-4`, `gamma=0.99`,
  `gae_lambda=0.95`, `clip_range=0.2`, `ent_coef=0.0`
- tested batch sizes: `64`, `128`, `256`

Scenarios:

1. `garrido_all_current`: Garrido-native current risk level, thesis-faithful
   timing, all thesis risks active.
2. `garrido_downstream_cherry`: Garrido-native current risk level with only
   `R22/R23/R24` active. This is the clean downstream cherry-pick because Track
   B controls Op10/Op12 dispatch.
3. `adaptive_v2_current`: the current Track B adaptive benchmark, with Markov
   regimes and downstream-risk uplift.

Primary selection metric:

- `order_ret_excel_mean`, the Garrido Excel ReT metric.

Secondary checks:

- cost index
- shift mix
- best static comparator under the same run
- no promotion if the win only appears under a less defensible risk setup

## Stage B: hyperparameter tuning after Stage A

Only run this after Stage A identifies a viable scenario/batch-size pair.

Candidate grid:

- `learning_rate`: `1e-4`, `3e-4`
- `ent_coef`: `0.0`, `0.005`, `0.01`
- optional: `norm_reward=True` if the current gamma/reward-normalization grid
  shows a clear benefit.

Do not run every combination at confirmatory scale. First use `3 seeds x 30k`;
then promote only the best cell to `5-10 seeds x 60k`.

## Interpretation rule

This is a tuning screen, not a paper claim. A cell can be promoted only if it:

1. improves or matches ReT Excel against the current baseline;
2. does not create a hidden cost blow-up;
3. uses a risk scenario we can defend to Garrido and reviewers;
4. is later replicated at matched `60k` training scale.
