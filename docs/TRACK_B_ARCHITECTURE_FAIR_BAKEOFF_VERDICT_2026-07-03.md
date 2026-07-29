# Track B Architecture Fair Bakeoff Verdict (2026-07-03)

## Verdict

The same-run architecture bakeoff does **not** promote DMLPA/history over the plain PPO+MLP baseline.

Under the matched Track B protocol, plain `ppo_mlp` achieves the highest Garrido/Excel ReT:

| policy | Excel ReT mean | order-level ReT mean | cost index |
|---|---:|---:|---:|
| `ppo_mlp` | 0.005920 | 0.005694 | 0.663 |
| `ppo_dmlpa_positional` | 0.005871 | 0.005648 | 0.785 |
| `ppo_mlp_history` | 0.005832 | 0.005605 | 0.687 |

The result supports the Paper 1 framing: the main mechanism is not generic history/context, but the Track B decision surface reaching the operational bottleneck. DMLPA remains a useful robustness sidecar, not a replacement for the PPO+MLP spine.

## Protocol Verified

- Output: `outputs/experiments/track_b_architecture_fair_bakeoff_2026-07-03/full8d_v9_history_5seed_60k_h104/`
- Seeds: 1--5
- Training budget: 60k timesteps per seed/policy
- Evaluation: 12 episodes per seed
- Horizon: h104
- Risk regime: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Action contract: `track_b_v1`
- Observation: `v9`, used so all arms receive the same history-rich information surface
- Policies: `ppo_mlp`, `ppo_mlp_history`, `ppo_dmlpa_positional`

## Paired Checks

Paired episode-level comparison from `paired_architecture_comparison.csv`:

| candidate | reference | metric | delta mean | positive pairs |
|---|---|---|---:|---:|
| `ppo_dmlpa_positional` | `ppo_mlp_history` | `order_ret_excel` | +0.000039 | 36/60 |
| `ppo_mlp` | `ppo_mlp_history` | `order_ret_excel` | +0.000088 | 44/60 |
| `ppo_dmlpa_positional` | `ppo_mlp` | `order_ret_excel` | -0.000049 | 24/60 |

Seed-level recomputation gives the same qualitative result:

- DMLPA vs MLP-history: +0.000039 mean, 4/5 seeds positive, CI crosses zero.
- DMLPA vs plain MLP: -0.000049 mean, only 2/5 seeds positive, CI crosses zero.
- MLP-history vs plain MLP: -0.000088 mean, 0/5 seeds positive, CI below zero.

## Interpretation

This is the fairest version of David's history question currently in the repository: all policies see the same history-rich v9 observation and run under the same seeds, budget, horizon, and evaluation protocol.

The result says:

1. Adding history naively (`ppo_mlp_history`) hurts relative to plain MLP.
2. Attention over that history (`ppo_dmlpa_positional`) recovers part of the loss and beats the history-only MLP, but still does not beat plain MLP.
3. Therefore, for Paper 1, DMLPA/history should be framed as tested and non-promoted, not as an unresolved threat to the architecture choice.

The Garrido-facing message should be careful: history is scientifically interesting for prevention, but current evidence does not show that it improves the Track B result. Prevention still requires a separate lead-lag/forecast-ablation audit.
