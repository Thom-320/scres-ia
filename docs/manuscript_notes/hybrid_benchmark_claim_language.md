Recommended claim language for the current manuscript revision:

- `control_v1` is implemented and cleanly separates PPO training reward from `ReT_thesis`, which is retained as a reporting metric.
- Current PPO results are preliminary but already differentiated by stress regime.
- Under `increased + stochastic_pt`, PPO is competitive with the best fixed baseline but not superior in control reward.
- Under `severe + stochastic_pt`, PPO outperforms the best fixed baseline in control reward while maintaining comparable service.
- The observation space should be described as a practical Gymnasium-compatible operational snapshot and a useful Markovian approximation for control, while explicitly acknowledging a residual partial-observability caveat.

Do not use the following language unless formal inference is added later:

- `statistically significant`
- `Markov property is proven`
- `PPO solves the adaptive control problem`

Suggested sentence for Section 4.2.5:

> The current PPO gains under `control_v1` should be interpreted as stress-regime dependent rather than uniformly dominant: PPO is competitive under moderate stress and superior under the current severe-stress benchmark, but this does not yet justify a globally validated superiority claim across all fixed baselines and operating conditions.
