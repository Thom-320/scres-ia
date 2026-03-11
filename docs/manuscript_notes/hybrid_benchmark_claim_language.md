Recommended claim language for the current manuscript revision:

- `control_v1` is implemented and cleanly separates PPO training reward from `ReT_thesis`, which is retained as a reporting metric.
- Current PPO results are preliminary but promising under a narrow operational weight regime.
- Adaptive shift switching appears feasible, but robustness still requires longer runs, more seeds, and harsher stress tests.
- The observation space should be described as a practical Gymnasium-compatible operational snapshot and a useful Markovian approximation for control, while explicitly acknowledging a residual partial-observability caveat.

Do not use the following language unless formal inference is added later:

- `statistically significant`
- `Markov property is proven`
- `PPO solves the adaptive control problem`

Suggested sentence for Section 4.2.5:

> The current PPO gains under `control_v1` should be interpreted as preliminary and locally robust within the explored weight region rather than as a globally validated superiority claim over all fixed baselines and stress conditions.
