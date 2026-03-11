**Garrido Meeting Package**

Generated: 2026-03-11 11:20

**Show in this order**

0. `10_intro_slides_outline.md`
   Use this as the opening structure. The story should begin with the transition from DES to sequential control, not with PPO or with library names.

1. `01_static_shift_baselines.png`
   This is the first evidence slide after the intro. It proves that the environment is differentiated before any learning claim: shift control materially changes service levels.

2. `02_ppo_training_reward_curve.png`
   Use this only as a PPO diagnostic. Do not call it cross-validation loss. Say: “for PPO we track training reward and held-out evaluation, not supervised cross-validation loss.”

3. `03_best_regime_summary.png`
   This is the adaptive-control slide. Present it as preliminary evidence under a narrow weight regime, not as a final superiority claim.

4. `04_policy_comparison.png`
   Use this only if someone asks what “winning regime” means operationally.

5. `05_action_mix.png`
   Use this if Garrido or David asks whether PPO collapsed to a trivial static policy.

**One paragraph to say out loud after the intro**

The DES is already validated and the RL interface is operational. Before claiming anything about PPO, we verified that the static shift baselines are clearly differentiated, which confirms that shift allocation is a meaningful control lever. We also separated the training objective from the reporting metric: `ReT_thesis` is retained as an evaluation metric, while `control_v1` is used for learning because direct control needs an operational reward. Current PPO results are preliminary but promising in a narrow regime, and the stronger 500k-step stochastic-PT runs are already in progress.

**Observability line**

We treat the exposed observation as a practical Gymnasium-compatible operational snapshot and a useful Markovian approximation for control, while explicitly acknowledging a remaining partial-observability caveat.

**Long-run benchmark status**

- increased + stochastic_pt is running (PID 89604, output: outputs/benchmarks/control_reward_500k_increased_stopt)
- severe + stochastic_pt is running (PID 89638, output: outputs/benchmarks/control_reward_500k_severe_stopt)
