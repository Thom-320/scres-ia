**Garrido Meeting Package**

Generated: 2026-03-11 12:19

**Show in this order**

1. `01_static_shift_baselines.png`
   This is the first slide. It proves that the environment is differentiated before any learning claim: shift control materially changes service levels.

2. `02_ppo_training_reward_curve.png`
   Use this only as a PPO diagnostic. Do not call it cross-validation loss. Say: “for PPO we track training reward and held-out evaluation, not supervised cross-validation loss.”

3. `03_best_regime_summary.png`
   This is now the main results slide. Say clearly that under `increased + stochastic_pt` PPO is competitive but not superior, while under `severe + stochastic_pt` PPO beats the best static baseline.

4. `04_policy_comparison.png`
   Use this only if someone asks what “winning regime” means operationally.

5. `05_action_mix.png`
   Use this if Garrido or David asks whether PPO collapsed to a trivial static policy.

**One paragraph to say out loud**

The DES is already validated and the RL interface is operational. Before claiming anything about PPO, we verified that the static shift baselines are clearly differentiated, which confirms that shift allocation is a meaningful control lever. We also separated the training objective from the reporting metric: `ReT_thesis` is retained as an evaluation metric, while `control_v1` is used for learning because direct control needs an operational reward. The completed 500k-step stochastic-PT runs now show a clean pattern: under increased risk PPO is competitive with the best static baseline, and under severe risk adaptive switching produces a reward advantage without collapse.

**Observability line**

We treat the exposed observation as a practical Gymnasium-compatible operational snapshot and a useful Markovian approximation for control, while explicitly acknowledging a remaining partial-observability caveat.

**Long-run benchmark outcome**

- increased + stochastic_pt: PPO is competitive but not superior in reward (-172.05 vs -170.10); service is effectively matched (83.8% vs 83.7%).
- severe + stochastic_pt: PPO beats the best static baseline in reward (-380.98 vs -385.59) while maintaining comparable service (63.1% vs 63.2%).
