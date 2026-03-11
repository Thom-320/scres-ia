# Seed-Level Inference for 500k Control-Reward Runs

This note uses paired seed-level reward means (`ppo_eval` vs. best fixed static policy) rather than pooled episode rows.

## control_reward_500k_increased_stopt

- Best static policy: `static_s2`
- Shared seeds: 11, 22, 33, 44, 55
- Mean reward difference (`PPO - best_static`): -1.948
- Bootstrap CI95: [-9.949, 8.514]
- Exact sign-flip p-value: 0.812
- Interpretation: PPO is worse than static_s2 by -1.948 control-reward points on shared seed means; bootstrap CI95 [-9.949, 8.514], exact sign-flip p=0.812.

## control_reward_500k_severe_stopt

- Best static policy: `static_s3`
- Shared seeds: 11, 22, 33, 44, 55
- Mean reward difference (`PPO - best_static`): 4.608
- Bootstrap CI95: [-0.277, 9.493]
- Exact sign-flip p-value: 0.188
- Interpretation: PPO is better than static_s3 by 4.608 control-reward points on shared seed means; bootstrap CI95 [-0.277, 9.493], exact sign-flip p=0.188.

## Use in the paper

- Treat this as minimal inferential support, not a full significance section.
- Keep the manuscript language at `preliminary`, `competitive`, or `stronger under severe stress`.
- Do not claim formal statistical significance unless you intentionally elevate the inference section later.
