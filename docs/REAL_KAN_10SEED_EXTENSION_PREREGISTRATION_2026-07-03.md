# Real-KAN 10-Seed Extension Preregistration -- 2026-07-03

## Purpose

The 5-seed/60k PPO+Real-KAN Track B sidecar is positive and is now the strongest
architecture response to Garrido's KAN concern. This extension tests whether that
result survives the same 10-seed discipline used for the canonical PPO+MLP headline.

This is still an architecture sidecar. It does not automatically reframe Paper 1
unless it beats PPO+MLP under the same metric and the cost tradeoff remains
defensible.

## Existing Evidence To Freeze

Already completed:

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`

Protocol:

- Official `pykan` KAN via `RealKANFeaturesExtractor`
- seeds `1..5`
- `train_timesteps=60000`
- `eval_episodes=12`
- `max_steps=104`
- `observation_version=v7`
- `action_contract=track_b_v1`
- `risk_level=adaptive_benchmark_v2`
- `reward_mode=control_v1`

Observed 5-seed result:

- PPO+Real-KAN `order_ret_excel_mean = 0.005926`
- Best static `order_ret_excel_mean = 0.005428`
- 5/5 seeds positive vs best static
- Higher shift-utilization cost than the static comparator and the current
  PPO+MLP paper-facing profile

## New Run

Run only fresh seeds `6..10`:

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_10seed_extension_6_10_60k_h104/`

Same protocol as seeds `1..5`:

- Official `pykan` KAN via `RealKANFeaturesExtractor`
- seeds `6..10`
- `train_timesteps=60000`
- `eval_episodes=12`
- `max_steps=104`
- `observation_version=v7`
- `action_contract=track_b_v1`
- `risk_level=adaptive_benchmark_v2`
- `reward_mode=control_v1`

## Decision Rule

After the extension lands, merge seeds `1..5` and `6..10` into a dated verdict.

Report all of:

- mean and seed-level `order_level_ret_mean`
- mean and seed-level `order_ret_excel`
- seed signs vs best static
- comparison to the canonical PPO+MLP 10-seed Excel-ReT anchor
- shift-utilization cost and any dispatch-cost-sensitive interpretation

Promote Real-KAN only to "architecture sidecar confirmed" if:

- at least 8/10 seeds beat the best static on `order_ret_excel`, and
- mean `order_ret_excel` remains at or above the PPO+MLP 10-seed anchor, and
- the higher cost is disclosed rather than hidden.

Do not make Real-KAN the manuscript spine unless a subsequent same-run or
cost-sensitive comparison shows it is superior to PPO+MLP on the operational
tradeoff, not merely marginally higher on ReT.
