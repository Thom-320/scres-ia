# Real-KAN Sidecar Confirmatory Verdict -- 2026-07-03

## Artifact

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`

## Protocol Verified

- Architecture: PPO + official `pykan` `kan.KAN` via `RealKANFeaturesExtractor`
- Seeds: `1..5`
- Training timesteps: `60,000`
- Evaluation episodes: `12` per seed
- Horizon: `h104`
- Risk regime: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Observation: `v7`
- Action contract: `track_b_v1`
- Raw-material flow: requested `kit_equivalent_order_up_to`, canonicalized to
  `bom_total_units_order_up_to`

## Result

Real-KAN confirms the Track B signal and is now the strongest architecture
sidecar against Garrido's KAN concern.

From `policy_summary.csv`:

- PPO+Real-KAN `order_level_ret_mean_mean = 0.005700`
- Best static (`s2_d1.50`) `order_level_ret_mean_mean = 0.005214`
- Gap: `+0.000485`
- PPO+Real-KAN `order_ret_excel_mean = 0.005926`
- Best static (`s2_d1.50`) `order_ret_excel_mean = 0.005428`
- Gap: `+0.000498`
- PPO+Real-KAN `flow_fill_rate_mean = 0.969644`
- Best static (`s2_d1.50`) `flow_fill_rate_mean = 0.670677`
- PPO+Real-KAN `assembly_cost_index_mean = 0.970566`
- Best static (`s2_d1.50`) `assembly_cost_index_mean = 0.666667`

Seed-level deltas against the best static are uniformly positive:

| Seed | Delta `order_level_ret_mean` | Delta `order_ret_excel` | Delta flow fill | Delta cost index |
|---:|---:|---:|---:|---:|
| 1 | +0.000483 | +0.000494 | +0.297361 | +0.332799 |
| 2 | +0.000441 | +0.000454 | +0.292051 | +0.333333 |
| 3 | +0.000477 | +0.000489 | +0.299167 | +0.331998 |
| 4 | +0.000532 | +0.000547 | +0.306186 | +0.188034 |
| 5 | +0.000494 | +0.000508 | +0.300071 | +0.333333 |

## Interpretation

This result answers the narrow KAN feasibility question strongly: a literal
official-KAN policy/value approximator can be trained in the corrected Track B
PPO loop and beats the static frontier across all five seeds.

It does **not** automatically replace the manuscript's PPO+MLP spine:

- It is slightly above the current PPO+MLP Excel anchor (`0.005926` vs about
  `0.005898`), but the comparison is not yet same-run paired against PPO+MLP
  inside this bundle.
- It achieves the resilience gain with materially higher shift-utilization
  cost (`0.971` vs the static `0.667`, and higher than the canonical PPO+MLP
  cost profile reported in the paper-facing bundle).
- It does not change the core mechanism: both PPO+MLP and PPO+Real-KAN win only
  when the Track B action surface reaches the downstream bottleneck.

## Decision

Promote Real-KAN from "smoke" to **strong architecture sidecar**. Do not promote
it to the Paper 1 headline unless a same-run paired comparison shows that it
beats PPO+MLP on the same seeds/evaluation protocol without unacceptable cost
deterioration.

Paper-safe wording:

> A confirmatory PPO+Real-KAN sidecar using the official pykan implementation
> also beats the dense static frontier in Track B and slightly exceeds the
> current PPO+MLP Excel-ReT anchor, although at higher shift-utilization cost.
> This supports the robustness of the bottleneck-aligned control result and
> motivates KAN as a response-letter or extension architecture, rather than
> changing the manuscript's main mechanism claim.
