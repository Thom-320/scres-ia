# Real-KAN 10-Seed Extension Verdict — 2026-07-03

## Verdict

Real-KAN is confirmed as a strong Track B architecture sidecar, but it does not
replace the paper's PPO+MLP spine.

The 10-seed merge clears the ReT bar cleanly: Real-KAN beats the best static
comparator in all 10 seeds and beats the canonical PPO+MLP Excel-ReT anchor in
all 10 matched seeds. The cost caveat is also clean: Real-KAN uses substantially
more shift/dispatch capacity than PPO+MLP.

## Artifacts

- Seeds 1--5:
  `outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`
- Seeds 6--10:
  `outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_10seed_extension_6_10_60k_h104/`
- Canonical PPO+MLP 10-seed anchor:
  `docs/track_b_q1_stats_2026-07-02_final_10seed/ppo_episode_metrics_10seed.csv`

## Protocol Check

Both Real-KAN bundles match except for the seed block:

- Algorithm/feature extractor: PPO with official `pykan` `KAN` via
  `RealKANFeaturesExtractor`
- Seeds: `1..5` and `6..10`
- Train timesteps: `60000`
- Eval episodes: `12`
- Reward: `control_v1`
- Risk: `adaptive_benchmark_v2`
- Observation: `v7`
- Action contract: `track_b_v1`
- Horizon: `max_steps=104`
- Raw-material flow mode: requested `kit_equivalent_order_up_to`, canonical
  `bom_total_units_order_up_to`
- Optimizer settings: learning rate `3e-4`, `n_steps=1024`, batch size `256`

## 10-Seed Metrics

Merged over seeds 1--10:

| Contrast | Real-KAN | Comparator | Delta | Seed signs |
|---|---:|---:|---:|---:|
| Excel ReT vs best static | 0.005938 | 0.005440 | +0.000499 | 10/10 |
| Order-level ReT vs best static | 0.005713 | 0.005227 | +0.000486 | 10/10 |
| Excel ReT vs canonical PPO+MLP | 0.005938 | 0.005898 | +0.000041 | 10/10 |

Seed-clustered CI95:

- Excel ReT delta vs best static: `[+0.000483, +0.000515]`
- Order-level ReT delta vs best static: `[+0.000471, +0.000502]`
- Excel ReT delta vs canonical PPO+MLP: `[+0.000025, +0.000057]`

Seed-level Excel ReT comparison vs canonical PPO+MLP:

| Seed | Real-KAN | PPO+MLP | Delta |
|---:|---:|---:|---:|
| 1 | 0.005940 | 0.005921 | +0.000019 |
| 2 | 0.005917 | 0.005906 | +0.000011 |
| 3 | 0.005933 | 0.005920 | +0.000013 |
| 4 | 0.005933 | 0.005862 | +0.000071 |
| 5 | 0.005908 | 0.005855 | +0.000053 |
| 6 | 0.005960 | 0.005880 | +0.000080 |
| 7 | 0.005932 | 0.005908 | +0.000024 |
| 8 | 0.005963 | 0.005891 | +0.000072 |
| 9 | 0.005959 | 0.005924 | +0.000035 |
| 10 | 0.005939 | 0.005911 | +0.000029 |

## Cost Caveat

Real-KAN's mean shift-utilization cost index is `0.982`, versus `0.667` for the
best static comparator and approximately `0.68` for canonical PPO+MLP. The ReT
lift is therefore real, but it is bought with a much more aggressive action
profile.

## Manuscript Implication

This is a direct response to Garrido's KAN concern: a real, official KAN feature
extractor can run inside PPO and can slightly improve Excel ReT under the Track
B contract.

It should not become the Paper 1 headline unless the manuscript is reframed
around an explicit cost/benefit tradeoff for architecture. The current Q1 spine
remains stronger as:

> Conservative PPO+MLP already establishes the bottleneck-aligned control
> mechanism; Real-KAN is a confirmed architecture sidecar that improves ReT
> modestly at materially higher utilization cost.
