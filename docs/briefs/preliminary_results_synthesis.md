**Preliminary Results Synthesis**

This note consolidates the three benchmark lanes that already exist in the repository so the current bottleneck is explicit.

## 1. Executive read

| Lane | Setup | Main outcome | What it means |
| --- | --- | --- | --- |
| `PPO + ReT_thesis` | `500k` timesteps, `risk_level="current"`, `stochastic_pt=False` | PPO raises `reward_total` to `254.97`, but drops `fill_rate` to `0.845` and collapses to `99.99% S1` | The training objective is misaligned for control learning. |
| `PPO + control_v1` short scan | `50k` timesteps, `risk_level="increased"`, `3` seeds, weight sweep | PPO only wins in a narrow region, mainly `w_bo=5.0`, `w_cost=0.03` | The operational reward is better, but still weight-sensitive. |
| `PPO + control_v1` long run | `500k` timesteps, `stochastic_pt=True`, `5` seeds | Under `increased`, PPO is competitive but not better than best static; under `severe`, PPO becomes better than best static on reward | Adaptation shows value mainly under stronger stress. |

## 2. Evidence by lane

### A. `PPO + ReT_thesis` is not a good training lane

Source: `outputs/evaluations/ppo_shift_control_ret_thesis_formal/aggregate_metrics.json`

| Policy | Reward total mean | Fill rate | Backorder qty total | Mean ReT | Shift mix |
| --- | ---: | ---: | ---: | ---: | --- |
| `default` | `239.09` | `0.9915` | `35,868.61` | `0.97957` | `0.0% S1`, `100.0% S2`, `0.0% S3` |
| `random` | `239.19` | `0.9902` | `41,321.04` | `0.97968` | `33.76% S1`, `32.93% S2`, `33.31% S3` |
| `trained` | `254.97` | `0.8452` | `654,146.39` | `0.98067` | `99.99% S1`, `0.01% S2`, `0.0% S3` |

Interpretation:

- PPO improves the reported scalar reward, but does so by learning an operationally poor policy.
- Service deteriorates sharply while the policy collapses to the cheapest shift regime.
- `ReT_thesis` remains useful as a reporting metric, but these results argue against using it as the main training objective.

### B. `control_v1` short robustness scan fixes part of the problem, but not all of it

Source: `outputs/benchmarks/control_reward_local_robustness/summary.json`

Benchmark setup:

- `train_timesteps=50_000`
- `risk_level="increased"`
- `seeds=[11, 22, 33]`
- Weight sweep over `w_bo in {3, 4, 5}` and `w_cost in {0.01, 0.02, 0.03}`

Most informative rows from the comparison table:

| Weights | Best static | PPO reward | Best static reward | PPO fill rate | Best static fill rate | PPO wins? |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `w_bo=5.0`, `w_cost=0.01` | `static_s3` | `-268.42` | `-219.60` | `0.7908` | `0.8303` | `No` |
| `w_bo=5.0`, `w_cost=0.02` | `static_s2` | `-239.19` | `-223.17` | `0.8152` | `0.8274` | `No` |
| `w_bo=5.0`, `w_cost=0.03` | `static_s2` | `-219.80` | `-225.77` | `0.8304` | `0.8274` | `Yes` |

Interpretation:

- `control_v1` is clearly more usable than `ReT_thesis` for training.
- However, PPO does not improve uniformly across nearby reward weights.
- The short-run lane says the reward is directionally better, but still sensitive enough that small coefficient changes alter the conclusion.

### C. `control_v1` long runs are the current source of truth

Sources:

- `docs/artifacts/control_reward/control_reward_500k_increased_stopt/summary.json`
- `docs/artifacts/control_reward/control_reward_500k_severe_stopt/summary.json`
- `docs/artifacts/control_reward/control_reward_500k_seed_inference/seed_inference.md`
- `docs/manuscript_notes/control_reward_500k_source_of_truth.md`

Locked setup:

- `train_timesteps=500_000`
- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- `stochastic_pt=True`
- `5` seeds

#### `increased + stochastic_pt`

| Policy | Reward total mean | Fill rate | Backorder rate | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `ppo` | `-172.05` | `0.8379` | `0.1621` | `12.0% S1`, `24.6% S2`, `63.4% S3` |
| `static_s2` | `-170.10` | `0.8374` | `0.1626` | `0.0% S1`, `100.0% S2`, `0.0% S3` |
| `static_s3` | `-178.47` | `0.8344` | `0.1656` | `0.0% S1`, `0.0% S2`, `100.0% S3` |

Seed-level inference vs. best static (`static_s2`):

- Mean reward difference: `-1.948`
- Bootstrap CI95: `[-9.949, 8.514]`
- Exact sign-flip `p=0.812`

Interpretation:

- PPO matches service, but not reward.
- Under moderate stress, adaptive control is not yet clearly better than the best fixed policy.

#### `severe + stochastic_pt`

| Policy | Reward total mean | Fill rate | Backorder rate | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `ppo` | `-380.98` | `0.6314` | `0.3686` | `41.3% S1`, `26.8% S2`, `32.0% S3` |
| `static_s3` | `-385.59` | `0.6324` | `0.3676` | `0.0% S1`, `0.0% S2`, `100.0% S3` |
| `static_s2` | `-385.85` | `0.6271` | `0.3729` | `0.0% S1`, `100.0% S2`, `0.0% S3` |

Seed-level inference vs. best static (`static_s3`):

- Mean reward difference: `+4.608`
- Bootstrap CI95: `[-0.277, 9.493]`
- Exact sign-flip `p=0.188`

Interpretation:

- PPO becomes better than the best fixed baseline on reward while keeping service effectively comparable.
- The effect is still preliminary inferentially, but it is the clearest sign in the repo that adaptive switching matters under high stress.

## 3. What these results say about the bottleneck

The current evidence points to a three-part diagnosis:

1. The first bottleneck was reward alignment. `ReT_thesis` as a training objective pushed PPO toward a cheap, low-service solution.
2. After moving to `control_v1`, the next bottleneck became reward sensitivity. PPO can improve, but the conclusion depends on the exact service-cost tradeoff.
3. After longer training, the remaining bottleneck is regime dependence. PPO is not uniformly superior; the advantage emerges mainly when stress is severe enough that fixed policies become insufficient.

## 4. Practical conclusion

The repository does not support the claim that "PPO failed completely." A more precise reading is:

- `PPO + ReT_thesis` is the wrong lane for control learning.
- `PPO + control_v1` is the correct lane, but moderate-stress results are only competitive.
- The strongest existing adaptive-control evidence is the `500k`, `severe`, `stochastic_pt=True` benchmark.

That means the next experiments should be judged against this already-frozen baseline, not by restarting the reward discussion from zero.
