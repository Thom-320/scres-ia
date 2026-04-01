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

Source: historical local robustness scan under `control_v1` (pre-audit benchmark family).

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

### C. `control_v1` long runs are now a comparator lane, not the primary source of truth

Sources:

- `outputs/paper_benchmarks/paper_control_v1_500k/summary.json`
- `outputs/paper_benchmarks/paper_ret_seq_k020_500k/summary.json`
- `outputs/paper_benchmarks/paper_ret_seq_k010_500k/summary.json`
- `docs/manuscript_notes/control_reward_500k_source_of_truth.md`

Locked setup:

- `train_timesteps=500_000`
- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- `stochastic_pt=True`
- `5` seeds

#### `increased + stochastic_pt` (`control_v1`, auditable comparator)

| Policy | Reward total mean | Fill rate | Backorder rate | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `ppo` | `-629.37` | `0.7820` | `0.2180` | `45.5% S1`, `27.8% S2`, `26.6% S3` |
| `static_s2` | `-617.98` | `0.7924` | `0.2076` | `0.0% S1`, `100.0% S2`, `0.0% S3` |

Interpretation:

- In the current auditable codebase, `control_v1` PPO is competitive but not better than `static_s2`.
- This lane remains useful as an operational comparator, not as the current repo winner.

#### Current winner under the post-audit paper-facing family

`ReT_seq_v1, κ=0.20` is the current leading auditable lane:

| Policy | Reward total mean | Fill rate | Backorder rate | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `ppo` | `133.08` | `0.7883` | `0.2117` | `72.9% S1`, `13.4% S2`, `13.7% S3` |
| `static_s2` | `132.53` | `0.7922` | `0.2078` | `0.0% S1`, `100.0% S2`, `0.0% S3` |

Interpretation:

- `ReT_seq_v1` with `κ=0.20` beats `control_v1` on the cross-mode comparable metrics inside the current auditable paper family.
- The lane is still best described as competitive rather than conclusively dominant.

## 3. What these results say about the bottleneck

The current evidence points to a three-part diagnosis:

1. The first bottleneck was reward alignment. `ReT_thesis` as a training objective pushed PPO toward a cheap, low-service solution.
2. `control_v1` remains useful as an operational comparator, but the post-audit codebase does not support it as the leading lane.
3. The remaining bottleneck is matched-family evaluation discipline: the repo still mixes historical bundles, current paper-facing bundles, and an unfinished `RecurrentPPO` lane.

## 4. Practical conclusion

The repository does not support the claim that "PPO failed completely." A more precise reading is:

- `PPO + ReT_thesis` is the wrong lane for control learning.
- `control_v1` is now best treated as an operational comparator lane.
- The strongest current auditable paper-facing evidence is the `ReT_seq_v1, κ=0.20` family under `v1 + thesis + stochastic_pt=True`.

That means the next experiments should be judged against the current auditable paper-facing bundles, not against the historical `*_stopt` artifacts.
