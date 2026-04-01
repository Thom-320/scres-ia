**Preliminary Results Synthesis**

This note consolidates the benchmark lanes that now define the paper story after the Track B long run completed.

## 1. Executive read

| Lane | Setup | Main outcome | What it means |
| --- | --- | --- | --- |
| Track A thesis-faithful | `v1/v4`, upstream control only, post-audit bundles | PPO and RecurrentPPO do not beat strong static baselines | RL failure is structural, not just an algorithm choice. |
| Track B smoke | `100k`, `3` seeds, `v7 + track_b_v1` | PPO beats all static baselines decisively | Minimal downstream control opens real adaptive headroom. |
| Track B long run | `500k`, `5` seeds, `v7 + track_b_v1` | PPO still beats `s2_d1.00` and `s3_d2.00` decisively | The positive Track B result is stable enough for the main paper claim. |

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

### B. Track A remains a valid negative baseline family

Sources:

- `outputs/paper_benchmarks/paper_control_v1_500k/summary.json`
- `outputs/paper_benchmarks/paper_ret_seq_k020_500k/summary.json`
- `outputs/benchmarks/final_recurrent_ppo_v4_control_500k/summary.json`

Representative readings:

| Lane | Policy | Fill rate | Interpretation |
| --- | --- | ---: | --- |
| `paper_control_v1_500k` | PPO | `0.7820` | Worse than `static_s2` |
| `paper_ret_seq_k020_500k` | PPO | `0.7883` | Competitive, but still below `static_s2` |
| `final_recurrent_ppo_v4_control_500k` | RecurrentPPO | `0.7514` | Memory does not rescue Track A |

Interpretation:

- Track A is no longer a “which reward wins?” story.
- It is now the negative result family showing that upstream-only control does not create enough leverage for RL to beat strong static baselines.

### C. Track B changes the conclusion

Sources:

- `outputs/benchmarks/track_b_smoke_initial_2026-03-31/summary.json`
- `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1/summary.json`
- `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1/comparison_table.csv`

Track B smoke (`100k x 3`):

| Policy | Reward total mean | Fill rate | Order-level ReT |
| --- | ---: | ---: | ---: |
| `s2_d1.00` | `177.78` | `0.9649` | `0.4952` |
| `s3_d2.00` | `170.34` | `0.9869` | `0.4544` |
| `ppo` | `250.17` | `0.99996` | `0.9268` |

Track B long run (`500k x 5`):

| Policy | Reward total mean | Fill rate | Order-level ReT | Shift mix |
| --- | ---: | ---: | ---: | --- |
| `s2_d1.00` | `178.24` | `0.9659` | `0.4886` | `0/100/0` |
| `s3_d2.00` | `171.06` | `0.9876` | `0.4584` | `0/0/100` |
| `ppo` | `254.21` | `0.99996` | `0.9503` | `77.8/15.7/6.5` |

Interpretation:

- The smoke was not a fluke.
- PPO remains above the best static policy at `500k x 5`.
- The positive result is large enough that Track B, not Track A, is now the main paper lane.

## 3. What these results say about the bottleneck

The current evidence now supports a four-part diagnosis:

1. The first bottleneck was reward alignment. `ReT_thesis` as a training objective pushed PPO toward a cheap, low-service solution.
2. `control_v1` remains useful as an operational comparator, but it does not rescue Track A.
3. The structural bottleneck was the action/control contract: Track A controls upstream capacity while the active bottleneck is downstream.
4. Once downstream control is exposed in Track B, PPO learns a genuinely superior policy.

## 4. Practical conclusion

The repository no longer supports a purely negative story. The precise reading is:

- `PPO + ReT_thesis` is the wrong lane for control learning.
- Track A is a valid negative benchmark family and mechanistic diagnosis.
- Track B is the strongest current auditable paper-facing evidence.
- The strongest current evidence is `ReT_seq_v1, κ=0.20` under `v7 + track_b_v1 + adaptive_benchmark_v2`.

That means the paper should be written around:

- Track A as the thesis-faithful failure mode,
- Track B as the minimal MDP repair,
- and the resulting PPO gain against strong static baselines.
