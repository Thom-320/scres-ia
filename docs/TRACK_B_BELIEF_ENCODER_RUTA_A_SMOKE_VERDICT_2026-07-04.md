# Track B Ruta A belief-encoder smoke verdict — 2026-07-04

## Objective

Test whether the stronger "Ruta A" belief design produces preventive learning:

```text
full v10 observation -> pretrained R24 belief encoder -> PPO features_extractor -> PPO fine-tuning
```

This is distinct from the earlier scalar sidecar, which appended two frozen
probabilities to the observation. Ruta A transfers a pretrained representation
into the policy network itself.

Primary metric remains Garrido Excel ReT: `order_ret_excel_mean`.

## Pretraining signal

The encoders were pretrained on full v10 observations (`101` dimensions) with
future R24 labels:

| Encoder | Held-out AUC R24 1w | Held-out AUC R24 2w |
|---|---:|---:|
| MLP belief encoder | 0.7558 | 0.7672 |
| Real-KAN belief encoder | 0.7599 | 0.7724 |

This is a real supervised signal: much stronger than the earlier logistic
belief head over a small curated feature subset.

## PPO+MLP Ruta A result

Artifact:

```text
outputs/experiments/track_b_belief_encoder_ppo_3seed_30k_2026-07-04/
```

Protocol:

- architecture: `ppo_mlp_belief_encoder`
- observation: `v10`
- action contract: `track_b_v1`
- reward: `control_v1`
- risk profile: `adaptive_benchmark_v2`
- seeds: 1, 2, 3
- training: 30k timesteps
- evaluation: 8 episodes per seed
- horizon: 104 weeks
- PPO hyperparameters: `n_steps=1024`, `batch_size=64`

| Lane | ReT Excel | CI95 | Cost index | Reading |
|---|---:|---:|---:|---|
| PPO+MLP v10 raw memory, 30k | 0.005811 | [0.005720, 0.005902] | 0.763 | raw memory alone |
| PPO+MLP scalar belief balanced, 30k | 0.005743 | [0.005676, 0.005809] | 0.640 | scalar belief hurts ReT |
| PPO+MLP scalar belief unweighted, 30k | 0.005652 | [0.005519, 0.005784] | 0.669 | calibrated scalar belief hurts more |
| PPO+MLP scalar belief balanced bs64, 30k | 0.005799 | [0.005710, 0.005889] | 0.747 | protocol-corrected scalar belief still below v10/v7 |
| PPO+MLP scalar belief unweighted bs64, 30k | 0.005820 | [0.005734, 0.005906] | 0.689 | slight v10 gain, still below v7 and Ruta A |
| PPO+MLP v7 calibration, 30k | 0.005843 | [0.005836, 0.005850] | 0.734 | budget-matched v7 control |
| **PPO+MLP Ruta A belief encoder, 30k** | **0.005902** | **[0.005865, 0.005940]** | **0.747** | training signal |

### Interpretation

Ruta A rescues PPO+MLP from the failure of the scalar belief sidecar. This
remains true after correcting the scalar-belief PPO runs to the canonical
`batch_size=64`: the bs64 scalar sidecars improve over the earlier
`batch_size=256` scalar runs, but they do not reach the v7 calibration baseline
and do not approach Ruta A.

At the same 30k smoke budget, Ruta A beats:

- raw v10 memory,
- scalar belief balanced,
- scalar belief unweighted,
- scalar belief balanced/unweighted with corrected PPO batch size,
- and the v7 30k calibration control.

So the correct training conclusion is:

> The R24 belief representation can help PPO+MLP when it is inserted as a
> pretrained shared encoder, not merely appended as two frozen probabilities.

This is a training/performance signal, not yet a prevention claim.

## PPO+MLP Ruta A prevention audit

Artifact:

```text
outputs/experiments/track_b_belief_encoder_ppo_counterfactual_r24_2026-07-04/
```

Fixed-RNG counterfactual:

```text
R_full - R_reset(pre-R24)
```

with the same seeds/eval episodes as the smoke run and R24 event anchors.

| Policy | Risk | Pairs | Positive pairs | Positive pair rate | Mean R full | Mean R reset | Mean delta ReT Excel |
|---|---|---:|---:|---:|---:|---:|---:|
| PPO+MLP Ruta A | R24 | 158 | 14/158 | 8.9% | 0.005902 | 0.005900 | +0.00000153 |

### Interpretation

This does **not** establish preventive learning.

The mean delta is slightly positive, but the positive-pair rate is only 8.9%.
That means most event-level counterfactual comparisons do not show value from
the pre-R24 action window. Under the agreed gate, this is not robust prevention.

The more defensible reading is:

> PPO+MLP Ruta A improves ReT Excel at smoke scale, but the improvement is not
> yet explained by consistent preventive action before R24 events.

## Current status

| Lane | Status |
|---|---|
| PPO+MLP scalar belief | Negative/weak for PPO after batch-size correction; no counterfactual escalation |
| PPO+MLP Ruta A | Positive training signal; no robust R24 prevention in counterfactual |
| Real-KAN scalar belief | Positive training signal; scalar-sidecar counterfactual did not show prevention |
| Real-KAN Ruta A | Small training signal vs raw v10; no robust R24 prevention in counterfactual |

## Stop/continue rule

Continue only if a lane shows both:

1. a clear ReT Excel training signal relative to its architecture-matched
   baseline, and
2. a counterfactual prevention signal with positive mean delta and a non-tiny
   positive-pair rate.

PPO+MLP Ruta A satisfies (1) but fails (2). It should not be escalated tonight.

## Real-KAN Ruta A result

Artifact:

```text
outputs/experiments/track_b_belief_encoder_real_kan_3seed_30k_2026-07-04_v4/
```

Protocol:

- architecture: `real_kan_belief_encoder`
- observation: `v10`
- action contract: `track_b_v1`
- reward: `control_v1`
- risk profile: `adaptive_benchmark_v2`
- seeds: 1, 2, 3
- training: 30k timesteps
- evaluation: 8 episodes per seed
- horizon: 104 weeks
- PPO/KAN hyperparameters: `n_steps=1024`, `batch_size=256`

| Lane | ReT Excel | CI95 | Cost index | Reading |
|---|---:|---:|---:|---|
| Real-KAN v10 raw memory, 30k | 0.005915 | [0.005906, 0.005924] | 0.982 | raw memory KAN |
| Real-KAN scalar belief balanced, 30k | 0.005935 | [0.005929, 0.005941] | 0.899 | scalar belief strongest cost-aware KAN |
| Real-KAN scalar belief unweighted, 30k | 0.005937 | [0.005932, 0.005943] | 1.000 | scalar belief max-ReT KAN |
| **Real-KAN Ruta A belief encoder, 30k** | **0.005924** | **[0.005894, 0.005954]** | **0.900** | small signal vs raw v10, weaker than scalar belief |
| Real-KAN fixed-RNG v7, 60k | 0.005946 | [0.005941, 0.005951] | 1.000 | larger-budget KAN reference |

Seed means for Real-KAN Ruta A:

| Seed | ReT Excel |
|---:|---:|
| 1 | 0.005945 |
| 2 | 0.005895 |
| 3 | 0.005933 |

### Interpretation

Real-KAN Ruta A has a small training signal relative to raw v10 Real-KAN and a
materially lower cost index. However, it does not beat the simpler scalar-belief
Real-KAN sidecar and does not reach the fixed-RNG v7 60k Real-KAN reference.

Because it did improve over raw v10, it earned the agreed counterfactual test.

## Real-KAN Ruta A prevention audit

Artifact:

```text
outputs/experiments/track_b_belief_encoder_real_kan_counterfactual_r24_2026-07-04/
```

Fixed-RNG counterfactual:

```text
R_full - R_reset(pre-R24)
```

| Policy | Risk | Pairs | Positive pairs | Positive pair rate | Mean R full | Mean R reset | Mean delta ReT Excel |
|---|---|---:|---:|---:|---:|---:|---:|
| Real-KAN Ruta A | R24 | 158 | 1/158 | 0.6% | 0.005924 | 0.005925 | -0.00000103 |

### Interpretation

This is a clear negative for prevention. The pre-R24 learned actions reduce
ReT on average under the counterfactual, and only one of 158 event-level pairs
is positive.

## Final overnight verdict

Across the tested preventive-learning lanes:

| Lane | Training/ReT signal | R24 prevention signal |
|---|---|---|
| PPO+MLP scalar belief | no sufficient signal after bs64 correction | not escalated |
| Real-KAN scalar belief | yes, especially for KAN | no prevention |
| PPO+MLP Ruta A belief encoder | yes | no robust prevention |
| Real-KAN Ruta A belief encoder | small yes vs raw v10 | no prevention |

Therefore the honest conclusion is:

> Track B policies can use risk-belief representations to improve Garrido Excel
> ReT in some architectures, especially KAN and Ruta A PPO, but the overnight
> fixed-RNG counterfactuals do **not** show robust preventive learning before
> R24 events.

No current lane has enough preventive signal to justify more overnight compute.
Further work should be a new design, not another blind rerun of the same
belief-sidecar/Ruta-A variants.
