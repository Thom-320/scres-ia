# Track B Risk-Belief Sidecar Smoke Verdict — 2026-07-04

## What was tested

This smoke tested the first implementation of the preventive-belief idea:

> observed risk memory + frozen R24 belief head + PPO policy, evaluated with
> Garrido Excel ReT.

The belief head was trained from observed historical memory only and appended
two probabilities to the `v10` observation:

- `P(R24 starts within 1 week)`
- `P(R24 starts within 2 weeks)`

Effective observation dimension:

```text
v10 = 101
belief = 2
total = 103
```

Protocol:

- seeds: 1, 2, 3
- training: 30k timesteps per seed
- evaluation: 8 episodes per seed
- horizon: 104 weeks
- action contract: `track_b_v1`
- reward: `control_v1`
- risk profile: `adaptive_benchmark_v2`
- primary reported metric: `order_ret_excel_mean`

Artifacts:

```text
outputs/experiments/track_b_risk_belief_ppo_3seed_30k_2026-07-04
outputs/experiments/track_b_risk_belief_real_kan_3seed_30k_2026-07-04
outputs/experiments/track_b_risk_belief_ppo_unweighted_3seed_30k_2026-07-04
outputs/experiments/track_b_risk_belief_real_kan_unweighted_3seed_30k_2026-07-04
```

## Result

| Lane | ReT Excel | CI95 | Cost index | Reading |
|---|---:|---:|---:|---|
| PPO+MLP v7, 30k calibration | 0.005843 | [0.005836, 0.005850] | 0.734 | budget-matched v7 control |
| PPO+MLP v10 raw memory, 30k | 0.005811 | [0.005720, 0.005902] | 0.763 | raw memory alone |
| PPO+MLP-belief balanced, 30k | 0.005743 | [0.005676, 0.005809] | 0.640 | belief lowers cost, hurts ReT |
| PPO+MLP-belief unweighted, 30k | 0.005652 | [0.005519, 0.005784] | 0.669 | calibrated belief hurts more |
| Real-KAN v10 raw memory, 30k | 0.005915 | [0.005906, 0.005924] | 0.982 | raw memory KAN |
| Real-KAN-belief balanced, 30k | 0.005935 | [0.005929, 0.005941] | 0.899 | belief improves KAN and lowers cost |
| Real-KAN-belief unweighted, 30k | 0.005937 | [0.005932, 0.005943] | 1.000 | calibrated belief improves ReT, not cost |
| PPO+MLP v7, 60k confirmatory | 0.005921 | [0.005906, 0.005936] | 0.700 | larger-budget paper reference |
| Real-KAN v7, 60k confirmatory | 0.005946 | [0.005941, 0.005951] | 1.000 | larger-budget KAN reference |

## Paired seed signs

PPO+MLP-belief balanced vs PPO+MLP v10 raw memory:

| Seed | v10 raw | belief | delta |
|---:|---:|---:|---:|
| 1 | 0.005797 | 0.005776 | -0.000021 |
| 2 | 0.005738 | 0.005675 | -0.000063 |
| 3 | 0.005897 | 0.005777 | -0.000120 |

PPO+MLP-belief unweighted vs PPO+MLP v10 raw memory:

| Seed | v10 raw | belief | delta |
|---:|---:|---:|---:|
| 1 | 0.005797 | 0.005560 | -0.000236 |
| 2 | 0.005738 | 0.005611 | -0.000127 |
| 3 | 0.005897 | 0.005784 | -0.000114 |

Real-KAN-belief balanced vs Real-KAN v10 raw memory:

| Seed | v10 raw | belief | delta |
|---:|---:|---:|---:|
| 1 | 0.005914 | 0.005929 | +0.000015 |
| 2 | 0.005907 | 0.005938 | +0.000031 |
| 3 | 0.005923 | 0.005938 | +0.000015 |

Real-KAN-belief unweighted vs Real-KAN v10 raw memory:

| Seed | v10 raw | belief | delta |
|---:|---:|---:|---:|
| 1 | 0.005914 | 0.005932 | +0.000018 |
| 2 | 0.005907 | 0.005937 | +0.000030 |
| 3 | 0.005923 | 0.005943 | +0.000020 |

## Interpretation

The result is asymmetric:

- For **PPO+MLP**, the frozen belief scalars do not help. They reduce resource
  use in the balanced run, but ReT Excel drops in all three seeds. The
  unweighted/calibrated run drops even further, so the PPO negative is not
  explained away by calibration.
- For **Real-KAN**, the same belief scalars help: ReT Excel improves in all
  three seeds under both balanced and unweighted heads. The balanced head also
  lowers cost materially from 0.982 to 0.899; the unweighted head maximizes ReT
  slightly more but returns to near-maximal cost.

This is the first evidence that the risk-belief idea may be useful specifically
when paired with KAN's nonlinear feature representation. It is not yet evidence
of prevention. It is evidence that adding an explicit learned R24 belief can
improve the KAN policy's ReT at smoke scale. The cost tradeoff depends on the
belief-head calibration.

## Calibration caveat

The current belief head uses `LogisticRegression(class_weight="balanced")`.
That was useful for discrimination, but it distorts probability calibration:

| Target | Base rate | Mean predicted probability |
|---|---:|---:|
| R24 within 1 week | 0.342 | 0.492 |
| R24 within 2 weeks | 0.627 | 0.505 |

The unweighted rerun fixed this calibration issue:

| Target | Base rate | Mean predicted probability |
|---|---:|---:|
| R24 within 1 week | 0.342 | 0.342 |
| R24 within 2 weeks | 0.627 | 0.627 |

That did **not** rescue PPO+MLP. It did keep Real-KAN positive, with slightly
higher ReT but no cost reduction. Therefore the PPO negative is probably not
just a calibration artifact. The KAN positive is more robust than the original
balanced run, because it survives after fixing probability calibration.

## Verdict

Do not promote the belief sidecar as a paper result yet.

But do continue the lane:

- `PPO+MLP-belief`: negative at smoke scale under both balanced and calibrated
  belief heads.
- `Real-KAN-belief`: promising at smoke scale. It improves ReT in 3/3 seeds
  under both heads. Balanced gives the best ReT/cost tradeoff; unweighted gives
  the highest ReT.

Next step: if we continue this lane, scale **Real-KAN-belief balanced** first
to 5 seeds × 60k as the cost-aware candidate, and optionally keep
Real-KAN-belief unweighted as the max-ReT candidate. Do not scale PPO+MLP-belief
unless the design changes from appended frozen probabilities to a true shared
auxiliary representation/loss.
