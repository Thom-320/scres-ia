# CF20 Learning Repair Sprint Results (2026-06-29)

## Goal

Test whether PPO can learn useful dynamics once initialized near the CF20 low-buffer/S1 optimum, and compare PPO against a persistent-policy bandit/CEM baseline.

## Setup

Case: `CF20` (`R21/R22/R23/R24` all increased).

Action space: fine-discrete Track A:

```text
frac ∈ {0,0.01,0.025,0.05,0.075,0.10,0.125,0.15,0.20,0.25}
shift ∈ {S1,S2,S3}
```

Primary metric: Excel ReT.

Static optimum:

```text
f0.075_S1
Excel ReT = 0.268779
resource  = 0.0375
```

Action-application gate passed:

```text
requested f0.075_S1
applied continuous_its_frac  = 0.075
applied continuous_its_shift = 1
resource_composite           = 0.0375
```

## PPO Results

Artifact: `outputs/experiments/cf20_learning_repair_1seed_2026-06-29/summary.json`

| arm | Excel ReT | resource | learned action | verdict |
|---|---:|---:|---|---|
| PPO scratch, `ReT_excel_plus_cvar` | 0.216960 | 0.206 | mixed `f0.1_S3` / `f0_S1` | degraded |
| PPO warm-start `f0.075_S1`, `ReT_excel_plus_cvar` | 0.268779 | 0.0375 | `f0.075_S1` | maintains best static |
| PPO warm-start `f0.05_S1`, `ReT_excel_plus_cvar` | 0.261393 | 0.025 | `f0.05_S1` | below best static |
| PPO scratch, `ReT_excel_delta_bootstrap` | 0.216960 | 0.314 | mostly `f0.25_S2` | degraded |
| PPO warm-start `f0.075_S1`, `ReT_excel_delta_bootstrap` | 0.268779 | 0.0375 | `f0.075_S1` | maintains best static |
| PPO warm-start `f0.05_S1`, `ReT_excel_delta_bootstrap` | 0.261393 | 0.025 | `f0.05_S1` | below best static |
| PPO scratch, `ReT_excel_terminal_shaped` | 0.216960 | 0.562 | `f0.125_S3` | degraded |
| PPO warm-start `f0.075_S1`, `ReT_excel_terminal_shaped` | 0.268779 | 0.0375 | `f0.075_S1` | maintains best static |
| PPO warm-start `f0.05_S1`, `ReT_excel_terminal_shaped` | 0.261393 | 0.025 | `f0.05_S1` | below best static |

## CEM / Persistent-Policy Baseline

Artifact: `outputs/experiments/cf20_learning_repair_cem_only_2026-06-29/summary.json`

With enough samples, CEM finds the same static optimum:

```text
CEM best = f0.075_S1
Excel ReT = 0.268779
resource = 0.0375
```

## Diagnosis

The issue is not action application and not primarily reward blindness.

The issue is that CF20's optimum is a **persistent static needle**:

```text
hold f0.075_S1 for the whole episode
```

PPO from scratch explores weekly mixed action sequences. Those sequences do not provide clean credit for the persistent low-buffer/S1 policy. Reward variants that add future-credit shaping did not fix discovery from scratch.

Warm-start confirms PPO can maintain the optimum once initialized there, but does not improve beyond it.

## Decision

For CF20 Track A:

```text
recommended_claim = static_optimum_no_dynamic_headroom
```

Interpretation:

- There is no evidence of dynamic headroom over `f0.075_S1`.
- CEM/static search is the more honest baseline for this problem.
- PPO should not be used as a claim engine here unless a future environment introduces genuine time-varying headroom.

## Implication for Next Work

Do not scale CF20 PPO on Track A. If the goal is a positive learning result, move to:

1. H4 retained-vs-reset, or
2. Track B / downstream controls, or
3. a non-stationary campaign with verified static-oracle headroom.
