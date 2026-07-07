# Track B risk-belief Sprint 1 status — 2026-07-04

## Decision

The preventive-learning lane should test **both**:

- `PPO+MLP-belief`, because PPO+MLP is the efficient spine.
- `PPO+RealKAN-belief`, because Real-KAN is the strongest architecture sidecar and offers spline interpretability.

This is the fair answer to the question: if PPO+MLP is more efficient, why not give it the same historical-risk memory before claiming KAN is necessary?

## Fixed-RNG counterfactual baseline

The fixed-RNG pre-risk counterfactual finished for PPO+MLP and Real-KAN. Excluding R14 from the causal reading because it depends on production, the non-R14 aggregate is:

| Policy | Weighted pre-risk delta | Positive pairs |
|---|---:|---:|
| PPO+MLP | +0.00000490 | 158/1046 |
| Real-KAN | -0.00000033 | 54/1046 |

Reading:

- PPO+MLP has a small pre-risk signal.
- Real-KAN does not show causal prevention in its current form.
- Real-KAN's ReT advantage appears to come mostly from high-capacity operation, not fine anticipatory timing.

## Supervised risk-belief smoke

Implemented:

`scripts/audit_track_b_risk_belief_predictor.py`

This builds observed-history features and labels:

- `mem_weeks_since_last_Ri`
- `mem_count_Ri_8w`
- `mem_count_Ri_26w`
- `mem_ewma_Ri_8w`

Targets:

- risk `Ri` starts in the next 1/2/4/8 weeks.

Key result:

- R11/R13 are often too saturated in short horizons.
- R24 at 1-2 weeks is the cleanest initial target.
- `memory_only` reaches AUC about `0.619` for R24 at 1 week, without forecast/regime features.

This is enough signal to justify a small RL smoke with memory features.

## Observation contract added

Added experimental observation version:

`v10 = v9 + observed historical memory for R11/R13/R24`

Fields added:

- `mem_weeks_since_last_R11`, `mem_count_R11_8w`, `mem_count_R11_26w`, `mem_ewma_R11_8w`
- `mem_weeks_since_last_R13`, `mem_count_R13_8w`, `mem_count_R13_26w`, `mem_ewma_R13_8w`
- `mem_weeks_since_last_R24`, `mem_count_R24_8w`, `mem_count_R24_26w`, `mem_ewma_R24_8w`

This is historical, observed information only. It is not a forecast oracle.

Local smoke tests passed for:

- PPO+MLP with `observation_version=v10`
- Real-KAN with `observation_version=v10`

## VPS runs launched

Two 3-seed x 30k v10 risk-memory smokes are running on the VPS:

- `outputs/experiments/track_b_v10_risk_memory_ppo_3seed_30k_2026-07-04/`
- `outputs/experiments/track_b_v10_risk_memory_real_kan_3seed_30k_2026-07-04/`

Watcher:

`watch-track-b-v10-risk-memory-smoke`

## Gate

Promote only if:

1. `order_ret_excel_mean` improves or does not materially degrade versus the fixed-RNG v7 counterpart.
2. The improvement is not purely a cost explosion.
3. A follow-up memory ablation (`memory_zeroed` or `memory_scrambled`) degrades the policy.
4. The event-aligned audit shows pre-risk action for frequent risks, especially R24.

Until then, the claim remains:

> We have adaptive learning. Preventive learning is an active experimental lane, not yet a settled claim.
