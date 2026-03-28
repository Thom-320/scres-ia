# Reward Function Design — SCRES+IA

## Overview

The repository now freezes `ReT_seq_v1` with `κ=0.20` as the primary training reward for the shift-control benchmark lane. This document records why that sequential resilience reward replaced `control_v1` as the default while still keeping `control_v1` as a historical comparator and `ReT_thesis` as an audit metric.

## The Problem: Reward Misalignment

The resilience metric from Garrido-Rios (2017), ReT, aggregates four sub-metrics:

- **Re(APj)** — Autotomy: how quickly the system detects a disruption (Eq. 5.1)
- **Re(RPj)** — Recovery: how quickly service resumes (Eq. 5.2)
- **Re(DPj, RPj)** — Non-recovery: penalty for prolonged disruption (Eq. 5.3)
- **Re(FRt)** — Fill rate: fraction of demand satisfied (Eq. 5.4)

These combine into a per-order resilience score (Eq. 5.5), which is then averaged across all orders in the simulation horizon.

### Why ReT Fails as a Training Objective

When used as the RL training reward, ReT creates a **cost-avoidance incentive** that dominates the service signal:

| Metric | ReT-trained agent | Expected behavior |
|--------|-------------------|-------------------|
| Shift allocation | 99.99% S1 | Mixed S1/S2/S3 |
| Fill rate | 0.845 | ≥ 0.83 |
| Control behavior | Cost minimization | Service-cost tradeoff |

The agent learns that running a single shift (S1) minimizes the cost component of ReT faster than the fill-rate penalty accumulates. This produces a **numerically high ReT score** but an **operationally poor policy**.

## The Frozen Solution: ReT_seq_v1

The selected reward is a sequential resilience index that reconciles Garrido's thesis metric with a trainable operational objective:

```
r_t = SC_t^0.60 × BC_t^0.25 × AE_t^0.15
```

Where:
- `SC_t = 1 - new_backorder_qty / new_demanded`
- `BC_t = 1 - min(1, pending_backorder_qty / cumulative_demanded_post_warmup)`
- `AE_t = 1 - κ(S_t - 1) / 2`
- `κ = 0.20`

The theoretical mapping is:
- `SC_t` → thesis Eq. 5.4 `Re(FRt)` as the step-level service-resilience term
- `BC_t` → sequential recovery proxy aligned with thesis Eq. 5.2
- `AE_t` → explicit cost-efficiency extension motivated by thesis Section 8.6.2
- geometric aggregation → reduced compensability between service, recovery, and efficiency dimensions

### Why κ = 0.20

The current paper trio favors `κ=0.20` as the pragmatic leader against `static_s2` on cross-mode comparable metrics:
- `κ=0.10` is too permissive toward `S3`
- `κ=0.20` yields the most defensible shift mix and best comparable service/resilience profile
- `κ=0.30` trends toward collapse-prone `S1` behavior and is not the repo default

## Historical Comparator: control_v1

The operational control reward was designed to directly penalize the two quantities the shift-control agent can influence:

```
r_t = -(w_bo × B_t/D_t + w_cost × (S_t - 1))
```

Where:
- `B_t / D_t` = step-level service loss (backorders / demand)
- `S_t - 1` = shift cost (0 for S1, 1 for S2, 2 for S3)
- `w_bo = 4.0` = service-loss weight
- `w_cost = 0.02` = shift-cost weight

### Weight Selection Rationale

The ratio `w_bo / w_cost = 200` encodes the operational priority:

> In a military food supply chain, failing to deliver rations to forward-deployed units is approximately 200× worse than the cost of running an additional shift.

This was validated empirically through a weight sweep (50k timesteps, 3 seeds, `increased` risk):
- At `w_bo=5.0, w_cost=0.03`: PPO first beats the best static baseline
- At `w_bo=3.0, w_cost=0.01`: PPO undertreats service loss
- The locked configuration `w_bo=4.0, w_cost=0.02` represents a robust middle ground

### Why w_disr = 0.0

A disruption penalty term `w_disr × disruption_fraction` was implemented but set to zero because:
1. Disruptions are **exogenous** — the agent cannot prevent them
2. Penalizing disruptions creates noise in the reward signal without improving policy quality
3. The agent already responds to disruptions indirectly through backorder penalties

## Empirical Validation

### Behavioral comparison

| Metric | ReT_thesis agent | control_v1 agent |
|--------|-----------------|-----------------|
| Shift mix (increased) | 99.99% S1 | 12% S1, 25% S2, 63% S3 |
| Fill rate (increased) | 0.845 | 0.838 |
| Adaptive behavior | None (collapsed) | Genuine shift switching |

### Performance under stress

| Scenario | control_v1 PPO vs best static | Interpretation |
|----------|-------------------------------|----------------|
| Increased | −1.95 (CI: [−9.95, +8.51]) | Competitive |
| Severe | +4.61 (CI: [−0.28, +9.49]) | Advantage emerges |

## Role of ReT in the Paper

`ReT_thesis` and `ret_thesis_corrected` are **retained as reporting and audit metrics** for two reasons:

1. It provides thesis-aligned comparison with Garrido-Rios (2017)
2. It captures the multi-dimensional resilience concept (autotomy, recovery, fill rate)

However, it is explicitly **not the training objective** because its structure incentivizes cost minimization over service maintenance when used for RL.

## Continuous Thesis Bridge: Cobb-Douglas

The repository also keeps a continuous Cobb-Douglas bridge for the thesis resilience logic:

```
ReT_cd_v1 = FR_t^0.70 × AT_t^0.30
```

Where:
- `FR_t = 1 - new_backorder_qty / new_demanded`
- `AT_t = 1 - disruption_fraction`

This lane is not the main paper contract, but it is the cleanest way to convert the piecewise thesis ReT into a smooth reward for PPO:
- `FR_t` preserves the thesis service-resilience signal from Eq. 5.4
- `AT_t` keeps the disruption-state effect in the reward rather than in a case split
- weighted geometric aggregation preserves non-compensability while removing the original discontinuities

### Why the raw Cobb-Douglas form is preferred

For this repo, the bounded raw product is better than adding a sigmoid:

```
ReT_cd_sigmoid = σ(0.70 ln(FR_t) + 0.30 ln(AT_t))
```

Because `FR_t` and `AT_t` already live in `(0, 1]`, the log score is always `≤ 0`. That means the sigmoid output is always `≤ 0.5`, even in the best case (`FR_t = 1`, `AT_t = 1`). In practice:
- `ReT_cd_v1` keeps the natural `(0, 1]` range of the bounded geometric mean
- `ReT_cd_sigmoid` compresses the usable reward scale and weakens the learning signal

### Repo recommendation

- Keep `ReT_seq_v1` as the primary paper-facing reward because it aligns better with controllable shift decisions and explicit efficiency tradeoffs.
- Keep `ReT_cd_v1` as the thesis-to-continuous ablation and methodological bridge.
- Keep `ReT_cd_sigmoid` only as a documented comparison showing why the sigmoid wrapper is not the right default for this normalized DES setting.

## Future Extensions

### PBRS (Potential-Based Reward Shaping)

An optional PBRS extension (Ng et al., 1999) is implemented as `control_v1_pbrs`:

```
r_shaped = r_base + γ × Φ(s') − Φ(s)
Φ(s) = α × fill_rate − β × backorder_rate
```

This preserves the optimal policy while injecting service-quality awareness into step-level rewards. It is treated as a phase-2 methodological upgrade, not as a blocker for the main paper.

## References

- De Moor, B. J., Gijsbrechts, J., & Boute, R. N. (2022). Reward shaping to improve the performance of deep reinforcement learning in perishable inventory management. *European Journal of Operational Research*, 301(2), 535–545.
- Ng, A. Y., Harada, D., & Russell, S. (1999). Policy invariance under reward transformations. *Proceedings of the 16th International Conference on Machine Learning*, 278–287.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.
- Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.
