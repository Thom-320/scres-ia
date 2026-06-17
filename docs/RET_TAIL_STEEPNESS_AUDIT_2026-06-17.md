# ReT_tail_v1 Steepness Audit - 2026-06-17

## Purpose

David suggested making the reward steeper.  `ReT_tail_v1` already uses a
Cobb-Douglas log-linear form, so the tested steepness variants are monotone
post-transforms of the bounded base reward, not raw `exp(R)`.

## Variants

- `identity`: `R = R_base`
- `power`: `R = R_base^gamma`, `gamma in {1.25, 1.5, 2.0}`
- `exp_norm`: `R = (exp(beta * R_base) - 1) / (exp(beta) - 1)`,
  `beta in {2, 4}`

## Smoke Gate

Command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
.venv/bin/python scripts/reward_surface_audit.py \
  --rewards ReT_tail_v1 \
  --profiles increased,severe \
  --panel-cfis 31,51,71 \
  --policy-set with_crossed \
  --replications 1 \
  --ret-tail-transform-grid identity,power:1.25,power:1.5,power:2.0,exp_norm:2,exp_norm:4 \
  --output-root /tmp/scresia_reward_steepness_audit_panel3 \
  --force
```

All six variants passed the static surface gate on this 3-Cf smoke.  However,
`exp_norm`, especially `beta=4`, degraded correlation with `flow_fill_rate` and
mean all-order ReT relative to `identity` and the low-power variants.

## Recommendation

- Keep `identity` as the default.
- Use `power:1.25` and `power:1.5` as the first serious steepness ablations.
- Keep `power:2.0` as an aggressive ablation.
- Treat `exp_norm` as secondary only; do not promote it unless it wins on
  held-out external metrics.
