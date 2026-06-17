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

## Full-Panel Gate

Command:

```bash
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
.venv/bin/python scripts/reward_surface_audit.py \
  --rewards ReT_tail_v1 \
  --profiles increased,severe \
  --panel-cfis 31-90 \
  --policy-set with_crossed \
  --replications 1 \
  --ret-tail-transform-grid identity,power:1.25,power:1.5,power:2.0,exp_norm:2,exp_norm:4 \
  --output-root outputs/benchmarks/reward_surface_audit_track_a_exhaustion_20260617 \
  --force
```

Output:

`outputs/benchmarks/reward_surface_audit_track_a_exhaustion_20260617/reward_surface_summary.csv`

All six variants passed the static surface gate on the full `Cf31-90` panel.
The best-by-reward static policy was `crossed_uniform_I168_S1` for every
variant/profile pair.  It ranked:

- `increased`: 3rd by `ret_p10_all`, with p10 `0.6834` vs top p10 `0.7011`.
- `severe`: tied 1st by `ret_p10_all`, with p10 `0.6667`.

Correlation summary:

| variant | profile | rho p10 | rho flow | rho stockout | pass |
|---|---|---:|---:|---:|---|
| identity | increased | 0.671 | 0.573 | -0.636 | yes |
| identity | severe | 0.786 | 0.641 | -0.523 | yes |
| power 1.25 | increased | 0.671 | 0.573 | -0.636 | yes |
| power 1.25 | severe | 0.786 | 0.630 | -0.518 | yes |
| power 1.5 | increased | 0.660 | 0.562 | -0.624 | yes |
| power 1.5 | severe | 0.786 | 0.630 | -0.518 | yes |
| power 2.0 | increased | 0.663 | 0.552 | -0.619 | yes |
| power 2.0 | severe | 0.768 | 0.619 | -0.496 | yes |
| exp_norm 2 | increased | 0.647 | 0.533 | -0.607 | yes |
| exp_norm 2 | severe | 0.768 | 0.619 | -0.496 | yes |
| exp_norm 4 | increased | 0.623 | 0.505 | -0.585 | yes |
| exp_norm 4 | severe | 0.707 | 0.577 | -0.439 | yes |

`exp_norm`, especially `beta=4`, still passes but degrades correlations with
flow fill, mean all-order ReT, and stockout relative to `identity` and low-power
variants.

## Recommendation

- Keep `identity` as the default.
- Use `power:1.25` as the first serious steepness scout; it preserves the
  full-panel surface almost exactly while making the bounded reward slightly
  steeper for PPO.
- Use `power:1.5` as the second scout if `power:1.25` is flat.
- Keep `power:2.0` as an aggressive ablation.
- Treat `exp_norm` as secondary only; do not promote it unless it wins on
  held-out external metrics.
