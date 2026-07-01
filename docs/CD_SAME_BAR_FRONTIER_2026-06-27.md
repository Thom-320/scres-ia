# Cobb-Douglas Same-Bar Frontier Gate

Date: 2026-06-27

## Decision

If a policy is trained with a Cobb-Douglas reward, the primary evaluation
resilience bar for that lane is the same Cobb-Douglas index:

- primary: `cd_sigmoid_mean`
- training candidates: `ReT_garrido2024_raw`, `ReT_garrido2024`, `ReT_cvar_cd`
- continuity metric: `mean_ret_excel_formula`
- secondary panel: flow fill, service-loss CVaR, lost/unattended orders, resource composite

This does not replace the Garrido/Excel lane. It defines a separate same-bar
Cobb-Douglas lane where static and dynamic policies are scored on one metric.

## Static Screen

Runner: `scripts/screen_cd_same_bar_frontier.py`

Artifacts:

- `outputs/experiments/cd_same_bar_frontier_grid_2026-06-27/`
- `outputs/experiments/cd_same_bar_frontier_robust_2026-06-27/`
- `supply_chain/data/cd_same_bar_frontier_recommendation_2026-06-27.json`

The screen evaluates the full Track-A thesis grid:

```text
6 inventory levels x 3 shift levels = Discrete(18)
```

No RL is used in this gate.

## Candidate Results

Five-seed robust screen, 52 weekly steps, regimes `current`, `increased`, `severe`.

| Candidate | Eligible | Top Policies By Regime | CD Spread | Wrong-Regime Penalty | Notes |
|---|---:|---|---:|---:|---|
| `phi2_psi1_det` | yes | `S1_I168`, `S1_I504`, `S3_I336` | 0.1680 | 0.0445 | Best headroom; no stochastic PT or demand change. |
| `phi1_psi1_spt` | yes | `S1_I168`, `S1_I168`, `S2_I168` | 0.1329 | 0.0313 | Minimal extension; only stochastic PT. |
| `phi2_psi1p5_spt` | no | `S1_I168`, `S1_I672`, `S2_I1344` | 0.1589 | 0.0324 | Reject as primary because severe moves to max inventory. |

## Recommended Lane

Primary C-D headroom lane:

```text
risk_frequency_multiplier = 2.0
risk_impact_multiplier = 1.0
stochastic_pt = false
demand_mean_multiplier = 1.0
ret_g24_shift_cost = 1.0
ret_g24_kappa_train_frac = 1.0
primary metric = cd_sigmoid_mean
```

Why this one:

- The best static policy is not the expensive `S3_I1344` corner.
- The optimum is interior in inventory in every regime.
- The optimum changes by regime, so dynamic control has a real target.
- It creates the largest robust C-D spread among the tested eligible cells.
- It changes only risk frequency, which is easier to justify than changing demand or action space.

Conservative sensitivity:

```text
risk_frequency_multiplier = 1.0
risk_impact_multiplier = 1.0
stochastic_pt = true
demand_mean_multiplier = 1.0
```

This lane only adds processing-time stochasticity and produces the intuitive
shift frontier `S1_I168 -> S2_I168` under severe.

## Continuous Buffers

Continuous buffers were not evaluated in this gate. The current faithful Track-A
wrapper is explicitly `Discrete(18)`. Changing to continuous buffers is a new
action-space claim, not a small environment tweak.

Recommended order:

1. Train/evaluate the C-D same-bar lane above against its static frontier.
2. If the dynamic policy still cannot beat the C-D frontier, run a static-only
   continuous-buffer diagnostic.
3. Only then implement a continuous-buffer RL action surface.

## Claim Boundary

This gate does not prove RL wins. It proves there is a Cobb-Douglas static
frontier worth attacking. The next run should compare dynamic policies against
the C-D best static in the same environment, with Excel ReT reported as a
secondary continuity metric.
