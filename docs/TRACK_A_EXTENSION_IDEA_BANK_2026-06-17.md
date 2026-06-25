# Track A Extension Idea Bank - 2026-06-17

Track A remains constrained to Garrido's decision variables: inventory buffer
level `I_{t,S}` and short-term capacity `S`.  These ideas are allowed only as
scenario/reward/training extensions; they do not add new operational levers.

## Reward Shape

- Keep `ReT_tail_v1` as the base reward.  It is a bounded Cobb-Douglas reward:
  `SC^w_sc * RC^w_rc * CE^w_ce`.
- Do not use raw `exp(R)` as the default.  Cobb-Douglas already computes
  `exp(log_reward)`.
- Audit these monotone steepness transforms before training:
  - `identity`
  - `power`: `R_base^gamma`, `gamma in {1.25, 1.5, 2.0}`
  - `exp_norm`: `(exp(beta * R_base) - 1) / (exp(beta) - 1)`,
    `beta in {2, 4}`
- Use `power` as the preferred steepness family because it preserves order,
  range `[0, 1]`, and interpretability.

## Garrido-Aligned Scenario Extensions

- `stochastic_pt` mean-preserving processing-time variability:
  `spread in {0.0, 1.0, 1.5, 2.0}`.
- Demand variability / forecast error, implemented as mean-preserving noise
  before it is used for policy claims.
- Variable cost scenarios inspired by Garrido et al. 2024:
  inventory holding `ci`, production/shift `cp`, backlog `cb`, and cancellation
  or lost-order costs where available.  Garrido et al. 2024 explicitly include
  average inventory `zeta`, accumulated backorders `epsilon`, spare production
  capacity `phi`, time-to-fulfill `tau`, and cost deviation `kappa` in a
  Cobb-Douglas/sigmoid resilience index, so cost and demand variability are
  defensible scenario/reward extensions.
- Realistic disruption extensions, all evaluated against the same static
  baselines:
  - material shortages
  - absenteeism
  - machine breakdowns
  - reprocessing orders
  - power or communications delays

## History / Learning Extensions

- `recurrent_ppo` is the LSTM baseline for hidden-state memory.
- `dmlpa_ppo` is the Transformer-over-history baseline using frame stacking.
- These are carriers of `L_{t-1}` in the formal representation
  `R_t = f(S_t, D_t, L_{t-1})`.

## Promotion Rule

Every extension must first pass a static reward/surface audit.  PPO results are
judged only on external resilience metrics, not on `reward_total`.
