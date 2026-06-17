# Track A Preflight Review - 2026-06-17

## Scope

Track A is still constrained to Garrido's decision variables:

- `I_{t,S}`: strategic inventory buffer level.
- `S`: short-term manufacturing capacity / shifts.

No Track A claim may use `continuous_it_s`, Track B downstream dispatch, ROP
controls, or any extra operational lever.  Those are later extensions.

## Thesis-Faithful Backbone

The Paper 1 / thesis-faithful backbone remains:

- `risk_occurrence_mode = thesis_periodic`
- `raw_material_flow_mode = kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier = 2.0`
- `inventory_period_mode = thesis_strict`
- `action_space_mode = thesis_factorized`
- `stochastic_pt = True`
- `stochastic_pt_spread = 1.0`

`m1.0`, `m0.5`, or other material-shortage settings are stress/scarcity
extensions, not "more faithful" thesis baselines.

## Reward Status

`ReT_ladder_v1` is preserved as a historical baseline.  It must not be mutated.

`ReT_tail_v1` is the current Track A training candidate because it is aligned
with tail/recovery metrics instead of aggregate mean service:

```text
SC_t = 1 - new_backorder_qty / max(new_demanded, 1)
RC_t = 1 / (1 + pending_backorder_qty / D8_t)
CE_t = sqrt(CAP_EF_t * INV_EF_t)
R_base = SC_t^w_sc * RC_t^w_rc * CE_t^w_ce
```

Interpretation:

- `SC_t`: service continuity, close to Garrido 2017 Eq. 5.4 `Re(FR_t)`.
- `RC_t`: smooth backlog recovery / containment proxy for the dynamic
  recovery idea in Garrido 2017.
- `CE_t`: cost/efficiency extension.  This is not in the 2017 thesis baseline,
  but is aligned with Garrido et al. 2024's Cobb-Douglas resilience/cost logic
  and with the thesis discussion that cost must enter the optimum-resilience
  problem.

Garrido et al. 2024 define a factory-resilience index with five output
variables:

- `zeta`: average accumulated inventory.
- `epsilon`: average accumulated backorders.
- `phi`: average spare production capacity.
- `tau`: average time to meet net requirements.
- `kappa`: cost deviation / total cost term.

Their Cobb-Douglas/log-linear form increases with inventory and spare capacity,
and decreases with backorders, fulfillment time, and costs.  They then restrict
the output range with a monotone sigmoid.  `ReT_tail_v1` is not a copy of that
factory APP index, but it is a Track A supply-chain analogue: service
continuity, backlog recovery, and un-gated cost efficiency are the DES-observable
terms available without adding decision variables beyond Garrido 2017.

The selected base weights are the full-panel static-audit winner:

```text
w_sc=0.30, w_rc=0.60, w_ce=0.10,
cap_kappa=0.40, inv_kappa=0.25, tail_boost=0.0
```

## Steepness

David's question was whether to use `exp(R)` to make the reward steeper.  The
answer is: not as the default.

`ReT_tail_v1` is already Cobb-Douglas:

```text
R_base = exp(w_sc log(SC) + w_rc log(RC) + w_ce log(CE))
```

So the professional steepness ablation is a monotone post-transform:

```text
R = R_base^gamma, gamma in {1.25, 1.5, 2.0}
```

This preserves the static policy order and keeps the reward in `[0, 1]`.

`exp_norm` is also implemented:

```text
R = (exp(beta R_base) - 1) / (exp(beta) - 1), beta in {2, 4}
```

but it is secondary.  It can distort correlations with flow/mean resilience and
must not be promoted unless it wins on held-out external metrics.

## Required Static Gate Before PPO

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

Gate:

- Best-by-reward must be top 3 by `ret_p10_all`.
- `best_reward_p10 >= 0.5 * top_p10`.
- `rho_ret_p10_all > 0`.
- `rho_flow_fill_rate > 0`.
- `rho_stockout_week_pct < 0`.

No PPO/Recurrent/DMLPA run should be promoted if this static gate fails.

Current result: the full-panel gate passed for all six variants.  The preferred
Kaggle scout is `power:1.25`; `identity` remains the default baseline and
`power:1.5` is the next ablation.  `exp_norm` passed but had weaker correlations
with the external metrics, so it stays secondary.

## Runnable Preflight

Use this before launching Kaggle:

```bash
.venv/bin/python scripts/track_a_preflight_check.py \
  --algo ppo_mlp \
  --risk-level increased \
  --ret-tail-transform power \
  --ret-tail-gamma 1.25
```

The preflight verifies:

- `thesis_factorized`, not `continuous_it_s`.
- `ReT_tail_v1`, not a legacy reward.
- `m2.0`, `thesis_periodic`, `kit_equivalent_order_up_to`.
- `stochastic_pt=True`, spread `1.0`.
- `n_envs=8`, `norm_reward=True`, and held-out `eval_seed_base`.
- `ReT_tail_v1` metadata exists and reward is bounded.

## First Kaggle Scout Contract

After the static gate passes, launch scouts only under the strict Track A
contract:

```bash
.venv/bin/python scripts/run_track_a_exhaustion_sweep.py \
  --label-prefix track_a_tail_screen \
  --algos ppo_mlp recurrent_ppo dmlpa_ppo \
  --action-space-modes thesis_factorized \
  --reward-profiles ret_tail \
  --risk-levels increased severe \
  --pt-profiles stoch_pt_hist \
  --use-cf-risk-profile \
  --panel-cfis 31-90 \
  --train-timesteps 100000 \
  --eval-episodes 20 \
  --max-steps 260 \
  --n-envs 8 \
  --norm-reward \
  --eval-seed-base 900000 \
  --device auto \
  --ret-tail-transform power \
  --ret-tail-gamma 1.25
```

Run the same scout for `identity` and optionally `power:1.5` if the static gate
supports them.  Do not treat `reward_total` as a victory metric.

## Victory Metrics

Primary external metrics:

- `ret_p10_all`
- `ret_mean_all_orders_zero_unfulfilled`
- `flow_fill_rate`
- `stockout_week_pct`
- dynamic ReT components where available

The sweep summary now reports static baselines by each metric separately, not
only the fill-mean baseline.

## Extension Bank Before Continuous Variables

Allowed Track A extensions, each compared against the best static under the
same scenario:

- stochastic processing-time spread: `{0.0, 1.0, 1.5, 2.0}`.
- demand variability / forecast error, mean-preserving first.
- variable costs `ci/cp/cb`, inspired by Garrido et al. 2024.
- material shortage, absenteeism, machine breakdowns, reprocessing, power or
  communications delays.
- history carriers: `recurrent_ppo` and `dmlpa_ppo`.

## Stop Rule

Close Track A if no strict run improves the best static by at least:

- `+0.02` in `ret_mean_all_orders_zero_unfulfilled` or `flow_fill_rate`, or
- a clear `ret_p10_all` improvement with lower stockout and no relevant fill
  loss.

Only after that should `continuous_it_s` or Track B become the main path.

## External Anchors

- Stable-Baselines3 recommends observation/action normalization and explicit
  history when the process has delayed observations/actions.
- Stable-Baselines3 PPO already normalizes advantages, but `VecNormalize`
  reward normalization is still useful as a value-target scale transform.
- SB3-Contrib `RecurrentPPO` is PPO with LSTM support; it tests hidden-state
  memory, not new supply-chain decision variables.
- Garrido et al. 2024 use demand variability, Monte Carlo simulation,
  heuristics, and a Cobb-Douglas resilience index, supporting the use of
  Cobb-Douglas and cost/variability extensions as logical extensions rather
  than thesis-baseline claims.
