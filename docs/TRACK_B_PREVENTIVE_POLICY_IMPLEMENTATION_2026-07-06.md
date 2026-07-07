# Track B Preventive Policy Implementation - 2026-07-06

## Objective

Implement a deliberately preventive Track B policy lane: not just stronger
adaptive dispatch, but a policy that has a chance to act before risk events
when a non-privileged belief signal rises.

The gate remains adversarial:

1. Training/evaluation must report Garrido Excel ReT (`order_ret_excel_mean`)
   and tail resilience (`order_ret_excel_cvar05_mean`).
2. A candidate only becomes "preventive" if the fixed-RNG counterfactual
   `R_full - R_reset(pre-risk)` has positive mean delta and non-tiny positive
   pair rate, roughly >= 0.5.
3. If it improves ReT/CVaR but fails the counterfactual, it is adaptive or
   tail-resilient, not preventive.

## Chosen Lane

Use the selected Case C environment:

- enabled risks: `R22,R23,R24`
- `R24` frequency multiplier: `3.0`
- `R22/R23` impact multiplier: `1.5`
- horizon: `104` weeks
- action contract: `track_b_v1`
- training reward base: `control_v1`

This is the strongest current headroom lane. Case A is thesis-faithful but has
little avoidable exposure because risks are nearly continuous. Case B exposes
the branch/exposure mechanism, but Case C adds enough stress to make adaptive
headroom visible without collapsing absolute ReT.

## Observation Contract

New observation config: `v10_no_forecast`.

This uses v10 operational memory fields but masks the explicit forecast fields:

- `risk_forecast_48h_norm`
- `risk_forecast_168h_norm`

The same mask is applied inside the belief reward wrapper before the belief
head sees the observation. This prevents a subtle leakage path where the agent
would see masked observations but reward shaping could still use raw forecast
channels.

## Architecture

Screen first with PPO+MLP.

Rationale: PPO+MLP is the strongest optimizer in the final Case C confirmation.
Real-KAN is reserved for a second pass only if PPO+MLP produces a real
preventive signal. This keeps novelty (KAN) from obscuring the main causal
question.

## Belief Target

Two Case C no-forecast belief datasets were generated:

- `R24` 1w: base rate 0.6893, held-out AUC 0.524
- `R22` 4w/8w: base rates 0.1705/0.3346, held-out AUC about 0.560

`R24` under frequency x3 is too close to a tautological target. The implemented
preventive lane therefore uses the `R22` 4-week belief head as the trigger for
early preparation.

Belief artifacts:

- `outputs/experiments/track_b_case_c_v10_no_forecast_belief_pretrain_r22_2026-07-06/mlp_belief_trunk_r22_case_c.pt`
- `outputs/experiments/track_b_case_c_v10_no_forecast_belief_pretrain_r22_2026-07-06/mlp_belief_trunk_r22_case_c_head.pt`

## Reward Shaping

New mode: `belief_conditioned_tail_pbrs`.

It extends the prior belief-conditioned PBRS:

```text
Phi =
  - alpha * pending_norm
  - beta * lost_norm
  + kappa * p_adv * readiness
  - rho * (1 - p_adv) * resource_posture
  - exposure * p_adv * (1 - readiness)
  - backlog_age * p_adv * oldest_backorder_age_norm
  - tail * p_adv * mean(rolling_backorder_rate_4w, backorder_rate)
```

The new terms are intentionally risk-conditioned. They are meant to push the
agent to reduce exposure while belief is elevated, without rewarding permanent
expensive posture during calm periods.

## Active Screen

Launched in tmux:

```bash
tmux has-session -t track_b_preventive_tail_grid_case_c
tail -80 outputs/experiments/track_b_preventive_tail_grid_case_c_3seed_30k_2026-07-06.run.log
```

Grid:

- `belief_v2_control`
- `preventive_conservative`
- `preventive_balanced`
- `preventive_aggressive`

Protocol:

- seeds `1,2,3`
- train timesteps `30000`
- eval episodes `8`
- batch size `64`
- n_steps `1024`
- observation `v10_no_forecast`
- Case C selected

If one cell shows training/tail signal without collapse, the next step is the
fixed-RNG `R_full - R_reset(pre-R22/pre-R24)` counterfactual on that checkpoint.
