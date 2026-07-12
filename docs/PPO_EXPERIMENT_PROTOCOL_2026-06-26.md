# PPO Experiment Protocol — Beat Garrido's Best Static (2026-06-26)

> **SUPERSEDED (2026-06-26) by `docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md`:** the algorithm is now
> **DQN** (not PPO) and the primary claim is the **decision-frontier** claim (learning helps only
> where a frontier exists). The **two-env design** (faithful + headroom) and the **win/dominance
> definition** below are KEPT and carried into Contract v2 §6–§7. Read v2 first.

Pre-registered protocol. Frozen **before** any confirmatory PPO run. No reward/observation/env/
regime is changed using held-out results (anti-fishing).

## Claim / win (co-primary, both reported)

A **dynamic PPO policy** on the thesis [6 inventory × 3 shift] surface beats/matches **Garrido's
best static config (computed in our DES)** on resilience, **using strictly fewer resources** —
judged on two bars:

1. **Excel-eval bar:** every policy scored on `ret_excel` (Garrido Eq. 5.5) + the full metrics
   panel. The agent trains on the **dense operational reward** (`control_v1`). Win =
   dynamic ≥ best static on `ret_excel` and `flow_fill_rate` (and ≤ on `service_loss_auc`,
   `lost_rate`, `ttr`) at strictly lower **surge-hours + buffer-units**.
2. **Cobb-Douglas same-bar:** train AND evaluate on `ReT_garrido2024` (cost-aware index). Win =
   dynamic > best static on the index.

"Best static" = best of the 18 configs in the **Part-2 static panel** per regime (the thesis names
no single joint optimum). Note: strict on-time fill is degenerate under the faithful env (every
order late, delay 54 > LT 48), so the resilience bars are `ret_excel` + `flow_fill_rate` +
`service_loss_auc` + `ttr`, NOT strict fill.

## Two environments

- **Env-faithful** (1-to-1 fair fight): the frozen `THESIS_FAITHFUL_ENV_FREEZE_2026-06-26`. Honest
  thesis-close arena; expect mostly an **efficiency** win (match resilience at lower resource).
- **Env-finetuned** (headroom): the faithful env + pre-registered realism knobs justified by the
  thesis's OWN limitations/future work — **non-stationary/variable demand** (§8.5.1),
  **stochastic_pt**, **risk frequency φ + impact ψ up**, optional **surge_inertia** (§8.6 lead-time).
  φ/ψ calibrated with the Part-2 panel to a **headroom band** (best static off-saturation,
  regime-dependent optimum) and FROZEN before PPO. [φ/ψ values: filled by Part 4.]

## Two rewards
See `PPO_REWARD_FREEZE_2026-06-26.md`. A = `control_v1` (operational, eval Excel bar). B =
`ReT_garrido2024` (Cobb-Douglas same-bar, faithful calibration).

## Policy conditions (baseline ladder)
1. **Original** static `S1_I0` (thesis Cf0).
2. **Best static** per regime (from the Part-2 panel — the strong bar to beat).
3. **Threshold heuristic** — state-dependent non-neural rule (lean by default; raise buffer/shift
   when backlog/disruption crosses a threshold). Separates "reactive" from "learned".
4. **Frozen RL** — trained checkpoint, no eval-time updates (separates neural control from learning).
5. **Dynamic PPO** — the proposed model.
   (Secondary: retained vs reset online learners for the `L_{k-1}` memory question.)

## Metrics panel (report ALL)
`supply_chain/episode_metrics.compute_episode_metrics` + resources: `ret_excel`, `ret_thesis`,
`ret_continuous`, Cobb-Douglas sigmoid index, `fill_rate`, `flow_fill_rate`, `lost_rate`,
`service_loss_auc`, `ttr_mean/p95`, `backlog_age`, CTj/RPj/DPj p50/p90/p99, `delivered_rations`,
`shift_hours`, `surge_hours`, `strategic_buffer_units`, `unit_*_per_ration`, plus per-step
service-loss mean/p95/**CVaR95** (from the comparison harness).

## Design / statistics
- **Regimes:** `current`, `increased`, `severe` (env-faithful); the frozen headroom regime
  (env-finetuned). Held-out evaluation seeds, common random numbers across all policies (paired).
- **Replication:** ≥10 training seeds; statistical unit = seed.
- **Inference:** paired bootstrap CIs (dynamic − best static) per metric; Holm correction across
  metrics; report effect sizes + CIs, not just p-values. Pareto-dominance flags
  (`strict_service_resource_dominates`) reused from `compare_garrido_dynamic_vs_static.py`.

## Stop-rule
- If, in **env-faithful**, dynamic does not beat the best static on resilience, the defensible win
  is the **efficiency** result (match resilience at strictly lower resource). Report it as such.
- If **env-finetuned** (at the frozen headroom) shows no dynamic > static gain on either bar after
  the pre-registered budget, report the null honestly; do NOT re-tune the regime against results.
- One frozen headroom regime only; no escalation against held-out outcomes.

## Pre-flight gate (Part 5)
Before any confirmatory run: env builds under both rewards × both envs; `compute_episode_metrics`
runs; the full static panel runs; PPO trains a few hundred steps; the dominance harness emits the
table. Pipeline verified end-to-end, no compute committed to a broken setup.
