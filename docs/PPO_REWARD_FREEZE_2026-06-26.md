# PPO Reward Freeze (2026-06-26)

> **SUPERSEDED (2026-06-26) by `docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md`:** the primary training
> reward is **`control_v1`** (operational) with `control_v2_backlog` as a pre-registered sensitivity;
> `ReT_garrido2024` / Cobb-Douglas are **OUTCOME bars only**, not training rewards (no circularity).
> The algorithm is **DQN**. The reward provenance/calibration notes below remain valid reference.

Two **co-primary** reward designs (user decision 2026-06-26). Both are frozen here; weights are
tuned on training-seed sensitivity only (anti-fishing). Every policy is **evaluated** on the full
metrics panel + both bars regardless of which reward trained it.

## Reward A — operational (dense) → evaluated on the Excel bar

- **Training reward:** `control_v1` (existing) =
  `−( w_bo·service_loss_step + w_cost·shift_cost_step + w_disr·disruption_fraction )`,
  with `service_loss_step = new_backorder_qty/new_demanded`, `shift_cost_step = (S−1)`.
- **Frozen weights:** `w_bo = 4.0`, `w_cost = 0.02`, `w_disr = 0.0` (`config.py` BENCHMARK_*).
  Ratio 200:1 — in a military SC, failing to deliver rations ≫ the cost of a shift.
- **Why dense (not raw Excel ReT):** the raw Excel ReT is sparse/delayed and quirky; `control_v1`
  is a smooth per-step proxy that drives toward low service-loss at low shift cost — which
  correlates with the Excel bar (which rewards short recovery RP).
- **Evaluation bar:** `ret_excel` (Garrido) + the full panel (`episode_metrics`) + resource use.
  Win = dynamic ≥ best static on `ret_excel` / `flow_fill_rate` / `service_loss_auc` at **strictly
  lower resource** (surge-hours + buffer-units).
- **Known gap (documented):** `control_v1` has no explicit *inventory* holding cost, so it does not
  directly pressure the buffer down (it pressures shifts via `w_cost`). The buffer-efficiency claim
  is therefore *evaluated* in the panel; if the trained policy over-buffers, the refinement is a
  `control_v2` with an inventory term (`−w_i·inventory/I1344`) — deferred to avoid editing the
  concurrently-modified env file. Optional direct-Excel sensitivity: `ReT_excel_delta`.

## Reward B — Cobb-Douglas same-bar (train = evaluate)

- **Training + eval index:** `ReT_garrido2024` — the 5-variable cost-aware Cobb-Douglas
  (Garrido 2024 Eq. 3/6): `R = ζ^a·ε^-b·φ^c·τ^-d·κ̇^-n` (ζ inventory+, ε backorders−, φ spare
  capacity+, τ time−, κ̇ cost−), **with the shift cost in κ̇** (`ret_g24_shift_cost = 0.5`,
  `ret_g24_kappa_train_frac = 0.2`). "Same bar": the objective IS the resilience index.
- **Calibration:** re-derived on the **faithful env** (thesis_window + kit_equivalent m2.0 +
  figure_6_2 + obs v4, shift_cost 0.5) →
  `supply_chain/data/ret_garrido2024_calibration_faithful_2026-06-26.json`. (Audit 2026-06-18: the
  faithful exponents remain close in sign and scale, but the faithful cost reference changes
  materially: `kappa_ref = 752,609` vs the legacy `1,412,199`. The env default now points to this
  faithful file so omitted flags do not silently load the stale calibration.)
- **Training variant:** train on `ReT_garrido2024_train` (κ̇ at the reduced fraction, better
  gradient, same optimum) and evaluate on `cd_sigmoid_index`; OR train+eval both on the sigmoid for
  the strictest same-bar reading. The pilot used the latter.

## Reporting (both rewards, both bars)
Every run reports: `ret_excel`, `ret_thesis`, `ret_continuous`, the Cobb-Douglas sigmoid index,
`flow_fill_rate`, `lost_rate`, `ttr_mean/p95`, `service_loss_auc`, CTj/RPj/DPj quantiles, plus
resources (shift/surge-hours, buffer-units, unit-cost-per-ration) and per-step CVaR95/p95
service-loss (from the comparison harness).
