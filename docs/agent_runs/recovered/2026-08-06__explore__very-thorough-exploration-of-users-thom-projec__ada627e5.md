# Explore — Very thorough exploration of <HOME>/Projects/research/scres-ia across ALL branches (u

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `ada627e5f50f29531`
- **session**: `155226df-d03c-4d02-9e9f-9e30785a6895`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-06T02:30:10.382Z`
- **effort**: `medium` · **version**: `2.1.221`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/155226df-d03c-4d02-9e9f-9e30785a6895/subagents/agent-ada627e5f50f29531.jsonl`
- **turns**: 112 · **assistant messages**: 4

## Task given

```text
Very thorough exploration of <HOME>/Projects/research/scres-ia across ALL branches (use `git branch -a` and `git log --all`).

I need a COMPLETE inventory of every resilience metric / endpoint this project has ever defined, and what was measured with each. For each metric report: exact name, the file and function that computes it, its definition in one sentence, and every sealed result that used it as PRIMARY.

Specifically hunt for and report on:
1. `ret_excel`, `ret_excel_visible`, `ret_excel_full_ledger`, `ret_excel_risk_conditional`, `ret_excel_request_snapshot_v2`, `ret_thesis` — where each is computed, how they differ, and what is known to be defective about each.
2. The Cobb-Douglas index: `R_cobb_douglas`, any `cobb_douglas` module, the exponent derivation rule, the five variables it uses. Report `supply_chain/cobb_douglas.py` or wherever it lives, in full detail.
3. `service_first_metric.py` — SERVICE_FIRST_V2_COMPONENTS, `claimant_fills`, `service_first_resilience_v2`, the lexicographic key.
4. `worst_claimant_fill`, `flow_fill_rate`, `lost_orders`, `backorder_qty_final`, `service_loss_auc`, `ret_visible`, `ret_order`, CVaR variants (`ret_excel_plus_cvar`, `ReT_cvar_cd`, `ret_visible_cvar10`).
5. Any temporal/AUC resilience panel: `resilience_temporal.py`, `compute_temporal_resilience_panel`.
6. `episode_metrics.py` — list EVERY key the panel returns.

Then, the critical part: **for each metric, find every artifact where a HEADROOM measurement (H_regime, H_PI, H_obs, Delta_N, or an argmax-moves test) was computed with that metric as the endpoint, and report the NUMBER.** I want to know which metrics produced non-zero headroom and which produced zero.

Search `results/**/result.json` for `primary_metric`, `metric`, `endpoint` fields and cross-tabulate against `claim_status` and any H_regime / H_PI / H_obs value in the same artifact.

Also find every document that RETRACTS or criticises a metric: grep docs/ for RETIR, RETRACT, defect, defecto, premia el abandono, cadence, censura, service-blind, ciego.

Report exact file paths, exact numbers, exact claim_status strings. Be exhaustive — this is for deciding which endpoint could change our conclusions.
```

## Final answer

# Complete resilience-metric inventory — `scres-ia`

Scope checked: all 90 local+remote branches (`git branch -a`, `git log --all`), 154 `result.json` on the current branch (`codex/expanded-contract-comparators-v2`) + 101 more that exist only on other branches (255 union). **No metric name, endpoint value, or module exists on another branch that does not exist here** — I verified by `git grep`ing every `primary_metric|metric|endpoint|governing_metric|canonical_metric` value across every ref; the off-branch artifacts (program_t, program_u, program_q2, q_r1) are almost all `BURNED_*` / `EXPLORATORY_NO_CLAIM` and carry no headroom claim. Exceptions noted in §7.

---

## 1. The ReT family — where each is computed and what is broken

All order-level ReT lives in **`<HOME>/Projects/research/scres-ia/supply_chain/ret_thesis.py`** (603 lines). A byte-identical fork exists at `<HOME>/Projects/research/scres-ia/research/thesis_native_timing/canonical_ret.py`. The panel that surfaces them is **`<HOME>/Projects/research/scres-ia/supply_chain/episode_metrics.py`**.

| key | computing function | one-line definition |
|---|---|---|
| `ret_excel` | `compute_order_level_ret_excel_request_snapshot_ledger` (ret_thesis.py:440) via `compute_episode_metrics` (episode_metrics.py:229) | Mean of Garrido's raw-workbook per-order formula over the **workbook-visible** population (completed, non-lost rows), un-clipped; `Bt`/`Ut` are request-time snapshots. **`ret_excel` is a pure alias of `ret_excel_visible`** (episode_metrics.py:229-230). |
| `ret_excel_visible` | same object, same call | Identical value to `ret_excel`; the two keys are set from one variable. |
| `ret_excel_visible_clipped_0_1` | episode_metrics.py:231, `_mean` of per-order values clipped to [0,1] | Same population, each per-order value clamped into the thesis's stated range. |
| `ret_excel_full_ledger` | `compute_order_level_ret_excel_formula` (ret_thesis.py:237) → episode_metrics.py:233 | Mean over **every generated order**, unfulfilled/lost scored 0.0 (`excel_unfulfilled` branch, ret_thesis.py:143-144). Removes the censoring. |
| `ret_excel_risk_conditional` | `_excel_ret_details` (episode_metrics.py:103-109) | Mean restricted to orders whose branch was `excel_autotomy` / `excel_recovery` / `excel_risk_no_recovery` — i.e. only risk-touched orders, excluding the no-risk fill-rate branch. |
| `ret_excel_request_snapshot_v2` | the contract *version string* selecting `compute_order_level_ret_excel_request_snapshot_ledger`; `ret_excel_contract_version` default at episode_metrics.py:146 | Not a separate number — it is the **ledger convention** under which `ret_excel` is produced. The predecessor `ret_excel_visible_v1` (`compute_order_level_ret_excel_visible_ledger`, ret_thesis.py:305) reconstructs `Bt`/`Ut` at **completion** time instead of request time and remains callable only to reproduce quarantined runs. |
| `ret_thesis` | `compute_order_level_ret` (ret_thesis.py:203) using `compute_ret_per_order` (ret_thesis.py:67) | The thesis's *piecewise* Eq. 5.5 (`Re^max·AP/LT` / `0.5/RP` / `Re^min·(DP−RP)/CT` / fill-rate), clipped to [0,1], over all demanded orders. |
| `ret_continuous` | `compute_periods_continuous_ret` (ret_thesis.py:189) | Smooth surrogate `Re_j = (AP+τ)/(AP+RP+DP+τ)`, τ=48 h, over served orders. |

### Known defects, by metric — `docs/RET_METRIC_DEFECTS_2026-07-29.md` (status `DEVELOPMENT_CORRECTIVE_AUDIT`)

**`ret_excel` — three independent defects.**

1. **Step-cadence dependence (repaired prospectively).** On one identical trajectory: 0.004369 (one step / 672 h) → 0.004401 (168 h) → 0.005623 (24 h) → 0.005981 (1 h). **37 % spread, monotone in cadence**, physics identical (same fill 0.99650, same 689,182 delivered, same `OPTj`/`OATj`/`APj` in all 311 orders); **`RPj` differs in 175 of 311**. Carrier is `_op_down_since` being reset at every step boundary. After repair, all five cadences give **0.004424198300**, spread exactly 1.0 (`results/metric_audit/ret_cadence_corrective_v2/result.json`). Consequence: `ret_excel` levels **may never be compared across artifacts of different cadence**.
2. **The autotomy branch is dead code.** `GARRIDO_FULFILLMENT_DELAY_HOURS = 54.0` (`supply_chain/config.py:119`) against `LT = 48`, so `CTj ≤ LTj` is unsatisfiable: `excel_case_pct_autotomy = 0.00` and `fill_rate_on_time = 0` in **every** configuration. Delay sweep (`results/metric_audit/ret_defects_v1/result.json`): delay 54 → ReT **0.004424**; delay 48 → **0.980513**; delay ≤ 47 → 0.980576. **A six-hour change moves ReT by 221.6×.** All signal therefore flows through `RPj`, the cadence-dependent quantity.
3. **Unbounded, non-monotone tail from a dimensional error.** `0.5/RPj` is unclipped and `RPj` accumulates hours while `R24` is a *rations* indicator. Order j=24 on tape R2r/1530011: 192 h late, `RPj = 0.006765 h`, **ReT = 73.9082** — the episode's highest. In R1r the most-delayed order scores 0.000462 while the highest scorer sits at the physical minimum `CTj = 54 h`. **ReT is not monotone in lateness in either direction.** Scope: R1r 0/3,279 orders > 1.0; **R2r 7/3,108 > 1.0, max 73.91**, inflating the family mean 6 %. Removing that tail by any of three independent repairs flips the R2r verdict from `NOT_SEPARATED` to `MPC_AHEAD` (Δ +0.0120…+0.0131, CI wholly above zero).
4. **Policy-dependent censoring.** Warm-up is endogenous and arm-dependent (631 h static vs 943 h MPC on tape 1530011), so scored populations differ (262 vs 253). Omitted-order fraction ranges **3.9 %–18.6 %** across postures (`docs/METRIC_PANEL_2026-07-29.md`).
5. **It rewards abandonment.** `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`, artifact `results/sensitivity/contention_headroom_v1_2/result.json`: across nine allocation splits, `ret_excel` and `flow_fill_rate` are **exactly inverted** — the ReT-maximising split delivers **50 %** of rations (ReT 0.0047 at split 0.1, 0.0051 at 0.9) while the ReT-minimising split delivers **80 %** (ReT 0.0004 at split 0.5, fill 0.795). A 12× "resilience gain" bought with 30 points of service, repeating in all six cells.

**`ret_excel_full_ledger` — defective as a repair.** Audited in `contention_headroom_v1_3`: removing the censoring **does not remove the U** (0.0024 / 0.0003 / 0.0025). Also points opposite to `ret_excel` on the sealed R2r confirmation (−0.004483, CI [−0.006600, −0.002388] vs +0.012475).

**`ret_excel_visible_clipped_0_1` — worst of the family on its own test.** Same audit: in R2r it goes 0.367 → **0.027** → 0.353, a **14× U** — clipping *aggravates* the abandonment preference. The author explicitly retracts it: *"El endpoint que yo mismo propuse para el paper (`ret_excel_visible_clipped_0_1`) falla su propia prueba."* Also measured **dead as a tiebreaker** — no two policies ever tie on (gate, fill), so it never breaks anything inside `service_first_resilience_v1`.

**`ret_excel_visible_v1` — formally superseded and quarantined.** `docs/RET_EXCEL_VISIBLE_V1_CONTRACT_2026-07-11.md` carries a retraction header: *"Superseded 2026-07-14 … quarantined as metric-development only because its OAT-derived `Bt/Ut` reconstruction is not source-validated. It cannot support a Paper-2 null, positive, `H_PI`, `H_obs`, comparator ceiling or terminal boundary."* Still the `canonical_metric` of two sealed screens (§5).

**`ret_excel_request_snapshot_v2` — status is PROVISIONAL, not primary.** `docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_UNRATIFIED_PROMOTION_2026-07-17.md` records that commit `4111cbc` edited the frozen contract in situ to promote it from *provisional* to *frozen researcher-defined primary*; the promotion was **reverted byte-for-byte to `ff5e4a8`**. The machine authority `research/paper2_exhaustive_search/metric_governance_audit.json` still asserts `CANONICAL_RET_EXCEL_REQUEST_SNAPSHOT_V2__PROVISIONAL__RESCORE_REQUIRED` and `PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION`. The 07-17 code change (`same_time_precedence`, `force_reconstruct`, causal boundary moved from `(j, OPTj)` to request time) was verified numerically inert: `results/metric_audit/v2_metric_freeze_equivalence/` — 3,289 rows per stratum, **0 differences, max|Δ| = 0** (`claim_status: V2_METRIC_UNCHANGED_SINCE_FREEZE`).

**`ret_thesis` — collapses.** Under risk **100 % of orders fall into the `recovery` bucket** (mechanically explained by the dead autotomy branch); under R3, 99 % fall into fill-rate. In the abandonment audit it is only *partially* repaired — flat in R2r, still U-shaped in R1r+R2r.

**Retired implementation.** `docs/RESILIENCE_METRICS.md` documents a **step-level ReT approximation** still in `env_experimental_shifts.py` (case proxies `1−disruption_frac`, `1/(1+disruption_frac)`, 0.0, `1−backorder_qty/demanded`) that computes per 168 h step, not per order, and uses `Re=1` instead of `Re=0.5`.

---

## 2. Cobb-Douglas — `supply_chain/cobb_douglas_resilience.py` (414 lines)

Port of Garrido, Pongutá & García-Reyes (2024), IJPR, DOI 10.1080/00207543.2024.2425771, §3.4 Eq. (2)–(6), Algorithm 2 lines 33-38.

**The index** (`resilience_index`, line 209):
```
linear = Σ_x sign(x) · exponent_x · ln(max(x, FLOOR_x))
R_cobb_douglas = 1 / (1 + exp(−linear))            # Eq. (6)
```
Returns `{"R_cobb_douglas", "linear_score", "term_zeta", …}`.

**The five variables** (`VARIABLES`, line 139) with signs (`SIGNS`, line 141):

| variable | Garrido | MFSC mapping (`CobbDouglasRecorder.sample`, line 306) | sign |
|---|---|---|---|
| `zeta` | mean inventory `I_t` | on-hand rations across 5 stock points + raw material ÷ `NUM_RAW_MATERIALS` (12-component BOM) | **+1** |
| `epsilon` | mean backorders `B_t` | `pending_backorder_qty` | −1 |
| `phi` | mean spare capacity `U_t` | `max(Θ_t − P_t, 0)` from Table 6.20 `theoretical_capacity_rations` | **+1** |
| `tau` | mean `NR_t / min(GR_t, Θ_t)` | Algorithm 2 lines 23 & 36 | −1 |
| `kappa_dot` | set-relative cost | `|S|·κ(s) / Σ_S κ(s)` (`kappa_dot`, line 226) | −1 |

**Exponent derivation rule** (`derive_exponents`, line 146): **`exponent_x = 0.20 / ln(x_max)`** (`SHARE_PER_TERM = 0.20`, "each function argument was equated to 1/5"). `GARRIDO_2024_EXPONENTS = {zeta 0.024, epsilon 0.026, phi 0.04, tau 0.06, kappa_dot 0.1771}` is kept **for verification only, never used to score** — those encode a 36-week model with inventories in the thousands, ours run in the millions. `x_max ≤ 1.0` raises rather than clamps. `conditioning()` reports `1/ln(x_max)` amplification (zeta ≈ 0.07 stable; tau ≈ 5.4 unstable). `assert_terms_bounded` enforces |term| ≤ 0.20; `FLOORS = {zeta/epsilon/phi/tau: 1.0, kappa_dot: 1e-9}` — a 1e-4 floor on tau once drove its term to −9.9 against an intended 0.20.

**κ** = `c_p·P + c_h·H + c_l·L + c_u·U + c_i·I + c_b·B + c_o·O`, all `UNIT_COSTS = 1.0` (Garrido §3.1 assumption 6). `c_o` (overtime) is **structurally 0** — the MFSC DES has no overtime process. `kappa_from_components` allows repricing without replay.

**Two disclosed hazards, in the module docstring:** (a) **zeta enters positively** — more inventory raises R, so without a costed ledger the index *rewards* overstocking; (b) **`kappa_dot` is set-relative** — R depends on the comparison set, so the set must be frozen and declared. `score_comparison_set` (line 377) enforces the set discipline.

**Its own defect** (`docs/METRIC_PANEL_2026-07-29.md`): **Cobb-Douglas is service-blind — it "prices no lost order: an order never served leaves the backorder queue and stops costing anything."** Also flagged in the abandonment doc as *"ciega al servicio por otra vía"*. Resolution limit: range between splits is ~1 % (`docs/RESULTADO_HEADROOM_COBB_DOUGLAS_2026-07-31.md` §4 — *"Ordena bien; no es un microscopio."*).

Sibling module: **`supply_chain/factory_resilience.py`** — the pure Eqs. 3-6 with `PUBLISHED_EXPONENTS`, **no** MFSC mapping, deliberately requiring strictly-positive APP components.

---

## 3. `supply_chain/service_first_metric.py` (130 lines)

**v1** — `SERVICE_FIRST_METRIC_ID = "service_first_resilience_v1"`, `SERVICE_FIRST_COMPONENTS = ("no_lost_orders", "flow_fill_rate", "negative_backorder_qty_final", "ret_excel_visible_clipped_0_1")`. `service_first_key` (line 30) returns `(float(lost <= 0.0), fill, −backorders, ret)`, compared as a **tuple**, never collapsed.

**v1 is defective and audited** — `docs/AUDITORIA_SERVICE_FIRST_METRIC_2026-08-01.md`, sealed at `results/metric_audit/service_first_v2/result.json` (`SERVICE_FIRST_V2_AUDIT_SEALED`, seal `0e37fe2faa3fd695…`, 6 falsifiers pass): **`lost_orders` measures queue overflow, not abandonment.** `BACKORDER_QUEUE_CAP = 60` (`config.py:45`); an order is only labelled lost when the backlog *overflows* 60. Measured open-at-horizon orders by split: **60.0 / 59.5 / 42.8 / 58.7 / 60.0** — four of five pinned at the cap. A policy holding its queue at 60 abandons up to 60 units and records **zero** losses. Components 3 and 4 are dead (no two policies tie on (gate, fill)).

**v2** — `SERVICE_FIRST_V2_METRIC_ID = "service_first_resilience_v2"`, `SERVICE_FIRST_V2_COMPONENTS = ("worst_claimant_fill", "flow_fill_rate", "negative_backorder_qty_final", "ret_excel_visible_clipped_0_1")`.

- **`claimant_fills(sim)`** (line 91): `{name: delivered[name]/demanded[name]}` from `sim.cssu_delivered` / `sim.cssu_demanded`; returns `{}` when there is no claimant partition; a zero-demand claimant scores 1.0.
- **`service_first_key_v2`** (line 103): **`(worst, fill, −backorders, ret)`** where `worst = min(claimant_fills.values())`, degenerating to aggregate fill with a single claimant.
- The rejected repair is documented in-source: unserved **quantity** share is exactly `1 − flow_fill_rate`, which would have collapsed components 1 and 2.
- **v2 is a STIPULATED normative endpoint, never evidence that abandonment is bad** (`endpoint_status` string in the sealed artifact).

---

## 4. Other endpoints

| metric | file:function | definition |
|---|---|---|
| `worst_claimant_fill` | service_first_metric.py:91/103 | Minimum per-claimant delivered/demanded ratio. |
| `flow_fill_rate` | episode_metrics.py:299 | `delivered_rations / demanded_rations` (quantity-weighted), 1.0 if no demand. |
| `lost_orders` / `lost_rate` | episode_metrics.py:266-267 | Count / fraction of orders with `lost = True` — i.e. **queue overflow past `BACKORDER_QUEUE_CAP = 60`**. |
| `backorder_qty_final` | episode_metrics.py:268 | `sim.pending_backorder_qty` at horizon. |
| `service_loss_auc_ration_hours` / `_per_order` | episode_metrics.py:208-217, 269-270 | Σ over orders of `max(0, end − (OPTj+LTj)) × quantity`, `end = OATj` or horizon. |
| `fill_rate` | `ret_thesis.compute_fill_rate_from_orders` (line 44) | Thesis Eq. 5.4, `1 − (B_t + U_t)/D_t`, order-count based, clamped [0,1]. |
| `ret_visible` | program_o / paper-2 lane naming for the visible-ledger score (`contracts/program_o_*`, `results/program_o/fixed_clock_hobs_*`) | The paper-2 governing endpoint under which L1–L4 were measured. |
| `ret_order` | `research/paper2_exhaustive_search/op11_fair_probe.py:145`, `supply_chain/program_g.py` | **Stylized reduced-model order adapter**, explicitly *not* the full-DES endpoint. `contracts/op11_fair_allocation_conversion_probe_v1.json`: *"This is the reduced-model metric, NOT the canonical DES endpoint."* Program I's entire GSA runs on it. |

**CVaR variants** — all secondary, none a gate:
- `ret_excel_cvar05`, `ret_excel_cvar10` — episode_metrics.py:239-240, `_tail_mean(ret_values, frac, lower_tail=True)`, mean of the lower 5 %/10 % of per-order visible values.
- `ret_excel_plus_cvar` — `env_experimental_shifts.py:2430 _compute_ret_excel_plus_cvar`. **Train-side reward only**: `ret_excel_delta_step − α·CVaR(rolling service-loss window)` where service loss = `new_backorder_qty/new_demanded`. Note this is a CVaR of a *service-loss proxy*, not of ReT itself.
- `ReT_cvar_cd` — `env_experimental_shifts.py:2183 _compute_ret_cvar_cd`, registered in the reward-mode list at :108. Mean-CVaR over the Garrido-2024 CD index.
- `ret_visible_cvar10` — appears only as a **reported / non-blocking secondary** in `contracts/program_o_hobs_prelearner_v1.json`, `program_o_full_des_hpi_translation_v1.json`, `program_o_state_rich_*`, `program_q_frozen_policy_replication_v1.json` (`secondary_nonblocking`), `program_s_product_mix_risk_interaction_gsa_v1.json`, `program_s_raw_matrix_schema_v1.json`. The PI directive is quoted verbatim in `program_o_relevant_risk_sensitivity_v1.json`: *"REPORTED (never a gate)"*.

**Temporal / AUC panel** — `supply_chain/resilience_temporal.py:101 compute_temporal_resilience_panel`, version string `risk_cluster_daily_v1`, wired in via `compute_episode_metrics(..., include_temporal_panel=True)`. Clusters risk events with a 1-week gap rule, builds a daily service history over a 4-week due-lookback, and per cluster computes baseline (median over 2 weeks), `service_loss_auc_ration_hours`, `maximum_service_drop`, and `system_ttr_hours` (first 7 consecutive days at ≥95 % of baseline service **and** ≤1.05× baseline backlog). `TEMPORAL_METRIC_KEYS` (line 223) = `temporal_service_loss_auc_ration_hours`, `temporal_maximum_service_drop`, `system_ttr_mean`, `system_ttr_p95`, `system_ttr_n_clusters`, `system_ttr_n_recovered`, `system_ttr_n_censored`, `system_ttr_censored_fraction`. Docstring: *"These diagnostics never replace or optimize the canonical Garrido ReT."*

---

## 5. `episode_metrics.py` — EVERY key the panel returns

`compute_episode_metrics` builds `panel` at line 221. Full list, in order:

**Counts:** `n_orders`, `n_served`, `n_lost`, `n_late`
**Resilience:** `ret_excel`, `ret_excel_visible`, `ret_excel_visible_clipped_0_1`, `ret_excel_full_ledger`, `ret_excel_contract_version`, `ret_excel_visible_n`, `ret_excel_omitted_n`, `ret_thesis`, `ret_continuous`, `ret_excel_cvar05`, `ret_excel_cvar10`, `ret_excel_p05`, `ret_excel_p10`, `ret_excel_p25`, `ret_excel_p50`, `ret_excel_p75`, `ret_excel_p90`, `ret_excel_p95`, `ret_excel_rolling_4w_mean`, `ret_excel_rolling_4w_min`, `ret_excel_rolling_4w_final`, `ret_excel_risk_conditional`, `ret_excel_risk_conditional_n`, `ration_ret_excel`, `excel_case_pct_fill_rate`, `excel_case_pct_autotomy`, `excel_case_pct_recovery`, `excel_case_pct_risk_no_recovery`, `excel_case_pct_unfulfilled`
**Service:** `fill_rate`, `fill_rate_on_time`, `lost_orders`, `lost_rate`, `backorder_qty_final`, `service_loss_auc_ration_hours`, `service_loss_auc_per_order`
**Recovery times:** `rpj_mean`, `rpj_p95`, `ttr_mean`, `ttr_p95` (the latter two are **backward-compatible aliases of `rpj_*`** with an in-source warning: *"New work must use rpj_* or system_ttr_* and must not call these TTR"*), `backlog_age_mean`, `backlog_age_max`
**Order-time distributions:** `apj_p50`, `apj_p90`, `apj_p99`, `apj_positive_p50`, `apj_positive_p90`, `apj_positive_p99`, `ctj_p50`, `ctj_p90`, `ctj_p99`, `rpj_p50`, `rpj_p90`, `rpj_p99`, `dpj_p99`
**Throughput:** `delivered_rations`, `demanded_rations`, `flow_fill_rate`
**Merged by `merge_resource_metrics` (line 310):** `shift_hours`, `surge_hours`, `strategic_buffer_units`, `end_state_inventory` (optional), `unit_surge_hours_per_ration`, `unit_buffer_units_per_ration`
**Optional temporal block:** the 8 `TEMPORAL_METRIC_KEYS` above plus `temporal_panel_version` and `temporal_cluster_records`.

`METRIC_KEYS` (line 337) is the declared tuple, 63 entries — note it **omits** `ret_excel_contract_version`, `n_orders`/`n_served`/`n_lost`/`n_late`, and `end_state_inventory`.

---

## 6. HEADROOM × ENDPOINT cross-tabulation — the numbers

`H_regime` is defined identically across runners as `mean_r[max_a V(a,r)] − max_a[mean_r V(a,r)]` — the value of knowing the regime when picking a **constant** (`scripts/run_mixed_risk_downstream_headroom.py:13`, `run_contention_headroom_v1.py:9`, `run_surface_gates_v1.py:131`, `run_cobb_douglas_headroom_v1.py:120`, `run_estar_capacity_sweep_v1.py:196 boot_h_regime`). `H_PI` / `H_obs` / `eta` are in `supply_chain/headroom_sensitivity.py:100 headroom_at` — `H_PI = mean(oracle − static)`, `H_obs = mean(observable − static)`, `eta = H_obs/H_PI`, **computed on `ret_order`**.

### Exactly-zero headroom

| artifact | primary endpoint | H | claim_status |
|---|---|---|---|
| `results/headroom/cobb_douglas_v1/result.json` | **`R_cobb_douglas`** | **H_regime = 0.0** (LCB95 0.0, level 0.5495) | `NO_HEADROOM_EVEN_UNDER_A_SOUND_METRIC` |
| same artifact, side metrics | `flow_fill_rate` | **0.0** (level 0.6880) | — |
| same artifact, side metrics | `ret_excel_risk_conditional` | **0.0** (level 0.0027) | — |
| `results/metric_audit/contention_service_first_v2/result.json` | **`service_first_resilience_v2`** (leading component `worst_claimant_fill`) | **H_regime = 0.0**, CI95 [0.0, 0.0], `argmax_moves_across_regimes: false` | `NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT` |
| `results/headroom/g1_ablation_ci_zero/result.json` | `R_cobb_douglas` (contrast `ret_excel_risk_conditional`) | **H_regime_cobb_douglas = 0.0** | `HALTED_FALSIFIER_FAILED` |
| `results/sensitivity/expedite_headroom_v2/result.json` | **`service_first_resilience_v1`** (diagnostic `ret_excel_risk_conditional`) | no positive H recorded | `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` |
| `results/headroom/estar_capacity_sweep{,_v1_1,_v2_replay_20260805}` | `flow_fill_rate` | budget=1200 **H_regime = 0.0** [0.0, 0.0010115536609307195]; budget=600 **0.0002480439221355102** [0.0, 0.0017124767592135623] | `ARGMAX_MOVES_WITHOUT_VALUE`, then `HALTED_FALSIFIER_FAILED` |
| `results/garrido_risk_headroom_sensitivity_v1/result.json` | `ret_excel_request_snapshot_v2` | R1_frequency group: `H_profile_raw = 0.0` CI [0,0], `H_profile_safe = 0.0` CI [0,0], `door_pass: false` | `H_PI_established: false`, `H_obs_established: false` |
| `results/sensitivity/contention_headroom_v1_3` (per-cell) | `flow_fill_rate`, `lost_orders`, `ret_excel_omitted_n` under FIFO_PARTIAL/non-fungible | **0.0** each | `CONTENTION_DOES_NOT_OPEN_THE_DOOR` |

**Superseded / inflated variant, kept as evidence:** `results/headroom/estar_capacity_sweep_INFLATED_H_REGIME/result.json` reports **0.002275611553734236** (budget 1200) and **0.002533888987794164** (budget 600) — ~10× the corrected values, same `flow_fill_rate` endpoint.

### Non-zero but tiny (the ~1e-4 campaign) — `ret_excel_risk_conditional` primary

`results/sensitivity/contention_headroom_v1{,_1,_2}/result.json`, `claim_status: CONTENTION_DOES_NOT_OPEN_THE_DOOR`, `mechanism_check/program_o_reference/H_PI = 0.15151`:

| cell | `flow_fill_rate` | `ret_excel_risk_conditional` | `ret_excel_visible_clipped_0_1` |
|---|---:|---:|---:|
| FIFO_PARTIAL, non-fungible | 0.0 | **1.5271791080907066e-04** ← `best_non_fungible` | 1.5271791080909147e-04 |
| FIFO_PARTIAL, fungible | 3.739303394961757e-04 | 3.140243658605879e-05 | 6.266259842276078e-04 |
| R24_AGE_PARTIAL, non-fungible | 0.0 | 1.4090690967484033e-04 | 1.4090690967488717e-04 |
| R24_AGE_PARTIAL, fungible | 9.511113539029115e-05 | **6.787358230600634e-05** ← `best_fungible` | 4.1580950826480834e-04 |
| SPT_FULL, non-fungible | **1.5061798861750863e-03** | 4.098937147919675e-05 | 2.8849951749044395e-04 |
| SPT_FULL, fungible | 0.0 | 0.0 | 0.0 |

`v1_3` (extra metrics, same seeds): `best_non_fungible` 1.2272948293104696e-04, `best_fungible` 1.4585548689558317e-05; FIFO_PARTIAL non-fungible `ret_excel_full_ledger` 3.13089285764484e-04, `ret_thesis` 1.5077914357466277e-04, `ret_excel_visible_clipped_0_1` 6.723654124756173e-04. Under *fungible*: `lost_orders` H_regime **0.02777777777777779**, `ret_excel_omitted_n` **0.375**, `ret_excel_visible_n` **0.4305555555555429** — i.e. the censoring channel carries orders of magnitude more "headroom" than the resilience score itself.

Other `ret_excel_risk_conditional` primaries:
- `results/sensitivity/bottleneck_asymmetry_v1/result.json` — `claim_status: HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION`. `best_cell/H_PI = 2.0942379471449724e-03`; `fase_1a_reference/H_regime = 0.000153`; `program_o_reference/H_PI = 0.15151`. **Every `subst=True` cell is exactly 0.0** (9/9); non-substituting cells range 5.306182999219779e-04 … 2.0942379471449724e-03.
- `results/sensitivity/perrisk_and_new_node_v1/result.json` — `H_with/H_regime = H_without/H_regime = 8.392069517451119e-06`, **`headroom_gain = 0.0`**, `HALTED_FALSIFIER_FAILED`.
- `results/sensitivity/perrisk_crn_v1/result.json` — `targeted_headroom/H_regime = 2.7721951609216422e-05` over factors `["freq_R23","freq_R24"]`, `HALTED_FALSIFIER_FAILED`.
- `results/sensitivity/second_order_risk_search_v1/result.json` — `risk_search/H_regime = 5.1360069233338974e-05`, `DEVELOPMENT_SECOND_ORDER_AND_RISK_SEARCH`.
- `results/sensitivity/backlog_confirmation_v1/result.json` — `REFUTED_BACKLOG_SENSOR_WAS_SELECTION_NOISE`.

### The head-to-head metric bake-off — `results/sensitivity/multi_metric_headroom_v1/result.json`

4,375 runs, 16 metrics + Cobb-Douglas on the **same** runs, weekly cadence, `claim_status: DEVELOPMENT_MULTI_METRIC_HEADROOM`, 5 falsifiers pass. `H_regime` per metric per regime-set:

| metric | all_seven | mixed | pure | H/SD (doc) |
|---|---:|---:|---:|---:|
| `flow_fill_rate` | **0.004634913222360537** | 0.005511468220997906 | 0.0034661732241773047 | 0.028 |
| `ret_continuous` | 0.0010909711176456116 | 0.0008233909920247751 | 0.000710115759240737 | 0.004 |
| `ret_excel` | 0.0003446766384620492 | 0.0001947700408312103 | 0.0005445521019696864 | **0.002** |
| `cobb_douglas_index` | 0.0002385429030883346 (LCB95 0.0001599258108253565) | 0.0001819821045399017 | 0.00030061062060193766 | 0.021 |
| `ret_excel_full_ledger` | 0.0002011178026503635 | 0.00022019106588856696 | 9.734218070622802e-05 | — |
| `ration_ret_excel` | 0.00019940389190220564 | 0.00022111767086005485 | 9.836649139988163e-05 | — |
| `ret_excel_cvar10` | 5.213312121761515e-05 | 2.1212695696601238e-05 | 5.920769552203442e-05 | 0.001 |
| `ret_excel_cvar05` | 3.455646771551024e-05 | 7.275323633871574e-06 | 3.242500474438109e-05 | 0.001 |
| `ret_excel_risk_conditional` | (doc) 0.000307, LCB95 0.000190 | — | — | **0.131** ← best |
| `ret_thesis` | (doc) 0.000135, LCB95 0.000075 | — | — | 0.069 |
| `ret_excel_rolling_4w_min` | (doc) 0.000005 | — | — | 0.030 |

Headline from `docs/RESULTADO_METRICAS_HEADROOM_2026-07-31.md`: **normalised headroom is 65× larger under `ret_excel_risk_conditional` than under `ret_excel`**; the canonical metric is among the worst; **CVaR variants are at the bottom and the "tails would show more magnitude" hypothesis is explicitly refuted**; the largest *raw* headroom belongs to `flow_fill_rate` (25× `ret_excel`) — *"Es incómodo para el encuadre del paper y por eso va aquí."*

`results/sensitivity/mixed_risk_downstream_v1/result.json` (same family, `DEVELOPMENT_MIXED_RISK_HEADROOM`): `H_all_seven = 1.82427500198451e-04`, `H_mixed_only = 1.732538753363433e-04`, `H_pure = 1.2129775406943821e-04`.

### The one large, custody-verified ceiling — `ret_visible` / `ret_excel_request_snapshot_v2`

`papers/paper2/results_table.md` (machine-generated):

| Level | Cell | Point | Bound |
|---|---|---|---|
| L1 physical opportunity | all (safe oracle) | **H_PI = 0.15151** | LCB95 = 0.11562, fungible null = 0 (exact) |
| L2 classical H_obs | rho75_share90 | LCB95 = **+0.06595** | 44/48 favorable; CVaR10 LCB −0.0086 (pt +0.0350) |
| L2 classical H_obs | rho90_share75 | LCB95 = **+0.04303** | 42/48; CVaR10 LCB −0.0155 (pt +0.0195) |
| L2 classical H_obs | rho90_share90 | LCB95 = **+0.05860** | 46/48; CVaR10 gate met |
| L3 learned H_OL | rho75_share90 | +0.07576 | LCB95 +0.04323 |
| L3 learned H_OL | rho90_share75 | +0.06261 | LCB95 +0.03659 |
| L3 learned H_OL | rho90_share90 | +0.10455 | LCB95 +0.06630 |
| **L4 neural premium Δ_N** | rho75_share90 | **−0.00165** | LCB95 −0.00879, 1/10 seeds |
| **L4 neural premium Δ_N** | rho90_share75 | **−0.00273** | LCB95 −0.01400, 0/10 seeds |
| **L4 neural premium Δ_N** | rho90_share90 | **−0.00150** | LCB95 −0.00828, 2/10 seeds |

Exact value `0.15151378920653932` is pinned in `contracts/program_o_fixed_clock_physical_hobs_validation_v1.json`, `program_o_hobs_prelearner_v1.json`, `program_o_state_rich_*`. `docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md`: *"The only material ceiling in the entire search."*

Supporting `ret_excel_request_snapshot_v2` runs:
- `results/program_o/exact_transducer_screen_v1/result.json` — `mean_h_pi` per cell: **0.18588934318655767, 0.1566362589433537, 0.20707833758898508, 0.14609217083762807, 0.0**.
- `results/program_o/exact_transducer_validation_v1/result.json` — `observed_h_pi`: **0.16448529209167592, 0.12845452083896094, 0.16203099747639588, 0.13544172661506512**.
- `results/program_o/gate_o0_full_action_transducer_v1/{screen,validation}/result.json` — `raw_h_pi = safe_h_pi` per profile, screen 0.08353856285463623…**0.18937653378992736**, validation 0.10176033482622304…**0.2014358869194488**; **all three `null_profiles` are exactly 0.0** in both.
- `results/program_o/affected_order_bound_v1/result.json` — `h_obs_established: false`, `h_pi_established: false`.
- `results/program_m/hpi_screen_v1/result.json` (`governing_metric: ret_excel_request_snapshot_v2`) — `h_pi_mean` over 19 cell summaries: 0.0, 6.673582533954965e-06 (×2), 0.0025733627989237957 (×2), 0.009835807201249061 (×2), 1.6016799136294817e-05 (×2), 0.009378245039521437 (×2), 0.019973587199269168 (×2), 0.0004715246728913408 (×2), 0.015432131300006126 (×2), **0.03624534184608839** (×2).
- `results/program_m/hpi_validation_v1/result.json` — `observed_h_pi`: 0.02737500037647318 (×2), **0.04114780825740077** (×2), 0.011581271684172043 (×2); `authorization/h_obs: false`.

### Surface / search lane (`auc_regret_norm`, not a resilience metric)

`results/surface_gates{,_v2}/result.json` — `g1_h_regime/H_regime = 0.003802243800697269` (LCB95 1.08e-16 at n_boot 1000, 1.3519330929717688e-05 at n_boot 5000). `results/surface_gates_extended{,_v2}` — **0.02829448338047058**, LCB95 0.014715148015042251 — a **7.4×** increase from adding decision variables. All four: `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. `search_ladder{,_ordered,_v2,_v2_ordered,_v3,_v4}` and `search_surrogates{,_smoke}` all use `primary_metric: auc_regret_norm` with statuses `NEURON_BEATS_THE_FULL_CLASSICAL_LADDER` / `NEURON_LEADS_BUT_NOT_ABOVE_EVERY_MEMORY_ARM` / `APPROXIMATOR_IS_NOT_THE_INGREDIENT_RETENTION_IS` / `NEURAL_SURROGATE_PREMIUM_IN_THE_OUTER_LOOP`.

### `worst_claimant_fill` as primary — never produced positive headroom

`results/headroom/contention_policy_class` → `HALTED_FALSIFIER_FAILED`; `..._smoke` → `POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT`; `g3_obs_conversion` → `STOP_G3_OBS_UNDERPOWERED`; `g3_obs_conversion_smoke` → `HALTED_FALSIFIER_FAILED`; `g3_obs_conversion_v2` → `STRUCTURED_CONTROL_SUFFICES_G3_OBS`; `g3_obs_conversion_v2_replay_20260803` → `REPLAY_OF_BURNED_BLOCK_NO_NEW_CONFIRMATION`; `g3c_preflight_burned` → `PREFLIGHT_HALTED_FALSIFIER_FAILED` (+`contract_status_for_fresh_seeds: DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`); `g3c_preflight_grid_v2` → `STOP_G3C_GUARDRAIL`. **No sealed positive H under this endpoint.**

### Argmax-moves tests

`results/metric_audit/contention_service_first_v2/result.json` `argmax_by_regime` — the cleanest instrument comparison in the repo:

| endpoint | R2r base/×3/×3×2 | R1r+R2r base/×3/×3×2 |
|---|---|---|
| `flow_fill_rate` | 0.5 / 0.5 / 0.5 | 0.5 / 0.5 / 0.5 |
| **`v2` (service_first)** | **0.5 / 0.5 / 0.5** | **0.5 / 0.5 / 0.5** |
| `ret_excel_clipped` | 0.9 / 0.9 / 0.9 | 0.1 / 0.1 / 0.1 |

`docs/RESULTADO_SERVICE_FIRST_V2_CONTENCION_2026-08-01.md` retracts the family attribution: across three seed blocks the *side* of the U is unstable (5,200,001+: 0.1/0.9; 5,600,001+: 0.1/0.1; 6,400,001+: 0.9/0.1) — **what is stable in all 18 cells is that `ret_excel` always picks an extreme (0.1 or 0.9, never 0.5) and every sound endpoint always picks 0.5.**

`docs/RESULTADO_HEADROOM_COBB_DOUGLAS_2026-07-31.md` §3: Cobb-Douglas and ReT disagree on the optimum in **6/6 regimes** (0.5 vs 0.1); and ReT's argmax **changes with step cadence** — 0.9 for R1r+R2r under `sim.run()` in Phase 1A, 0.1 under daily stepping. Cadence dependence therefore **inverts the recommended policy**.

`results/metric_panel/panel_v1.json` (`docs/METRIC_PANEL_2026-07-29.md`, `DEVELOPMENT_SCREEN_NO_CLAIM`): at posture 168/0/168, S1 and S3 have **identical fill (0.99634), zero lost, zero unresolved**, yet `ret_excel` (0.005568 → 0.005594), `ret_excel_full_ledger` and `ret_excel_cvar10` all pick S3 at **2.15× the cost** (κ 834,377 vs 388,544), while `R_cobb_douglas` picks S1 (0.5969 vs 0.5400). `agreement_is_floor_robust: true` from thresholds 20,000 to 200,000.

`results/program_g/terminal_metric_audit/verdict.json` (endpoint `ret_order`): `passing_observable_policies: []`; cover vs best periodic static — `ret_order` −0.02317453625750402 CI [−0.028162880439771405, …], `worst_cssu_fill` 0.6731 vs 0.8465 (−0.173), `unfulfilled_rations_at_horizon` 11,116.90 vs 7,746.62. `forbidden_claims: ["full_des_confirmation", "cobb_douglas_rescue", "virgin_mfsc_confirmation"]`.

### Sealed prospective confirmation on `ret_excel` (bounded)

`docs/RET_METRIC_REPAIR_CONFIRMATION_V1_OUTCOME_2026-07-30.md`, `results/metric_audit/ret_metric_repair_confirmation_v1/result.json` (`PROSPECTIVE_CORRECTIVE_CONFIRMATION`), contract SHA-256 `c1efdc20fc9d75743a5789ab3ddf90108d7f18cf1230dcd62fb5a85815dc441e`:

| family | Δ (MPC − frozen static) | paired CI95 | tapes | verdict |
|---|---:|---|---:|---|
| R1r | **−0.00001954** | [−0.00004940, −0.00000021] | 5/16 | `NOT_CONFIRMED` |
| R2r | **+0.01247474** | [+0.00910860, +0.01590910] | 15/16 | `PASS_MATERIAL_REPAIRED_MPC` |

**Endpoint-dependence on the same R2r tapes:** bounded `ret_excel` +0.012475; canonical `ret_excel` +0.012516; **`ret_excel_full_ledger` −0.004483** [−0.006600, −0.002388]; `ret_thesis` +0.000370 [−0.001084, +0.001731]; flow fill +0.002340 (CI straddles 0); delivered rations **−25,399**; strategic material **−99,072**. Terminal: `PASS_MATERIAL_REPAIRED_MPC_R2R / NOT_CONFIRMED_R1R / HOLD_RESOURCE_ADJUSTED_OR_FULL_LEDGER_SUPERIORITY / NO_GO_NEURAL_OR_KAN_AUTHORIZATION`.

### Program I — `ret_order`, off-contract

`research/paper2_exhaustive_search/global_headroom_sensitivity_design_and_results.md`: mean observable advantage **0.0100–0.0114**, LCB95 **0.0072–0.0087** (below the 0.01 requirement), worst-CSSU fill deteriorates **0.126–0.128** → fails the fairness guardrail. Local branching restricted-oracle means: production posture **0.00003951** [0.00003255, 0.00004674]; Op9 dispatch cadence **0.00001127**; Op10/Op12 split **0.00001045** — all "too small, horizon unstable". The 64-cell atlas's apparent **64/64 `H_PI = 0`** under direct visible-ledger optimisation is explicitly disowned as *"an unguarded sparse-ledger shed-to-win degeneracy"* — every cell selects `HOLD^4`, whose zero visible rows score 1.0 while **all 48 generated orders are lost**. Its two `H_obs ≥ 0.01` cells fail lost-order non-inferiority (+0.4792 [+0.1875,+0.8333] and +0.1667 [+0.0208,+0.3542]).

`contracts/program_q_frozen_policy_replication_v1.json` / `program_s_product_mix_risk_interaction_gsa_v1.json` define `Delta_N = learner − max(classical)` with `LCB95(Δ_N) ≥ 0.01` for premium, equivalence if CI95 ⊂ [−0.01, +0.01]. `research/paper2_exhaustive_search/program_q_power_preopen_v5_verdict_20260718.json` gives `Delta_N_equivalence` power 0.528 (N=128), 0.653 (160), 0.7595 (192), 0.8755 (256), all at `H_OL = 1.0`.

---

## 7. Documents that retract or criticise a metric

Primary (read in full for this report):
- `docs/RET_METRIC_DEFECTS_2026-07-29.md` — the three ReT defects; contains two explicit in-document corrections of earlier drafts (`_cumulative_down_hours` → `_op_down_since`; "most-delayed order gets best score" → false).
- `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md` — *"Retiro dos afirmaciones que hice antes de medir"*; kills `ret_excel_visible_clipped_0_1` as a candidate endpoint.
- `docs/AUDITORIA_SERVICE_FIRST_METRIC_2026-08-01.md` — `lost_orders` is queue-overflow, not abandonment; `RECONCILIADO 2026-08-01` header.
- `docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_UNRATIFIED_PROMOTION_2026-07-17.md` — the unratified promotion, reverted.
- `docs/RET_EXCEL_VISIBLE_V1_CONTRACT_2026-07-11.md` — supersession/quarantine header.
- `docs/RESULTADO_SERVICE_FIRST_V2_CONTENCION_2026-08-01.md` — *"la atribución por familia se retira"*; `f6` falsifier retracted and rebuilt (*"Quinto falsador en tres días que probaba un correlato"*).
- `docs/METRIC_PANEL_2026-07-29.md` — the three-mechanism failure table (censoring / case collapse / no lost-order price).
- `docs/RESULTADO_METRICAS_HEADROOM_2026-07-31.md` — CVaR hypothesis refuted; a self-corrected 300× → 1.0× cost estimate.
- `docs/RET_METRIC_REPAIR_CONFIRMATION_V1_OUTCOME_2026-07-30.md` — the "not allowed" claim list.
- `docs/RET_REPAIR_PREREGISTRATION_2026-07-29.md`, `docs/RESILIENCE_METRICS.md` (step-level approximation), `docs/RET_TAIL_STEEPNESS_AUDIT_2026-06-17.md`, `docs/TRACK_B_CVAR_METRIC_NOTE_2026-07-06.md`, `docs/CVAR_GATE_INSTRUMENT_AUDIT_VERDICT_2026-07-17.md` + `..._INTERPRETATION_CORRECTION_2026-07-17.md`, `docs/PROGRAM_G_TERMINAL_METRIC_AUDIT_VERDICT_2026-07-12.md`, `docs/PROGRAM_G_METRIC_TRIANGULATION_2026-07-12.md`, `docs/COBB_DOUGLAS_FACTORY_METRIC_AUDIT_2026-07-13.md`, `docs/ERRATA_FIDELITY_REFERENCE_V4_2026-07-31.md`, `docs/AUTOTOMY_PROXY_CORRECTION_2026-07-30.md`, `docs/EXPANDED_CONTRACT_COMPARATORS_RECLASSIFICATION_2026-07-29.md`, `docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md`.

The full keyword hit list is 60+ files under `docs/`; the ones above are the ones that actually change a metric's standing.

---

## 8. Bottom line for the endpoint decision

- **Only one endpoint has ever produced large, custody-verified headroom: `ret_visible` / `ret_excel_request_snapshot_v2` in Program O** — H_PI 0.15151 (LCB95 0.11562), H_obs LCB95 +0.043…+0.066, H_OL LCB95 +0.037…+0.066. Its Δ_N is negative in all three cells. And the governing status of that ledger convention is still `PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION`.
- **Every "sound" endpoint gives exactly zero.** `R_cobb_douglas`: 0.0. `service_first_resilience_v2` / `worst_claimant_fill`: 0.0, CI [0,0]. `flow_fill_rate`: 0.0 in the Cobb-Douglas run, 0.0 or ≤1.5e-3 elsewhere. Their argmax is 0.5 in 6/6 regimes.
- **The endpoint most likely to *change* your conclusion is `ret_excel_risk_conditional`** — 65× more normalised headroom than `ret_excel` (H/SD 0.131 vs 0.002), with a stated mechanism (it excludes the decision-independent fill-rate branch). But it inherits every `ret_excel` pathology except dilution: same cadence carrier, same censoring, same unbounded `0.5/RPj` tail, and its argmax still lands on 0.1 rather than 0.5.
- **`flow_fill_rate` carries the largest raw headroom of any endpoint measured on the same runs** (0.004635, 25× `ret_excel`) and is not a resilience index — flagged in-repo as uncomfortable for the paper's framing.
- **CVaR is closed as a headroom-finder**, on measurement: `cvar05`/`cvar10`/`p05` sit at H/SD ≈ 0.001, and `ret_visible_cvar10` is contractually `REPORTED (never a gate)`.

Two things I could not fully close: `results/headroom_gsa/all_cells_reconstruction.json` (56 Morris + 40 GP + 1 confirmation cell) was not enumerated cell-by-cell, and the q_r1 successor lane (`contracts/q_r1_ret_service_successor_draft_v1.json`, endpoint `early_ret_complete_cohort` / `early_ret_visible`, SESOI 0.01, status `DRAFT_NO_SEEDS_NO_EXECUTION_AUTHORITY`; `results/q_r1/successor_confirmation_v1/*` = `PROSPECTIVE_CONFIRMATION`) lives only on `codex/paper-b-cf1-cf20-replication` and exposes no `H_*` field under the key names you listed — worth a dedicated pass if that successor endpoint is a live candidate.
