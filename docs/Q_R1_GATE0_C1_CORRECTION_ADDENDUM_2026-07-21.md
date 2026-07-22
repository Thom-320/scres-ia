# Q-R1 Gate 0 — C1 correction addendum (full-cohort endpoint + honest restatement)

**EXPLORATORY_NO_CLAIM. Burned roots 7570801–24 only. No fresh seeds. Canonical ReT untouched.**

This is the first gate of the approved two-track plan (Track B). It closes the metric-population
hole the five external audits flagged (`early_ret_2w` scored only *completed* cohort orders, dropping
unresolved/lost from the mean) by adding a **full-cohort endpoint** and reporting it beside the
visible one, under the causally-coherent natural continuation. It changes nothing under
`results/q_r1/cold_start_replication_v1/`.

## What was added (`scripts/run_q_r1_c1_natural_continuation_diagnostic.py` v2)
- `early_ret_full_2w` — the 12-order early cohort mean with **unresolved and lost orders scored 0**
  (denominator = all cohort orders). Derived from existing fields
  (`early_ret_2w * early_visible_rows / early_generated_orders`); the canonical visible metric is
  left byte-identical.
- Cohort-aligned guardrails already present (early unresolved, early worst-product, early
  service-loss) surfaced as deltas.
- **Actual downstream utilization** (`actual_loaded_departures`, `actual_payload`,
  `actual_downstream_vehicle_hours`) forwarded from the transducer panel, so a matched-resources
  audit can separate "better decision" from "used more of the reserved fleet".
- **Action-vs-belief distinction** recorded explicitly (see below).

## Result (κ = 0.90, 264 pairs)

| Endpoint | A: natural continuation | C: frozen splice (the burn) |
|---|---:|---:|
| Δ early_ret **VISIBLE** (completed-only) | +0.02256 (CI [+0.0190, +0.0260]) | +0.02264 |
| Δ early_ret **FULL-COHORT** (unresolved/lost = 0) | **+0.02303 (CI [+0.0193, +0.0268])** | +0.02261 |
| Δ unresolved (mean / max) | +0.0038 / 2 | +0.2008 / 3 |
| Δ worst-product fill | −0.00196 | −0.01774 |

**Headline:** the retained cold-start effect is **not** a metric-gaming artifact. With every cohort
order kept in the denominator, the natural-continuation advantage is **+0.02303, LCB95 +0.0193 > 0** —
if anything slightly *stronger* than the visible endpoint (dropping unresolved orders was not helping
the retained arm). Under natural continuation the service damage the splice produced is gone
(unresolved +0.0038 with CI straddling 0; worst-fill −0.002). At κ=0.75 the full-cohort Δ is +0.01395
and natural-continuation unresolved is **negative** (−0.053, an improvement) with worst-fill positive.
The iid null (κ=0.5) is exactly 0 on every endpoint.

## Honest caveats (carried forward, per the audit corrections)
- **Action-equivalence, not belief-equivalence.** A≡B calendars in 264/264 pairs. But all 264 pairs
  have *different* retained-vs-reset initial beliefs (max |Δbelief| 0.40); the belief only changes the
  first two actions in **119/264** pairs. So the ITT favorable count is 119/264 acted; the ~92%
  favorable figure is conditional on the action changing. Do not claim the posteriors converged.
- **Actual utilization differs where decisions differ.** Scheduled ("charged") resources are exactly
  equal (`resources_max_abs = 0`), but actual payload differs in 24/264 pairs (median 0, mean |Δ|
  321) — exactly the pairs where the retained arm decided differently. Legitimate under a fixed-clock
  reserved fleet, but now disclosed and to be formalized as a utilization-parity report in Gate 1.
- **Burned-seed diagnostic.** No PASS is claimed. A fresh-seed prospective replication under this
  corrected estimand (full-cohort primary, cohort-aligned guardrails, a domain-justified unresolved
  margin) is required before any Paper-3 claim.

## Self-checks (all pass)
Burn crosscheck vs the frozen replication rows **0.0 bit-exact**; forced-prefix idempotence 0
failures; iid null exactly 0; scheduled resources exact.

## Next (Gate 1)
Freeze the strongest structured MPC* (exact/stratified 6-state integration, fail-closed fallback,
objective aligned to the full-cohort/worst-product terms, frozen cross-arm CRN bank) before the
residual is recomputed.
