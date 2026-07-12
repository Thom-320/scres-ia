# Program I Carril A (full-DES GSA) — verifier audit — 2026-07-12

Codex executed the full-DES two-lane GSA (the 6th dictamen's Carril A) in parallel and closed it
STOP (`dd3b9217`). Verified here (read-only); the STOP is LEGITIMATE.

## Audited (all PASS)
- **Discipline chain intact.** Morris (`results/program_i/morris/verdict.json`) interpretation =
  `PHYSICAL_SENSITIVITY_ONLY_REQUIRES_COUNTERFACTUAL_BRANCHING` — it ranks ret_excel physical
  sensitivity but does NOT authorize RL; `rl_authorized` gated behind branching. This is the exact
  sensitivity≠headroom discipline (the project's thesis), correctly enforced.
- **Exact-replay branching, anti-DRA-1-confound.** 4,320 branches: 60 fresh tapes × 4 prefix-balanced
  states × 3 actions × 3 families × {4,8}wk. Every branch passed raw-state + flow-ledger identity
  BEFORE action; **max mass residual = 0.0** across all families. CRN/mass verified.
- **All 3 realistic decision families STOP.** Production/shifts, Op9 dispatch, Op10/12 transport:
  oracle−constant ReT ≈ 1e-5 (≪ 0.01 gate by 250–1000×), service-loss reduction 0.4–1.2% (<5%),
  horizon agreement 67–81% (<90%); resource-equal only for transport. `promote_to_observable=false`,
  `promote_to_rl=false` for every family. `confirmation_opened=false` (1110001+ sealed); no learner.
- **Gates + selection rule correct** (`global_sensitivity_v1.json`): oracle ΔReT≥0.01, support
  0.15/0.85, lowest-amplitude-never-max-outcome. `decision_right_catalog_v1.json` freezes the typed
  Op3–Op13 surface with the dependent-input (Q/ROP/capacity) caveat noted.

## Reconciliation (the two-lane answer is now complete)
- **Carril A (Codex, full DES, ret_excel):** on the realistic Garrido decision families, global
  sensitivity finds controls that move the response, but exact branching shows their state-contingent
  value is practically negligible (~1e-5). STOP, no adaptive headroom.
- **Program I stylized GSA (mine, fast lane, ret_order):** Morris → information/risk-magnitude INERT,
  only scarcity×concurrency moves headroom; GP-located H_obs>0 region is a spatial-fairness violation.
- **Together:** both lanes, by independent methods and metrics, confirm the central finding —
  editing Garrido's decision variables/risks does not create deployable adaptive headroom; physical
  sensitivity ≠ adaptive value. The full-DES lane the PI asked to open is DONE and it is a boundary
  result, not a new training lane. Manuscript is the remaining artifact.
