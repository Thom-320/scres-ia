# Enmienda de preregistro — Fase 1B: expedición con presupuesto escaso

**Status:** `DESIGN_FROZEN_NOT_RUN`

This document is an executable addendum to
`docs/PREREGISTRO_EXPEDICION_HEADROOM_2026-07-31.md`. It resolves the missing
physical effect size and action interface. It does not alter the thesis-native
lane; every expedition result belongs to the extended DES.

## 1. Physical assumption and price

The nominal eligible transport legs are `op8_pt`, `op10_pt` and `op12_pt`, each
24 h in the active downstream contract. An expedition is a priority transport
mode that reduces the **next eligible leg** by a fixed 12 h:

```text
pt_expedited = max(0, pt_nominal - 12 h)
budget_charge = 24 h per armed expedition
```

The 12 h reduction is our declared half-time assumption, not a thesis datum. It
is the primary effect size because it is interpretable against the nominal 24 h
leg. Fixed sensitivities of 6 h and 18 h are reported descriptively and are not
searched to select a headline result. No expedition changes capacity, demand,
risk frequency, risk impact or inventory mass.

## 2. Action and conservation contract

The simulator exposes:

```text
arm_expedition(leg in {op8, op10, op12})
```

An arm reserves 24 h from `expedite_budget_remaining`. It is consumed by the next
eligible invocation of that leg's processing-time hook. A request without 24 h
remaining is rejected and cannot create a negative budget. Requests may queue
behind an earlier request for the same leg when a disruption delays the next
invocation; each invocation consumes exactly one request and reductions never
stack. `B = 0` must be bitwise identical to the same episode with the expedition
feature disabled.

The runner must record, per episode:

- budget granted, charged, remaining and rejected arms;
- armed and applied leg/time pairs;
- nominal and expedited PT;
- order-level CT changes and flow/mass residuals.

## 3. Calendar comparison

For each budget `B ∈ {0, 168, 336, 672, 1344}` h/year, the number of possible
expeditions is `floor(B / 24)`. All arms use the same exogenous seed/tape and
the same number of charges. The calendar is closed during the first 42 days
(1008 h), which is a fixed conservative action epoch after the first eligible
Op8/Op10/Op12 hooks in the baseline, and during the final 14 days, which is a
fixed settlement window. No arm may spend budget outside that action window.

- **Constant:** one fixed leg and evenly spaced dates; the finite grid of leg and
  phase choices is optimized on calibration episodes only.
- **Tape oracle:** the same admissible candidate dates/legs, selected with access
  to the complete exogenous risk tape for the episode using the frozen risk-overlap
  score. It is an upper bound for that calendar heuristic, not an outcome-optimal
  oracle, not a deployable policy and not a claim about future-event observability.
  The name `clairvoyant` is retained only for continuity with the parent
  preregistration.
- **Placebo:** uniformly shuffled candidate dates/legs with the same total charge.

The tape-oracle and constant schedules are not allowed to inspect order-level
outcomes when defining the candidate grid. Selection on evaluation outcomes is a
falsifier failure.

## 4. Endpoints and reading rule

Primary adjudication endpoint: `service_first_resilience_v1`, the frozen
lexicographic service-first key in
`docs/PREREGISTRO_METRICA_SERVICE_FIRST_2026-08-01.md`.

`ret_excel_risk_conditional` remains a diagnostic headroom signal only. It may
not authorize a timing or learning claim unless the service-first components
are non-inferior.

Secondary construct-sensitivity endpoints:

- `cobb_douglas_index`, labelled as the researcher-constructed factory-level
  construct;
- `flow_fill_rate`, lost orders, backorders and service-loss AUC;
- canonical workbook ReT variants for continuity, never alone as a learning
  objective because of the abandonment result.

For the numeric service component define:

```text
H_PI_fill(B) = best tape-oracle flow-fill(B) - best constant flow-fill(B)
```

The remaining service-first components are reported as paired deltas:
`Delta_no_lost`, `Delta_fill`, `Delta_backorder` and `Delta_ret`. No weighted
scalar may replace them.

Promotion to a timing-headroom result requires, at the primary 12 h effect:

- `H_PI_fill ≥ 0.01` and paired `LCB95 > 0`, or a strictly positive
  lexicographic service-first win with a declared non-inferiority bound;
- tape-oracle > placebo on the same budget;
- budget, PT, mass and CRN falsifiers all pass;
- no increase in lost orders, no decrease in flow fill and no increase in
  final backorder under the paired confirmation comparison;
- any ReT improvement is reported only as a secondary component.

If the tape oracle equals placebo, the result is open-loop value of spending, not
information value. If the primary effect fails, the 6/18 h sensitivities remain
diagnostic and cannot authorize PPO. No result from this contract may be described
as the optimum of the DES reward surface: the oracle is deliberately restricted to
the predeclared exogenous-risk overlap score.

## 5. Falsifiers

| falsifier | failure meaning |
|---|---|
| `f1_budget_conserved` | the policy spent more than `B` or charged different arms |
| `f2_pt_effect_real` | applied expedition did not shorten the selected leg by 12 h |
| `f3_next_leg_only` | one arm modified more than the next eligible leg |
| `f4_zero_budget_identity` | `B=0` changed any baseline output or RNG path |
| `f5_same_exogenous_tape` | CRN/tape differs across arms |
| `f6_placebo_same_charge` | placebo used a different number or size of expeditions |
| `f7_no_future_in_deployable_arms` | constant/placebo read future risk or outcome fields |
| `f8_no_abandonment_win` | primary gain is purchased by lost orders/service collapse |
| `f9_seeds_virgin` | any confirmation seed overlaps a burned block |

## 6. Required artifacts before execution

- `scripts/run_expedite_headroom_v2.py`;
- unit tests for budget conservation, next-leg consumption and `B=0` identity;
- a sealed seed manifest using `5_800_001+` only after the custody audit;
- `results/sensitivity/expedite_headroom_v2/result.json`;
- a result note carrying `PASS`, `HOLD` or `NO_GO` without retroactive metric
  selection.
