# Adjudication — constraint-aware c256/c1024 escalation

**Date:** 2026-07-23
**Pre-registration:** `docs/Q_R1_CONSTRAINT_AWARE_C256_ESCALATION_PREREGISTRATION_2026-07-22.md`
(committed `fba4a1d5`, **before** this result)
**Execution:** VPS `ovh-agent-lab`, 4 shards, burned roots 7570801–7570824
**Claim status:** `BURNED_DEVELOPMENT_NO_CLAIM`

---

## Verdict — the stated prior was correct: the family is vacuous

**`CONSTRAINT_AWARE_FAMILY_IS_NUMERICALLY_IDENTICAL_TO_THE_FROZEN_SCENARIO_COMPARATOR` —
no distinct `RETAINED_SAFE_EFFECT` comparator can be built from it.**

The pre-registration fixed two outcomes in advance:

1. *"If the constraint-aware pair passes → the family becomes eligible for a separate
   safe-effect freeze."* It passes:

   | pair | agreement | mean err | q95 err | N | pass |
   |---|---|---|---|---|---|
   | `constraint_aware_h4_c256_wf0.70` vs c1024 | 0.96875 | 0.000110 | 0.000247 | 96 | **True** |

2. The **stated prior**: *"the wf0.70 floor was inactive in the burned c64 Pareto… If that
   holds at c256, then `constraint_aware_h4_c256_wf0.70` is numerically the same controller
   as the already-frozen `scenario_h4_c256_wf0.00`… That would itself be the finding."*

The prior holds **exactly**. Comparing the two families cell-by-cell over the 96 raw
convergence rows:

- cells where constraint-aware picks a different action than scenario: **0 / 96**
- max |gap| in planning value error between the families: **0.0**

The constraint-aware comparator is **bit-identical** to the frozen scenario comparator. The
`wf0.70` service floor never binds, so "constraining for safety" and "optimising pure ReT"
produce the same controller.

## Consequence

- **No separate freeze is warranted.** Freezing this family would re-freeze the already-frozen
  scenario c256 controller under a second name. Per the pre-registration, eligibility is not a
  freeze; here the freeze is declined because it would be a duplicate, not because it failed.
- **`RETAINED_SAFE_EFFECT` cannot get a distinct structured comparator by tightening the
  service floor.** The `wf0.70` floor is inactive and the neighbouring `wf0.80` floor is
  already known infeasible (`inf` value error, 0.0 agreement). There is no floor value in
  between that both binds and stays feasible in this reduced model.
- A genuinely binding deployment-safety constraint would have to come from a different
  mechanism (e.g. Garrido-validated product-coupled physics), not from this knob.

## Incidental replication — no red flag

The run also re-scored the scenario family, which reproduced the frozen receipt exactly:

| pair | agreement | mean err | q95 err |
|---|---|---|---|
| `scenario_h4_c256_wf0.00` vs c1024 (this run) | 0.96875 | 0.00010953 | 0.00024667 |
| frozen receipt `15d8d00b…` | 0.96875 | 0.00010953 | 0.00024667 |

The pre-registration required that a disagreement here be reported; there is none. The frozen
comparator reproduces on an independent VPS execution.

## Provenance

`selection_performed`, `learner_return_used`, `retained_minus_reset_used_for_selection` all
`false`. Burned roots, no fresh seeds, no learner. Development evidence closing a
comparator-construction question, not a claim.
