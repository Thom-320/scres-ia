# Program O — H_obs state-rich fit: claim-boundary record (CORRECTED)

**Date:** 2026-07-15
**Status:** `H_OBS_NOT_ESTABLISHED — NOT A TERMINAL BOUNDARY.` The state-rich fit stopped on
the frozen actual-use resource gate; the information placebos **were never executed**. No
positive and no "state-has-no-value" boundary can be claimed from this run.

> **Correction notice.** The prior version of this file (commit d29c29b) claimed a terminal
> `BOUNDARY_CERTIFICATE` on the grounds that "every controller fails the information placebos →
> rich state adds nothing → value saturates at the belief → state-rich increment ≈ 0
> (overdetermined)." **That conclusion was an error and is fully retracted.** `information_placebos`
> is `null` in 40/40 result rows and `pre_placebo_rule_pass` is `False` in 40/40; the screener
> (`scripts/screen_program_o_state_rich_fit.py:589`) computes placebos **only** for configurations
> that first clear every non-placebo primary gate, and 0/40 did. `information_placebos_pass: False`
> is the uninitialized default, not an experimental result. The placebos never ran, so there is
> **no evidence** about whether rich operational state adds value over the regime belief. Reading a
> default flag as a result was the mistake. Credit to the concurrent audit for catching it.

---

## 1. Established quantitative ceiling (H_PI) — custody-verified (unchanged)

safe H_PI **0.15151**, simultaneous safe LCB95 **0.11562**, exact fungible-null **0.0**,
25,177-episode parity, conserved throughput. Commit `6ad6f10`, verdict `98ce2ce`,
`result_sha256 f5f2da8d…`. This remains solid and is not affected by the error below.

## 2. What the state-rich fit actually shows

Run `program-o-state-rich-fit-v1-20260715`, commit `041dcef`, result `d67ac97a` (transfer-verified),
producer exit 0, burned fit tapes `7420001–48` only, **sealed validation `7420049–96` never opened**
(`validation_seed_accessed: False`). Terminal label `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`,
`stability.passing_cells: []`.

**Verified facts (retained):**
- All 10 controllers capture material ReT vs the full open-loop frontier (belief-DP per cell:
  0.038 / 0.068 / 0.074 / 0.102), with `metric_guardrails_pass: True` and `reserved_capacity_equal: True`.
- All fail the **actual-use resource gate**: `strict_actual_use_pass: False` and the matched-resource
  frontier is empty (`eligible_calendar_count: 0`) — every controller out-transports the entire
  65,536-calendar frontier. Mechanism (belief-DP, ρ75s90): at equal production (Δ=0) and equal
  **reserved** fleet (5,376 charged-hours, only ~2,280 used → **42 % utilization**), the belief policy
  delivers +~22,000 otherwise-stranded rations (worst-product-fill +0.26) by filling idle, already-charged
  freight.
- Only belief-MPC reached the state-perturbation certificate (passes 8/40 rows → its actions are
  functionally state-dependent) — but this measures dependence, **not incremental value**.
- The **information placebos never ran** (gated behind the resource gate) → the state-value question
  is **unmeasured**.

## 3. Two open scientific issues — both genuine, neither resolved

1. **Resource estimand (contested, but the frozen STOP is correct as scored).** It is scientifically
   reasonable to ask whether "actual transport use" is the right resource under a fixed-clock fleet
   charged whether loaded or empty (2,280 / 5,376 used). But the frozen contract
   (`contracts/program_o_state_rich_comparator_fit_v1.json`) made **actual use binding and stated that
   reserved-capacity equality could not override it**. Under that frozen contract the `STOP` is correct
   and may **not** be retroactively relabeled a false-negative. Switching to reserved-capacity fairness
   now would be a **new prospective estimand**, not a re-scoring of this run.
2. **State value (unmeasured).** Because the placebos never executed, we do **not** know whether the
   observable ReT signal requires the rich operational state or reduces to the belief. This is the
   experiment that was gated out.

## 4. Disposition

- H_obs **NOT established**. Do **not** open sealed validation, authorize a learner, claim Paper 2, or
  begin Paper 3.
- Do **not** claim a terminal "state-has-no-value" boundary — that was the retracted error.
- Program O retains a custody-verified full-DES H_PI and a development-stage observable signal that is
  **potentially compatible with reserved resources but unconfirmed**.

## 5. Legitimate continuation (a new run, not a rescue of this one)

A **new preregistered diagnostic on burned tapes `7420001–48` only** that (a) actually **executes the
information placebos** and state certificate for the candidate controllers, and (b) reports H_obs under
**both** resource estimands separately — fixed-clock reserved-capacity vs pay-per-use transport — so the
two are never conflated. That can determine whether observable value exists under the fixed-clock
interpretation. It cannot retroactively rescue this run. The construct question (Garrido Q13) remains
open and non-blocking.

Custody: fit result `d67ac97a` transfer-verified; H_PI verdict `f5f2da8d`; sealed tapes intact.
