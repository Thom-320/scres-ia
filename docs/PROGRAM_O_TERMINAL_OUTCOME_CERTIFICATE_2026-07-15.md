# Program O — H_obs claim-boundary record (state-rich fit + dual-resource diagnostic)

**Date:** 2026-07-15
**Status:** `OBSERVABLE STATE VALUE MEASURED (development) — H_obs NOT YET CONFIRMED.` The frozen
state-rich fit stopped on the actual-use resource gate; a preregistered dual-resource diagnostic
then executed the placebos that the fit had gated out. Under the **fixed-clock reserved-capacity**
resource convention there is a **stable, state-dependent observable signal**; under **pay-per-use**
there is not. Everything now hinges on one freight-economics fact. Sealed validation `7420049–7420096`
remains unopened.

> **History.** Commit d29c29b claimed a terminal boundary ("state adds nothing"); that was an error —
> the fit's `information_placebos` were `null` (never ran, gated behind the resource gate). Retracted
> in 022abd0. The dual-resource diagnostic (below) has now **actually run the placebos** and shows the
> opposite: the rich operational state **does** add material value over the belief under fixed-clock.

---

## 1. Established quantitative ceiling (H_PI) — custody-verified (unchanged)

safe H_PI **0.15151**, simultaneous safe LCB95 **0.11562**, exact fungible-null **0.0**,
25,177-episode parity, conserved throughput. Commit `6ad6f10`, verdict `98ce2ce`, `result f5f2da8d…`.

## 2. State-rich fit (source run) — STOP on the actual-use gate

Run `program-o-state-rich-fit-v1-20260715` (commit `041dcef`, result `d67ac97a`, sealed tapes never
opened) → `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`. All 10 controllers reach material ReT (0.038–0.102)
with metric guardrails clean and reserved capacity equal, but out-transport the full frontier
(`strict_actual_use_pass: False`). The `STOP` is **correct as scored** under the frozen contract
(actual-use binding); it may not be retroactively relabeled. The information placebos never ran.

## 3. Dual-resource diagnostic — the placebos executed

Run `program-o-dual-resource-diagnostic-v1-20260715`, commit `a9733d0`, result `e48606e7`
(transfer-verified), producer exit 0, burned fit tapes `7420001–48` only, `validation_seed_accessed: False`.
Status `DIAGNOSTIC_STABLE_SIGNAL_FIXED_CLOCK_ONLY`. Post-hoc development diagnostic; does **not** confirm
H_obs or relabel the source STOP.

**belief-MPC (configs 3 and 4)** pass the fixed-clock state-rich increment on a connected component of
three cells — **rho75_share90, rho90_share75, rho90_share90** (spans both ρ and share levels) — beating
**every** information placebo with paired one-sided LCB95 > 0:

| Estimand (real observation minus placebo, LCB95) | rho75s90 | rho90s75 | rho90s90 |
|---|---|---|---|
| **Total observable** (vs no-state / stale-t2 / swapped / cross-tape) | +0.058…+0.145 | +0.050…+0.174 | +0.085…+0.186 |
| **Incremental operational state, given belief** (vs belief-only / stale-op / swapped-op) | **+0.037** | **+0.025** | **+0.073** |
| Incremental belief, given state (vs stale-belief / operational-only) | +0.008…+0.010 | +0.008…+0.011 | +0.016…+0.017 |

`diagnostic_gates`: `incremental_operational_state_pass: True`, `incremental_belief_pass: True`,
`fixed_clock_total_observable_pass: True`, `state_counterfactual_pass: True` — but
`pay_per_use_state_rich_increment_pass: False`, `pay_per_use_total_observable_pass: False`. No controller
passes under pay-per-use.

**Finding:** at equal production and equal **reserved** fleet, the rich operational state provides
**material, state-dependent observable value over the regime belief** (development stage). This directly
overturns the retracted "state adds nothing." The value is real only if idle reserved-fleet utilization
is free — i.e. only under fixed-clock.

## 4. The single decisive open fact

**Is the downstream fleet fixed-clock reserved (charged whether loaded or empty — under which the signal
is real) or pay-per-use (under which it is a resource confound)?** This is a Garrido/thesis freight-model
question and is now the critical path. Program O has no other blocking uncertainty.

## 5. Disposition and the legitimate route to a terminal outcome

- Do **not** open sealed validation yet; do **not** authorize a learner; H_obs **not confirmed**;
  no Paper 2 / Paper 3 claim. This is a development signal on burned tapes.
- **Do not open the sealed tapes under the fixed-clock contract until the fleet convention is
  independently justified** — otherwise it is post-hoc selection of the convention that passes.
- **If the freight model is fixed-clock reserved:** freeze belief-MPC (selected on burned tapes),
  preregister a fixed-clock H_obs **validation** contract, open sealed `7420049–96` once. A clean OOS
  pass → H_obs established → classical-observable Paper 2 candidate → then the neural-learner gate.
- **If pay-per-use:** the boundary stands (perfect-information ceiling real; observable conversion is a
  transport-utilization effect).

Custody: diagnostic `e48606e7`, fit `d67ac97a`, H_PI verdict `f5f2da8d`; sealed tapes intact.
