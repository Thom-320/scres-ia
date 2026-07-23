# Thesis-native dispatch-lever screen — verdict

**Date:** 2026-07-23
**Contract:** `contracts/thesis_native_dispatch_lever_screen_v1.json` (frozen `fba3e728`, before dev seeds)
**Instrument:** `scripts/run_thesis_native_dispatch_screen.py` (`d10fe8ad`), 3,096 direct-SimPy rows,
520-week horizon, dev seeds 7540001–7540012, elapsed 3,415 s
**Claim status:** `DEVELOPMENT_SCREEN_NO_CLAIM`

---

## Verdict

**`STOP_DISPATCH_LEVER_NO_STATE_DEPENDENT_HEADROOM` — 0 of 6 cells pass any gate.**

The strongest finding is not the failure — it is *where* the optimum sits:

> **In every one of the six risk cells, the best constant dispatch posture is multiplier 1.0 —
> the thesis's own U(2400,2600)/day setting.** Garrido's fixed dispatch quantity is already
> optimal among all seven tested multipliers (0.5–2.0), in every tested risk regime, and no
> state-dependent policy (observable or clairvoyant with 168 h anticipation) improves on it.

| cell | robust constant | best-constant ReT | H_PI oracle (mean / LCB95) | H_obs LCB95 | material arms | gate |
|---|---|---|---|---|---|---|
| R1_current (control) | **mult 1.0** | 0.0059 | 0.0 / 0.0 | 0.0 | 0 | fail |
| R2_current | **mult 1.0** | 0.7320 | no admissible arm | — | 0 | fail |
| R2_OAT_R22_increased | **mult 1.0** | 0.7050 | no admissible arm | — | 0 | fail |
| impact_R22_psi1.5 | **mult 1.0** | 0.7297 | +0.00077 / −0.00059 | — | 0 | fail |
| impact_R22_psi2 | **mult 1.0** | 0.7266 | no admissible arm | — | 0 | fail |
| R2_OAT_R24_increased | **mult 1.0** | 0.6019 | no admissible arm | — | 0 | fail |

Gates (frozen): H_PI_safe LCB95 ≥ 0.02; H_obs LCB95 ≥ 0.015; ranking reversal with ≥2
materially optimal actions. Observed: max H_PI mean +0.00077 (**26× below** the gate), zero
material two-mode arms anywhere, zero reversals.

## The mask hides nothing

"No admissible arm" could conceal headroom, so the unconstrained check was run: best raw
two-mode minus best constant, ignoring every guardrail —

| cell | best raw two-mode Δ |
|---|---|
| R1_current | +0.00000 |
| R2_current | −0.00011 |
| R2_OAT_R22_increased | −0.00077 |
| impact_R22_psi1.5 | +0.00077 |
| impact_R22_psi2 | −0.00047 |
| R2_OAT_R24_increased | +0.00048 |

In 4/6 cells the best switching policy is *worse* than the thesis constant even before
guardrails. The headroom is absent, not masked.

## Why (mechanism, consistent with the pre-registered honest prior)

The contract recorded before execution: *"R22 pauses transport servers but does NOT destroy
in-transit cargo … the value channel is post-outage surge drainage, not loss avoidance.
Headroom may well be null."* That is what the data shows. Dispatch is conservation-capped
(`min(target, on-hand)`): during an outage nothing can move regardless of target; after
recovery, the backlog drains at the rate stock arrives from upstream — which is governed by
the assembly line (2,564/day at S1), not by the dispatch target. Raising the target above
1.0 mostly cannot be executed (no stock to dispatch); lowering it only starves service
(mult 0.5 → ReT 0.377 in the smoke). The op9 lever has almost no *executable* authority in
either direction: the binding constraint the F11 audit identified is real, but its bindingness
comes from upstream flow balance, not from the dispatch cap parameter itself.

## Self-checks (all passed)

Neutral-action mapping introspected exact (op9_q bounds 2400/2600 at mult 1.0); risk-event
timeline CRN-identical across all arms per (cell, seed) — the run aborts otherwise; obs and
oracle switching live in every R2 cell (the liveness check caught a real bug in smoke:
`RiskEvent.affected_ops` vs `ops`); R1 control clean (H_PI exactly 0.0, no reversal — no
instrument artifact).

## Disclosures

- `resource` is not emitted by this env path; the admissibility mask used the 6 recorded
  guardrails. Conservative for a STOP (a resource guardrail could only remove arms, never
  add headroom). Any future PASS would need it restored.
- Development screen on 12 dev seeds; holdout 7540101–48 remains sealed and unopened.
- One lever (op9_q). op10/op12 (track_b_v1), op7 batch threshold, and the op11 CSSU split
  (`set_cssu_allocation_action`, 24 h latency) were deferred to a v2 and are NOT closed by
  this verdict — but the mechanism finding (authority limited by upstream flow, not by the
  cap parameter) lowers the prior for op10/op12, which sit even further downstream.

## Consequence

The user's preferred thesis-native lane ("Track B LOC dispatch"), sanctioned by Garrido's
"everything except ops 1–2" statement, is now **measured** rather than deferred: under
thesis-native risks at current/increased levels, the dispatch-quantity lever has no
state-dependent headroom for ANY policy — learned or classical — to convert. Combined with
the buffer×shift invariance (45 profiles), the 4⁸ answer-key result, and the (pending) 8⁸
ceiling, the thesis-native environment is closing on every tested decision frontier with
the same shape: **the thesis's own constants are already optimal; adaptive control has
nothing to add**. That is Paper 1's central claim, now with its strongest evidence yet.
