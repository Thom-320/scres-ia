# DRA-2 verification criteria (frozen by verifier BEFORE the build, 2026-07-11)

Role: Codex builds DRA-2 (finite-convoy batch-release control at Op7–Op8) end-to-end;
this document freezes, in advance, the checks the verifier will run and the pass/fail
lines — so the audit cannot be accused of moving goalposts (the V4→V5 lesson). It
encodes the DRA-1 failure modes as up-front gates. It does NOT design the experiment
(the DRA-2 preregistration/contract is Codex's build artifact).

## Why DRA-2 could differ from DRA-1 (the thing to actually verify)
DRA-1 died because the action created NO persistent intertemporal opportunity cost
(`reallocate_unused` made it non-committal; homogeneous sinks; redistribution without
throughput). DRA-2's whole premise is that dispatching a finite convoy NOW makes it
unavailable for the next cycle. **The central verification question is therefore: does
`DISPATCH_NOW` vs `HOLD` actually create a persistent, non-recoverable difference in
tomorrow's feasible action set?** If not, DRA-2 is DRA-1 in a convoy costume.

## Pre-committed verification gates (all must hold for a credible DRA-2 result)

### G-A. Intertemporal commitment is REAL (anti-V1-A)
- Dispatching consumes a finite convoy slot that is genuinely unavailable for the full
  round-trip (48 h); HOLD genuinely preserves a usable slot/inventory.
- Independent check: from an identical state, `DISPATCH_NOW` and `HOLD` must yield
  DIFFERENT feasible action sets and resource states at t+1 and t+2 (not silently
  reconverged). Verifier will assert `A(X_{t+1}|dispatch) ≠ A(X_{t+1}|hold)` in a
  material fraction of live epochs. If the convoy state reconverges (like DRA-1's
  reallocated pool), FAIL — DRA-2 is moot.

### G-B. The regime generates LIVE epochs (anti-V2-A)
- Both actions feasible AND consequential in ≥20% of sampled epochs (the review's
  bar). Verifier will independently count `dispatch_live` epochs under the frozen
  treatment regime. If ~0 (as the generic R2r regime gave for DRA-1), the frontier is
  meaningless → FAIL until the regime is fixed.
- `dispatch_feasible` alone is not the numerator: exact CRN branches must diverge in
  physical/resource state or in the next feasible action set.

### G-C. Multi-step branching, not one-step
- Intertemporal value cannot be measured by a single 1-day pulse (DRA-1's method).
  Verifier requires the exact finite-horizon SEQUENCE oracle (7-day binary, 2^7=128
  per state on a stratified subset) in addition to one-step branching. The verdict
  must rest on the sequence oracle, not the pulse.

### G-D. No resource purchase
- The adaptive policy must be compared at EQUAL vehicle-hours / departures (or
  Pareto-accounted). Verifier will check that any ReT gain is not bought by dispatching
  more convoys. If the winner simply runs more departures, FAIL (resource purchase,
  not intelligence).
- Frozen primary implementation: compare each dynamic candidate with the best
  calibration-static policy having no more departures and no more vehicle-hours than
  the candidate. Pareto reporting is secondary and cannot, by itself, support a win.

### G-E. Standard integrity (as in DRA-1)
- Bitwise identity when the convoy extension is disabled (frozen-proxy regression).
- Mass AND convoy-slot conservation over full episodes.
- CRN: exact prefix + exogenous identity across branches (fail-closed asserts).
- Observations contain no future risk/repair/demand/regime.
- Cluster inference BY TAPE; a frozen minimum practical δ (the review suggests
  ΔReT≥0.01 and service-loss −5%) — verifier will hold the gate to the PREREGISTERED
  threshold, and flag if none was set (D1's Gate-C had no practical floor, which is why
  the negligible +8.5e-05 could not formally close it).

### G-F. Diversity + convertibility (the promotion bar)
- HOLD and DISPATCH_NOW each optimal in ≥15% of held-out states; oracle service-loss
  −5% CI95>0; ReT co-directional; depth-3 tree recovers ≥50% and beats best constant
  on virgin tapes; positive on ≥70% virgin tapes; tail not worsened. Only then PPO.

## Standing flags carried from DRA-1
- **UNPUSHED**: as of this freeze, HEAD is 12 commits ahead of origin/codex, 175 ahead
  of origin/main — DRA-1 + V1–V6 are local only. Building DRA-2 on top grows an
  unpushed pile; recommend pushing the DRA-1 branch before stacking DRA-2 (outward
  action, needs PI authorization; not the verifier's to perform).
- **DRA-1 verdict stands**: `STOP_NO_DYNAMIC_ORACLE_HEADROOM`, airtight (V5), asymmetry
  0.8% disclosed. DRA-2 does not reopen it.
- **No PPO / no virgin tapes** until G-A…G-F pass, per the frozen contract.

---
## Verification result — implementation smoke (commit 9ad0ac7, verifier audit)

Independent audit against the frozen gates. **Implementation smoke PASSES; DRA-2 is a
genuinely different (better) test than DRA-1.**

- **G-A intertemporal commitment: PASS (independently verified).** From an identical
  feasible state, `DISPATCH_NOW` sets `op8_convoy_available=False` and the convoy is
  unavailable for exactly the contracted 48 h (`actual_return_at = 960.0 = t0+48`,
  route_wait=0); `HOLD` preserves it. R22 physically pauses outbound/return via
  `_op8_route_progress`. This is the real, persistent opportunity cost DRA-1 lacked —
  dispatching now genuinely removes the convoy from the next cycle. (Note: my first
  coarse probe mis-read the return as 54 h; a 1 h-resolution re-check showed 48 h exact
  — my sampling artifact, corrected before reporting. The V4→V5 lesson, self-applied.)
- **Dispatch feasibility smoke: 49.8%. G-B strong liveness: PENDING.** The smoke
  percentage is not the scientific G-B numerator.
- **G-C multi-step oracle: implemented** (exact 2^k sequence search; 512 rollouts /
  4 sequence-states in smoke; full run 2^7).
- **G-E integrity: PASS** — identity when `op8_dispatch_mode` off (default
  thesis_full_batch runs, ret_excel 0.4829); crn_pass, mass_pass, convoy_conservation_pass,
  prefix_identity_pass all true; observation excludes future info. 73 tests pass.
- **Authorization gate is REAL** — the runner fails closed for n_tapes>4 unless a
  contract-bound PI-autonomy record validates; calibration_opened=false,
  virgin_tapes_opened=0, ppo_trained=false.
- **G-F diversity — PROMISING but only smoke.** One-step optima split HOLD 10 /
  DISPATCH 6 (62/38), both materially optimal — UNLIKE DRA-1's 97% single-action
  dominance. **This is the first lever to show genuine action diversity at smoke.**
  HARD CAVEAT: 16 smoke states on 4 disposable tapes is NOT a result; the diversity/
  headroom verdict requires the gated 60-tape calibration + full sequence oracle, and
  must clear a PREREGISTERED practical δ (G-E) and equal-vehicle-hours (G-D).

**Verifier status:** DRA-2 implementation is sound and correctly gated. No defects
found (one self-inflicted measurement artifact caught and corrected). Blocked, as it
should be, on Garrido's face-validation of the returning-convoy extension before the
scientific run. The promising smoke diversity does NOT authorize any claim yet.

---
## Gate refinements adopted from Codex's cross-review (2026-07-11) — pre-calibration

Codex's three pendings sharpen the frozen gates; the verifier ADOPTS them before any
60-tape calibration is authorized:

- **G-B strengthened.** The reported 49.8% smoke fraction is DISPATCH-*feasibility*, not
  a true live fraction. The scientific G-B now requires, per epoch,
  `live(s)=1{HOLD and DISPATCH both feasible AND yield physically DIFFERENT trajectories}`
  — auto-verified to differ in ≥1 of: future convoy availability/ETA, staged inventory,
  in-transit inventory, next feasible action. The 49.8% must NOT be presented as the G-B
  result until this stronger definition is measured on the calibration regime.
- **G-D estimand frozen (no formulation shopping).** The contract currently allows
  "resource-equivalent OR Pareto"; that lets one retrospectively pick the favorable
  framing. Before calibration, freeze ONE primary estimand:
  `ΔReT | equal vehicle-hours/departures budget` (Pareto as a disclosed secondary only).
- **G-A materialized as a machine preflight.** Turn my code-read + manual probe into an
  auditable verdict: on eligible fixtures, assert
  `A(X_{t+1}|DISPATCH) ≠ A(X_{t+1}|HOLD)` and emit pass/fail. "Intertemporal commitment
  exists" must be evidence, not a reviewer's reading.

**Commit-count correction:** HEAD is 15 commits ahead of origin/codex (not 12) — the
unpushed pile is growing; push before stacking more.

**DRA-1 claim wording (locked):** "Under the tested contract — daily allocation with
automatic unused-capacity reallocation — CSSU allocation shows scarce, sporadic,
operationally negligible dynamic value." Do NOT claim perfect A/B symmetry or that 0.25
is intrinsically optimal (0.8% SHA asymmetry disclosed; label-swap test retained).

---
## Verification — calibration gate result (commit 6e5d5b9, tag dra2-stop-2026-07-12)

Independent audit of `resource_gate_verdict.json`. **STOP_DRA2_PRE_RL_GATE is legitimate
and DRA-2 is the FIRST family with material, resource-matched dynamic headroom.**

- All my corrections were applied: strong_liveness_pass, diversity_pass,
  resource_envelope_pass all True; prefix-balanced states; realized-pattern diversity.
- **Passes are real:** oracle ReT +0.02662 CI95[+0.0165,+0.0373] (≥0.01); service-loss
  −5.17% CI95[3.27,7.05] (≥5%); lost-orders Δ=0 (no shedding). This is ~300× the D1/DRA-1
  headroom (8.5e-05) — genuinely different.
- **Resource-matching VERIFIED genuine (not a purchase):** the dominance comparator
  `threshold_5000/wait_72h` uses FEWER resources than the dynamic candidate (departures
  12.25 ≤ 12.55; unavailable-hours 574 ≤ 592.4) and the dynamic still wins +0.02662.
  `resource_envelope_pass: True`. The win is not bought with more convoys.
- **STOP reason is the horizon-sufficiency gate (my Decision 1.5), and it is CORRECT +
  DISCIPLINED.** 7d/10d first-action agreement 91.67% (direction fairly stable) BUT
  headroom-magnitude stability FAILED — the clairvoyant optimal *value* is
  horizon-unstable, so the 7-day labels don't characterize the true value and training a
  tree/PPO on them would be dishonest. The frozen rule required BOTH; only one held → STOP.
  Goalpost NOT moved despite the 91.67% temptation. observable_tree_authorized=false,
  ppo_authorized=false, holdout/virgin/PPO untouched.

**Verifier verdict:** DRA-2 is the first genuine crack toward adaptive value — real
clairvoyant, diverse, resource-matched headroom — correctly stopped short of PPO because
the signal is not horizon-converged. The clairvoyant +0.02662 is an H_PI upper bound, NOT
a deployable result (the observable-tree conversion was never run). 

**On DRA-2b (Codex's external-review question): WARRANTED and disciplined.** The finding
"material headroom but horizon-unstable" points exactly to a NEW study with a long
decision horizon FIXED EX ANTE (e.g., match/exceed the backlog-clearing timescale; freeze
the sequence horizon before any calibration). Conditions: new preregistration, new tapes,
does NOT alter this STOP, NOT presented as a confirmatory continuation of DRA-2. If DRA-2b
shows STABLE, resource-matched, observable-convertible headroom → it is the first
legitimate path to PPO. If not → 7th boundary result. This is the honest door to RL.
