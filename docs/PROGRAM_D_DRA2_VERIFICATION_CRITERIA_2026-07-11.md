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
