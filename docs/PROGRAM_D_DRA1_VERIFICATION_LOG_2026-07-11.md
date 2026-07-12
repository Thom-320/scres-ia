# DRA-1 adversarial verification log (Claude Code = verifier, 2026-07-11)

Role split: Codex builds DRA-1 end-to-end; this log records independent
adversarial verification of each stage against `contracts/cssu_allocation_v1.json`
and `docs/PROGRAM_D_DRA1_V3_PREREGISTRATION_2026-07-11.md`. Verifier does not
modify physics code.

## V1 — Allocation primitive (`supply_chain/cssu_allocation.py`, commit 07b3904)

**PASS on conservation.** Independent probe over 216 cases
(stock×capacity×allocation×demand grid):
- 0 pool-enlargement violations (`total_dispatched ≤ min(stock, capacity)` always);
- 0 mass leaks (`dispatched_A + dispatched_B + unused = available` exactly);
- 0 over-service (`dispatched ≤ demand` per node);
- zero-sum split correct under joint scarcity (a=0.75 → A=1950/B=650);
- `stable_cssu_destination` is SHA-keyed, consumes no simulator RNG (CRN-safe),
  ~50/50 balance (1986/2014 over 4000 orders).

**FINDING V1-A (design risk, not a bug) — the allocation action is a no-op
except under JOINT scarcity.** Because `reallocate_unused=True` (contract default)
sends a destination's unused share to the other's unmet demand, the split
`allocation_a` only changes outcomes when BOTH destinations are simultaneously
capacity-constrained. Verified: with only B scarce, a=0.75 and a=0.25 both yield
B=2100 (identical).

Consequence for the gate: the localized-threat regime (R22/R23 taking one
node/lane DOWN) tends to make the allocation decision MOOT — when CSSU-A is
destroyed you serve B regardless. The allocation lever is actually live only
under:
1. joint scarcity (both nodes up, total demand > total capacity);
2. recovery / backlog-clearing after a localized hit (rush the recovering node
   vs maintain the healthy one);
3. demand imbalance (R24 surge at one node) with both nodes up.

**Required of the next build stage (allocation actuator + state sampling):** the
DRA-1 branching/oracle must sample states from regimes 1–3, and the frozen
localized-threat regime must actually GENERATE joint scarcity / post-hit recovery
(e.g., upstream R21/R3 scarcity or R24 surge concurrent with both CSSUs up), NOT
merely destroy a node. Otherwise DRA-1 will `STOP_NO_OBSERVABLE_SPATIAL_HEADROOM`
as a false negative — the lever looking dead because the test never placed it in a
live regime, exactly the failure mode the whole program is built to avoid on the
comparator side. This is the single highest-risk item for DRA-1 fidelity.

**Status confirmed by contract:** `allocation_action_integrated_with_des: false`,
`verdict: PENDING_ALLOCATION_ACTUATOR` — the actuator/DES wiring is the next stage;
V1-A must be addressed there.

## V2 — Allocation actuator + localized risks (commit a6068d5)

**PASS on:** 13/13 physical tests; aggregate-mode default is run-to-run reproducible
(ret_excel 0.006487, deterministic) and the frozen regression suite passes (identity
when split off); per-CSSU mass conservation holds over full episodes (raw/ration
residual 0); the V1-A `jointly_constrained → cssu_allocation_live_epochs` classifier
is implemented. Split is correctly opt-in (`cssu_topology_mode == "split_v1"`).

**FINDING V2-A (BLOCKER for a meaningful frontier) — the allocation lever is
100% MOOT under the tested R2r regime.** Independent run, `split_v1`, CF11
R2r-increased, 52 wk: **`cssu_allocation_live_epochs = 0` of 648 (0.0%).** Every
epoch is classified `allocation_moot` — i.e. `allocation_a` never changes dispatch.
This is V1-A realized at integration: with the thesis capacity≈demand balance,
splitting ~2500/day demand across two CSSUs leaves each node ~1250 < ~2500 capacity,
so both are always serviceable and the split never binds; R22/R23 that take a node
DOWN make it moot rather than live. **A static frontier run in this state would show
all 9 policies producing ~identical ReT → a false `STOP_NO_OBSERVABLE_SPATIAL_HEADROOM`.**

Required before opening discovery tapes: **empirically demonstrate that the frozen
DRA-1 treatment regime generates a MATERIAL fraction of `allocation_live` epochs**
(joint scarcity / post-hit recovery / localized-R24 with both paths up). Codex's
contract states this requirement; V2-A shows it is NOT met by a generic R2r regime,
so the materialized regime must be checked to satisfy it. Suggested guard: the
frontier smoke should assert `live_fraction ≥ some preregistered minimum` and abort
if the regime is allocation-dead.

**FINDING V2-B (secondary, confirm wiring).** Setting
`sim.cssu_daily_capacity_override` to 2000→600 left `total_unattended_orders`
unchanged (502↔503) — the override did not bind on the split dispatch path, so
scarcity could not be induced that way. Either the override targets a different path
or it is not wired into `_cssu_daily_capacity()`; Codex to confirm. (Not blocking,
but it is one lever intended to create the scarcity V2-A needs.)

## Open items to verify as they land
- [ ] Allocation actuator wired into DES; bitwise identity when split disabled.
- [ ] Destination-tagged demand conserves the aggregate demand tape exactly.
- [ ] Per-node mass balance in a full episode (A+B+in-transit+delivered = input).
- [ ] Localized R22/R23 degrade the correct node/lane AND the regime yields
      live-allocation states (V1-A).
- [ ] Observations contain no future risk/regime/demand.
- [ ] Static frontier: best-admissible constant selection honors guardrails.
- [ ] Branching: bitwise prefix + exogenous identity asserts; cluster by tape.
- [ ] PPO-vs-best-static contrast per V3 §7 (CI95>0, clipped>0, ≥70% tapes,
      ≥8/10 seeds, lost within limit, capacity conserved, placebo-separated).
