# Program D — Systematic discovery of the dynamic decision rights that matter (PREREGISTRATION, frozen 2026-07-11)

> **2026-07-11 update / provenance chain:** the D1 execution details are superseded by
> `docs/PROGRAM_D_D1_V2_PREREGISTRATION_2026-07-11.md` (risks-on correction after the v1 screen
> was found to run with `risks_enabled=False` — see
> `docs/PROGRAM_D_D1_V1_AUDIT_2026-07-11.md`). Two external committee verdicts and
> their reconciliation are archived under `docs/external_assessments/` and integrated
> there (sharper gate thresholds, binding lost-order/backlog guardrails, family ranking
> #1 CSSU / #2 rationing=D1 / #3 batch-timing, CRN bitwise-identity requirement,
> biblio fixes, "Discovering Decision Rights Before RL" C&IE framing). This document
> stands as the program-level frame; the D1-v2 prereg governs the current experiment.

Status: **frozen before any experiment.** Combines two converging plans:
(1) the systematic decision-rights sweep across Op3–Op13 (the full legal
surface, not hand-picked contracts); (2) the structural filter — a lever can
only carry DYNAMIC value if "always-max/always-on" is infeasible or
self-defeating BY CONSTRUCTION, not merely by an added price — plus the cheap
gate instrument (authority screen + clairvoyant branching, zero PPO, zero
virgin tapes). Metric primary = `ret_excel` (Garrido Excel ReT), never
`ret_thesis`, never `order_level_ret_mean`.

## 0. Why this is not a repeat of the four stopped programs

Tracks A / B-P / C and Program L, plus the downstream-reserve v2/v3 gates, all
measured constants-optimal (≤ a few % headroom) and STOPPED before PPO. Their
common physics: unlimited storage (§6.5.5), no holding cost in ReT, weekly-mean
metric ⇒ always-max is feasible and optimal ⇒ a constant wins. **The lesson is
not "RL never wins here"; it is "we kept testing levers where always-max is
feasible."** Program D targets the levers where it is NOT.

The **structural criterion** (gate before any lever enters the screen):
> A candidate lever qualifies only if there is no single setting that is
> feasible-and-weakly-dominant in every state — i.e. the decision is a genuine
> allocation/tradeoff (zero-sum service, latency-vs-throughput under a real
> shared constraint, or an intertemporal exposure tradeoff), not a
> monotone "more is better" knob.

## 1. The legal decision surface (thesis §6.3.3 pp.84–87, verified first-hand)

PI/author domain rule: everything from the WDC/CDC downward is a legitimate
decision variable; **Op1 (MLA contracting) and Op2 (external suppliers) are
NOT** (first-tier suppliers are outside the MLS, Fig. 6.1). Candidate rights:

| Op | Decision right | Structural-criterion verdict |
|---|---|---|
| Op3 | Q & ROP of shipment to plant | monotone-ish; screen but low prior |
| Op4 | expediting / capacity / route around a hit corridor | qualifies IF LOC capacity is finite |
| Op5–7 | per-station capacity, overtime, WIP, **maintenance window** | maintenance qualifies (intertemporal) |
| Op7–8 | **batch size & release timing** (5,000 threshold, 48 h headway) | qualifies IF shipments consume a real shared resource |
| **Op9** | **priority / admission / partial fulfilment of the standing queue** | **QUALIFIES — zero-sum, one order/day served, cap-60 evictions forced** |
| Op10 | convoy count/capacity & dispatch | qualifies IF convoy capacity is finite |
| Op11 | **allocation between the two CSSUs, reserve, lateral transshipment** | qualifies IF CSSUs are disaggregated |
| Op12 | route/mode/priority on the last vulnerable arc | qualifies IF capacity finite |
| Op13 | prioritisation & partial fulfilment of contingent (R24) orders | qualifies — allocation, not "max" |

Distinction held throughout (from plan 1): inventory/WIP/backlog are STATE;
demand and risk occurrence are EXOGENOUS; PT is physical (controllable only via
a concrete action — expediting, added capacity, alt-transport); ReT/CT/fill are
OUTCOMES. Only Q, ROP/headway, priority, allocation, routing, activated
capacity, and maintenance are DECISIONS.

## 2. Candidate levers, ranked (structural criterion + cost to test)

**D1 — Op9 rationing / sequencing / partial-fulfilment rule (§6.5.4). RANK 1.**
The standing 60-slot backlog is served one order/day by the daily freight
(`_op9_daily_freight_dispatch` releases `pending_backorders[0]`), and the cap-60
overflow evicts the sort tail as LOST (Ut). Both channels are driven by
`_backorder_priority_key`. Verified first-hand: this binds in the DEFAULT
op9_linked physics through (a) who-is-served-today and (b) who-is-shed-on-
overflow. Zero-sum, no "max", direct Ut/Bt→ReT coupling, NO new physics.
Cheapest and strongest candidate. Rule set (pure reorderings): SPT+contingent
(current default), FIFO+contingent, LPT+contingent, pure-SPT, pure-FIFO,
age-threshold (orders older than τ jump the queue), + a state-contingent
oracle at Phase 3.

**D2 — Op7/Op8/Op9 batch size & release timing. RANK 2.** The 5,000-ration
batch gate at Op7/Op8 and the daily-freight cadence at Op9 create standing
waits. Latency-vs-throughput is a genuine tradeoff ONLY if shipments consume a
real shared resource (vehicle slots/day, LOC exposure per departure).
REQUIRES a declared physical extension (finite convoy slots) before it
qualifies — flagged, not assumed.

**D3 — Op5–7 maintenance scheduling. RANK 3.** The 24 h/week maintenance (EC =
TC − 24 h, §6.3.2) is currently implicit/fixed. Defer vs advance ↔ later R11
breakdown exposure = intertemporal tradeoff, no "max". REQUIRES modeling the
maintenance↔R11 coupling (one declared extension).

**D4 — Op11 CSSU allocation / lateral transshipment. RANK 4.** Thesis names TWO
CSSUs; the env aggregates them. Disaggregating creates a spatial allocation
problem (which theatre node to serve/reserve). REQUIRES disaggregating Op11 +
a second demand stream — larger build; deferred behind D1–D3.

**Out of scope (already measured, or author-forbidden):** buffer levels, global
shift count, continuous dispatch-rate multipliers, Op1/Op2 anything.

## 3. The discovery instrument — five phases, per lever (no PPO until Phase 5)

**Phase 1 — Physical liveness.** Hold all else constant; force the lever
low/base/high on identical tapes+seeds (CRN); assert the flow actually changes
(inventory, WIP, queues, deliveries, CT, backlog, Ut, ReT). A lever that does
not move the flow is eliminated immediately. (For reordering levers like D1,
liveness = non-identity of Ut/Bt/ReT across rules on ≥1 tape.)

**Phase 2 — Static frontier / authority screen.** Evaluate all feasible
constant settings of the lever, paired CRN, ~30 calibration tapes. Report best
constant, worst constant, and the best−worst gap with paired CI95. This both
(a) tests authority (does the lever move the metric at all, statically) and
(b) fixes the strong same-contract comparator so we never manufacture a win
against a weak baseline. Authority present iff best−worst gap CI95 excludes a
preregistered δ_authority.

**Phase 3 — Counterfactual branching (clairvoyant bound).** At representative
sampled decision states: replay the exact prefix (strict-CRN streams make this
bitwise), then from the same state and same random future, branch every action
and simulate 4 weeks / until recovery; record the best action → build
Q(X_k, a). Promotes only if the optimal action VARIES with state in a material
fraction of states AND the per-tape oracle gain over the best constant ≥
δ_oracle with CI95. Reuses `scripts/run_l_branching_headroom.py` machinery.

**Phase 4 — Observable convertibility.** Train a SMALL cross-fitted tree (not
RL) to predict the best action from ALLOWED observations only (GroupKFold by
tape). A lever is promoted only if it clears ALL of:
1. physically changes outcomes (Phase 1 pass);
2. ≥2 actions are each individually optimal in ≥15% of states;
3. the dynamic oracle beats the best constant by ≥5% (Phase 3);
4. the cross-fitted observable tree captures ≥50% of that headroom;
5. beats the best constant with CI95 on held-out tapes;
6. holds resources equal or is Pareto non-dominated.

**Phase 5 — Interactions (only for promoted levers).** Fractional-factorial /
Morris screen of pairs for complementarity (e.g. Op9 priority × Op10 capacity;
Op7 batch × Op8 release; CSSU reserve × Op12 routing; maintenance × WIP).
ONLY after Phase 4 does any RL contract get defined — PPO learns over levers
that PROVED headroom, never an intuitive list.

## 4. Preregistered thresholds (frozen; δ_min agreed with PI before any look)

- δ_authority (Phase 2): best−worst constant gap ≥ 2% of the static ReT range,
  CI95 excluding 0.
- δ_oracle (Phase 3): per-tape clairvoyant gain over best constant ≥ 5% on
  service_loss_auc AND co-directional on ret_excel, paired CI95 > 0.
- action-variability (Phase 3/4 crit. 2): ≥2 actions each optimal in ≥15% of
  sampled states.
- convertibility (Phase 4 crit. 4): observable tree captures ≥50% of oracle
  headroom.
- Co-primary for D1 specifically: `service_loss_auc`, `total_unattended_orders`
  (Ut), with `ret_excel` — disclosed because rationing acts on the ledger
  directly.

## 5. Stop discipline & honest prior

Four decision classes measured ≤ a few % headroom; prior for any single lever
is low. Two features distinguish D1–D3: forced tradeoff (no always-max) and
direct ledger coupling. If the full Tier-1 screen (D1–D3) returns null, that is
NOT wasted: it upgrades the boundary paper's coverage claim from "four
hand-picked decision classes" to "the thesis-complete decision surface below
the CDC exposes no dynamically convertible frontier" — a materially stronger
result. No PPO run is authorized for any lever until its Phase 4 observable
gate passes. No env iteration except at the Phase 2/3 calibration stage. Git
timestamp before reading any Phase-3+ result; virgin universes untouched.

## 6. Execution order (this program)

1. D1 lever implementation (bitwise-safe default) — DONE alongside this doc.
2. D1 Phase 1 (liveness) + Phase 2 (authority screen) — first experiment.
3. D1 Phase 3 (branching) only if Phase 2 shows authority.
4. D2/D3 physical extensions designed only after D1 verdict.
5. ChatGPT-Pro external review folded in: any new candidate passes THIS §3
   screen, no exceptions.
