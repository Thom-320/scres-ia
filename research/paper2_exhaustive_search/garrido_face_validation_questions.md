# Garrido face-validation questions — canonical Q1–Q14 batch

Date: 2026-07-13
Status: **CANONICAL HUMAN-FACING BATCH**

This is the only face-validation question document to send to Garrido. It
consolidates the former Q1–Q10 draft and the Q11/Q12 addendum. The broader
`garrido_decision_questions.md` remains a source inventory, and
`garrido_questions_addendum_R09_APP_2026-07-13.md` remains an audit trail; neither
is a second sendable batch. Machine adjudication is governed by
`garrido_family_question_state_transitions.json`, and answers must be recorded
verbatim in `docs/GARRIDO_FACE_VALIDATION_RESPONSES_2026-07-13.md`.

Purpose: Programs D–K3 provide terminal or boundary evidence under their final contracts. The integrated bottleneck M/T/R extension is still numerically unresolved: its tested signal policy is null, but its equal-resource 11,184,811-calendar frontier has not been certified. The other candidate extensions are `blocked_domain_fact`. These questions are falsifiable and family-specific; they determine whether the proposed researcher-introduced physics is operationally licensable, but they do not by themselves close the current numerical M/T/R contract.

These questions should be sent to Garrido as a batch BEFORE any further computational work on the corresponding family. No family below may be implemented without its question being answered and recorded in a frozen intervention ledger (the DRA-2 pattern).

## Rules for interpreting an answer

- A “yes” is not permission to assume missing values. It must include an
  operational definition, units, a plausible range, timing/authority semantics,
  and a source record or explicit signed expert judgment.
- Silence, ambiguity, “possibly,” or an answer without the required operational
  fields is `UNANSWERED`; the family state does not change.
- A domain-positive answer changes a family at most to
  `DOMAIN_FACT_SUPPORTED_CONTRACT_REQUIRED`. It never makes the family
  `promotable`, never establishes `H_PI` or `H_obs`, and never authorizes a
  learner.
- A domain-negative answer blocks only the stated mechanism under the answered
  facts. It is not a global impossibility proof over unasked physics.
- Linked families bundled under one question are adjudicated independently.
  A positive answer for one linked family does not reopen the others.
- The canonical development endpoint is `ret_excel_request_snapshot_v2`,
  provisional pending Metric Clarification M1 below. Cobb–Douglas remains
  secondary construct sensitivity unless Q12
  supplies both the APP ledger and Garrido's explicit MFSC-domain endorsement;
  even then, a new frozen contract and recalibration are required before it can
  become a co-primary construct.

---

## Metric Clarification M1 — request-ledger timing (mandatory before virgin confirmation)

**Context:** The raw workbooks attach `sumBt`, `sumUt`, order number and `OPTj`
to each visible request row. Thesis Annex B describes the same barrier matrix as
associated with each request generated. The earlier Python contract instead
reconstructed `Bt/Ut` at `OATj`; that implementation is now quarantined.

**Question:** “In the original Simulink request barrier, were accumulated
backorders `sumBt` and unattended/lost orders `sumUt` frozen when order `j` was
generated at `OPTj`? If another completion, queue removal or loss occurred at
the identical simulation timestamp, did that event update `sumBt/sumUt` before
or after the snapshot attached to request `j`?”

**Effect of answer:** This does not create adaptive headroom. It validates or
corrects only the deterministic event-order convention of
`ret_excel_request_snapshot_v2`. Ambiguity keeps virgin confirmation blocked;
all development tapes and comparators must be rescored after any correction.

---

## Q1 — Multi-product mix (reopens F8 / multi_ration_product_mix)

**Context:** The thesis (Ch 8) explicitly compresses 21 real ration types into one homogeneous product (Cold Weather Combat Ration #1) and flags this as a limitation. The DES currently models a single product.

**Question:** "Among the 21 real ration types produced by the MFSC:
(a) Do they share a common bill of materials / raw-material kit, or do some require distinct ingredients or components?
(b) On the Op5–Op7 assembly line, is there a setup/changeover time when switching between ration types, and if so, approximately how long?
(c) Is substitution between ration types operationally authorized (e.g., a climate-compatible swap when the primary type is stocked out), and by what rule?
(d) Are there approximate demand shares for the main types, and do they vary by theatre/climate/season?"

**Linked operational field — approved substitution:** "May commanders satisfy a
requisition with another ration type without violating nutrition, climate,
mission or contract requirements, and when is that substitution decision made
and logged?"

**Reopens F8 if:** products have distinct BOMs OR nonzero setup OR partial (non-universal) substitution. These facts create the possibility of ranking reversal; binding shared capacity, reachable heterogeneous states, variable demand and a pre-commitment signal are still necessary.
**Closes only the product-choice dimension if:** all 21 types are perfectly substitutable with zero setup and identical BOM/rate. The contract then collapses to an aggregate replenishment problem; this does not prove that one particular aggregate base-stock policy globally dominates.

---

## Q2 — Finite fleet / persistent vehicle location (reopens F10)

**Context:** The thesis (Ch 6.5) takes vehicle availability and route planning as given ("assumed available"). The DES's DRA-2 lane modeled a single finite convoy on one leg (Op8) as binary HOLD/DISPATCH. A multi-destination finite fleet with persistent location is materially richer but is a researcher extension.

**Question:** "For the transport legs Op10 (Supply Battalion → CSSU) and Op12 (CSSU → Theatre):
(a) How many vehicles or convoys does the Supply Battalion actually have available?
(b) When a vehicle is dispatched to CSSU-A, is it unavailable for CSSU-B until it returns, and what is the approximate round-trip time?
(c) Can a vehicle be rerouted mid-trip, or is the destination fixed at the moment of dispatch?
(d) Are the vehicle pools for Op8, Op10, and Op12 shared (one fleet) or separate (dedicated per leg)?"

**Linked operational fields — reservation, lateral transfer and mode choice:**

- "Are Op10 and Op12 served by one finite shared fleet that must be reserved in
  advance; if so, what are its capacity, vehicle-hours, booking lead,
  dwell/cancellation/reassignment rules, link-specific degraded lead times and
  pre-booking signal characteristics?"
- "Do CSSUs laterally resupply one another; if so, is that route faster than SB
  resupply, what vehicle/load/travel/return/risk rules apply, and is its lift
  distinct from SB-to-CSSU transport?"
- "For Op10/Op12 delivery, which land, air or water modes are available; what are
  their payload, fleet count, one-way/turnaround time, mission eligibility and
  R22 exposure; what is known before dispatch; and does committing one mode
  remove that asset from later epochs?"

**Reopens F10 if:** there is a finite fleet (≥1 vehicle, preferably 1–2 to create real contention) serving ≥2 destinations with a round-trip commitment that makes the vehicle temporarily unavailable. Program I found scarcity×concurrency to be the live interaction inside its frozen spatial contract; that motivates this binding-resource test but is not a global theorem over all mechanisms.
**Closes F10 if:** vehicle availability is effectively infinite (the thesis assumption) or each destination has a dedicated vehicle with no contention.

---

## Q3 — Demand observability under disruption (reopens F1 / censored_demand)

**Context:** The thesis (Ch 6.5) assumes late orders become backorders (cap-60 pending list, SPT service). The DES records every requested order quantity. Programs K2/K3 (replenishment) assumed fully observed demand and found no observable conversion. A lost-sales/censored-demand contract is materially different but requires that the true demand be genuinely unobserved in some states.

**Question:** "When the theatre cannot receive rations (CSSU down, line of communication cut, or stockout):
(a) Is the quantity of rations actually demanded recorded as a number, or is only the existence of the shortfall known?
(b) Does the Supply Battalion learn the true quantity that was needed, and if so, how and with what delay?
(c) Is there any disruption mode in which the demand signal is lost or censored rather than merely deferred?"

**Linked operational field — multi-echelon reporting and authority:** "What
inventory/backlog/consumption/convoy reports move from each CSSU to SB, with
what latency/error, which decisions are central versus local, and are 24-hour
central shipments irrevocable?"

**Reopens F1 if:** demand quantity is genuinely unobserved under at least one disruption mode (the posterior over censored demand becomes a belief state with proven adaptive value — Huh et al. 2009, Zhang et al. 2020).
**Closes F1 if:** the MFSC always records the full requested quantity even under stockout — then censoring is false and F1 reduces to K2/K3 (closed).

---

## Q4 — Route/LOC observability at dispatch (reopens F4 / regime_lead_time)

**Context:** R22 (line-of-communication disruption) is thesis-native, affecting Op4/Op8/Op10/Op12 with ~24h mean rehabilitation. DRA-2b assumed the route/convoy state was directly observable. If R22 status is only inferable from missing convoy arrivals (imperfect observation), the dispatch decision becomes a POMDP with a materially different information structure.

**Question:** "When a line of communication is hit by an R22 disruption:
(a) Does the Supply Battalion know immediately that the route is cut, or does the disruption manifest only as a delayed or missing convoy arrival (i.e., information from the absence of an expected event)?
(b) Is there any partial leading indicator (e.g., intelligence, weather, prior incident pattern) available before dispatch that correlates with route status?"

**Linked operational field — alternate-route recourse:** "Does the MFSC operator
choose among at least two routes from Op8 toward the same downstream demand
using one finite shared fleet; if so, what are each route's payload,
outbound/return times, R22 exposure, commitment and reassignment rules,
degradation persistence and warning available before dispatch, and are 36 h
each way, +24 h degraded, persistence 0.85, prevalence 0.25 and signal accuracy
0.85 plausible?"

**Reopens F4 if:** route status is imperfectly observed at dispatch time (creates a POMDP). The value hinges entirely on this — if route status is observable in real time, F4 reduces to DRA-2b (closed).
**Closes F4 if:** R22 status is always known immediately when it occurs.

---

## Q5 — High-consequence failure mode for alarm maintenance (reopens F7)

**Context:** Program J found a maximum screened restricted-oracle ret_excel
effect of `0.00013383238214118324` on the serial Op5–Op7 line. Clearing the
Paper 2 practical gate of `0.01` therefore requires at least
`0.01 / 0.00013383238214118324 = 74.72×` that best observed effect (and more
once observable conversion and guardrails are imposed). A materially new alarm
family must supply that order of endpoint liveness; merely increasing signal
accuracy or failure frequency reparameterizes the closed J/I mechanism.

**Question:** "Among the station-failure risks (R11 workstation breakdown ~2h repair; R21 natural disaster ~120h recovery on Op3/5/6/7/9; R3 black-swan 672h fixed downtime):
(a) Is there any failure mode where an observable precursor (alarm, wear indicator, intelligence) predicts the failure sufficiently in advance to act, AND
(b) The downstream inventory cannot cover the resulting downtime (i.e., the failure has ret_excel consequence, not just service loss)?

**Reopens F7 if:** such a precursor-predictable, ret_excel-consequential failure exists. The candidate would then face a Whittle-index / condition-based threshold comparator plus all maintenance calendars.
**Closes F7 if:** no such mode exists — then alarm maintenance reduces to J (closed).

---

## Q6 — Integrated shared resource identity (closes or reopens the bottleneck family)

**Context:** The bottleneck M/T/R contract freezes one scalar response-team week and treats M/T/R as allocation destinations. This is equal in the contract's scalar budget, but the real cross-stage fungibility of that team and the reserve issue/replenishment accounting remain unvalidated. Its exact equal-resource frontier is still unresolved. A domain-defensible version requires ONE real, named resource that is mutually exclusively committed across manufacturing, LOC, and theatre response — not three separate budgets summed.

**Question:** "In the real MFSC, is there a single named resource (a specific team, vehicle, crew, or skill pool) that is allocated mutually exclusively among (a) plant/assembly recovery, (b) line-of-communication repair, and (c) theatre mission response — such that committing it to one precludes the others for a real activation + dwell time? Where is it based, what skills, travel, activation and minimum dwell apply, and how does one unit causally change recovery time at each target?

For the proposed forward reserve, when an R24 requirement is served and the reserve orders back up to 10,000 units from Op9: (1) is replenishment only a stock-conserving internal transfer using non-scarce routine order-processing and transport, or does each request/transfer consume a separately finite departure, convoy/vehicle-hour, handling crew-hour, fuel/budget or order-processing slot; (2) is that resource shared with Op10/Op12 or with the M/T/R response team; (3) is the 5,000-unit R capability a maximum per R24 event, per week or per episode, and does each issue consume team or handling time; and (4) is the initial 10,000-unit stock prepositioned before the episode and how is it costed? If any separate resource is finite, please give its capacity, units, lead/turnaround, concurrency and cancellation/reassignment rules."

**Licenses a domain-valid M/T/R extension if:** such a single mutually-exclusive resource exists with a real identity and causal effect on recovery time at each target, and every separately finite reserve issue/replenishment resource is named and budgeted. The family would still need an exact same-contract frontier or valid resource-restricted ceiling under the real resource semantics.
**Blocks the current cross-stage interpretation if:** the three response functions use separate, non-transferable resources. That answer removes the proposed physical mechanism; it does not retroactively turn an uncomputed numerical ceiling into a computed one.

---

## Q7 — R21/R3 shared Maintenance Battalion restoration teams (reopens restoration sequencing)

**Context:** R21 disables Op3/Op5/Op6/Op7/Op9 concurrently and R3 disables Op5/Op6/Op7/Op9. The thesis names the Maintenance Battalion, but its executable recovery clocks are independent and parallel. Therefore current-physics sequencing has exact zero action liveness. A historical registry called a one-crew extension `FALSIFIED` through restoration-order invariance, but no machine proof supports that global claim; the extension is domain-blocked, not closed.

**Question:** "After one R21 or R3 event, does the Maintenance Battalion have fewer cross-trained recovery teams than disabled sites, forcing repairs to serialize? If so: (a) how many teams exist and where are they based; (b) what travel and activation times apply; (c) what operation-specific repair-time distributions apply; (d) are repairs preemptible; (e) what damage assessment is available before dispatch and with what lead/error; and (f) do the thesis's 120/672-hour values represent team repair work or autonomous downtime?"

**Reopens restoration sequencing if:** the team pool is genuinely scarce, assignments are mutually exclusive for nonzero travel/repair time, and a nonprivileged damage assessment arrives before commitment. The first executable screen must face every fixed restoration permutation, upstream/downstream/critical-path/WSPT rules, exact finite-state DP and resource-constrained MPC.

**Blocks the extension if:** recoveries are autonomous/parallel, teams are at least as numerous as disabled sites, or no pre-dispatch observation can change assignment. The null cell is the thesis-default independent parallel recovery.

---

## Q8 — Op7 inspection authority and latent quality state (reopens inspection effort)

**Context:** The thesis already specifies Op7 quality control and R14 Binomial defects at `p=.03` (current) and `.08` (increased), with detected items returned to Op6; it does not specify 100% detection, while the current DES treats every generated defect as detected. The current DES has no inspection action, so incremental inspection `H_PI=H_obs=0`. On a full 2,564-unit producing shift, the probability of zero generated R14 defects is about `1.21e-34` or `1.42e-93`; merely raising the rate does not create a predictive state and reduces to closed buffer/shift routes.

**Question:** "Is Op7 100% inspection or a sampling plan, and does the stated `0.003125 h/ration` include inspection plus packaging? May inspection intensity vary? If so, give allowable sampling fractions or inspector counts, minutes/test, reassignment lead/minimum dwell, sensitivity/specificity/false rejects, whether R14 is latent or detected nonconformance and whether it persists by lot/shift/crew/machine/supplier, what signal is available before allocating inspection, the consequence of an escaped defect, rework time/yield/repeat/scrap rules, and the mandatory AQL or qualified-delivery standard."

**Reopens inspection effort if:** there is a persistent pre-action quality state, a deployable leading signal, variable inspection authority with measured effort-response, a conserved inspection/line resource and a real escaped-defect consequence. The evaluator must CRN-couple exogenous per-unit/lot latent draws and enforce `latent defective = inspected TP + inspected FN + uninspected defective`, `latent conforming = inspected TN + inspected FP + uninspected conforming`, `escaped defective = inspected FN + uninspected defective`, and `inspected TP + inspected FP = reworked + scrapped + held/queued`; only qualified/reworked output may fulfill demand.

**Blocks the extension if:** inspection is fixed, defects are independent with known constant `p`, no pre-action signal exists, or escaped defects have no distinct operational consequence.

---

## Q9 — Component-specific R13 and finite Op4 load priority (reopens kit balancing)

**Context:** The thesis names 12 raw materials and models R13 over 12 planned deliveries, but the executable DES aggregates all inputs and applies a common Op2 delay. The decision-right catalog incorrectly routed Op4 raw-material transport to DRA2, which is the finished-ration Op7–Op8 convoy lane. No component-specific Op4 action exists, so current-kernel `H_PI=H_obs=0`.

**Question:** "In R13, do the Binomial delayed deliveries identify named `rm1...rm12` suppliers/components, and may WDC choose the component composition or priority of a capacity-limited Op4 shipment? Please specify component-level on-hand and pipeline observability, BOM consumption, mixed-load/minimum-lot rules, payload weight/volume, dispatch and return capacity, expedition authority/resource, and whether any component substitution is permitted."

**Reopens kit balancing if:** named component shortages are observed before a finite mixed-load commitment and the kit-limiting component can change across states under equal total component and lift resources.

**Blocks the extension if:** R13 is a common delay, shipments must remain proportional kits, component stocks stay proportional, lift is nonbinding, or component identity is unavailable before dispatch.

---

## Q10 — R14 detected-lot disposition (reopens rework versus replacement)

**Context:** The thesis mandates return of detected defects to Op6. The DES's `r14_defect_mode` is a constructor/fidelity configuration, not a sequential decision. Under the current equal-rate, unit-yield rework semantics, rework saves raw material and weakly dominates discard, so the absent disposition action has `H_PI=H_obs=0`.

**Question:** "When Op7 detects R14 non-conforming rations, is return to Op6 mandatory, or may the operator choose lot hold/rework versus scrap-and-replace? Please specify when defect quantity is known, rework duration and capacity, yield, material requirement, replacement lead and raw-material use, partial-batch release rules, disposal cost, quality acceptance constraints, and whether the original R14 event remains attributed to the replacement order."

**Linked operational field — finite storage (a separate family bundled here only
to keep one twelve-question packet):** "What are usable capacities at WDC,
assembly, SB and CSSUs, which space is dedicated/fungible, what happens on
overflow, and what resource/lead is required to relocate or reconfigure stock?"

**Reopens disposition if:** the decision is operationally discretionary and validated rework time/yield/capacity versus replacement material/lead can reverse the preferred choice across observed states while preserving quality and risk attribution.

**Blocks the extension if:** return to Op6 is mandatory or unit-yield same-rate rework remains weakly dominant.

---

## Q11 — Mission-expiry / admission triage (reopens R09)

**Context:** The thesis assumes late orders become backorders (never abandoned).
R09 requires hard deadlines + permanent abandonment + triage authority — a
non-work-conserving admission problem, materially distinct from D1 (closed).

**Question:** "For theatre requirements — especially R24 contingent-demand
orders:
(a) Does an unfilled ration requirement have a HARD deadline after which it is
PERMANENTLY abandoned (mission moves/ends, rations no longer needed), rather
than backordered indefinitely?
(b) If so, what is the deadline distribution, and is it TIGHTER than the
24–120 h R21/R23/R22 recovery timescales?
(c) Does the logistics agency hold doctrinal authority to triage/decline which
orders enter the fulfillment pipeline (admission control)?
(d) Do R24 orders carry mission-criticality CLASSES beyond the single
contingent-priority flag?"

**Linked operational field — mission loadout and carried autonomy:** "Before
deployment, may logistics allocate a fixed total ration loadout among mission
cohorts using observed mission-duration and resupply plans; what are the
pack/mass limits, decision authority, signal timestamps and accuracy,
demand-ledger semantics, and return/transfer rules?"

**Reopens R09 if:** hard deadlines tighter than recovery times exist AND triage
authority exists. The mission-loadout family is adjudicated separately and also
requires a conserved total loadout, observed cohort information before an
irrevocable allocation, and sealed demand-ledger semantics.

**Blocks R09 if:** orders always backorder, deadlines are looser than recovery,
or no triage authority exists. Under those facts it collapses to D1; treating
an expired denominator order as if it never existed is forbidden.

---

## Q12 — APP cost ledger and MFSC scope of the Cobb–Douglas construct

**Context:** The Garrido et al. 2024 factory-resilience index needs five
variables: inventory, backorders, spare capacity, time-to-fulfil and cost. The
MFSC DES computes inventory/backorders/time but its current cost ledger omits
hiring, firing and overtime (the APP labour costs). The rejected CD v1 runner
does not supply those facts. Governance therefore keeps Cobb–Douglas secondary
unless a validated APP ledger, a sourced calibration and Garrido's explicit
MFSC-domain endorsement exist.

**Question:** "For the Op5–Op7 assembly plant:
(a) Is the workforce/shift level (S1/S2/S3) a genuine PERIOD-BY-PERIOD decision,
with hiring, firing, and overtime costs — or is the balanced-line workforce
fixed?
(b) What are the RELATIVE cost coefficients (even approximate ratios) for
regular production, overtime, holding inventory, backorders, hiring, and
firing?
(c) Is there installed-vs-used capacity (spare capacity) as a measurable line
quantity?
(d) Would you ENDORSE applying your published factory-resilience index to the
MFSC Op5–Op7 sub-system as a CO-PRIMARY construct alongside ReT, with
re-calibration to MFSC variable ranges — or is the index scoped only to the
standalone factory model?"

**Makes the CD-APP lane eligible for a new co-primary contract only if:**
workforce/overtime is a real period-by-period decision, the answer supplies or
identifies a validated cost ledger and measurable spare capacity, AND Garrido
explicitly endorses the MFSC mapping. Even then, the index is not promoted by
the answer alone: a versioned APP contract, MFSC-range recalibration, resource
ledger and full comparator/pre-learner gates remain mandatory.

**Keeps the CD-APP lane secondary if:** workforce is fixed with no
hire/fire/overtime decision, the APP ledger or spare-capacity variable is not
available, or Garrido scopes the published index to the standalone factory.
This answer cannot alter the canonical ReT endpoint or rescue a canonical null.

---

## Q13 — Non-fungible ration classes sharing Op5–Op7 (validates or collapses the Program O construct)

**Context:** Program O is a disclosed two-class researcher extension motivated
directly by the Ch 8 limitation named in Q1 — the thesis compresses 21 real
ration types into one homogeneous product. Program O restores the suppressed
feature by modelling **two non-substitutable classes (P_C, P_H)** that share the
single capacity-constrained Op5–Op7 assembly line and must be given a weekly
production split. Under this construct the full-DES perfect-information gap is
**custody-verified and material** (safe H_PI 0.152, simultaneous LCB95 0.116) and
is **exactly zero when the two classes are made fungible** (the causal null). The
entire value therefore rests on genuine non-fungibility at a shared bottleneck.
This question asks Garrido to (i) confirm the construct is domain-defensible and
(ii) supply the real calibration values so the result can be re-expressed for the
true MFSC rather than the researcher stand-in. Unlike Q1–Q12 this is not a
"would this reopen a family" query — the gap already exists and is audited.

**Question:** "Among the 21 real ration types produced by the MFSC:
(a) Do **two or more classes exist such that a theatre requisition for one
CANNOT be filled by another** (e.g., religious/halal/kosher, medical/therapeutic,
or climate-specific rations) — genuine non-substitutability, not merely
different SKUs?
(b) Do those classes **share the same capacity-constrained Op5–Op7 assembly
resource**, so producing one consumes capacity the other needs in the same
period — or does each class have dedicated, non-contending capacity?
(c) Is the **demand mix between classes uncertain and time-varying with
persistence** (one class dominant for a stretch, then shifting)? If so, what are
realistic values for the dominant-class share and the persistence of the dominant
regime? (Program O currently screens dominant-share ∈ {0.75, 0.90} and regime
persistence ∈ {0.75, 0.90}.)
(d) Is the class mix **partially observable in advance** of the weekly production
commitment through requisition or forecast signals — with what lead and error —
or is it known exactly, or not at all?
(e) Does **switching assembly between classes incur a setup/changeover** time or
cost? (Program O currently assumes zero setup and identical BOM, mass and rate.)
(f) Is a **weekly allocation of a small number of production batches** (Program O
uses three batches over an eight-week horizon) a realistic decision cadence and
granularity, or is the real decision continuous, daily, or annual?
(g) Is a **two-class reduction a defensible abstraction**, or is the real
structure many small classes (which changes the allocation combinatorics)?"

**Validates the Program O construct — and licenses an MFSC-representativeness
claim — if:** at least two classes are mutually non-substitutable (a), share a
capacity-constrained bottleneck (b), and the mix is uncertain, persistent (c) and
imperfectly observable before commitment (d). The certified H_PI and any
observable-headroom result may then be re-calibrated to the real (share,
persistence, setup, cadence) and reported as representative of the MFSC. Non-zero
setup (e) does not collapse the construct; it must be added to the physics and
re-run.

**Collapses the Program O construct to the exact fungible null (H_PI = 0; no
Program O paper) if:** all 21 types are mutually substitutable at fulfilment
(a false), OR each class has dedicated non-contending assembly capacity (b false),
OR the class mix is deterministic or known exactly in advance (c/d false). Under
any of these the two-class distinction carries no information value and Program O
reduces to the aggregate single-product problem already modelled.

**Non-blocking note.** This question refines *external calibration and the
MFSC-representativeness claim*; it is **not** a gate on the internal research. The
two-class extension is disclosed and self-motivated by the thesis's own 21→1
compression, so the state-rich observable-headroom gate proceeds on the researcher
extension regardless of Garrido's timing. A "collapses" answer retires Program O
as a real-MFSC claim (converting it to a boundary/abstraction result); a
"validates" answer upgrades it to a calibrated MFSC claim. It does not authorize a
learner or a metric change — the full pre-learner gate chain still applies.

---

## Q14 — Downstream fleet contracting model (resolves the Program O H_obs resource estimand)

**Context:** Program O's dual-resource diagnostic (result `e48606e7`) showed that under a
**fixed-clock reserved-fleet** cost model a state-dependent observable controller (belief-MPC)
captures material adaptive value — incremental operational-state value LCB95 **+0.025 to +0.073**
over a connected 3-cell component, beating every information placebo — but under a **pay-per-use**
transport model it does not. The controllers use ~**2,280 of 5,376** charged downstream vehicle-hours
(**42 % utilization**) and deliver otherwise-stranded rations by filling idle capacity. Whether that
idle-capacity use is **free** (fixed-clock) or a **purchased resource** (pay-per-use) is the single
fact that decides whether Program O has real observable headroom or a transport-utilization artifact.
This is the critical-path question; all internal experiments are exhausted.

**Question:** "For the downstream transport legs Op10 (Supply Battalion → CSSU) and Op12
(CSSU → Theatre):
(a) Is freight capacity **reserved and paid on a fixed schedule** (vehicles/hours contracted and
charged whether loaded or empty), or is cost **incurred per trip / per unit hauled** (pay-per-use)?
(b) If fixed-schedule: is there routinely **idle reserved capacity** (unused vehicle-hours inside the
contracted envelope) that additional deliveries could fill at no incremental transport cost?
(c) If pay-per-use: what is the marginal cost structure (per trip, per vehicle-hour, per ration-km)?
(d) Are the 112 daily dispatch slots and 5,376 downstream vehicle-hours a hard reserved entitlement, a
soft target, or a metered resource?"

**Resolves Program O to the POSITIVE branch if:** the fleet is fixed-clock reserved with routinely idle
capacity (a + b) → filling idle reserved capacity is free → the diagnostic's fixed-clock signal is real
→ execute a preregistered out-of-sample validation of the frozen belief-MPC controller on the sealed
tapes `7420049–7420096`. A clean OOS pass establishes classical H_obs > 0 (then the neural-learner gate).

**Resolves Program O to the BOUNDARY branch if:** transport is pay-per-use, or the reserved envelope is
tight with no idle capacity (a / c / d) → the observable ReT gain is a purchased-transport effect → the
perfect-information ceiling (H_PI 0.152) does not convert to resource-honest observable value; Program O
terminates as a boundary instance.

**Does not, by itself, authorize** opening the sealed tapes or a learner — it selects which of the two
preregistered branches executes. Until answered, the sealed tapes stay closed (opening them under the
fixed-clock contract first would be post-hoc selection of the convention that passes).

---

## How to use these questions

1. Send all fourteen sections as one batch. Linked families within one section are
   adjudicated independently.
2. Record each answer verbatim in
   `docs/GARRIDO_FACE_VALIDATION_RESPONSES_2026-07-13.md`, including source
   evidence and sign-off.
3. Apply only the deterministic transition in
   `garrido_family_question_state_transitions.json`. An incomplete answer is
   `UNANSWERED_NO_STATE_CHANGE`.
4. A domain-supported family may proceed only to design of a versioned contract,
   followed by the full G0–G9 pre-learner gate chain on fresh tapes and the
   strongest comparator in `lit_comparator_obligations.md`.
5. If all answers block their corresponding extensions, those introduced
   mechanisms cannot be promoted. A terminal outcome B still additionally
   requires the active numerical M/T/R contract to be exactly bounded or
   explicitly excluded from the scientifically licensed envelope; domain
   answers cannot substitute for that computation.

## What these questions do NOT authorize

- None of these questions, even if answered "reopens," authorizes training a neural learner. Each reopened family must still pass physical liveness, the 0.01 H_PI gate, observable conversion (G5), the null-cell falsification (G8), and the full action-feedback certificate before any learner is trained.
- None authorizes reopening a terminal lane (D–K3) by changing its architecture, reward, horizon, or metric.
- None authorizes a metric switch, a resource-purchase win, a shed-to-win result,
  or a Cobb–Douglas rescue. Q12 can only establish eligibility to preregister a
  separately validated APP co-primary construct under the conditions stated
  above.

## Registry wording preserved verbatim

The following prompts are copied byte-for-byte from the current
`approach_registry.json`. They are included to prevent the consolidated human
packet from silently narrowing any operational field. The numbered sections
above govern presentation; the mapping JSON governs adjudication.

- `multi_ration_product_mix_setup_substitution`: Which of the 21 ration types share Op5-Op7, what are their BOMs, line rates, batch/changeover/minimum-run rules, mission and climate substitution matrix, product-demand shares and timestamps, and which operational signal is available before the mix is committed?
- `regime_lead_time_advance_transport_reservation`: Are Op10 and Op12 served by one finite shared fleet that must be reserved in advance; if so, what are its capacity, vehicle-hours, booking lead, dwell/cancellation/reassignment rules, link-specific degraded lead times and pre-booking signal characteristics?
- `censored_demand_active_information_acquisition`: During a CSSU/theatre stockout, is complete requested quantity recorded or only issued quantity; can a report be requested before commitment, what does it reveal, how early, and at what operational resource cost?
- `lateral_cssu_transshipment`: Do CSSUs laterally resupply one another; if so, is that route faster than SB resupply, what vehicle/load/travel/return/risk rules apply, and is its lift distinct from SB-to-CSSU transport?
- `multi_echelon_information_lag_only`: What inventory/backlog/consumption/convoy reports move from each CSSU to SB, with what latency/error, which decisions are central versus local, and are 24-hour central shipments irrevocable?
- `queue_admission_priority_with_abandonment`: Does an unfilled ration order become useless after a mission/deployment window, is it cancelled and logged as lost, can logistics reject/defer/reroute it, and what observed deadline classes and frequencies apply?
- `alarm_signal_finite_maintenance`: Is a condition signal available before an actual Op5-Op7 breakdown and can a finite shared crew prevent/shorten it; what are signal errors/lead, preventive efficacy/duration, repair duration and finished-stock cover?
- `r21_r3_shared_mab_restoration_sequencing`: After an R21 or R3 event, does the Maintenance Battalion have fewer cross-trained recovery teams than disabled sites, forcing repairs to serialize; if yes, what are team count, bases/travel/activation, operation-specific repair distributions, preemption, damage-assessment lead/error, and whether 120/672 hours is work content or autonomous downtime?
- `finite_storage_space_allocation`: What are usable capacities at WDC, assembly, SB and CSSUs, which space is dedicated/fungible, what happens on overflow, and what resource/lead is required to relocate or reconfigure stock?
- `inspection_effort_vs_throughput`: Is Op7 100% inspection or sampling; may intensity vary; what inspector minutes, sensitivity/specificity/false rejects, lot persistence and pre-action signals, escaped-defect consequence, rework yield/time and AQL apply?
- `multimodal_last_mile_mode_choice`: For Op10/Op12 delivery, which land, air or water modes are available; what are their payload, fleet count, one-way/turnaround time, mission eligibility and R22 exposure; what is known before dispatch; and does committing one mode remove that asset from later epochs?
- `alternate_route_recourse_same_destination`: Does the MFSC operator choose among at least two routes from Op8 toward the same downstream demand using one finite shared fleet; if so, what are each route's payload, outbound/return times, R22 exposure, commitment and reassignment rules, degradation persistence and warning available before dispatch, and are 36 h each way, +24 h degraded, persistence 0.85, prevalence 0.25 and signal accuracy 0.85 plausible?
- `integrated_production_maintenance_routing_recovery_resource`: Is there one truly fungible resource allocated mutually exclusively among plant repair, LOC/vehicle recovery and theatre response; where is it based, what skills/travel/activation/dwell apply, how does one unit causally change recovery time at each target, and is emergency-reserve stock a separately budgeted and replenished resource?
- `mission_loadout_carried_autonomy_allocation`: Before deployment, may logistics allocate a fixed total ration loadout among mission cohorts using observed mission-duration and resupply plans; what are the pack/mass limits, decision authority, signal timestamps and accuracy, demand-ledger semantics, and return/transfer rules?
- `component_specific_r13_kit_balancing_op4_expedite`: In R13, do delayed deliveries identify named rm1...rm12 components, and may WDC choose the component composition or priority of a capacity-limited Op4 shipment; what BOM, inventory/pipeline visibility, payload, return, minimum-lot, expedition and substitution rules apply?
- `r14_detected_lot_disposition`: When Op7 detects R14 nonconforming rations, is return to Op6 mandatory or may the operator choose hold/rework versus scrap-and-replace; what timing, duration, capacity, yield, material, lead, release, disposal, quality and R14-attribution rules apply?
- `demand_shaping_or_substitution`: May commanders satisfy a requisition with another ration type without violating nutrition, climate, mission or contract requirements, and when is that substitution decision made and logged?
