# Domain-facts adjudication under PI autonomy — 2026-07-17

**PI directive (2026-07-17, recorded):** "No podemos depender de Garrido. Garrido nos dio
total autonomía." Accordingly, every gate previously written as
`GATE_GARRIDO_WRITTEN_SIGNOFF` or `BLOCKED_PENDING_GARRIDO_*` is REPLACED by
`GATE_DOMAIN_FACTS_ADJUDICATED`: a researcher-made adjudication of the pending domain
questions, grounded in (a) the thesis text itself, (b) public military-logistics doctrine,
and (c) published literature. This document is that adjudication.

**Rules of use (binding):**
1. Adjudications are RESEARCHER decisions, not Garrido confirmations. Every paper that
   relies on one must disclose it as such ("domain fact adjudicated from doctrine, not
   confirmed by the thesis author").
2. An adjudication UNBLOCKS prospective contracts only. It reopens no closed verdict, opens
   no seed, and authorizes no learner by itself. The unexploited-mechanism ledger rule is
   untouched: each mechanism still needs its own contract, null, and fresh seeds.
3. If Garrido ever answers, his answer supersedes the adjudication; any contract built on a
   contradicted adjudication is re-examined prospectively (never retro-invalidated — its
   results stand as conditional on the stated assumption).
4. Evidence grades: STRONG (thesis text + convergent doctrine), MODERATE (doctrine or
   analogue, thesis silent), WEAK (plausible, thin evidence — not sufficient to unblock).

Amends, per PI directive: dictamen C1/C5 in
`docs/PROGRAM_S_CRITICAL_REVIEW_2026-07-17.md` (the alarm annex changes from
`BLOCKED_PENDING_GARRIDO_ADVANCE_SIGNAL_FACTS` to
`UNBLOCKED_BY_ADJUDICATION_AF-1_CONDITIONAL` with the generator constraints below; the S4
Garrido gate becomes `GATE_DOMAIN_FACTS_ADJUDICATED` + disclosure clause).

---

## AF-1 — Advance signal (the S-P alarm annex; Program P's frozen open question)

**Question:** do military logistics decision-makers receive advance, imperfect warning of
disruptions and demand surges?

**Adjudicated answer: YES, with per-risk structure — grade MODERATE-STRONG.**

- **Thesis anchor:** §8.4 closes with Garrido's own recommendation: any change in demand
  "must be quickly incorporated into the buffering strategy, which implies that the SC
  should have a fine-tuned cross-functional **forecasting process in advance**" (extracted
  text, lines 6355–6357). An anticipatory information channel is thesis-ALIGNED, not
  smuggled.
- **Doctrine:** US Army predictive/anticipatory logistics programs describe anticipating
  demand from operational plans and automated advance alerts when planned resupply is
  insufficient ([Army: Predictive Logistics is the Way of the Future](https://www.army.mil/article/282488/predictive_logistics_is_the_way_of_the_future),
  [Army SBIR PORTAL](https://armysbir.army.mil/topics/predict-optimize-recommend-and-track-for-adaptive-logistics-portal/),
  [Defense One: The future of military logistics is predictive](https://www.defenseone.com/ideas/2025/02/future-military-logistics-predictive/402939/)).
- **Per-risk structure (constraint on any alarm generator):**
  - **R24 contingent demand:** surges originate in PLANNED operations; logistics staff
    participate in operational planning days-to-weeks ahead. Highest defensible lead and
    accuracy (internal plan signal, not intelligence).
  - **R21 natural disasters:** weather-forecast-grade warning; 24–72 h lead with
    forecast-skill accuracy is defensible.
  - **R22 LOC attacks:** least predictable; only low-accuracy threat-level warning is
    defensible. A high-accuracy R22 alarm would be researcher fiction.
  - **R11/R14 internal failures:** condition-monitoring analogue only; short lead.
- **Consequences:** the S-P alarm annex may be UNBLOCKED conditional on: (i) the alarm
  generator is specified per-risk following the asymmetry above (no uniform 0.85-accuracy
  oracle across risk classes); (ii) alarm quality levels are justified against this
  adjudication in the contract text; (iii) alarm-ON cells remain a separate stratum whose
  claims are labeled "conditional on the adjudicated signal model".

## AF-2 — Q11/R09: mission deadlines, triage, admission/eviction authority (strongest reopener)

**Question:** beyond the cap-60 list, does a hard TEMPORAL expiry exist, with criticality
classes and admission/eviction authority?

**Adjudicated answer: YES — grade STRONG.**

- **Thesis anchor:** the ReT construct itself defines a temporal expiry: "Bt accumulates
  overtime, and after a certain period (**order cancellation time**), Bt are categorized as
  Ut" (Eq 5.4 context, extracted text line 3084). The temporal deadline is IN the thesis
  construct; the Simulink cap-60 eviction was the implementation.
- **Doctrine:** UMMIPS ranks every requisition by a two-digit Priority Designator =
  Force/Activity Designator × Urgency of Need, with binding time standards, and Required
  Delivery Dates that override standards when within 8 days
  ([DSCA UMMIPS glossary](https://samm.dsca.mil/glossary/uniform-materiel-movement-and-issue-priority-system-ummips),
  [OSD time-definite delivery standards](https://www.acq.osd.mil/log/LOG_SD/TDD_Standards.html),
  [DLA ADC 57 RDD edit](https://www.dla.mil/Portals/104/Documents/DLMS/ADC/ADC057_RDDEditFinal.pdf)).
  MILSTRIP includes requisition modification and cancellation transactions — admission,
  re-prioritization, and cancellation authority are real.
- **Consequences:** a deadline-and-triage contract (the D1-successor variant the reopener
  note already scoped: temporal deadline + admission/eviction authority, NOT the exhausted
  cap-60 constant-authority variant) is unblocked for prospective preregistration.
  Criticality classes within contingent demand are doctrinally grounded (UND levels).

## AF-3 — Q6/Q7: shared restoration capacity (Maintenance Battalion)

**Question:** are post-disruption recoveries independent parallel clocks (model assumption)
or a shared, finite, priority-allocated capacity?

**Adjudicated answer: SHARED AND PRIORITY-ALLOCATED — grade STRONG.**

- **Thesis anchor:** Figure 6.1 places a Maintenance Battalion (MaB) inside the Logistics
  Brigade structure, while Tables 6.6b/6.7b model recoveries as independent exponential
  clocks — the thesis itself contains the tension.
- **Doctrine:** maintenance doctrine (ATP 4-33) is built on cross-leveling maintenance
  assets across the battlefield, pass-back maintenance when workload exceeds capacity, and
  commander-prioritized battle-damage repair — i.e., a finite shared pool with explicit
  priorities, not per-incident independence
  ([ATP 4-33 Maintenance Operations](https://sgtsdesk.com/wp-content/uploads/2019/11/ATP-4-33-Maints-Ops-July-2019.pdf),
  [Army: Sustaining Brigade Logistics](https://www.army.mil/article/239938/sustaining_brigade_logistics_concepts_of_support_sops_critical_to_long_term_sustainment_of_bct),
  [FM 4-30.31 Recovery and BDAR](https://www.globalsecurity.org/military/library/policy/army/fm/4-30-31/fm4-30-31.pdf)).
- **Consequences:** the shared-restoration mechanism (Garrido-opportunities ledger item 4)
  is unblocked for a prospective contract: concurrent disruptions competing for restoration
  crews, with a priority rule as the decision surface. This is a NEW-physics contract
  (O-v2-eligible territory), never a re-run of closed programs.

## AF-4 — Q13: ration classes, substitutability, shared line

**Question:** how substitutable are the 21 ration types, and do they share the assembly line?

**Adjudicated answer: PARTIAL SUBSTITUTABILITY, SHARED LINE — grade MODERATE-STRONG.**

- **Thesis anchor:** "the MFSC assembles 21 types of combat rations based on the
  nutritional requirements of troops and climatic conditions" (§6.3.1, extracted line
  3365); one single factory (the MFSC) assembles all of them — the shared Op5–Op7 line is
  thesis-native. The modeled product is the 'Cold weather combat ration #1'.
- **Public record (Spanish forces):** ration families RIC (menus A/B), RED, RIL, emergency
  rations; cold-weather feeding is handled by adding 500-kcal supplement modules to a base
  ration ([Sanidad Militar review of the Spanish combat ration](https://scielo.isciii.es/scielo.php?script=sci_arttext&pid=S1887-85712014000400010),
  [Revista Ejércitos: Raciones Militares (V) España](https://www.revistaejercitos.com/articulos/raciones-militares-v-espana/)).
  Menus WITHIN a family are substitutable; ACROSS climatic/dietary classes substitution is
  limited (caloric/packaging requirements; religious/medical constraints are hard).
- **Consequences:** Program O's two-class non-fungible physics (P_C/P_H sharing Op5–Op7)
  is CONFIRMED-REPRESENTATIVE in direction (its H_PI ceiling 0.152 keeps construct
  meaning). Program S's product framing stands. Magnitudes (class shares, mix persistence)
  remain researcher-set and must stay reported-as-assumptions.

## AF-5 — Q14: downstream fleet economics (fixed-clock vs pay-per-use)

**Question:** do distribution convoys run on reserved schedules (consuming vehicle-hours
loaded or empty) or pay-per-use?

**Adjudicated answer: FIXED-CLOCK RESERVED — grade MODERATE.**

- **Doctrine:** LOGPAC resupply is "the standard, preferred" method: centrally organized
  convoys on predetermined routes and timetables, typically every 24 h, with organic unit
  transport ([GlobalSecurity: LOGPAC operations](https://www.globalsecurity.org/military/library/report/call/call_99-6_logpac.htm),
  [FM 10-27-4 Methods of Supply](https://www.globalsecurity.org/military/library/policy/army/fm/10-27-4/fm10-27-4_ch3.pdf)).
  Scheduled organic transport = capacity reserved whether loaded or empty.
- **Thesis-side evidence:** the real-Excel utilization observed in the audit (~42%) is
  consistent with reserved organic capacity, inconsistent with marginal-cost pay-per-use.
- **Consequences:** the fixed-clock-reserved branch of the closed Program O dual-resource
  fork is the doctrinally supported one. Per the reopener note this is NON-DECISIVE for the
  closed verdicts (the OOS STOP and the corrective validation stand untouched); its value
  is FRAMING: Paper 2 may state that the fixed-clock assumption used in the diagnostic is
  the doctrinally realistic one, with this adjudication cited.

## AF-6 — op11: multi-CSSU competition with observable allocation authority

**Question:** do multiple CSSUs compete for scarce deliveries under an observing allocator?

**Adjudicated answer: YES — grade MODERATE.**

- **Doctrine:** commanders explicitly allocate scarce resources and set priorities of
  support across subordinate units; sustainment C2 gives the allocator visibility of unit
  status ([MWI: the division's function in LSCO](https://mwi.westpoint.edu/more-than-just-another-echelon-defining-the-divisions-function-in-large-scale-combat-operations/),
  [Joint doctrine on authorities](https://www.jcs.mil/Portals/36/Documents/Doctrine/fp/authorities_fp.pdf)).
- **Consequences:** the op11 probe's `HOLD_PENDING_DOMAIN_FACT` can move to
  "adjudicated-supported extension": multi-CSSU allocation is a REALISTIC decision surface,
  not researcher fiction. The probe's own DEVELOPMENT-NEGATIVE result (fairness eats the
  headroom) is untouched; any successor is a new contract.

## AF-7 — M2: acceptance criterion (mean vs tail)

**Question:** is operational acceptance mean resilience, or does doctrine impose worst-case
floors?

**Adjudicated answer: MEAN PRIMARY (Garrido's construct), TAIL FLOORS EXIST IN DOCTRINE —
grade MODERATE.**

- Time-definite delivery standards are percentile-based (e.g., delivery within standard a
  stated fraction of the time) — doctrinal service floors exist
  ([OSD TDD standards](https://www.acq.osd.mil/log/LOG_SD/TDD_Standards.html)).
- **Consequences:** keeps the frozen taxonomy intact: canonical mean ReT stays the primary
  (thesis construct); CVaR/worst-product remain class-C deployability guardrails, reported
  but not silent kill-switches. This matches the corrected CVaR interpretation
  (CORRECTED_INTERPRETATION_NUMERICAL_AUDIT_RETAINED) and requires no contract change.

---

## What this document does NOT do

- It does not reopen `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`, the O-R calibration
  verdict, the 45-profile invariance screen, or any other closed verdict.
- It does not open any seed and does not authorize any training run.
- It does not upgrade Program S's S1 honest prior (flat H_PI^safe remains the expected
  outcome of C2's invariance logic); it only removes the external-dependency blocks.

## Gate replacement summary

| Old gate | New state |
|---|---|
| S4 `GATE_GARRIDO_WRITTEN_SIGNOFF` | `GATE_DOMAIN_FACTS_ADJUDICATED` (this doc) + paper disclosure clause |
| S-P annex `BLOCKED_PENDING_GARRIDO_ADVANCE_SIGNAL_FACTS` | `UNBLOCKED_BY_ADJUDICATION_AF-1_CONDITIONAL` (per-risk generator constraints binding) |
| D1-deadline successor (Q11) | eligible for prospective preregistration citing AF-2 |
| Shared-restoration mechanism (ledger #4) | eligible for prospective preregistration citing AF-3 |
| op11 `HOLD_PENDING_DOMAIN_FACT` | `ADJUDICATED_SUPPORTED_EXTENSION` citing AF-6 (probe result unchanged) |
| Paper 2 fixed-clock framing (Q14) | cite AF-5 as doctrinal support; closed verdicts untouched |
