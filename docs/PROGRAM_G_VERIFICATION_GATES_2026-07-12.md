# Program G verification gates (frozen by verifier BEFORE the build, 2026-07-12)

Role split unchanged: Codex builds Program G (charter
`docs/PROGRAM_G_STRUCTURED_SPATIAL_HEADROOM_CHARTER_2026-07-12.md`, contract
`contracts/program_g_structured_spatial_headroom_v1.json`, charter commit `8d555b9`); this freezes
the independent machine checks and pass/fail lines the verifier will run, so the audit cannot move
goalposts after the fact (the V4→V5 lesson; the Program F post-terminal-audit lesson). PI authorized
Program G as the FINAL experimental campaign before the manuscript. The charter is endorsed as
STRUCTURALLY sound and genuinely distinct from DRA-1, with one discipline item frozen in §B.

## A. Endorsed — the charter is faithful and distinct (verified)

- **Genuinely NOT a renamed DRA-1.** DRA-1's null came from homogeneous CSSUs + `reallocate_unused`
  (auto-reassigned unused capacity → non-committal action, no intertemporal cost). Program G fixes
  exactly that: `automatic_unused_capacity_reallocation: false`, `location_persists: true`,
  `return_required_before_reuse: true`, finite reserve moving with the convoy, and heterogeneous
  A/B (separate stock/backlog/tempo/routes). Dispatch to A genuinely denies B and persists → real
  spatial commitment (scarcity) + real intertemporal coupling (reachability #5) + heterogeneity
  (ranking reversal #6). This is the DRA-1 counterfactual, not a rescue.
- **Anti-p-hacking selection.** Factorial Base/T/TR/TS/TRS/TRSC is for COMPONENT ATTRIBUTION, all
  arms published, environment selection entirely from exact branching + observable rollout, RL never
  consulted (`selection_by_neural_performance_forbidden`, `ppo_trained_before_selection: false`).
- **Independent holdout (the Program F lesson applied).** Program F could only claim cross-fitted
  conversion because 950001+ was never opened. Program G separates screen 980001 / calib 990001 /
  **holdout 1000001** / virgin 1010001, and the gate requires tree−static CI95>0 on the independent
  holdout — a genuine out-of-sample test, not cross-fit.
- **Value-of-information isolation.** Signal must beat SHUFFLED and DELAYED placebos; a signal that
  arrives after convoy commitment does not qualify as advance information.
- **Conjunctive learner gate, stricter than F.** oracle ΔReT ≥ 0.02 (CI95>0); service ≥5%; ≥2
  actions optimal ≥15%, none >85%; tree ≥30% conversion to AUTHORIZE a learner (≥50% for the
  stronger managerial claim — two-tier, honest); ≥2 ADJACENT plausible cells (robustness, not one
  lucky cell); resource envelope; lost/tail non-inferiority. STOP_PROGRAM_G_NO_OBSERVABLE_SPATIAL_
  HEADROOM otherwise; no post-hoc change to signal/tempo/fleet/MCID/semantics. `forbidden_rescues`
  includes "add a second convoy after seeing a null" — the anticipated escalation, pre-barred.
- Physical invariants 1–7, incl. #6 CRN trajectory-hash identity and #7 A/B-mirror symmetry.

## B. FROZEN discipline item — domain-envelope freeze (apply the Program F lesson)

The domain parameters (tempo persistence/transitions, signal lead/sensitivity/FP, convoy capacity,
outbound/return times, reserve capacity/location, S2 labor/delay) are `PENDING_DOMAIN_FREEZE` and
`build_gate.authorized: false`. Per the Program F chosen-physics lesson, these must NOT be a single
hand-picked cell chosen where the gate passes. Required before screen tapes 980001+ open (one of):
1. thesis anchors / Garrido face-validation of the envelope, relabeled researcher-imposed-face-
   validated with a contract SHA + authorization record; OR
2. the envelope declared as a frozen GRID with the "≥2 adjacent plausible cells" adjacency defined
   BEFORE screening, selection by the frozen learnability rule (not neural/ReT-max), all cells
   reported. The charter's factorial + adjacency requirement already leans this way; freeze it
   explicitly so cell/arm selection cannot be gamed.

## C. Machine checks the verifier will run when the env/runners land (stage by stage)

- **CRN identity at ONSET (the Program F lesson).** Assert same-prefix+same-action → identical
  trajectory hash (#6), and across the six actions the exogenous threat AND demand consumed are
  bitwise identical, recorded at ONSET (policy-independent) — NOT from a post-outage log that
  truncates near the horizon. Realized damage/response may differ; threat/demand may not. Fail-closed.
- **A/B mirror symmetry (#7).** Swapping A/B labels + mirroring actions preserves aggregate outcomes
  to tolerance — machine-checked; a broken mirror means an asymmetric artifact, not real headroom.
- **Mass/vehicle conservation.** A+B demand preserves the aggregate demand tape; convoy count /
  location / load / outbound / return conserve the vehicle resource; reserve movement conserves
  rations and consumes convoy capacity; no inventory created by any action.
- **No-privileged-observation (fail-closed whitelist, like Program E/F).** Observation excludes
  latent tempo labels, exact future demand/threat, oracle labels, future outcomes; signals are
  reliability-limited and event-keyed. Exact-whitelist schema test + future-event-mutation invariance.
- **Dispatch masking is real.** Convoy dispatch masked when unavailable / route closed / no physical
  load — not silently remapped to HOLD.
- **Placebo separation executed.** Signal beats shuffled AND delayed placebos on the branching/tree
  estimand, with the delayed placebo arriving after commitment; machine-verified, not prose.
- **Full-contract comparator.** Statics and convex mixtures receive the SAME six actions and physical
  constraints (the Track B lesson); comparator frozen on calib/holdout, never reselected on virgin.
- **Endpoint is rollout value, not classification accuracy** (the Program E/F lesson).
- **Pre-RL-before-learner + tape discipline.** No learner trained, no virgin (1010001+) tape opened,
  until every §learner-eligibility gate passes in ≥2 adjacent cells and weights/tree/analysis frozen.
- **Two-tier claim honored.** 30% conversion authorizes a LEARNER only; 50% required for a managerial
  claim; the 5% service MCID and 0.02 ΔReT are not lowered post hoc.

## D. Standing

DRA-1, DRA-2, DRA-2b, Program E, Program F stay terminal; no reuse of their tapes/weights/thresholds/
cells. Program G is the FINAL experimental campaign before manuscript freeze. Every outcome is
pre-committed and publishable: if it passes, it is the project's FIRST observable adaptive-control
eligibility (the honest place RL could finally win); if it STOPs, it is the 8th boundary result and
sharpens the phase diagram. Neither is a rescue. The manuscript is already complete on D + DRA-1/2/2b
+ E + F; Program G is an additive stronger-result attempt.
