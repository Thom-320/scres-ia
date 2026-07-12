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

## E. External convergent validation + three applied diagnostics (2026-07-12 dictamen)

A second independent expert dictamen (ChatGPT-Pro, 2026-07-12), reasoning from first principles and
the repository's own boundary results, re-derived **the same environment as Program G** — down to
identical specifics: two heterogeneous CSSUs A/B with one finite shared convoy (no auto-reallocation,
location persists), operational-tempo regimes low/routine/surge with advance operational signals, the
minimal six-action `(convoy_destination∈{A,B,HOLD}, shift∈{S1,S2})` contract, the Base/T/TR/TS/TRS/
TRSC factorial, and the same gate numbers (oracle ΔReT ≥ 0.02, tree ≥30% conversion to authorize a
learner / ≥50% for a managerial claim, ≥2 adjacent plausible cells, signal-vs-placebo). Two
independent sources converging on the same design **before any PPO** is itself strong anti-p-hacking
evidence and should be cited in the Program G / Paper-2 methods as convergent design validation.

The dictamen contributes three diagnostics Program G's charter does not yet name. They are frozen
here as ADDITIONS that make the pre-RL audit STRICTER (never easier), slotted between the oracle
(pre-RL step 5) and the tree (step 6), reported for every factorial arm and cell:

1. **Observable-information gate `I(O_t; a*) > 0` (pre-tree, the cheap Program-E predictor).** On the
   exact-CRN branch oracle, for each state record the clairvoyant best action `a*(X)` and the
   deployable observation `O_t`. Estimate mutual information `I(O; a*)` (e.g. balanced-accuracy of
   predicting `a*` from `O` above the marginal-class base rate, or empirical MI on discretized `O`),
   and require it to beat a **shuffled-O placebo** with CI95>0. If `I(O; a*) ≈ 0`, the headroom is
   not observably convertible BY CONSTRUCTION and the arm/cell STOPs before any tree or learner —
   this is exactly the failure Program E paid to discover episodically. Report `I(O; a*)` alongside
   `H_PI` and `H_obs_tree`; a passing tree conversion with near-zero `I(O; a*)` is contradictory and
   must be investigated, not promoted.
2. **Equilibrium-policy / action-perturbation test (ref: contextual-bandit equilibrium diagnostic).**
   Take the best full-contract static; retrospectively flip a small fraction ε∈{5%,10%,15%} of its
   actions to the clairvoyant-best and measure ΔReT(ε). If flipping ε yields negligible improvement,
   the static is near-equilibrium and there is little learnable margin regardless of algorithm.
   Report the ΔReT(ε) curve per arm; a flat curve is a STOP signal that no learner can rescue.
3. **Designated second family (recorded fallback, NOT opened yet).** If the spatial family STOPs at
   `STOP_PROGRAM_G_NO_OBSERVABLE_SPATIAL_HEADROOM`, the pre-committed next structure is Op5–Op7
   DISAGGREGATED with a per-station observable degradation state, state-dependent (condition-based)
   failure hazard, ONE finite repair/maintenance crew (real opportunity cost: repairing Op5 now
   forgoes Op6), intermediate WIP with block/starvation, and action = assign crew to {Op5,Op6,Op7,
   wait}. This requires disaggregating the currently-fused Op5–Op7 and making R11 endogenous to
   condition; it is a separate charter, not a Program G rescue, and needs its own Garrido domain
   freeze. Recorded so the fallback is pre-committed, not invented after a null.

These do not change any promotion threshold; they add upstream STOP signals that would have caught
Program E's null before training. `H_PI ≠ H_obs ≠ H_learnable` — the gate now measures all three.

## F. DOMAIN ENVELOPE V1.1 — pre-build amendment adopted (2026-07-12, Codex out; verifier owns)

An external review found a physical contradiction in the minimal envelope (weekly action vs 48h
convoy cycle → convoy idle 120h/wk, so "persistent convoy location" was false). Verified first-hand
(120h idle; enumerable oracle 393/1,569; 5,000 = real `RATIONS_PER_BATCH` anchor). Adopted as
**V1.1** — `contracts/program_g_domain_envelope_v1_1.json`,
`docs/PROGRAM_G_CHARTER_PREBUILD_AMENDMENT_V1_1.md`,
`docs/PROGRAM_G_INTERVENTION_LEDGER_2026-07-12.md`,
`docs/PROGRAM_G_DOMAIN_SIGNOFF_TEMPLATE_2026-07-12.md`. The charter's guardrails are unchanged; only
the physical/informational ontology is corrected. Build stays UNAUTHORIZED pending Garrido sign-off.

Corrections folded into the audit (these SUPERSEDE the minimal-envelope readings in §A–§C):
- **Convoy decisions every 48h when available** (not weekly); persistent state = convoy availability +
  reserve depletion + deployed-ration location + A/B backlog. Machine-verify persistence across epochs.
- **Emergency-reserve overlay**: normal MFSC flow remains; convoy moves only the 10,000-ration reserve
  (two 5,000 loads, no in-episode replenishment). Verify normal flow unaffected + A+B demand conserved.
- **Static frontier is NOT the six constants**: it must include strategic prepositioning postures
  (10k@SB, 5k+5k, 10k@A, 10k@B) AND exact open-loop schedules (all ≤2-departure sequences over 4/8wk,
  selected on calibration only). Primary contrast `π_obs − max(posture, open-loop, constant)` within a
  matched resource envelope. Comparing a learner to "always-A/HOLD" is the forbidden Track B error.
- **Signal = sensitivity/FPR** (moderate 0.70/0.20, high 0.85/0.10), lead 7/14d, binary weekly per-CSSU
  — NOT scalar accuracy. **Wrong-CSSU placebo ADDED** to the shuffled+delayed set (isolates value of
  knowing WHERE). Verify signal beats all three placebos.
- **S1 fixed in the primary spatial screen**; S2 enters only after a liveness preflight (Program L
  showed S1/S2/S3 identical for positive buffers). Verify S2 is not occupying the action space unproven.
- **`Base/T/TR/TS/TRS/TRSC` is a nested mechanism ladder, not a factorial** — report contrasts
  `T−Base, TR−T, TS−T, TRS−TR, TRS−TS, TRSC−TRS`; do not overclaim orthogonality.
- **Primary grid = 16 connected-region cells** (2 signal × 2 lead × 2 surge-weight × 2 **commonality**),
  not 24. Commonality (A/B concurrency) is the decisive omitted axis. Promotion needs a **connected
  component of ≥2 cells**, trained/evaluated on the uniform distribution over that component — never the
  max-ReT cell. Persistence 6–8wk is a later sensitivity, not primary selection.
- **Sequence G0→G5** frozen: G0 physics (no tapes); G1 central cell all arms; G2 TRS/TRSC over 16 cells;
  G3 promotable connected region + 60 calib tapes + tree/hysteresis + exact open-loop comparator +
  resource frontier; G4 holdout 1000001+ only after full freeze; G5 rollout/MPC → bandit → MaskablePPO.
- **Intervention-ledger discipline**: no [X] researcher-imposed parameter enters a tape until Garrido
  confirms it; each has a falsifier that runs in G0/G1 before any learner. The 3 diagnostics in §E
  (`I(O;a*)`, equilibrium-perturbation, second-family fallback) still apply, per arm/cell.

## D. Standing

DRA-1, DRA-2, DRA-2b, Program E, Program F stay terminal; no reuse of their tapes/weights/thresholds/
cells. Program G is the FINAL experimental campaign before manuscript freeze. Every outcome is
pre-committed and publishable: if it passes, it is the project's FIRST observable adaptive-control
eligibility (the honest place RL could finally win); if it STOPs, it is the 8th boundary result and
sharpens the phase diagram. Neither is a rescue. The manuscript is already complete on D + DRA-1/2/2b
+ E + F; Program G is an additive stronger-result attempt.
