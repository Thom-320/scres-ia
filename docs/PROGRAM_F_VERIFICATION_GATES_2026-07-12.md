# Program F verification gates (frozen by verifier BEFORE the build, 2026-07-12)

Role split unchanged: Codex builds Program F (charter
`docs/PROGRAM_F_RISK_MITIGATION_PORTFOLIO_CHARTER_2026-07-12.md`, contract
`contracts/mfsc_risk_mitigation_portfolio_v1.json`, commit `b0088ac`); this freezes the
independent machine checks the verifier will run and the pass/fail lines, so the audit cannot
move goalposts after the fact (the V4→V5 lesson). The prereg is endorsed as STRUCTURALLY sound
and faithful to the 2026-07-12 dictamen, with ONE substantive gap flagged in §B below that must
be resolved before any tape universe (940001+) is generated.

## A. Endorsed — Codex prereg is faithful on these (verified against the dictamen)

- Three risk-specific mitigation families M/T/R on real thesis risk keys (R11 / R22-R23 / R24;
  Op5-7 / Op8-12 / Op9-11), targeting **specialization** — the one condition every prior lane
  left unprobed as a package.
- Two-token budget, six allocations `[[2,0,0],[0,2,0],[0,0,2],[1,1,0],[1,0,1],[0,1,1]]`, equal by
  construction — structurally kills the "PPO won because it spent more" failure mode.
- Exogenous/endogenous separation: threat + demand tape common across policies; mitigation changes
  realized damage only; `risk_and_demand_hashes_must_match_between_actions`. This is the DRA-1 /
  Track B-P DiD identification principle, correctly carried over.
- New disjoint universes 940001+/950001+/960001+; no Program E / DRA-2b reuse for estimation.
- Prelearner-BEFORE-learner: 480 branch states × 6 actions × {4,8}wk + 60×6³ sequence rollouts;
  classification accuracy explicitly demoted to diagnostic (Program E paid for that lesson);
  endpoint is complete-rollout value.
- Conjunctive gates all present (3 levers live; ≥2 actions optimal ≥15%; none >85%; oracle ΔReT
  ≥0.01 CI95>0; service-loss ≥5%; observable conversion ≥50% + beats best constant on holdout;
  horizon-stable; lost/tail non-deterioration; equal resources). `STOP_PROGRAM_F_PRELEARNER` on
  any failure; params frozen post-endpoint; `forbidden_rescues` list explicit and correct.
- Learner ladder (constrained lookahead + MaskablePPO first; recurrent needs observable-memory
  ablation; retained learning needs confirmed adaptive value); L(e−1) correctly deferred.

## B. FLAGGED GAP (blocks tape generation) — efficacy/signal/dwell are unvalidated free parameters

The dictamen §7-§8 and §10 were explicit that the mitigation efficacy functions `g_r`, the signal
reliability (FP/FN, lead), the context dwell, and the service MCID are **Garrido-gated** and that
the confirmatory cell must be chosen by a **learnability phase-diagram screen** (Morris / LHS /
fractional-factorial over context-persistence, signal quality, mitigation efficacy, budget
tightness, risk amplitude, switching cost) — "the lowest-amplitude realistic cell that
pre-satisfies liveness/diversity/oracle-headroom/observable-convertibility," NOT a guess.

Codex's contract instead **hard-freezes a single hand-picked cell** with no Garrido face-validation
and no screening phase:
- `condition_reduction_per_token: 0.20`, `r11_protection_factor_by_tokens: [1.0,0.75,0.50]`,
  `weekly_wear_increment: 0.12/0.05`;
- `transport realized_duration_factor_by_tokens: [1.0,0.65,0.40]`;
- `reserve max_issue_per_r24_by_tokens: [0,2500,5000]`, `capacity 10000`;
- `signals classification_accuracy: 0.75`, `lead 168h`; `dwell U[4,8] weeks`; MCID frozen at 5%.

Freezing before data is good discipline against post-hoc tuning — but these numbers have no thesis
or PI grounding, and the entire program's validity rests on them: if efficacies/persistence are
(even innocently) set where ranking reversal appears, the "headroom" is an artifact of chosen
physics, not a discovered property of the MFSC. `forbidden_rescues` bars *changing* efficacy after
reading calibration; it does not address the *initial* choice being an unvalidated free parameter.

**Required before 940001+ opens (verifier position, one of):**
1. **Garrido face-validation** of `g_r`, signal reliability, dwell, and MCID → relabel these fields
   `researcher_imposed_face_validated_by_PI` with a `contract_sha256` + authorization record
   (the DRA-2 autonomy-artifact pattern); OR
2. a **pre-learner phase-diagram screen** over the §8 axes with the confirmatory cell selected by
   the frozen learnability criterion (lowest-amplitude realistic cell pre-satisfying
   liveness/diversity/oracle-headroom/convertibility) — cell chosen BEFORE the confirmatory tapes,
   recorded, and NOT the PPO-maximizing cell.

Absent (1) or (2), a Program F "pass" would be contestable as chosen-physics. This is the single
open item; everything else is endorsed.

## C. Machine checks the verifier will run when the env/runners land

- **CRN identity (CRITICAL, where DRA-1's confound lived).** Assert, bitwise, that the exogenous
  risk tape and demand tape are IDENTICAL across all six allocations from a common prefix (the
  `risk_and_demand_hashes_must_match_between_actions` claim), fail-closed. Only realized damage
  `D_{r,t}` may differ.
- **Mass conservation on reserve.** Issued reserve depletes finite Op9 stock; replenishment only
  moves existing stock across Op10-12 at 168h lead; no ration is created by any action; unissued
  reserve stays costed. Assert conservation each branch.
- **No-privileged-observation.** Observation vector excludes latent context label, exact future
  event/duration, oracle labels, any future outcome. Signals are event-keyed, policy-independent,
  and reliability-limited (0.75 or the face-validated value).
- **Actuator liveness.** Each of M/T/R measurably changes a next-state signature (condition /
  realized R22-R23 duration / reserve stock + service), not a silent no-op.
- **Comparator discipline.** Best static and convex mixture selected on calibration/holdout, frozen;
  never reselected on virgin. Full-contract comparator (same eight-dim/observation contract), not a
  restricted frontier (the Track B lesson).
- **Endpoint is rollout value, never classification accuracy.** Tree/hysteresis promoted only on
  captured oracle headroom in complete rollouts.
- **Prelearner-before-learner + tape discipline.** No learner trained, and no virgin (960001+) tape
  opened, until every §7 promotion gate passes and weights/tree/heuristic/mixture/analysis frozen.

## D. RESOLUTION of §B (PI decision 2026-07-12): phase-diagram screen path

The PI chose **the phase-diagram screen** over Garrido face-validation. This resolves §B by
option (2): g_r / signal reliability / dwell are no longer a single hand-picked cell but **swept
axes**, and the confirmatory cell is selected by a frozen learnability criterion. A screen is only
honest if its selection rule and multiplicity are frozen BEFORE it runs — otherwise "screen N cells,
keep the one that passes" is exactly the p-hacking Program F exists to avoid. Frozen here:

- **Axes + ranges frozen before the screen (Codex to propose, verifier to freeze):** mitigation
  efficacy g_r (protection/condition/issue factors), signal correct-class probability, context
  persistence (dwell), budget tightness (tokens), risk amplitude (within thesis Table 6.12 anchors),
  switching/commitment cost. Exogenous threat rates/magnitudes stay at the `b1db8d1` thesis anchors.
- **Design:** Morris or Latin-hypercube / fractional-factorial over the axes; each cell evaluated on
  a dedicated **SCREEN tape universe, disjoint from calibration/holdout/virgin** (new seed block, NOT
  940001+/950001+/960001+). The screen NEVER touches virgin.
- **Selection rule (frozen, NOT max-ΔReT):** the confirmatory cell = **the lowest-amplitude
  realistic cell that pre-satisfies liveness + action-diversity + oracle-headroom + observable-
  convertibility** on the screen tapes. Ties → lowest risk amplitude, then lowest efficacy magnitude.
  Selecting the ΔReT-maximizing cell is a forbidden rescue.
- **Fresh-tape confirmation:** once selected, the cell's §7 promotion gate is evaluated on the
  SEPARATE calibration (940001+) and holdout (950001+) tapes — never on the screen tapes that chose
  it. Virgin (960001+) opens only after promotion passes.
- **Multiplicity honesty:** report how many cells were screened, the full screen surface (the phase
  diagram is itself a deliverable), and that selection used the frozen rule. If NO realistic cell
  pre-satisfies the four conditions, emit `STOP_PROGRAM_F_SCREEN` → 7th boundary result (phase
  diagram of boundary conditions), no confirmatory run.
- The service MCID stays 5% unless Garrido fixes another value before the screen (dictamen option).

Verifier will audit: axes/ranges frozen before screen tapes open; screen tapes disjoint from
calib/holdout/virgin; selection matched the frozen learnability rule (not ΔReT-max); confirmatory
cell re-evaluated on fresh calib/holdout; multiplicity reported. §C machine checks still apply to
the built physics.

## F. VERIFIER AUDIT — screen executed and CLOSED (STOP_PROGRAM_F_SCREEN, a1278de) — LEGITIMATE

Audited stage by stage against §D; numbers verified directly from `results/program_f/screen/
verdict.json` (not the prose), runner logic read, physics tests re-run. All PASS:

- **Discipline (§D):** design frozen before screen (`2b815d9`, SHA `300a2999…` matches doc);
  screen seeds 970001–970288 (288 unique) — **zero** overlap with calib/holdout/virgin
  (940001+/950001+/960001+ all `0` opened); `ppo_trained: False`. Selection key in the runner is
  the frozen lowest-amplitude lexicographic order `(risk, efficacy, |signal−0.75|, dwell,
  commitment, cell_id)` — NOT ΔReT; only admissible-AND-passing cells are candidates.
- **Conversion is rollout, not accuracy:** runner L160 `conversion = tree_delta / oracle_ci[0]`,
  L180 gate `conversion ≥ 0.50 AND tree_delta > 0`, tape-cross-fit sequential rollout. (Program E's
  accuracy-vs-value lesson honored.)
- **Physics sound:** 24/24 actuators live, 24/24 threat-CRN identity, max mass residual `0.0`;
  `test_program_f_physics.py` 4/4 green.
- **The STOP is real, not an artifact:** oracle headroom exists (16/24 cells; all **8 admissible
  two-token cells** have positive clairvoyant ΔReT with CI95 lower > 0 — FSC-24 ΔReT 0.02258
  CIlo +0.00671, DRA-2b magnitude), yet **observable-tree conversion passes 0/24** and every
  admissible tree rollout delta is NEGATIVE (−0.012 … −0.029). Best conversion overall (FSC-15
  0.471) is a 1-token boundary cell AND its oracle ΔReT 0.0079 < 0.01 — cannot pass either way.
  FSC-24 (max headroom) was correctly NOT promoted (that would be the forbidden ΔReT-max rescue).

Verdict: the chosen-physics objection (§B) is decisively resolved — the null holds across a
24-cell phase diagram spanning efficacy/signal/dwell/budget/amplitude/commitment, not one guessed
cell. This is the **7th boundary result**. Diagnostic ladder extended:
`physical effect → action diversity → clairvoyant headroom ↛ observable rollout value`.

## E. Standing

Standing: DRA-1, DRA-2, DRA-2b, Program E stay closed. Program F is ADDITIVE — the manuscript can
already be written on the six boundary results + DRA-2 near-miss + Program E null; Program F is a
pre-committed stronger-result attempt whose failure mode is an honest phase diagram (a 7th boundary
result), not a rescue. The dictamen's own decisive line holds: "no subimos multiplicadores hasta
que el archivo JSON diga algo agradable."
