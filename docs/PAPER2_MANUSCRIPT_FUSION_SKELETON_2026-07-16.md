# Paper 2 manuscript — fusion skeleton (2026-07-16)

**Status:** drafting plan; un-pauses the manuscript saved in
`docs/PAPER_1_MANUSCRIPT_IDEA_SAVED_2026-07-12.md` and folds in every custody-verified result
produced after 2026-07-12. This document maps each section to its evidence artifact so that no
number enters the manuscript without a populated, custodied source (standing rule).

**Working title:** *When not to train: a certified boundary for adaptive control in a military
supply-chain digital twin*

**Central claim (unchanged from the saved spine):** RL is warranted only after a DES exposes
observable, resource-adjusted, state-contingent value; physical authority and perfect-information
headroom are insufficient.

**Vehicle:** the paused LaTeX manuscript at `docs/manuscript_current/submission/elsevier/`
(sections 01–06 exist; the folder holds two competing section structures — 03_eligibility_framework
vs 03_methodology etc. — resolve to ONE structure below).

**Venue decision (user):** the template is Elsevier (EJOR / IJPE / Omega fit); the earlier stated
target was IJPR (Taylor & Francis, different template). Decide before heavy formatting work.

## Section plan → evidence map

1. **Introduction.** The question is not "can RL improve resilience" but "when is training
   warranted at all". Contribution: (i) an eligibility ladder (physical → perfect-information →
   observable → learned) with pre-registered gates; (ii) a machine-verified exhaustion certificate
   for a validated military DES; (iii) an adversarial-custody methodology that caught every false
   positive before promotion.
2. **Related work.** Keep saved section; add scenario-discovery / GSA line (Morris, Sobol, PRIM)
   and sim-optimization comparator literature.
3. **The MFSC case and the eligibility framework.** Saved sections 03/04 (thesis-faithful DES,
   canonical `ret_excel_request_snapshot_v2` endpoint, decision-right catalog Op3–Op13 =
   `contracts/decision_right_catalog_v1.json`).
4. **The ladder in action — negative results with teeth.** Include the D1 STATIC finding as the
   canonical example of "constant authority without material adaptive value": the thesis's own
   cap-60 overflow-triage mechanism (§6.5.4; conceptual "order cancellation time", thesis p. 75)
   is heavily exercised in Garrido's raw data (queue reaches its cap in all 20 Cf; 148–993 Ut
   orders per configuration; the ledger does not pin the total-order denominator, so no
   percentage is claimed), and a better CONSTANT rationing rule exists (`spt_flat` ≈ +0.0105
   over the thesis default, `results/program_d/d1_v3_visible_frontier`); the state-preferred
   rule DOES vary across states (≈75/16/7% shares) but that dependence adds only ≈ +0.0011 ReT
   and is not observably convertible. Framing anchor for the whole paper: thesis §6.5.2
   explicitly ASSUMES no post-hoc reaction (pure buffering); our results QUANTIFY the internal
   consequences of that assumption within the model's envelope — they do not empirically
   validate it for the real military chain. One subsection per closed family, each
   with its ceiling from `results/paper2_search/paper2_exhaustion_certificate_2026-07-15.json`
   (`quantitative_ceilings`): thesis-native envelope (DRA2b H_PI 0.0221 with H_obs≈0; Track C
   6.5e-05; route recourse ≤0.005), risk-magnitude invariance (screen e4a3d4a0: optimal posture
   invariant across all 45 Cf profiles, max H_profile_safe 6.9e-05 vs 0.01 bar), and the GSA
   boundary map (`docs/PROGRAM_I_HEADROOM_GSA_RESULT_2026-07-12.md`: information quality and risk
   magnitude formally inert; headroom responds only to scarcity×concurrency; the single OOS-stable
   H_obs>0 region fails the spatial-fairness guardrail).
5. **Case study — how development optimism dies in sealed validation (Program O).** The flagship
   methodological narrative: certified full-DES H_PI 0.152 (LCB95 0.116, fungible-null 0) →
   state-rich fit stopped by the resource gate (and the placebo-default misread, retracted in
   022abd0, as an honesty exhibit) → dual-resource diagnostic (fixed-clock development signal) →
   pre-registered fixed-clock-physical OOS validation FAILED prospective consistency (26/48 vs 34)
   → closed without rescue. Artifacts: `contracts/program_o_*`, result 09ec3f16, commit 897ebab.
6. **The wartime residual, disclosed.** The frozen `war_stress_timing_atlas_v1` (144 concomitant
   cells, within-cell `H_timing_safe`, expanded physical control incl. dispatch/HOLD/churn) is the
   exact remaining escape hatch — and it is compute-infeasible under its own frozen authorization:
   measured 1.2606 s/episode (H104 benchmark, seed 94700001), 86.5M projected episodes after the
   only certifiable exact reduction (duplicate-only, 153/50,202 = 0.30%,
   `war_stress_exact_policy_reduction_20260716.json`), vs caps of 2M episodes / 7 wall-days.
   Framed as a bounded, machine-verified residual with its unlock rule quoted — reviewers see
   exactly what would change the verdict. (If the lane is later executed, this section becomes the
   result chapter either way.)
7. **Discussion — when NOT to train.** Decision rules for practitioners: comparator strength
   (restricted comparators manufacture wins), guardrails (aggregate metrics hide theatre-starving),
   oracle-first gating (0 PPO runs wasted post-gate), sealed-seed one-shot validation, A11
   (populated fields, never labels/defaults — with the two retractions as evidence the standard
   has bite).
8. **Conclusion + exact reopeners.** The five Garrido questions (canonical batch
   `research/paper2_exhaustive_search/garrido_face_validation_questions.md`, Q1–Q14): Q11/R09
   mission expiry (strongest), Q6/Q7 shared restoration resource, Q13 Program-O construct realism,
   Q14 freight economics (scopes only), Q2/R03 bar-raised. The negative is conditional and
   falsifiable — that is its value.

## New assets needed

- Ladder diagram (physical→PI→observable→learned) with each program placed at its failure rung.
- Ceilings table auto-generated from the certificate JSON (script, not hand-copied numbers).
- Program O arc figure: development LCBs vs sealed-validation outcome.
- Wartime atlas cell grid (144 cells) with the compute-infeasibility boundary annotated.

## Rules

Every number cited must resolve to a custodied artifact path in this map. Spanish reporting to the
PI; manuscript in English. No result from `thesis-native-timing-oracle` (prototype, stale base) may
be cited. The paused-manuscript preservation rule is lifted only for fusion per this plan.
