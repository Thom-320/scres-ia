# Program S critical review (dictamen) — 2026-07-17

Reviewer: PI-side session (Claude), at the request of the PI. Scope: the Program S
proposal ("sensibilidad riesgo × producto × timing para Papers 2 y 3") AND the
in-flight draft implementation found uncommitted in the worktree
`/private/tmp/scres-program-s-product-risk-gsa` (branch
`codex/program-s-product-risk-gsa-v1`, base commit c2fa5cb):
`contracts/program_s_product_mix_risk_interaction_gsa_v1.json`, the S0 scripts and
tests, and the `supply_chain/` diffs. This review changes no verdict and opens no
seed. It is input to the auditor BEFORE any freeze-with-hashes.

**PI decisions recorded 2026-07-17 (binding for v1):**
1. The operational alarm is REMOVED from Program S v1 and deferred to a separate
   annex contract gated on written Garrido facts (see C1).
2. Session scope: dictamen + amendments; S0 implementation continues with the
   auditor (no duplicate implementation from this session).

## What the proposal and draft get right (keep, verbatim where frozen)

- Immutability of O / O-R / Q verdicts; no-rescue and no-post-result-addition
  prohibitions are present in the draft contract.
- Separation of deterministic liveness from thesis-rate incidence — exactly the
  successor correction demanded by `STOP_V1_1_BEFORE_G2_RARE_RISK_FIXTURE_NOT_POPULATED`
  (see `docs/PROGRAM_O_RELEVANT_RISK_SENSITIVITY_V1_1_PREFLIGHT_VERDICT_2026-07-17.md`).
- R14 as a defect-probability multiplier with thesis rework semantics and no
  invented repair duration.
- Declared strata (`THESIS_NATIVE_INDEPENDENT` vs `RESEARCHER_WARTIME_COUPLED`)
  with the replacement rule (coupled cluster replaces, never duplicates).
- Pre-generated risk tapes common across policies; CRN discipline.
- Garrido unexploited-mechanism ledger with theory gates and the rule that none of
  its entries can be added mid-v1 to rescue a result.
- Elevated S2 promotion bars (0.02 / 0.015) — stricter than the historical 0.01.
- Q retains VPS priority; S seeds (7510001+, 7520001–7520256, 7530001+) verified
  unreferenced anywhere in the repo (checked ledger + full-tree grep; the two hits
  in `war_stress_gsa_execution_preflight_v1.json` are incidental numerals).

## Required amendments before freeze (C1–C9)

### C1 — The alarm smuggles Program P (BLOCKING; resolved by PI decision)
`OperationalAlarm` with lead {0,24,72}h and designed balanced accuracy
{0.50,0.70,0.85} is a synthetic advance signal. The frozen v1.1 contract states:
`"program_p": "not authorized; requires external domain facts on deadlines,
criticality, shared line, and advance signal"`. The draft contract nevertheless
carries an `alarm_grid` in `s2`, alarm placebos, alarm observation fields, and
"real alarm beats placebos" inside `s4` gates. Any S2 headroom found with a
0.85-accuracy 72h-lead alarm can be manufactured by the alarm-quality assumption
rather than by physics — and the alarm generative model (how false positives /
negatives are sampled, conditioned on what) is not even specified.

**Amendment:** delete from v1: `s2.alarm_grid`, all alarm placebos, the
`alarm_*` fields of the cell schema, the alarm observation fields, and the two
alarm clauses in `s4`. Create `program_s_p_alarm_annex_v1` with
`status: BLOCKED_PENDING_GARRIDO_ADVANCE_SIGNAL_FACTS`; it may only be unblocked
by written Garrido answers (the exact question already exists in the exhaustion
certificate) and must then specify the alarm generator precisely.

### C2 — Mix-only action vs product-symmetric risks: the invariance problem (BLOCKING)
The action remains `Discrete(4)` product mix. But per the frozen v1 relevant-risk
map: R11/R14 hit the SHARED assembly ops, R21 is multi-op symmetric, R22 is
"product-blind downstream robustness control", and **R23 was excluded with a
recorded physical reason**: "forward-unit destruction hits Op11 downstream of the
mix, symmetric across products (PT=0 pass-through); effect on the MIX decision
expected nil -- excluded, disclosed". The draft reintroduces R23 into
`CROSS_ECHELON_SURGE` and into the S0 fixtures without addressing that reason —
which still applies unchanged to a mix-only action. The only genuinely
product-asymmetric channel is R24 (whose product-label rule is itself a declared
model assumption). The closed 45-profile thesis-native screen already showed the
shape this predicts: risk escalation moves ReT levels massively (0.53→0.20) while
the optimal action stays INVARIANT (max H_profile_safe 6.9e-05 vs a 0.01 bar).

**Amendments:**
(a) Remove R23 from `physical_masks.CROSS_ECHELON_SURGE` and from `s0.deterministic_fixtures`
    (keep the v1 exclusion text in the contract), OR add a written physical
    justification of why the exclusion reason stops applying — none exists for a
    mix-only action.
(b) Add a prespecified S1 early-exit: if within a stratum the Morris sweep yields
    max LCB95(H_PI^safe) below a frozen bar (recommended: 0.5 × the S2 promotion
    threshold, i.e. 0.01), that stratum terminates
    `STOP_S1_NO_CONNECTED_PHYSICAL_HEADROOM` without entering S2.
(c) State in the contract that the null (flat H_PI^safe under escalation) is the
    default expected outcome and is publishable within the "when NOT to train"
    line — this is the honest prior given the 45-profile invariance result.

### C3 — Stratum fallback is softened "escalate until it passes" (BLOCKING)
`s2.selection_order: ["thesis_native_first", "wartime..."]` is sequential
selection across strata AFTER observing outcomes — a relative of the forbidden
rule "increase risk until a policy wins" (frozen in
`garrido_risk_headroom_sensitivity_v1`). The lowest-severity-region and
φ=4-exploratory mitigations do not remove the sequential-selection problem.

**Amendment:** in v1 the wartime stratum is HYPOTHESIS-GENERATING ONLY. Both
strata always run and are always reported; promotion to S3/S4 is only possible
from `THESIS_NATIVE_INDEPENDENT`. A wartime-only signal requires a NEW contract
with fresh seeds (K→K2 pattern), never an in-contract fallback.

### C4 — Double claim on Paper 2: multiplicity vs Program Q (BLOCKING)
The draft carries `PASS_S_PAPER2_NEURAL_PREMIUM` and
`PASS_S_PAPER2_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT` while Program Q
(`FROZEN_POWER_PASS_N_256_PENDING_SEED_AUTHORIZATION`, seeds 7490001–7490256)
is already the frozen, powered confirmatory instrument for exactly those two
claims. Two live instruments for the same title = two shots at Paper 2.

**Amendment:** Q is THE Paper 2 confirmatory instrument. Rename S terminals to
`PASS_S_RISK_AWARE_ADAPTATION_PREMIUM` / `PASS_S_RISK_AWARE_ADAPTATION_CLASSICALLY_EQUIVALENT`
(keep BOUND/STOP labels analogous). Add an explicit precedence clause: if Q
passes, S can contribute a robustness/extension section to Paper 2; S may become
a Paper-2 instrument only via explicit amendment AFTER a Q STOP/BOUND terminal.

### C5 — Missing Garrido gate (BLOCKING for S4)
Standing project rule: a learner only via a NEW contract after written Garrido
sign-off. The draft `s4` contains no such gate (verified: no Garrido clause in
`s4`). Wartime coupling and (if ever unblocked) the alarm are precisely what
Garrido must bless.

**Amendment:** add `GATE_GARRIDO_WRITTEN_SIGNOFF` as a hard precondition of S4
(and of the S-P annex). S0–S3 may proceed without it; no learner training starts
without it.

### C6 — No compute benchmark gate: the war-stress lesson (BLOCKING for S1)
`s1_morris` has no measured-throughput feasibility gate (verified: no
benchmark/feasibility clause). The design implies ~500 Morris points ×
(65,536-calendar frontier + oracle + belief-MPC) × 12 tapes. The war-stress atlas
died at exactly this step (`STOP_BEFORE_SCIENTIFIC_EXECUTION_COMPUTE_INFEASIBLE`,
measured 1.26 s/ep). The exact transducer is validated risk-off only — never
benchmarked with risk skeletons.

**Amendment:** add a mandatory S1-pre gate: measure s/point on the risk-extended
transducer over a fixed pilot (burned tapes only), freeze a feasibility bar and a
PRESPECIFIED reduction ladder (fewer trajectories → fewer masks → stratum
deferral) before the Morris design freeze. Adaptive mid-run reductions remain
forbidden.

### C7 — Seed governance (BLOCKING before any 751x open)
The S blocks are free, but there is no unified 74x/75x registry and the virginity
auditor has the known false-positive defect (it flags the file that DECLARES
blocks as unopened — same defect currently blocking Q). Also note the training
universe 751100001–751350000 shares the "751" prefix with dev tapes 7510001+ —
the auditor must compare numeric ranges, never prefixes.

**Amendment:** fix the allowlist semantics once (shared task with Q), register
the S blocks in the same source-of-truth the auditor reads
(`program_s_seed_manifest_v1.json` must be consumed by, not parallel to, that
auditor), and re-freeze the auditor before 7490001 or 7510001 opens.

### C8 — Minor
- Paper 3 (9 arms, 0.8-diagonal persistence) stays a sketch; freeze only after an
  S4 PASS. The `7530001+` block stays declared-but-unopened.
- The `belief-DP on the transducer` comparator is scope risk; mark OPTIONAL.
- The draft contract `status` already reads `FROZEN_S0_IMPLEMENTATION_...` while
  these amendments are pending — a draft under review must carry
  `DRAFT_PENDING_INDEPENDENT_AUDIT`; the freeze-with-hashes happens after audit.
- S2 keeps the elevated bars (0.02/0.015) — correct, do not lower them.

### C9 — NEW (from reading the diff): S0 edits the shared base-class hot path
The uncommitted `supply_chain/supply_chain.py` diff changes the assembly
production path of `MFSCSimulation` itself: the single
`_record_assembly_product_output(can_produce)` call becomes a conditional split
into `_record_rework_product_output(rework_qty)` +
`_record_assembly_product_output(raw_produced_qty)`, plus a new
`_record_product_rework_started` hook at R14 defect removal. Risk-off equivalence
now RELIES on the invariant "rework_qty == 0 whenever risks are off". That is
plausibly true, but it is a change to frozen shared physics that every historical
program inherits — not a subclass-confined adapter change like the v1.1
pass-through.

**Amendment:** (a) an explicit fail-closed test that risk-off runs of the
UNMODIFIED-signature path reproduce the custodied Program O corrective-validation
matrices bit-exactly (max abs diff ≤ 2.22e-16, the G0 bar of v1.1); (b) a test
asserting `rework_qty == 0` for the entire risk-off episode set; (c) the base-class
diff is listed file-by-file in the contract's provenance section so the freeze
hashes cover it.

## Priority note

S0 is not merely Program S's first rung — it is the v1_2 successor demanded by
the v1.1 STOP. A PASS_S0 independently unblocks re-proposing the pending G2
escalation grid on the Program O mechanism (the original sensitivity question)
as a cheap successor contract, whatever happens to S1–S4.

## Disposition

Route C1–C9 to the auditor of `codex/program-s-product-risk-gsa-v1` BEFORE the
contract freeze-with-hashes. Nothing in this review reopens O, O-R, or their
STOPs; nothing authorizes seeds. Q retains VPS priority and remains the Paper 2
instrument.
