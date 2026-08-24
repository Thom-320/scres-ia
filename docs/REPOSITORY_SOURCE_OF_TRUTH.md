# Repository source of truth

<<<<<<< HEAD
This note freezes the current paper-facing repository story. Treat it as the
primary reference for README examples, manuscript edits, benchmark commands,
and reviewer-facing explanations unless a later note explicitly supersedes it.

> **Superseded lane notice (2026-07-10).** Everything below replaces the
> pre-Track-B version of this document, which described the
> `shift_control`/`ReT_seq_v1`/`v1` lane as the frozen paper backbone. That
> lane and its 500k bundles under `outputs/paper_benchmarks/` are HISTORICAL
> context only. The claim-by-claim authority is
> `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`.

## Canonical scientific story (Paper 1)

The manuscript in `docs/manuscript_current/submission/elsevier/` makes one
central claim:

> In a thesis-grounded military food-supply DES, PPO appears decisively better
> than a dense static frontier when that frontier varies only shift and
> downstream dispatch while holding the learner's upstream controls fixed.
> That advantage does **not** survive a same-contract challenge: a
> calibration-only constant full-contract policy exceeds the ten-checkpoint
> PPO mean on 60 untouched tapes (PPO minus static `−0.000018049`, two-way
> CI95 `[−0.000028615,−0.000008087]`). The paper's contribution is therefore
> benchmark and comparator-contract design, not demonstrated adaptive
> superiority. A small within-learner dispatch increment survives, but it does
> not restore superiority over static control.

The decisive artifact is
`docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`. Older positive
comparisons remain valid only relative to their explicitly restricted static
families. Do not aggregate them into a claim that PPO improves Track B
resilience over strong same-contract statics.

Clean-replication closure (2026-07-10, tapes 500061–500120, eval-only,
pre-registered): the internally consistent 5-seed joint bundle's post-hoc
+9.6e-6 edge does NOT replicate (+0.0000062, two-way CI95
[−0.0000066, +0.0000184], 47/60 tapes; stop rule FAIL). Correct language for
the fresh joint vs the full-contract static: "no detected difference at
current precision" — never "PPO retains a small advantage".
`docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md` (RESULT section).

The repository roles that must not be conflated:

- `thesis_faithful` lane (`docs/thesis_faithful/CONTRACT.md`,
  `scripts/run_thesis_faithful.py`): strict Garrido-Rios reproduction, a
  validation gate for the DES, not a training benchmark.
- **Track A** (`track_a_*` contracts): the thesis-grounded buffer/shift
  decision family. Boundary result: no tested learner converts the measured
  oracle headroom (claims registry C8).
- **Track B** (`track_b_v1`, 8D): the comparator-sensitivity lane — upstream
  qty/ROP + Op5 + shift + Op10/Op12 dispatch. It contains the restricted-
  frontier positive result and the decisive same-contract reversal.
- **Track B-P** (`track_bp_v1`, 11D; `supply_chain/track_bp_env.py`): the
  Paper-2 extension lane (strategic reserve postures under lead-time
  commitment). Outside Paper 1. See C28/C29.

## Frozen benchmark backbone (Track B canonical)

- Environment factory: `external_env_interface.make_track_b_env()`
- Action contract: `track_b_v1` (8D)
- Training reward: `control_v1`
- Observation: `v7` (note: 48-dim at the time seeds 1-5 were trained; 52-dim
  for seeds 6-10 — four tail fields appended between runs; disclosed and
  handled by exact slicing in held-out evaluation)
- Risk level: `adaptive_benchmark_v2`; horizon h104 (weekly steps, 168 h)
- Year basis: `thesis`; stochastic PT: on; learning rate 3e-4; 60k timesteps
- Primary metric: `ret_excel` (Garrido/Excel ReT). Never `ret_thesis`.

## Primary artifact bundles (current)

- Headline 10-seed paired dense-CRN stats:
  `docs/track_b_q1_stats_2026-07-02_final_10seed/`
- **Crossed held-out evaluation (Blocker 1, 2026-07-09):**
  `outputs/experiments/track_b_crossed_eval_2026-07-09/` — 10 checkpoints x
  60 fresh tapes (eval seeds 200001+), Excel ReT delta `+0.000486`, two-way
  CI95 `[+0.000456, +0.000517]`, 10/10 checkpoints and 60/60 tapes positive.
- Corrected decision-contract factorial (Blocker 2, mechanism gate):
  `outputs/experiments/track_b_factorial_{joint,upstream_shift,dispatch_only}_2026-07-09/`
- **Same-contract challenge (final gate, failed):**
  `outputs/experiments/track_b_same_contract_challenge_2026-07-10/` and
  `docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`.
- Frozen checkpoints: see `docs/REPRODUCIBILITY.md`.
- E3 cross-regime + dense-frontier: `docs/track_b_q1_stats_2026-07-02_final/`
  and `outputs/experiments/track_b_e3_dense_frontier_2026-07-02/` (use the
  conservative dense-best values; see C11 provenance note).
=======
**Effective date:** 2026-07-17
**Scientific status:** `CURRENT_IMPLEMENTED_PORTFOLIO_EXHAUSTED_NO_LEARNER_AUTHORIZED`

This document is the repository-level claim boundary. It supersedes the former
Track-A/Track-B narrative that presented `ReT_seq_v1` and stress-regime gains as
the current paper contribution.

## Binding headline

No tested learner has established deployable adaptive value under the current
contracts. No learner is authorized. Paper 2 is not confirmed, and Paper 3
remains blocked.

The strongest implemented mechanism is Program O, a nonfungible product-mix
extension of the full DES. It established a large, custody-verified
full-information ceiling, and its corrective observable validation established
a reproducible mean canonical-ReT advantage with genuine state-dependent
actions. It did **not** satisfy the frozen joint tail-safety contract.

The terminal label is:

`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`

## Program O evidence

### Full-information ceiling

The full-DES translation established:

- safe `H_PI = 0.1515137892`;
- simultaneous safe LCB95 `= 0.1156159089`;
- exact fungible-null `H_PI = 0`;
- 25,177 direct horizon-8 parity episodes;
- equal production and reserved downstream resources.

This is a physical opportunity ceiling, not observable adaptation and not a
learner result. The authoritative compact record is
`results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json`.

### Observable corrective validation

The corrective validation used fresh seeds `7430001-7430048`, frozen
development-selected full-frontier comparators, and studentized one-sided
max-t inference. Mean canonical-ReT passed in all connected cells:

| Cell | Mean delta ReT | Simultaneous LCB95 | Favorable tapes |
|---|---:|---:|---:|
| rho75/share90 | 0.09852 | 0.06595 | 44/48 |
| rho90/share75 | 0.07347 | 0.04303 | 42/48 |
| rho90/share90 | 0.09974 | 0.05860 | 46/48 |

All 27 information-placebo contrasts passed. Physical equality passed across
1,451 replays with zero failures. Action trajectories and state
counterfactuals passed in every cell.

The frozen joint contract nevertheless failed because simultaneous CVaR10
non-inferiority did not clear zero in two cells:

- rho75/share90: LCB95 `-0.0085776`;
- rho90/share75: LCB95 `-0.0155069`.

All other guardrails passed. The point estimates favored the controller, but
the preregistration required every guardrail LCB to be non-negative. The
contract forbids a second rescue, cell deletion, threshold relaxation,
controller change, or metric change.

The authoritative records are:

- `docs/PROGRAM_O_CORRECTIVE_HOBS_VALIDATION_VERDICT_2026-07-15.md`;
- `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`.

### Why a corrective run existed

The first prospective block, seeds `7420049-7420096`, was opened exactly once.
Its automatic adjudication was retracted because the executor reselected the
comparator on validation tapes and used an invalid unstandardized simultaneous
critical value across heterogeneous estimands. The burned trajectories and
custody remained valid. A single corrective validation was authorized to test
the same scientific contract with those adjudication defects repaired.

The first-run records are retained as historical evidence, not as the current
terminal verdict:

- `docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md`;
- `results/program_o/fixed_clock_hobs_validation_v1/independent_audit_v1.json`.
>>>>>>> origin/main

## Current claim boundary

<<<<<<< HEAD
Valuable but secondary or retired:

- `shift_control`/`ReT_seq_v1` 500k lane (historical; pre-Track-B)
- `ReT_thesis` as a training reward or reported metric
- KAN / DKANA / GNN as a contribution (sidecars only)
- SAC/TD3 beyond the screen-scale scope check
- Prevention/anticipation claims (retracted; boundary result only — C25/C26)
- H4 retained/reset as a central theory (small effect; future work)
- Track B-P reserve postures (Paper 2, gated; C28/C29)

## Document hierarchy

When documents disagree:

1. `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md` (claim-by-claim authority)
2. This file
3. `docs/REPRODUCIBILITY.md`
4. Dated verdict documents (`docs/*_VERDICT_*.md`, autopsies, audits)
5. Historical reports, manuscript notes, and meeting notes
=======
It is accurate to claim:

- material full-information product-mix headroom in the full DES;
- exact collapse of that headroom under the fungible null;
- observable, state-dependent mean canonical-ReT improvement over the frozen
  full open-loop comparator;
- failure to establish joint tail-safe classical `H_obs` under the frozen
  familywise contract.

It is not accurate to claim:

- safe joint `H_obs > 0` under the project contract;
- learned adaptive superiority;
- Paper 2 confirmation;
- Paper 3 authorization;
- a global impossibility theorem outside the implemented and preregistered
  portfolio.

The portfolio-level machine-readable boundary is
`research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json`.

## Metric and domain status

Program O used `ret_excel_request_snapshot_v2` as its frozen canonical primary
endpoint. Garrido face validation remains necessary to identify the intended
same-timestamp ordering of `sumBt`/`sumUt` and to establish how representative
the nonfungible product classes are of the MFSC. Those answers may refine
construct validity or justify a genuinely new preregistered contract. They do
not retrospectively reopen Program O.

## Historical lanes

Track A, Track B, Track B-P, Track C, Programs D through N, and older
`ReT_seq_v1` benchmark bundles remain useful provenance and bounded evidence.
They are not the current positive paper claim. Earlier gains must retain their
original comparator and contract qualifiers.

## Publication and execution authorization

- Current defensible paper route: a boundary paper separating physical
  headroom, observable mean conversion, and joint tail-safe deployability.
- New simulation: only after a genuinely new mechanism is justified and
  preregistered with new physics, observations, comparators, and seeds.
- Learner: not authorized under current contracts.
- Paper 3: blocked until a future contract establishes learned adaptive value.

## Provenance scope of this reconciliation

This small reconciliation changes the remote claim state and publishes compact
audited summaries with immutable hashes. It intentionally does not add the raw
calendar matrices or large custody bundles to Git history. Those remain
external custody artifacts identified by SHA-256 in the included audit files.
Accordingly, this PR makes the terminal claim boundary reviewable from GitHub;
it is not by itself a complete raw-data replication package.

## Document hierarchy

When documents disagree, use:

1. this file;
2. `paper2_current_boundary_certificate_20260716.json`;
3. the Program O corrective independent audit and verdict;
4. the full-DES HPI custody verdict;
5. dated historical verdicts and older claim registries.
>>>>>>> origin/main

## Program O-R learner stage (2026-07-17/18 addendum)

<<<<<<< HEAD
Preferred: "thesis-grounded reconstruction with forensic workbook replay and
throughput checks"; "restricted-frontier gain"; "same-contract static
challenge"; "comparator-family sensitivity"; "small within-learner dispatch
increment"; "boundary result"; "no detected difference at current precision".

Avoid: "validated digital twin"; "empirically validated"; the invented
"±15% validation threshold"; "prevention"/"anticipation"; "organizational
learning"/"path dependency"; "worst-case" for p99 statistics; "equivalent"
for a CI that spans zero; "regardless of algorithm choice"; "full 8D static
frontier" for the downstream 147-cell enumeration; "first DES–RL for SCRES";
"downstream dispatch access is the strongest (observed) lever" (retired by the
identified factorial); "only when the contract exposes dispatch"; "PPO beats
strong static control"; "adaptive advantage" or "bottleneck value" without an
explicit restricted-comparator qualifier.

## Program G terminal status (2026-07-12)

Program G is terminal under its stylized spatial-order contract:
`STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT`. The authoritative
artifacts are `docs/PROGRAM_G_TERMINAL_METRIC_AUDIT_VERDICT_2026-07-12.md` and
`results/program_g/terminal_metric_audit/verdict.json`. The corrective run used a 168-hour
week, the canonical cumulative ReT ledger, quantity-weighted ReT, 200 new calibration tapes,
and 400 locked terminal tapes. No observable policy passed the joint guardrails against the
best periodic static (`ABAB`). Earlier G5 language describing a virgin observable adaptive win
is historical for its service-loss proxy only and must not be used as the paper headline.

## Program H terminal status (2026-07-13)

Program H ended at `STOP_PROGRAM_H_NO_BELIEF_POLICY_PASS_INFORMATION_BOUND_REMAINS_LOOSE`.
The informative O0 filter did not yield a qualifying regret fitted-Q, belief-MPC, or point
rollout policy on 400 locked tapes. The best order-ReT delta was +0.00225 with CI95
[-0.00021,+0.00460], 19% favorable tapes, and 13.7% PI conversion. The exact full-tape ceiling
remained material (+0.01641), so formal information insufficiency was not established. Seeds
1080001+ were not opened and no Program H RL was trained. This is the last computational
extension of the stylized spatial contract.

## Paper 2 bottleneck-migration screen (2026-07-13)

The first thesis-wide Op3–Op13 response-team contract ended at
`STOP_NO_ADAPTIVE_BOTTLENECK_VALUE`. A signal-adaptive policy allocated one equal-cost team
among manufacturing, LOC and mission response, but lost to calibration-frozen constant M on
120 locked tapes: ReT delta -0.001309, CI95 [-0.006384,+0.003093], 53.3% favorable tapes, and
service-loss change -3.03% (worse). CRN, mass and equal team-hours passed. PPO was blocked and
1120001+ stayed unopened. This is a confirmed adaptive-negative result under a declared
high-authority extension; do not escalate the same efficacy/signal cell after observing it.

## Program O and O-R terminal status (2026-07-17)

This section supersedes any earlier sentence implying that no learner has shown state-dependent
value relative to open-loop scheduling.

- Program O remains `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`. Its mean canonical-ReT
  conversion passed; its frozen joint CVaR10 tail gate did not. The numerical audit shows that
  roughly +0.079 true tail improvement was needed for 80% joint power at 48 tapes. The later
  interpretation correction explicitly retracts the label "instrument defect": the gate was
  stringent and underpowered for small effects, not technically invalid.
- Program O-R remains `STOP_CALIBRATION_NOT_ELIGIBLE` at scientific commit `821c8d8`. On held-out
  calibration, RecurrentPPO beat the complete 65,536-calendar open-loop frontier in all three cells
  with simultaneous lower bounds +0.0366 to +0.0663, genuine feedback and exact resource/demand
  checks. It did not beat the best classical state-rich controller and its virgin block was not
  opened.
- Correct claim boundary: positive calibration evidence of learned adaptation over open-loop;
  neural premium, independent replication, tail-safe deployment and retained-learning value are
  not established.
- Program Q is a separate prospective frozen-policy replication. CVaR10 is secondary there, while
  resource equality, demand preservation, anti-shed/product floors, feedback and information
  placebos remain fail-closed identification gates. Its primary learner is now unconditionally the
  ten-checkpoint historical RecurrentPPO population frozen by SHA-256; collaborator architecture
  sandboxes are non-promotable sidecars and cannot delay or replace Program Q.

Authoritative artifacts are `docs/PROGRAM_O_R_TERMINAL_VERDICT_2026-07-17.md`,
`research/paper2_exhaustive_search/program_o_ret_calibration_v12_terminal_audit_20260717.json`, and
`contracts/program_q_frozen_policy_replication_v1.json`. Paper 3 remains unauthorized until Program
Q returns either `PASS_Q_NEURAL_PREMIUM` or
`PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT`.
=======
Terminal: `STOP_CALIBRATION_NOT_ELIGIBLE` (compound gate; commit `821c8d8`). Separated
preregistered estimands: **H_OL (learner vs complete 65,536 open-loop frontier) POSITIVE in all
3 cells** (LCB95 +0.037..+0.066, 41-44/48 favorable, feedback/placebos/resources/replay clean);
**Δ_N (learner vs best classical belief controller) ≈ 0** (LCB95 −0.008..−0.014). Reading:
*learned adaptation real; no neural premium over structured decision theory.* Virgin 7480101-48
sealed forever. Prospective replication frozen as
`contracts/program_q_frozen_policy_replication_v1.json` (N=128/cell by frozen power rule, block
7490001+, four terminal outcomes, David-challenger clause). See
`docs/PAPER2_CLAIM_LADDER_2026-07-18.md`.
>>>>>>> origin/main
