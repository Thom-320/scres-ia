# Repository Source of Truth

This note freezes the current paper-facing repository story. Treat it as the
primary reference for README examples, manuscript edits, benchmark commands,
and reviewer-facing explanations unless a later note explicitly supersedes it.

> **Paper 2 superseding addendum (2026-07-20).** Program Q is no longer
> prospective. It is terminal at `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION`:
> learned ReT superiority over the complete 65,536-calendar open-loop frontier
> replicated in all three cells, and practical equivalence to the strongest
> tested structured controller was demonstrated. The compound STOP is due to
> failure of the frozen worst-product-fill non-inferiority guardrail. The next
> Paper 2 attempt is prospectively governed by
> `contracts/paper2_learning_augmented_event_triggered_mpc_v1.json`: strengthen
> ReT-aligned MPC first, require residual and endogenous-timing headroom before
> training, then test a confidence-gated learning-augmented event-triggered MPC.
> No new result from that hybrid exists yet. If any essential gate fails, the
> executed Program Q decomposition manuscript is the publication route.

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

## What is not the main paper lane

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

## Required language discipline

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
- Program Q is terminal and independently replicated the ReT and equivalence endpoints. Its
  compound label remains `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION` because the frozen
  worst-product-fill non-inferiority guardrail failed in all three cells. Do not paraphrase the
  compound STOP as absence of learned adaptation, and do not paraphrase endpoint replication as a
  deployment-safety or neural-premium result.

Authoritative artifacts are `docs/PROGRAM_O_R_TERMINAL_VERDICT_2026-07-17.md`,
`research/paper2_exhaustive_search/program_o_ret_calibration_v12_terminal_audit_20260717.json`, and
`contracts/program_q_frozen_policy_replication_v1.json` and
`docs/PROGRAM_Q_TERMINAL_VERDICT_2026-07-18.md`. Paper 3 remains unauthorized under the new
master contract until the prospective Paper 2 hybrid establishes learned adaptive value; no
historical verdict is reopened.
