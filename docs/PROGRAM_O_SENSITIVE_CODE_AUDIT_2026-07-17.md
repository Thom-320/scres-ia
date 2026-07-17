# Program O sensitive-code audit — 2026-07-17

## Scope and governing status

This audit is based on the exact Program O-R training commit
`02aad62ff5997458ca427729134291f77abc5bca`. It covers:

- the O-R incremental environment, training partition, launcher, evaluator,
  full-DES replay auditor, and execution freeze;
- the relevant-risk v1.1 post-warm-up activation change and terminal G1
  artifact at `0d30b2323d03454083c03e07162fa3de17c6e0b6`;
- Claude Fable's commits `2f3d014` and `e581326` on
  `codex/garrido-risk-sensitivity`.

The historical verdict remains
`STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`. O-R training may finish. O-R
calibration and confirmation are **HOLD** until the evaluator defects below are
fixed prospectively, before opening tape `7480001`. Relevant-risk v1.1 remains
`STOP_V1_1_BEFORE_G2_RARE_RISK_FIXTURE_NOT_POPULATED`; G2 is not authorized.

## Findings

### F1 — PASS: O-R is not trained on one physical seed

The training design separates optimizer seeds from trajectory seeds:

- optimizer seeds: `8101..8110`;
- ten disjoint blocks of 25,025 skeletons;
- 25,024 eight-step episodes plus one reset-only sentinel per learner;
- 250,240 stepped training episodes total;
- three cells alternate round-robin inside every learner;
- calibration `7480001..7480048` and confirmation
  `7480101..7480148` do not overlap training.

The environment does not return tape ID, seed, true cell parameters, or latent
regime in the observation. The remaining generalization risk is overfitting to
the three-cell generator family, not memorization of one seed.

### F2 — PASS: frozen training source and live VPS custody

Every source hash in `program_o_ret_only_learner_v1` matches the worktree and
the execution-freeze contract hash is
`4471ffd4a1bc5997cc61481a6e0d42059abf78206197ba7be2959bead9a411fa`.

At audit time the VPS reported:

- commit `02aad62ff5997458ca427729134291f77abc5bca`;
- launcher SID/PGID `1450721`;
- ten live training processes and one watcher;
- approximately `177152..186880 / 200192` steps;
- no model output yet;
- no calibration or confirmation directory.

The training code, seed partition, terminal-only ReT reward, and risks-off
physics are internally consistent. The focused Program O suite passed 65/65.

### F3 — CRITICAL: trajectory feedback is computed but not gated

`scripts/evaluate_program_o_ret_learner.py:71-80` computes a per-model
`trajectory_audit(...)["passed"]`. Lines 247-253 store it. Lines 301-318 build
`provisional_primary_pass` without ever reading it.

Consequence: a learner that collapses to a fixed or phase-only calendar can be
reported as a provisional PASS despite the frozen requirements
`trajectory_feedback_required=true` and
`fixed_calendar_collapse_forbidden=true`.

Required correction before calibration: fail closed unless every frozen
feedback criterion is satisfied, and populate a top-level feedback gate.

### F4 — CRITICAL: frozen replacement controls are declared but absent

The contract names `modal`, `phase_only`, and `frequency_matched` comparators.
The evaluator enumerates open-loop calendars and the ten classical
configurations, but it neither constructs nor scores those three learner
replacement controls. Diversity of the learner's calendars is not equivalent
to demonstrating that state information caused its value.

Required correction before calibration: implement the three replacements on
the same tapes, define their frozen contrasts, and include them in the feedback
gate. Do not infer feedback from calendar diversity alone.

### F5 — CRITICAL: virgin confirmation can be opened directly

The CLI accepts `--phase confirmation` at lines 164-180 and immediately selects
the virgin seed range. It does not require:

- a populated calibration result;
- a calibration PASS;
- a passing direct full-DES audit;
- an independent authorization artifact.

The requirement exists only as prose in the execution freeze. Required
correction: confirmation must require explicit paths and hashes for all four
artifacts and reject any missing or failing predecessor.

### F6 — HIGH: physical integrity gates are incompletely implemented

The evaluator gates only `mass_residual` and `partition_residual`. It does not
explicitly gate:

- exact equality of gross production rights;
- charged dispatch slots and charged vehicle-hours;
- preservation of every generated order in the visible/full/lost/unresolved
  ledgers.

These quantities are available in `MATRIX_KEYS`. Required correction: compare
the learner and each selected comparator exactly on the frozen charged-resource
metrics, and add a populated demand-ledger identity check. Actual transport
utilization can remain secondary as specified.

### F7 — MEDIUM: evaluation custody is not executable end-to-end

The execution freeze requires raw-matrix checksums after evaluation, but the
evaluator does not emit them. It also exits zero regardless of provisional PASS
or FAIL. A zero exit is acceptable for result production only if a separate
fail-closed adjudicator consumes the populated result, direct replay, hashes,
and feedback controls. No such adjudicator is currently frozen.

### F8 — PASS with disclosed boundary: post-warm-up risk activation

The v1.1 change starts the parent DES with risks disabled, waits for the common
Program O warm-up event, then launches only the requested risk generators.
This preserves a shared neutral initial state and risks-off identity. Per-risk
RNG streams are initialized independently from the action path.

The test coverage for this sensitive change is too thin: the added test checks
only contract text. Future work needs executable tests for:

- no pre-warm-up risk event;
- correct enabled-risk subset;
- unchanged per-risk CRN under different actions;
- no duplicate generator launch;
- risks-off bit identity.

### F9 — STOP confirmed: G1 v1.1 does not authorize G2

The populated artifact reports:

- R11 PASS: 192 events, Op5/6;
- R14 PASS: 672 events, Op7;
- R24 PASS: 48 events, Op13;
- R21 FAIL: zero events;
- R22 FAIL: five events, only Op8/10 observed;
- R3: zero events.

The runner has no G2 command and refuses to overwrite a result. The STOP is
correct under the frozen contract. The scientific lesson is narrower: the
phi=1, twelve-tape stochastic-coverage fixture was underpowered for rare R21
and for union coverage of random R22 targets. A successor must separate
deterministic liveness/unit tests from descriptive incidence coverage; this
does not permit G2 under v1.1.

### F10 — CORRECTION REQUIRED in Claude commit `e581326`

The commit says R24 contingent demand has no `P_C/P_H` label. That is factually
incorrect for the implemented Program O model:

1. `_sample_calendar_demand_quantity()` adds pending R24 surge to the next
   scheduled order;
2. `_op13_demand()` assigns that order the already-frozen product label from
   `program_o_tape["order_products"]`;
3. G1 observed 13 contingent `P_C` orders and 11 contingent `P_H` orders.

R24 is therefore model-defined under the disclosed conditional convention
"the next scheduled order inherits the surge." It may remain excluded from an
MFSC-representative analysis pending Garrido's product-attribution fact, but
not because the code lacks a label. Do not merge `e581326` without correcting
its rationale.

### F11 — Claude v1 runner must remain superseded

The older runner at `2f3d014`:

- starts risks during Program O prefix construction;
- uses one seed and phi=4 for G1;
- does not read the actual `affected_ops` field correctly;
- can overwrite `g1_result.json`;
- has no closed-loop risk-aware controller.

It is not an admissible execution surface. The v1.1 STOP artifact supersedes
it. Any future G2 requires a newly frozen incremental risk adapter whose
risks-off action sequence matches the custodied `belief_mpc__3` calendar index
on every tape.

## Disposition

1. Let O-R training finish and retrieve/hash only its final checkpoints.
2. Do not open calibration or confirmation with the current evaluator.
3. Freeze a prospective evaluation-only amendment addressing F3-F7; do not
   retrain or alter learner hyperparameters/reward.
4. Keep relevant-risk v1.1 stopped before G2.
5. Correct or supersede Claude commit `e581326`; do not merge the old v1 runner.
6. Keep Program P on `HOLD_PENDING_DOMAIN_FACTS`.

