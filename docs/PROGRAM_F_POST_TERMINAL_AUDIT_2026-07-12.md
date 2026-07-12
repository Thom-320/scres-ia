# Program F post-terminal audit — 2026-07-12

## Final classification

`TERMINAL STOP — FINAL CRN/OBSERVATION AUDIT COMPLETE`

The audit does not reopen Program F, change any endpoint, select a cell, or
authorize calibration or learning. It resolves three presentation and
reproducibility items discovered after `STOP_PROGRAM_F_SCREEN`.

## 1. Runtime CRN identity

The original screen repeated the input `threat_sha256` across policies. That was
necessary but insufficient evidence that each simulation consumed the same
exogenous sequence.

The post-terminal audit therefore hashes events and demand from runtime logs:

- threat: event id, risk id, onset relative to treatment, unmitigated duration,
  affected operations, magnitude and context at onset;
- demand: order id, placement time, quantity, contingent flag and destination.

Coverage:

- all 24 screen cells;
- all 288 cell-tape combinations;
- every fixed-budget action in each cell;
- 1,152 prefix-balanced branch-state comparisons;
- zero calibration, holdout or virgin tapes;
- zero learners.

Result:

- static runtime CRN checks: `288/288 PASS`;
- branch runtime CRN checks: `1152/1152 PASS`;
- threat mismatches: `0`;
- demand mismatches: `0`.

### Superseded completion-based diagnostic

An independent disposable audit initially returned
`FAIL_PROGRAM_F_CRN_CONSUMPTION` on one interdiction tape. It reconstructed
"consumed" threats from `damage_events`, whose rows are appended only after the
mitigation-dependent outage finishes. Near the terminal horizon, shorter damage
logged one additional completed event. The input event onset was identical; the
completion-based population was not.

This was a logging-population artifact, not a difference in exogenous threats.
The authoritative audit records every base event at onset, before applying M/T/R,
and compares identical follow-up horizons. The failed diagnostic remains
preserved and is marked superseded rather than deleted.

## 2. No privileged observations

`OBSERVATION_KEYS` is now an exact whitelist. Automated tests require the
runtime observation to equal it and reject keys containing future, oracle,
latent-context, repair-duration or future-outcome information. A mutation of an
exact future event cannot alter the contemporaneous observation when the frozen
noisy lead signal is held constant.

The allowed observation contains only noisy scores, realized condition and risk
history, inventories, backlog, active/pending mitigation and episode phase.

## 3. Claim correction

The screen used grouped cross-fitting within the 288 screen tapes. Seeds 950001+
were never opened, so this is not independent holdout confirmation.

Authoritative language:

> No admissible cell achieved positive cross-fitted observable-policy rollout
> conversion within the preregistered screen.

Forbidden language:

> Program F failed an independent out-of-sample holdout confirmation.

No holdout was attempted because the screen produced no selectable cell.

## Consequence

The terminal `STOP_PROGRAM_F_SCREEN` is now audit-complete and paper-ready. It
remains a scoped result about the frozen 24-cell screen and depth-3 observable
policy class, not a universal impossibility theorem for reinforcement learning.
