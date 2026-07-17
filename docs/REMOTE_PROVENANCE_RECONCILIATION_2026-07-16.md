# Remote provenance reconciliation — 2026-07-16

## What happened

Three independent external reviews (2026-07-16) audited the **GitHub remote** and concluded that
the decisive next step was to *"close the still-open Program O state-rich H_obs fit, then run the
sealed validation"* — presenting it as the program's best current opportunity.

**That experiment already ran and terminally failed.** The reviews could not see it because the
remote was ~2 days stale:

| Fact | Local (true) state | What the remote showed the reviewers |
|---|---|---|
| Program O state-rich H_obs | Fit stopped on the resource gate → dual-resource diagnostic (fixed-clock development signal) → fixed-clock-physical OOS validation on sealed 7420049–96 (opened once): STOP 26/48 + a simultaneous-inference defect → **preregistered CORRECTIVE validation on fresh sealed 7430001–48: canonical mean advantage PASS in all 3 cells (LCB95 0.043–0.066; 42/44/46 of 48 favorable; 27/27 placebos pass; 1,451 physical replays, 0 failures) but the JOINT `ret_visible_cvar10` safety gate FAILS in 2 cells (simultaneous LCB95 −0.0086 / −0.0155; point estimates positive) → `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`, second rescue prohibited** (`results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`) | `PENDING_STATE_RICH_H_OBS_FIT` |
| Paper 2 boundary | Exhaustion certificate `00a9005` + OAT-scope retraction `087daa6`; wartime atlas `STOP_COMPUTE_INFEASIBLE` (4c9702e); D1/op11 rounds | Intermediate `PROGRAM_O_TERMINAL_OUTCOME_CERTIFICATE_2026-07-15.md` (H_PI established, H_obs pending) |
| `origin/main` tip | — | `e3389aa`, BEFORE the entire closure arc |
| Evidence commits (`897ebab`, `adbfb8f`, `00a9005`, `087daa6`, `4c9702e`, `eea30d8`, …) | present locally, custody-verified | **absent from every remote branch** |
| `REPOSITORY_SOURCE_OF_TRUTH.md` | superseded several eras ago | still tells the Track-A story (`ReT_seq_v1` as primary training reward) |
| PR #3 | — | OPEN draft, 1,527 files, unreviewable |

The reviewers' own governance complaint — *"the remote is not a coherent source of truth"* — is
therefore not cosmetic: **it caused our own auditors to recommend reopening a burned sealed
experiment.** This document plus the branch pushes below are the fix.

## Corrective facts binding any strategy discussion

1. **Program O is CLOSED at H_obs — but with a finer terminal shape than "no conversion"** (this
   section CORRECTS the first version of this document, which lossily reported the 26/48 stop as
   final): the corrective validation showed the observable controller converts a MATERIAL MEAN
   improvement with genuine feedback, equal resources, and all 27 placebos beaten — and fails only
   the joint lower-tail service-safety gate (CVaR10) in 2 of 3 cells. The honest paper claim is
   therefore: *H_PI large; mean observable conversion real; jointly SAFE conversion not
   established* — the frontier between physical value, average observable value, and deployable
   safe value. Per the frozen contract, a second rescue is prohibited; a Program O-v2 requires NEW
   operational physics from Garrido (substitution classes, distinct BOM/rates, setups, pre-commit
   mission signals, per-class deadlines), never a re-run of the same controller under the same
   assumptions.
2. **Q13/P1 (non-fungible ration classes) therefore changes FUNCTION:** a "yes" from Garrido makes
   the certified H_PI ceiling MFSC-representative (interpretation/framing) — it does NOT reopen the
   closed observable contract (certificate: "restores representativeness of the CEILING only").
3. The reviews' methodological recommendations that survive this correction and are adopted:
   M1 as blocking question (with the same-timestamp event-order sub-question and the Annex-B
   barrier-matrix anchor), the block structure and decision table for the Garrido packet, the
   H_learned vs H_neural estimand separation, the Paper-3 "retained beliefs, not weights" design —
   all recorded as FUTURE-CONDITIONAL designs, contingent on a door actually opening.

## Reconciliation actions (this commit + pushes)

- Pushed to origin: `codex/garrido-risk-sensitivity` (contains the full closure chain incl.
  `897ebab`, the risk screen, atlas freeze, GSA STOP, D1 rounds, op11 probe),
  `paper2-exhaustion-certificate`, `war-risk-interaction` (audit trail),
  `codex/program-o-state-rich-audit-20260715` (contains diagnostic `a9733d0`).
- Remaining for the PI (main-branch / outward decisions): replace
  `docs/REPOSITORY_SOURCE_OF_TRUTH.md` on `main` (draft: state = no learned adaptive value;
  Program O H_PI certified + H_obs CLOSED after sealed OOS; wartime atlas compute-infeasible;
  Paper 3 blocked), close or supersede PR #3 with per-program reviewable PRs, and cut an immutable
  tag for the Program O evidence chain.
- Large binary/matrix artifacts referenced by certificates should ship as a release bundle with
  SHA-256s, not as further Git history.

## Rule derived from this incident

**Every custody-verified terminal verdict must be pushed to the remote in the same working session
that produces it.** An auditor who can only see the remote will otherwise re-derive yesterday's
program state and spend real effort optimising a dead branch of the decision tree.
