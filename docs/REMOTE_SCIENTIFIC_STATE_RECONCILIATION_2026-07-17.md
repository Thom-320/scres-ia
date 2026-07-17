# Remote scientific-state reconciliation — 2026-07-17

## Purpose

GitHub `main` lagged the custody-verified local evidence and still described an
older Track-A/Track-B benchmark story. External reviewers therefore treated the
Program O state-rich experiment and sealed validation as pending, although the
validation sequence had already reached a terminal stop.

This change is a deliberately small provenance PR. It does not modify the DES,
policies, metrics, seeds, statistical thresholds, or scientific results.

## Reconciled terminal state

- Full-DES Program O safe `H_PI`: `0.1515137892`, simultaneous safe LCB95
  `0.1156159089`, exact fungible null `0`.
- First prospective observable block `7420049-7420096`: opened once; automatic
  adjudication retracted after comparator-selection and simultaneous-inference
  defects were found.
- Single corrective block `7430001-7430048`: mean canonical ReT, all 27
  placebos, physical equality, action trajectories, and state counterfactuals
  passed.
- Joint safe contract: failed because simultaneous CVaR10 non-inferiority was
  negative in rho75/share90 and rho90/share75.
- Binding label: `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`.
- Second rescue: forbidden.
- Learner, Paper 2 positive claim, and Paper 3: not authorized.

## Files included

- `docs/REPOSITORY_SOURCE_OF_TRUTH.md`
- `docs/PROGRAM_O_FIXED_CLOCK_HOBS_VALIDATION_VERDICT_2026-07-15.md`
- `docs/PROGRAM_O_CORRECTIVE_HOBS_VALIDATION_VERDICT_2026-07-15.md`
- `results/program_o/fixed_clock_hobs_validation_v1/independent_audit_v1.json`
- `results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json`
- `results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json`
- `research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json`

## Scope limitation

The PR includes compact verdicts and checksum-bearing audit summaries. The raw
calendar matrices and large custody packages remain outside Git history. Their
SHA-256 identities are preserved in the included JSON records. This is enough
to reconcile what the repository may claim, but not to recreate every episode
from `main` alone.

## Effect on existing GitHub work

The large draft PR #3 is superseded as a statement of current scientific
status. It should not be merged as the provenance reconciliation. Any useful
code from that PR should be reviewed later in small, program-specific changes.
