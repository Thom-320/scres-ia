# Evidence branch & tag index — 2026-07-16

Companion to `docs/REPOSITORY_SOURCE_OF_TRUTH.md` (merged via PR #4). Everything a clean clone
needs in order to audit the terminal state, with exact SHAs at publication time.

## Published scientific branches

| Ref | SHA (at index time) | Contents |
|---|---|---|
| `codex/garrido-risk-sensitivity` (science tip) | `ad43081c5` | Full closure chain: Program O terminal arc, Garrido risk screen, wartime timing atlas + GSA overlay freezes (`STOP_COMPUTE_INFEASIBLE`), D1 rounds, op11 probe (`DEVELOPMENT_NO_GO`), exhaustion + boundary certificates, reconciliation docs, Garrido question package |
| `paper2-exhaustion-certificate` | `087daa631` | Exhaustion certificate lineage + OAT-scope retraction (audit trail) |
| `war-risk-interaction` | `e5dc1354e` | Retracted GSA prototype + measured constraints. **Audit trail only — binding standard forbids merging or importing it** |
| `codex/program-o-state-rich-audit-20260715` | `6ce26bf5a` | Program O state-rich audit + dual-resource diagnostic chain |

## Published scientific tags

`paper2-boundary-2026-07-16` → `ad43081c5` · `program-o-terminal-2026-07-15` → `adbfb8f20`

These are **annotated published tags, not immutable**: the repository currently has no tag
protection rulesets, so they can technically be moved. Recommended owner action: add a ruleset
protecting `paper2-*` and `program-o-*` tags. Until then, verify the SHAs above rather than
trusting tag names.

## Files previously referenced by certificates but missing from `main`

Added in this PR (copied verbatim from `codex/garrido-risk-sensitivity`):

- `research/paper2_exhaustive_search/garrido_risk_timing_questions_v1.json`
- `research/paper2_exhaustive_search/GARRIDO_RISK_TIMING_EXACT_QUESTIONS_2026-07-15.md`

## Metric governance (formalized)

- **Governing promotion endpoint:** `ret_excel_request_snapshot_v2` (request-snapshot semantics).
  Evidence: exact reproduction of all 47,546 workbook formula cells given source snapshots
  (`research/paper2_exhaustive_search/excel_metric_reaudit_20260713.json` on the science branch)
  and thesis Annex B (per-request barrier matrix carrying generation time + cumulative
  backorders/losses, supporting OPTj-time snapshots).
- **`ret_excel_visible_v1` is QUARANTINED as a promotion endpoint** (OAT-time aggregation does not
  match the Annex-B request-time evidence). It remains in use as a *guardrail/diagnostic lens*
  (e.g., `ret_visible_cvar10` in the Program O corrective gate) — quarantine applies to primary
  claims, not to safety diagnostics.
- **Open question M1 (blocking for any virgin confirmation):** the source intention for event
  ordering at identical timestamps (delivery / list eviction / new request in the same Simulink
  instant). Asked verbatim in the Garrido package.

## Standing verdicts (pointers, not restatements)

Program O: `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`
(`results/program_o/fixed_clock_hobs_corrective_validation_v1/independent_audit_v1.json` on the
science branch; the frozen contract's `no_post_failure_changes` covers thresholds and guardrails).
Portfolio: `CURRENT_IMPLEMENTED_PORTFOLIO_EXHAUSTED_NO_LEARNER_AUTHORIZED`
(`research/paper2_exhaustive_search/paper2_current_boundary_certificate_20260716.json`).

## Known technical debt (separate PR)

`main` CI is red on one pre-existing test (fill_rate ≈ 0.4228915663; 340 pass / 1 fail) —
unrelated to the reconciliation merges; to be fixed in a dedicated technical PR.
