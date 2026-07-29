# Excel mark-semantics discriminator (2026-07-10)

**Question:** can Garrido's per-order risk marks be explained by a pure
order-window overlap rule? If not, are the violations compatible with the
material-genealogy mechanism tested in the DES?

**Instrument:** under overlap semantics marks are MONOTONE in the window —
if order A's [OPTj, OATj] contains order B's, every risk marked on B must be
marked on A. Superset violations reject a mark determined **only** by overlap
with fixed event intervals. They do not uniquely prove material genealogy:
operation-, route-, quantity-, or event-specific assignment can also violate
the null. Conversely, zero violations are consistent with overlap but do not
prove it exclusively.

The original six-Cf screen is retained as a historical artifact. The
reproducible script `scripts/audit_garrido_excel_mark_semantics.py` reruns the
test on all ten odd Cf cases. Even Cf cases remain untouched.

## Result — pure window-only attribution is rejected selectively

| Risk | Reproduced violation range, odd Cf | Defensible interpretation |
|---|---|---|
| R24 | 0.000-0.041 | Usually monotone, but Cf17 rejects a universal pure-window rule. |
| R14 | 0.000-0.001 | Strongly consistent with a monotone rule; mechanism not uniquely identified. |
| R11 | 0.052-0.146 | Mostly monotone with a non-window residual. |
| R23 | 0.005-0.566 | Strongly Cf-dependent; no single window classification is defensible. |
| R22 | 0.036-0.252 | Cf-dependent; pure window-only semantics rejected in some cells. |
| R21 | 0.306-0.690 | Pure window-only attribution strongly rejected; genealogy is a candidate. |
| R13 | 0.395-0.551 | Pure window-only attribution strongly rejected; genealogy is a candidate. |
| R12 | 0.576-0.790 | Pure window-only attribution strongly rejected; genealogy is a candidate. |

## Consequences (binding for the next sprint)

1. The 168 h R24 arm remains a **calibrated, mechanism-plausible benchmark**.
   This test alone cannot relabel its width or mechanism as Garrido-faithful.
2. Batch provenance and event-specific counterfactuals are justified as tests
   for R12/R13/R21, not proven as the workbook's unique generating semantics.
3. A hybrid attribution rule remains a candidate. It cannot be frozen until
   event coverage and predicted order-mark shares are reported on odd Cf cases.
4. No even-Cf run is authorized from this discriminator alone.
