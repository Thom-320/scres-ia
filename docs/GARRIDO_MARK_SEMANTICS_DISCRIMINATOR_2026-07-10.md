# Excel mark-semantics discriminator (2026-07-10)

**Question:** are Garrido's per-order risk marks a function of the order's
time window (overlap semantics) or do they travel with consumed material
(genealogy)? This decides whether the batch-provenance build is necessary
and for which risks.

**Instrument:** under overlap semantics marks are MONOTONE in the window —
if order A's [OPTj, OATj] contains order B's, every risk marked on B must be
marked on A. Superset violations are impossible under any window rule and
prove non-window attribution. Computed over all nested attended-order pairs
in six odd CFs (24k-41k pairs per CF; even CFs untouched).
Artifact: `outputs/audits/excel_mark_semantics/superset_violations_odd_cfs.json`.

## Result — Garrido's semantics are HYBRID, per risk family

| Risk | Violation rate (range across CFs) | Proven semantics |
|---|---|---|
| R24 | 0.000-0.004 | TIME WINDOW. The fixed-window arm is his mechanism, not a calibration. |
| R14 | 0.000-0.001 | Time window / ubiquitous quality exposure. |
| R11 | 0.056-0.126 | Mostly window (small residual, per-station columns). |
| R23 | 0.007-0.280 | Mixed, CF-dependent. |
| R22 | 0.037-0.235 | Mixed, CF-dependent (convoy-specific when the order's own leg was hit). |
| R21 | 0.350-0.624 | GENEALOGY — marks travel with lots/warehouse batches. |
| R13 | 0.371-0.446 | GENEALOGY — delayed supplier lots mark their consumers. |
| R12 | 0.587-0.739 | GENEALOGY — contract-delay lots mark their consumers. |

## Consequences (binding for the next sprint)

1. **R24's 168 h window is RE-LABELED from calibration to mechanism-consistent**
   (zero superset violations in every CF; the exact width remains to be
   estimated from his data, but window-ness itself is proven). Same for R14.
2. **The batch-provenance build is justified — but ONLY for R12/R13/R21 (and
   partially R22/R23)**. This narrows Codex's declared blocker to the risks
   where the data demands it, and matches the counterfactual finding that only
   4/15 R13 events materially alter availability (only consumed-late lots mark).
3. The next attribution candidate is therefore hybrid: window semantics for
   {R24, R14, R11, most R23}, lot-genealogy for {R12, R13, R21, R22-own-leg} —
   evaluated under the existing joint rule on odd CFs, then ONE run on evens.
