# Hybrid attribution candidate gate (2026-07-10)

> **Subsequent project decision:** the failed hybrid remains boundary evidence,
> but exact Simulink attribution recovery is no longer an RL training gate. See
> `docs/GARRIDO_OPERATIONAL_REFERENCE_DECISION_2026-07-10.md`.

## Executive decision

**Do not promote the proposed window-plus-genealogy hybrid. Do not inspect even
Cf cases.** The Excel monotonicity discriminator rejects a pure window-only rule
for R12/R13/R21, but that rejection does not uniquely identify genealogy. The
event-specific FIFO counterfactual was therefore tested as a candidate mechanism
for R13 across all five odd R1 Cf cases. It fails to reproduce the workbook mark
share consistently.

## Correction to the initial discriminator interpretation

The historical six-Cf artifact had the correct qualitative signal but overstated
identification. The reproducible all-odd rerun shows:

- R12: violation rate 0.576–0.790;
- R13: 0.395–0.551;
- R21: 0.306–0.690;
- R24: usually near zero, but 0.041 in Cf17;
- R23: 0.005–0.566 depending on Cf.

A violation proves only that a binary mark is not determined solely by overlap
between `[OPTj,OATj]` and fixed event intervals. It is compatible with material
genealogy, operation/path exposure, quantity assignment, or another event-order
mapping. Near-zero violations are consistent with a monotone window rule, not
exclusive proof of one. Consequently, the fixed 168 h R24 arm remains a
calibrated mechanism-plausible benchmark rather than a proven Garrido rule.

## Complete R13 FIFO gate on odd Cf cases

Each endogenous R13 calendar was frozen. Every R13 event was removed one at a
time while all other duration events, seeds, demand, and physics remained fixed.
Only positive increments of cumulative order-release debt were allocated to the
concrete counterfactual consumer; downstream identity reshuffling was reported
separately.

| Cf | R13 events audited | FIFO direct order share | Excel R13 share, same 20,000 h window | Gap |
|---:|---:|---:|---:|---:|
| 1 | 116/116 | 0.2302 | 0.2734 | −0.0433 |
| 3 | 117/117 | 0.7153 | 0.2673 | +0.4480 |
| 5 | 117/117 | 0.1552 | 0.2009 | −0.0457 |
| 7 | 92/92 | 0.8260 | 0.1785 | +0.6475 |
| 9 | 117/117 | 0.2885 | 0.2779 | +0.0106 |

Three cells are close; two fail catastrophically. Selecting only Cf1/Cf5/Cf9
would be post-selection. The candidate fails jointly.

## Calendar-clock sensitivity

An opt-in Op2 calendar-anchored release clock was tested because the inherited
start-to-start clock can shift future supplier releases after R13. It changed
upstream recovery timing but produced exactly the same direct FIFO mark shares
in Cf1, Cf3, and Cf7. The clock hypothesis does not explain the cross-Cf failure.

## Binding consequences

1. Do not implement a hybrid ReT attribution rule from the current evidence.
2. Do not run R12/R21 expansion merely to rescue the failed R13 candidate.
3. Do not estimate a new R24 window or inspect even Cf cases.
4. Historical conclusion at the time of this gate: keep RL blocked. This was
   later superseded by the operational-reference decision cited above.
5. The next information-bearing step is to recover Garrido's actual Simulink
   event-to-order assignment logic or obtain a domain validation of what each
   risk column records. Without that source semantics, further DES attribution
   tuning is an inverse problem with multiple observationally compatible rules.

Reproducible code:

- `scripts/audit_garrido_excel_mark_semantics.py`
- `scripts/audit_garrido_event_delayed_quantity.py`

Primary local artifacts:

- `outputs/audits/excel_mark_semantics/superset_violations_all_odd_cfs.json`
- `outputs/audits/garrido_event_delayed_quantity_cf{1,3,5,7,9}_r13_full/`
