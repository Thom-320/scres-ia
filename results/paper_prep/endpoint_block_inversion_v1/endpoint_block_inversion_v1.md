# Endpoint x block sign-inversion sensitivity (machine-generated; do not edit)

Re-aggregation of frozen artifacts; no new simulation, no new seeds. Paired percentile bootstrap, 20000 resamples, seed 20260824.

- Block A source: `results/metric_audit/ret_metric_repair_confirmation_v1/result.json@aa0f3c0181ce`
- Block B sources: 4 files, `results/step3_*/full/rows.json`

## Family R1r — frozen incumbent `static_op3rmI0_op5rmI0_op9rationsI336` (posture [0, 0, 336])

| Endpoint | Block A (corrective, n=16) | Block B (Step-3, n=12) | Sign |
|---|---|---|---|
| `ret_excel` | -0.0000195 [-0.0000496, -0.0000002] 5/16 | -0.0000217 [-0.0000472, +0.0000039] 3/12 | stable |
| `ret_excel_clipped_0_1` | -0.0000195 [-0.0000494, -0.0000002] 5/16 | -0.0000217 [-0.0000472, +0.0000039] 3/12 | stable |
| `ret_excel_full_ledger` | -0.0000194 [-0.0000486, -0.0000001] 5/16 | -0.0000214 [-0.0000467, +0.0000040] 3/12 | stable |
| `ret_excel_quantity_time_clipped_0_1` | -0.0000196 [-0.0000486, -0.0000003] 5/16 | -0.0000217 [-0.0000472, +0.0000039] 3/12 | stable |
| `ret_thesis` | -0.0000194 [-0.0000483, -0.0000003] 5/16 | -0.0000214 [-0.0000467, +0.0000040] 3/12 | stable |
| `flow_fill_rate` | +0.0000202 [+0.0000035, +0.0000437] 5/16 | +0.0000208 [+0.0000036, +0.0000395] 6/12 | stable |
| `delivered_rations` | +3756.3125000 [+634.1875000, +7964.8125000] 5/16 | +3947.9166667 [+825.4166667, +7463.1666667] 6/12 | stable |

Control — Block B against the best static *within the block* (not the frozen incumbent):

| Endpoint | best static in block | delta mean | favorable |
|---|---|---|---|
| `ret_excel_clipped_0_1` | `static_op3rmI0_op5rmI0_op9rationsI1344` | -0.0000217 | 3/12 |
| `ret_excel_full_ledger` | `static_op3rmI0_op5rmI0_op9rationsI1344` | -0.0000214 | 3/12 |

Sign inversions across blocks: none

Tape blocks disjoint: **True** (A: 1710001-1710016, B: 1420001-1421006).

Within-block a ReT-family sign disagreements (same tapes, same policy pair): none
Within-block b ReT-family sign disagreements (same tapes, same policy pair): none

## Family R2r — frozen incumbent `static_op3rmI336_op5rmI0_op9rationsI168` (posture [336, 0, 168])

| Endpoint | Block A (corrective, n=16) | Block B (Step-3, n=12) | Sign |
|---|---|---|---|
| `ret_excel` | +0.0125156 [+0.0090039, +0.0159550] 15/16 | -0.0121712 [-0.0182663, -0.0055031] 1/12 | **INVERTS** |
| `ret_excel_clipped_0_1` | +0.0124747 [+0.0091086, +0.0159091] 15/16 | -0.0121712 [-0.0182663, -0.0055031] 1/12 | **INVERTS** |
| `ret_excel_full_ledger` | -0.0044835 [-0.0066005, -0.0023880] 2/16 | +0.0084152 [+0.0064970, +0.0103105] 12/12 | **INVERTS** |
| `ret_excel_quantity_time_clipped_0_1` | +0.0123867 [+0.0092443, +0.0152587] 15/16 | -0.0121827 [-0.0182799, -0.0055125] 1/12 | **INVERTS** |
| `ret_thesis` | +0.0003700 [-0.0010845, +0.0017306] 9/16 | -0.0000206 [-0.0001458, +0.0001375] 5/12 | **INVERTS** |
| `flow_fill_rate` | +0.0023396 [-0.0014491, +0.0062599] 11/16 | +0.0047571 [+0.0005404, +0.0088970] 9/12 | stable |
| `delivered_rations` | -25399.0625000 [-29344.4593750, -21173.5484375] 0/16 | +42784.0833333 [+35818.5000000, +50403.5000000] 12/12 | **INVERTS** |

Control — Block B against the best static *within the block* (not the frozen incumbent):

| Endpoint | best static in block | delta mean | favorable |
|---|---|---|---|
| `ret_excel_clipped_0_1` | `static_op3rmI0_op5rmI0_op9rationsI1344` | -0.0342220 | 1/12 |
| `ret_excel_full_ledger` | `static_op3rmI0_op5rmI1344_op9rationsI1344` | -0.0009908 | 2/12 |

Sign inversions across blocks: `ret_excel`, `ret_excel_clipped_0_1`, `ret_excel_full_ledger`, `ret_excel_quantity_time_clipped_0_1`, `ret_thesis`, `delivered_rations`

Tape blocks disjoint: **True** (A: 1810001-1810016, B: 1422001-1423006).

Within-block a ReT-family sign disagreements (same tapes, same policy pair): `ret_excel` vs `ret_excel_full_ledger`; `ret_excel_clipped_0_1` vs `ret_excel_full_ledger`; `ret_excel_full_ledger` vs `ret_excel_quantity_time_clipped_0_1`; `ret_excel_full_ledger` vs `ret_thesis`
Within-block b ReT-family sign disagreements (same tapes, same policy pair): `ret_excel` vs `ret_excel_full_ledger`; `ret_excel_clipped_0_1` vs `ret_excel_full_ledger`; `ret_excel_full_ledger` vs `ret_excel_quantity_time_clipped_0_1`; `ret_excel_full_ledger` vs `ret_thesis`

