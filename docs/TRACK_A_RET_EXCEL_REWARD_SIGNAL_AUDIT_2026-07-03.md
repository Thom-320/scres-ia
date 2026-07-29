# Track A Excel-ReT Reward Signal Audit (2026-07-03)

## Corrected Verdict

The Garrido/Excel ReT endpoint remains the correct primary evaluation metric. The issue is not
the metric. This note originally described the 43.9% term below as "retroactive revaluation" of
old orders. That label was too broad and has been corrected after an independent audit froze
pre-step order snapshots with `copy.deepcopy`.

The narrow question -- "does time passing alone rescore an already unchanged old order?" -- is
answered **no**: the corrected frozen-snapshot audit finds 0.0% true retroactive rescoring.

The 43.9% measured here is still useful, but it means something different: **delayed credit from
orders that already existed before the decision step and were resolved during the step**. In other
words, many rewards in a weekly step come from orders placed in earlier weeks whose `OATj`/`CTj`
fields are filled in during the current step. That is real delayed consequence, not a bookkeeping
bug.

This may still contribute to Track A's learning difficulty through ordinary delayed credit
assignment, but the fix would not be to "clean" the Excel-ReT reward. If needed, the next
training-side hypothesis is to tune the return horizon (`gamma`, n-step/GAE behavior, or a
separate shaping reward) while keeping Excel ReT as the primary evaluation endpoint.

## Artifact

- Script: `scripts/audit_track_a_ret_excel_reward_revaluation.py`
- Output: `outputs/audits/track_a_ret_excel_reward_revaluation_2026-07-03/`
- Step decomposition: `outputs/audits/track_a_ret_excel_reward_revaluation_2026-07-03/step_decomposition.csv`
- Summary: `outputs/audits/track_a_ret_excel_reward_revaluation_2026-07-03/summary.json`

## Overall Results

| Quantity | Value |
|---|---:|
| Steps audited | 468 |
| Sum absolute total reward delta | 572.176 |
| Sum absolute old-order state-change contribution | 251.012 |
| Sum absolute new-order contribution | 321.639 |
| Old-order state-change share | 0.439 |
| Steps where old-order state-change dominates new contribution | 207/468 |

## By Regime

| Regime | Old-order state-change share | Old dominates steps |
|---|---:|---:|
| R13_phi1_psi1.5 | 0.479 | 20/52 |
| R13_phi4_psi1.5 | 0.793 | 30/52 |
| R13_phi8_psi1.5 | 0.502 | 28/52 |
| R14_phi1_psi1.5 | 0.433 | 8/52 |
| R14_phi4_psi1.5 | 0.580 | 43/52 |
| R14_phi8_psi1.5 | 0.580 | 44/52 |
| R24_phi1_psi1.5 | 0.349 | 4/52 |
| R24_phi4_psi1.5 | 0.357 | 11/52 |
| R24_phi8_psi1.5 | 0.529 | 19/52 |

## Interpretation

The Excel formula is cumulative by design, but the independent frozen-snapshot audit shows that
unchanged old orders are not being retroactively rescored merely because time advances. The live
object audit in this note captured mutable `OrderRecord` state changes during `env.step()`: for the
flagged old orders, the relevant pattern is `OATj: None -> timestamp`, i.e. the order was delivered
in the current step.

For Paper 1, this should be used only as a Track A learning-mechanism explanation:

> Track A's negative result is not merely "no headroom." The conservation-respecting static gate
> shows headroom, while PPO fails to convert it. Reward diagnostics rule out a retroactive
> rescoring bug, but show that Excel-ReT training rewards contain substantial delayed credit from
> earlier orders resolved in later weekly steps.

Do not use this audit to replace the manuscript's primary metric. The primary endpoint remains the
Garrido/Excel ReT formula.
