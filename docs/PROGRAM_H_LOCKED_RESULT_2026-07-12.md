# Program H — locked-test result + dual-bound status — 2026-07-12

Status: **Strong Case-A (information-limited) evidence on the LOCKED test 1070001; a CERTIFIED sub-δ
formal bound remains open because the ret_order objective is non-additive.** 1080001 sealed.

## Locked test 1070001 (n=150, ret_order)
| Quantity | vs ABAB | reading |
|---|---|---|
| ABAB ret_order | 0.5138 (baseline) | |
| belief cover (signal + demand) − ABAB | **−0.0262 [−0.0349, −0.0180]** | loses |
| QMDP-nowcast (perfect current tempo) − ABAB | **−0.0262 [−0.0349, −0.0180]** | loses (= belief) |
| J_PI − ABAB (rigorous loose ceiling) | **+0.0153 [0.0114, 0.0190]** | > δ_min |

- The best available-information policy — and a policy with a PERFECT nowcast of both current regimes —
  is materially WORSE than blind alternation ABAB (−0.026). Consistent with development (−0.021).
- The clairvoyant advantage (+0.015) lives almost entirely in inaccessible FUTURE transition timing.

## Dual bound: attempted, FAILED sanity, DISCARDED
The Brown–Smith–Sun information-relaxation dual was implemented but its penalty generator was in
service-loss units (thousands of rations) while ret_order ∈ [0,1]; the penalty dominated ~1000×, giving
`dual = 476` and violating the sanity chain `ABAB ≤ dual ≤ J_PI`. It is **invalid and discarded** (not
reported as a bound). The root cause is the ret_order objective's **non-additivity**: a valid dual needs
a penalty in ret_order units via a per-completion-week decomposition — a real implementation the quick
attempt did not land. `results/program_h/dual/verdict.json` records the failure honestly.

## Honest overall status
- **Certified rigorous bound available:** J_PI (+0.015) ≥ J*_obs. Because +0.015 > δ_min=0.01, the
  certified LOOSE bound does NOT by itself prove Case A.
- **Tight evidence (dev + locked):** the best non-anticipative policy — even with a perfect nowcast —
  loses to ABAB by ~−0.026. This strongly supports Case A (information-limited), but is a strong-policy
  argument, not a certified sub-δ upper bound.
- **The one remaining formal step:** a ret_order-unit dual (per-completion-week decomposition) OR the
  exact belief-MDP backward induction with the additive completion-week reward. Both are tractable given
  the tiny state; neither was landed this turn. This is disclosed, not hidden.

## What it means for the paper
The information-limited reading is strongly supported and publishable as-is: across D, DRA-1/2/2b, E, F,
G, and now the H belief audit, observable/learnable adaptive value is scarce because the decisive
information is inaccessible future state, not because the policies were weak — even a perfect current
nowcast cannot beat a robust static schedule under the thesis resilience metric. The certified sub-δ
bound would upgrade "strongly supported" to "proven"; it is the last, well-scoped computational item.
