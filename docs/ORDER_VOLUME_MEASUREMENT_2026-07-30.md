# Order volume measured — and the horizon assumption that inflated it

**Status:** `DEVELOPMENT_CORRECTION`. Reference regenerated as
`results/metric_audit/fidelity_reference_v3/` (sha `31ecf9f9dae8058a`), with the horizon
**measured per sheet** instead of assumed.

## The measurement asked for

| | generated / year | scored / year | scored fraction |
|---|---:|---:|---:|
| Garrido | **284.7** | **216.2** | 0.759 |
| ours | 311.0 | 274.7 | 0.884 |
| ratio | **1.09×** | **1.27×** | — |

Our generated rate is within 9% of his, and both are close to the thesis's own calendar:
demand every 24 h, six days a week, is 312 orders per year and we produce 311.

**So the order-volume hypothesis is refuted.** We do not generate twice his orders. The
scored rate is 1.27× his, driven by scoring a higher *fraction* — 88.4% against 75.9% —
not by generating more.

## The error that produced the hypothesis

`REFERENCE_HORIZON_YEARS` was hard-coded at 20 for every canonical sheet, citing §6.8.1's
*"20 years or 161,280 hours"*. Measured from each sheet's own `max(OPTj)`:

| sheet | max `OPTj` (h) | years |
|---|---:|---:|
| CF1 | 161,190 | **19.99** |
| CF10–CF16 | ~80,500 | **9.98–9.99** |

**Only CF1 and CF2 run 20 years. CF3–CF20 run 10**, exactly as Table 6.13 prescribes and
exactly as we had already recorded — the earlier note that "Cf1 and Cf2 ran to 19.84 years,
not the 10 Table 6.13 prescribes" was in our own memory and I did not apply it.

Dividing every sheet by 20 halved most of the reference and manufactured the discrepancy.
The corrected reference is **215.1 ± 9.0** scored orders per year in R1r and **217.3 ± 8.6**
in R2r, against the 119 / 108 the v2 reference claimed.

The horizon is now measured from `max(OPTj)` per sheet; the constant survives only as a
fallback for a sheet with no `OPTj` column.

## What this changes

| moment | R1r `d_k` (v2 ref) | R1r `d_k` (v3 ref) |
|---|---:|---:|
| `scored_orders_per_year` | 8.2 | **19.9** |
| `rpj_mean` | 30.1 | **42.1** |
| `ret_mean` | 8.1 | 7.8 |

| moment | R2r `d_k` (v2) | R2r `d_k` (v3) |
|---|---:|---:|
| `scored_orders_per_year` | **26.0** | 22.5 |
| `rpj_mean` | 11.1 | 17.0 |

The reference spreads tightened once the horizon stopped being wrong, so most `d_k` values
**rose**. `rpj_mean` remains the worst moment in R1r and is now clearly worst overall at
42 SD. The population moment is second in both families rather than first in R2r.

**Every aggregate distance computed against the v2 reference is wrong and must be
recomputed.** That includes today's `fidelity_sweep_v3_rate` and the matching sweep, whose
*shape* conclusions stand — those were measured on `CTj` quantiles directly, not on `d_k` —
but whose `d_k` columns do not.

## What survives, and what I would say now

The matching sweep's verdict is unaffected: it failed on mass at the minimum and on `CTj`
p50, both read straight off the distribution.

The recommendation I made — measure order volume before attempting a sixth mechanism —
was right to make and the answer is **no**: volume is not the common cause. The scored
*fraction* differing (88% against 76%) is a smaller and different question, and it points
at loss and warm-up exclusion rather than at demand.

And this is the sixth time in one day that a quantity fitted or assumed against one thing
turned out wrong when checked against another. The difference here is that the assumption
was **mine, made today**, in the very instrument built to catch that pattern.
