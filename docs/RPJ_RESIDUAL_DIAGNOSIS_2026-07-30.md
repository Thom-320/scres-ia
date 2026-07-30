# The `RPj` residual is not in `RPj` — it is the shape of the `CTj` distribution

**Status:** `DEVELOPMENT_DIAGNOSIS_NO_CONSTANT_CHANGED`. Follows
`RPJ_MODE_FINDING_2026-07-30.md`, which established that the thesis's `elapsed` formula
fits better than our shipped `disruption` default on all four measurements but leaves R1r
at 2.42× on the mean and 7.11× on the p95.

## The residual, located

| | ours (R1r, `elapsed`) | Garrido (R1r) |
|---|---:|---:|
| orders with a risk indicator | 0.806 | **0.9995** |
| `CTj` p50 | **54.0** | **101.4** |
| `CTj` p95 | **3246.0** | **2238.6** |
| `RPj` p50 | 54.0 | 99.5 |
| `RPj` p95 | 3246.0 | 455.6 |
| **`RPj / CTj` median** | **1.0000** | **0.9579** |

Two things fall out immediately.

**`RPj` is not independently wrong.** Our `RPj/CTj` is 1.0000 exactly — `RPj` *is* `CTj`,
every time. His is 0.9579. So our `RPj` inherits its distribution wholesale from `CTj`, and
the 2.42× and 7.11× are `CTj` errors wearing an `RPj` label. Fixing `RPj` in isolation is
not possible because there is nothing in `RPj` to fix.

(The exact 1.0000 has its own cause: risks in our model are frequent and long enough that
one is essentially always active at order placement, so `eff_risk_start` clamps to `OPTj`
and the elapsed formula returns the full cycle time. His 0.9579 says the first risk
typically begins shortly *after* placement.)

**The `CTj` distribution is the wrong shape, in both directions at once:**

| | ours | Garrido |
|---|---:|---:|
| mass exactly at the minimum | **64.1%** at 54.0 | p50 is 101.4, so the mass is in the middle |
| mass in the middle (54 < CTj ≤ 500) | 18.1% | — |
| mass in the tail (> 500) | 17.8% | — |
| p50 | 54.0 (**0.53×** his) | 101.4 |
| p95 | 3246.0 (**1.45×** his) | 2238.6 |

**Our median cycle time is half his and our tail is 45% longer.** We are bimodal — a spike
at exactly 54 and a long tail — where his distribution has its mass in the middle.

## This is the same root cause as two other findings

The spike at exactly 54 is the on-hand short-circuit; the long tail is the queue that
everything else falls into. That is the **two-path structure** already diagnosed as the
cause of the autotomy problem, and it is consistent with `scored_orders_per_year` being
2× his: we serve too many orders instantly and too few through the middle.

So three separate items on the priority list — the `RPj` residual, the autotomy share, and
the scored-order rate — **reduce to one defect**: the on-hand path serves 64% of orders at
a single instant value instead of a distribution, and everything that misses it waits far
too long.

## What this means for the fix

`on_hand_transit_mode = "modelled_legs"` was the right lever and it is not enough. Adding a
small handling draw spreads the spike by minutes; the gap needs mass moved from **54 into
the 100–500 h range**, which means fewer orders served instantly and more served after a
realistic wait, not a jittered constant.

That is a change to how orders are matched against stock, not a parameter. It needs its
own preregistration and it should be measured against all six moments at once — because it
will move the tail as well, and the tail is currently 45% too long in the same direction
the change would push it.

## What is now settled

- The shipped `RPj` mode is not the thesis's and fits worse. That stands on its own and is
  actionable independently.
- The residual after switching is **not an `RPj` defect** and must not be chased there.
- Three priorities collapse into one, which changes the sequencing: the on-hand matching
  is the single highest-leverage change available, and it is bigger than anything attempted
  today.
