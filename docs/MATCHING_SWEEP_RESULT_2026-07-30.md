# The matching sweep: nothing meets the declared acceptance

**Status:** `DEVELOPMENT_SWEEP_ACCEPTANCE_NOT_MET_NO_DEFAULT_CHANGED`. Executes
`PREREGISTRO_EMPAREJAMIENTO_ORDEN_STOCK_2026-07-30.md`. Artifact
`results/metric_audit/matching_sweep_v1/`. Roots 2,200,001+, three per cell, eight cells
per family. Defaults unchanged and verified byte-identical.

## Result

**Zero of sixteen cells pass**, in either family. The acceptance was declared before the
run — mass at the minimum below 0.30, `CTj` p50 and p95 within 1.5× of 101.4 and 2,238.6,
`rpj_mean` no worse — and it is the mass criterion that fails everywhere.

| family | cell | mass at min | `CTj` p50 | `CTj` p95 |
|---|---|---:|---:|---:|
| R1r | base `partial0 / blocking / legacy` | 0.645 | 54 | 2,078 |
| R1r | best: `op9_linked` (any pairing) | **0.525** | **54** | 2,226 |
| R2r | base | 0.518 | 54 | 2,809 |
| R2r | best: `op9_linked` (any pairing) | **0.354** | **78** | 2,493 |
| — | **required** | **< 0.30** | 68–152 | 1,492–3,358 |

## What the axes actually do

**`order_fulfillment_mode = op9_linked` is the only lever that matters**, and it is the one
that already existed. It removes the instant path and moves mass at the minimum from 0.645
to 0.525 (R1r) and 0.518 to 0.354 (R2r), and in R2r it is the only setting that moves the
median at all, 54 → 78.

**`partial_fulfilment` helps only in `legacy` mode**, where it moves R2r's median 54 → 72
and mass 0.518 → 0.475. Under `op9_linked` it changes nothing, because there is no partial
path left to take.

**`queue_blocking = skip_head` does essentially nothing**, and in one cell it makes things
worse: R2r `partial0 / legacy` goes from 0.518 to 0.607 mass at the minimum while
shortening the tail. Scanning past an unservable head lets more small orders through
quickly, which *adds* to the spike rather than filling the middle.

The declared falsifier — improving the median while worsening the tail proportionally —
did not fire in any cell.

## The residual, and where it points

**R1r's median stays at exactly 54.0 in all eight cells.** No matching arrangement moves
it. Even with every order queued for the daily freight, more than half still depart the
same day and finish at the minimum.

That says the constraint is not *how* orders are matched to stock but **how much stock
there is**. If the freight almost always has enough on hand to clear the day's order, the
queue never forms and the cycle time is the pipeline minimum whatever the matching rule.
Our chain is simply less congested than his in the middle of the distribution.

So the diagnosis moves upstream again: from the metric, to the delay, to the transport, to
the matching, and now to **production and replenishment volume**. Each step was measured
and each ruled out the one before it.

## What this settles

- The matching axes are implemented, default-preserving, and **measured to be
  insufficient**. They are kept because they are real and because `op9_linked` is a
  meaningful lever, but none is adopted.
- The preregistered acceptance was **not met**, and is reported as not met. No cell is
  promoted, and the criteria are not loosened to admit one — that would be the fitting the
  preregistration exists to prevent.
- `skip_head` is a genuine negative: it moves the distribution the wrong way.

## What I would not do next

Chase this further without first asking whether the target is right. Five mechanisms have
now been examined and each explained less than expected. Before a sixth, it is worth
checking whether our order *volume* per year being 2× his — already the largest single
discrepancy in R2r at 26 SD — is the common cause of both the mass at the minimum and the
short median. That is a measurement, not another change.
