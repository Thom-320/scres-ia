# The autotomy gap is fixable, and I mis-framed it

**Correction.** `PARTITION_HYPOTHESES_2026-07-30.md` called this "new physics, out of
scope for any reproduction claim". That was wrong. It is an implementation choice made by
an earlier model, and it is changeable. This document says exactly what to change.

## The mechanism, fully isolated

Two delivery paths, and the partition falls straight out of them:

| path | orders | `CTj` | risk-touched |
|---|---:|---|---:|
| on-hand | 143 | **exactly 48.0**, zero variance | 0 |
| pipeline | 73 | 72, 96, 120, … (24 h grid) | **73 — all of them** |

A risk depletes stock, the order can no longer be served on-hand, it falls to the
pipeline, and the pipeline is quantised to 24 h. So **risk ⟹ pipeline ⟹ a full day late,
always.** "On time" only ever means "served from stock", and it is produced by a flat
constant rather than by a modelled transit.

## The three defects, in priority order

**1. `demand_on_hand_fulfillment_delay` is a flat constant standing in for transit.**
It returns exactly 48.0 with zero variance. Garrido's on-schedule orders are at 48.00744
to 48.048 — a base plus small continuous processing. A constant cannot produce that, so
the on-hand path can never absorb a disruption: any added hour moves the order to a
different path entirely rather than to 48.1. Replacing it with the modelled transit
(Op10 + Op11 + Op12 service times) is the single change that makes autotomy reachable.

**2. A wrong inference is load-bearing in the dispatch design.**
`_op9_daily_freight_dispatch`'s docstring reads: *"strictly positive departure waits (no
CTj = 48.0 exactly, hence zero on-time orders in the workbooks)"*. The premise is right —
his data has no exact 48.0. The conclusion is false: 110 rows sit at 48.00744–48.048 and
**are** classified as autotomy. The design then targets departure waits "up to 6 h", which
is where 54 comes from, against his actual waits of 0.007–0.048 h — **27 seconds to 3
minutes**. The daily freight was built to reproduce a congestion moment and broke the
absorption moment.

**3. The daily freight is implemented as a gate, not a rate.**
The thesis says Op9–Op12 ship "at a daily freight rate (ROP = 24 h)" while demand is
2,400–2,600 rations per day — one day of demand per shipment. That is a *rate*: one order
arrives, one shipment leaves. Ours makes orders queue behind a slot, so the wait is up to
a full day rather than the processing time. Note `op9_dispatch_policy` already offers
`ready_headway` and it changes nothing here, because the quantisation is in the transport
legs, not in Op9's release.

## Does this block us? No.

Every comparison in the project is between arms sharing the same quantisation, so all of
it stands: Program Q, the H2/H3 confirmation, the buffer gate, the 90-configuration
reproduction, the v2 comparator, the prospective ReT confirmation, the 648-posture
frontier. None is invalidated by a defect that applies identically to every arm.

What it does block is narrower and should be stated plainly:

- any claim of **distributional fidelity on the autotomy moment**;
- any use of ReT's `APj/LT` branch, which is dead in our model;
- the delay sweep, which returned a clean negative because no constant can reach the
  moment.

## When can we continue? Now.

The fix is an improvement to pursue, not a prerequisite. It should be preregistered like
everything else — it changes every ReT figure — and the tooling already exists: the
fidelity reference, the six moments and the dominance comparison were built for exactly
this. The next sweep is over the **dispatch representation**, not the delay.

One caution that is about proportion rather than reluctance. The daily-freight congestion
was introduced to reproduce his queue and tail moments (`ΣBt ≈ 60`, the ~2,300 h CT p95).
Removing it will move those. The dominance comparison exists precisely so that trade is
measured across all six moments at once instead of being fitted on one, which is the error
that produced 54 in the first place.
