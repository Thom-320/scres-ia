# Why risk always implies lateness in our DES — both hypotheses tested

**Status:** `DEVELOPMENT_DIAGNOSTIC_NO_CONSTANT_CHANGED`. Follows
`FIDELITY_DELAY_SWEEP_2026-07-30.md`, which found that no delay in the calibration grid
produces a single autotomy case because on-time orders and risk-touched orders are
disjoint sets. Two hypotheses were named there. Both are now tested.

## H1 — "our risks have no sub-lead-time impact scale". **REFUTED.**

Measured over six episodes at delay 48, both families, 426 orders with a disruption:

| | ours | Garrido |
|---|---:|---:|
| min disruption (h) | **1.00** | 0.003 |
| p05 | 12.40 | 46.31 |
| median | 248.2 | 118.8 |
| share under 48 h | **10.3%** (44/426) | — |

We have plenty of short disruptions — one in ten is under the lead time, and the shortest
is one hour. Our disruptions are *longer on average* than his (median 248 h against 119 h),
but the claim that we lack a short-impact scale is false. A one-hour disruption still makes
an order late.

## H2 — "the pipeline carries no slack that could absorb a short disruption". **CONFIRMED, and the mechanism is quantisation.**

| | distinct `CTj` values | first values | step |
|---|---:|---|---|
| ours | **143** in 1,289 orders | 48, 72, 96, 120, 144, 168, 192, 216 | **exactly 24 h** |
| Garrido | **36,326** in 43,360 rows | 48.00744, 48.00744, … | essentially continuous |

**Our delivery is quantised to the daily dispatch cadence.** An order either makes its
truck (`CTj = 48`) or misses it and waits a full 24 hours (`CTj = 72`). There is no
intermediate outcome, so a one-hour disruption and a twenty-hour disruption incur the
*same* 24-hour penalty.

That is the partition, exactly. With `LT = 48` and 24-hour steps, an order is either at
48 with no disruption or at ≥72 — a full day late. `CTj = 48.007` cannot occur, so
"disrupted **and** on schedule" is not a reachable state.

## What Garrido's chain does instead

His autotomy rows are the direct measurement, and they are more extreme than either
hypothesis anticipated:

    absorbed disruption (APj):  0.45 h to 48.04 h,  median 3.4 h
    overshoot of those orders:  0.00744 h to 0.048 h  (max 2.9 minutes)

**He absorbs up to 48 hours of disruption with under three minutes of lateness.** That is
not last-mile slack in any ordinary sense — it is the order continuing to flow while an
operation is down, which is precisely the tail-autotomy picture of his Figure 5.2: the
chain sheds the affected part and the remainder keeps moving. Our model blocks instead.

## Consequence

The gap is **neither the fulfilment delay nor the risk scale**. It is that our delivery is
discrete where his is continuous, and that a disrupted operation blocks the order in our
model where it does not in his.

No constant can close this, which is why the delay sweep returned a clean negative. Closing
it would mean changing the dispatch representation — sub-daily granularity, or a path that
completes around a downed operation — and that is new physics, not calibration. It is
therefore out of scope for any reproduction claim and must be preregistered separately if
it is attempted at all.

**And it retires my own H1.** I proposed the risk-scale explanation first and it is wrong;
the data had short disruptions all along.
