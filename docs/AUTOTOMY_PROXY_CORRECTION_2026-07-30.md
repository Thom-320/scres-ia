# The autotomy finding was measured with the wrong proxy

**Status:** `CORRECTION`. Supersedes the central claims of
`PARTITION_HYPOTHESES_2026-07-30.md` and `FIDELITY_DELAY_SWEEP_2026-07-30.md`.

## The error

I used `RPj > 0` as the proxy for "an order was touched by a risk", and concluded that
on-time orders and risk-touched orders are disjoint sets, therefore autotomy is
unreachable at any delay.

**A risk-touched order that arrives on schedule has `RPj = 0` and `APj > 0`, by
construction of the classification.** The recovery period is only populated on the
recovery branch. So the intersection I measured — on time **and** `RPj > 0` — is empty by
definition, in any model, including Garrido's. I measured the absence of a set that cannot
exist and reported it as a physical finding.

The correct proxy is `APj > 0`, or the presence of `ret_risk_indicators`.

## What is actually true

Re-measured over three roots per cell, autotomy share = `count(APj > 0) / scored rows`:

| family | delay 42 | 47 | 48 | 49 | 54 | 60 | reference |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1r | 0.5006 | 0.5006 | 0.4994 | **0.000** | 0.000 | 0.000 | 0.00436 |
| R2r | 0.1480 | 0.1504 | 0.1504 | **0.000** | 0.000 | 0.000 | 0.00064 |

**The branch fires, and the problem is the opposite of what I reported: there is far too
much autotomy, not none.** At or below the lead time it fires for half the orders in R1r —
**115× the reference** — and for 15% in R2r, **236×**. At 49 h and above it stops entirely.

So the shape is a step at the lead time, not a structural impossibility. Both sides miss
the reference by two orders of magnitude, in opposite directions.

## What this changes

**Retracted:** "risk always implies lateness"; "on-time and risk-touched are disjoint";
"not one of the 24 cells produces a single autotomy case"; "no constant can close this".
All four rest on the bad proxy.

**Retracted with them:** the framing that this is a structural gap requiring new physics.
It is a *calibration* problem after all — the branch is reachable, and the delay controls
it sharply. That vindicates the original instinct to sweep the delay, and it means the
question is which delay gets the share near 0.44%, not whether any can.

**Survives, because it was measured independently of the proxy:**

- the delivery quantisation on a 24 h grid, and that `retry_when_ready` breaks it in R2r;
- the two delivery paths, on-hand at exactly 48.0 and pipeline on the grid;
- that R1r risks never touch transport, so the transport mode is a no-op there;
- the canonical reference itself, and every moment in it;
- the `RPj` cadence fix, the provenance stamping, and the contract consolidation.

**Newly open:** the grid straddles the reference between 48 and 49 with nothing in
between. A finer sweep in that interval is the obvious next measurement, and it is cheap.
The step is sharp enough that the reference share may sit at a delay very close to the
lead time — which would also explain why Garrido's `CTj` clusters at 48.007–48.048 rather
than at 48.0 or 54.

## Why this happened, and the cheap guard

The proxy was never verified against the canonical data, where it would have failed
immediately: Garrido's 110 autotomy rows all have `RPj = 0` and `APj > 0`. One check
against the reference would have caught it before it propagated into two documents and a
retracted framing.

The guard now exists as a test: any statistic claiming to count risk-touched orders must
reproduce the canonical autotomy count on Garrido's own rows.
