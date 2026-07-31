# Recurrence and overlap: refuted. And the physics was never the problem.

**Status:** `DEVELOPMENT_HYPOTHESIS_REFUTED_NEW_MECHANISM_IDENTIFIED`. Nothing changed.
Roots 2,300,001–4, escalated R1r, one shift, no buffers.

## 1. The hypothesis, tested directly

I proposed the R1r tail came from event recurrence and overlap rather than per-event
length. Arm P (parallel reading) is a clean instrument for this: it removes the serial
multiplier, so it collapses both coverage and concurrency.

| brazo | cobertura Op1 | cobertura Op2 | máx. R13 simultáneos | `rpj_p95` |
|---|---:|---:|---:|---:|
| S serial | 12.0% | **64.7%** | 1.8 | 2450 |
| P paralelo | 3.8% | **13.9%** | **1.0** | 2206 |

P cuts Op2 outage coverage by **4.6×** and removes overlap **entirely** (max concurrency
1.0 — no two R13 events ever active at once). The tail falls **10%**.

**The hypothesis is refuted.** Removing essentially all recurrence-driven overlap buys a
tenth of a 5.4× gap.

That now exhausts the list: metric, RPj mode, scored population, horizon, per-event
duration, the serial multiplier, outage coverage, and overlap have each been measured and
ruled out.

## 2. The measurement I should have made first

I never compared his `CTj` to ours. Doing it settles everything:

| | p50 | p95 | max |
|---|---:|---:|---:|
| **Garrido `CTj`** (9 hojas) | 101 | **2,239** | 52,137 |
| **nuestro `CTj`** | 54 | **2,450** | — |
| **Garrido `RPj`** | 99 | **456** | 1,156 |
| **nuestro `RPj`** | 54 | **2,450** | — |

**Our cycle times match his to 9% at the p95.** The physics was never wrong. His `RPj` is
**0.19–0.21 of his `CTj`** at the p95, consistently across all nine sheets; ours is exactly
1.00. The entire gap is `RPj` attribution, and every model-side hypothesis I have chased
since this morning was aimed at the wrong layer.

Note the signature: his `RPj` p50 (99) **equals** his `CTj` p50 (101), while his p95 is a
fifth of it. `RPj` tracks `CTj` for short orders and detaches for long ones.

## 3. The mechanism, and the thesis says it plainly

Algorithm 2, p.69:

> 2: **IF the impact of at least one Rcr ∈ Ω manifests within the interval [OPTj, OATj]**
> AND CTj > LTj,
> 3: THEN, RPj = (OATj – first-R⁰cr)

The onset must **manifest within** `[OPTj, OATj]`. `first-R⁰cr` is the first such onset
*inside the order's own window*.

`supply_chain.py:5933`:

```python
eff_risk_start = max(earliest_risk_start, order.OPTj)
order.RPj = max(0.0, order.OATj - eff_risk_start)
```

`earliest_risk_start` includes risks **already running when the order was placed**, and the
clamp then rewrites their onset to `OPTj`. That is precisely the case Algorithm 2 excludes:
a risk that began before `OPTj` did not manifest within the interval, so it must not seed
`R⁰`. The clamp converts every such order into `RPj = OATj − OPTj = CTj`.

And **66.7% of our risk-touched orders are placed with a risk already ongoing** (measured,
`884c035`). Those are disproportionately the long ones, because long orders are exactly the
ones sitting in the queue through saturated periods. So the clamp takes the orders whose
`CTj` is largest and assigns them the largest possible `RPj`.

That reproduces the signature in §2 from both ends: short orders have their causing risk
onset inside the window, so `RPj ≈ CTj` — matching his p50. Long orders under the thesis
rule either take a later in-window onset or fail the condition entirely — matching his
detached p95.

## 4. Status, and what I am not doing

This is a **hypothesis with textual backing and a matching signature**, not a result. It is
not implemented and must not be: removing the clamp changes every ReT figure in the
project, so it needs its own preregistration with the six-moment acceptance rule and a
declared prediction on the same scale as that rule — the defect I made in the last one.

Two things to carry in:

* the prediction is **two-sided and specific**: `rpj_p95` should fall toward ~456 while
  `rpj_mean` and the p50 stay near 99, and `autotomy_share` should become **nonzero**,
  since `RPj = CTj > LTj` is exactly what currently makes the `CTj <= LTj` autotomy branch
  unreachable. That last one is a free, independent test — a mechanism that fixes the tail
  without lighting up autotomy is the wrong mechanism;
* `ret_mean` sits at **1.7 SD (R1r) / 2.1 (R2r)** and outranks all of it.

## 5. Corrections carried

* «missing recovery duration» (884c035): wrong, the thesis fixes 168 h and 24 h and the
  code has both.
* «249.8 SD»: wrong by 17×, the correct figure is 14.6; my ad-hoc `d_k` dropped our own
  standard error.
* «recurrence and overlap»: refuted here, by the arm-P instrument.

Three wrong mechanisms in one day, all pointed at the model. The physics matched his all
along, and the one measurement that would have shown it — his `CTj` against ours — cost
one script.
