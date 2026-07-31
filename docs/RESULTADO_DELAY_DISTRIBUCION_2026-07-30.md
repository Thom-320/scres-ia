# Result — the delay-shape arms: HALTED by falsifier 4, and it caught a real defect

**Status:** `HALTED_FALSIFIER_FAILED_SPECIFICATION_DEFECT`. Executes
`docs/PREREGISTRO_DELAY_DISTRIBUCION_2026-07-30.md`. Artifact
`results/metric_audit/fulfillment_delay_distribution_v1/result.json`.
**No moment is reported**, per §5 of the contract.

## 1. What passed and what failed

| falsador | resultado |
|---|---|
| 1 — arm A reproduces the frozen block | **PASA**, exacto en los tres momentos |
| 2 — support ≥ 48.0074, no order below `LT` | **PASA**, 0 órdenes bajo `LT` |
| 3 — `CTj` deja de ser masa puntual | **PASA**, y con holgura |
| 4 — `p50(CTj)` realizado dentro del ±10% de 101,45 | **FALLA** |

Falsifier 3 is worth stating because it is the mechanism working exactly as designed:

| brazo | valores distintos de `CTj` | cuota modal |
|---|---:|---:|
| A constante | 231 | **60,5%** |
| D1 / D2 / D3 | ~2.567 | **1,3–1,5%** |

**The point mass is gone.** That part of the diagnosis is confirmed.

## 2. Why falsifier 4 failed, and it is my specification, not the shapes

| brazo | `p50` sorteado | `p50` realizado | objetivo |
|---|---:|---:|---:|
| D1 exponencial | 101,4 | **141,4** | 101,45 |
| D2 lognormal | 101,6 | **141,9** | 101,45 |
| D3 Weibull | 101,4 | **130,5** | 101,45 |

The delay is a **minimum**, not the cycle time: `remaining_delay = max(0, transit −
elapsed)`, and an order's realised `CTj` is the draw **plus whatever queueing it meets**.

I estimated the **draw** distribution against Garrido's **realised** `CTj` quantiles. Those
are not the same object. Queueing adds ~40 h at the median, so every shape overshoots by
29–40%.

Worse, the bias is not a fixed shift. Under arm A the drawn value is 54 and the realised
p50 is exactly 54 — 60.5% of orders finish at the minimum, so queueing adds nothing at the
median. With a longer mean delay, orders overlap more and the queue builds. **The
queueing term is endogenous in the delay**, so no closed-form correction to the parameters
fixes it.

## 3. What I am NOT doing

§3 of the contract forbids tuning `β`, `σ` or `k` outside the declared formulas, and §6
forbids selecting by result. **Re-parameterising and re-running would violate both**, and
it is exactly the behaviour the falsifier exists to prevent. The parameters are not the
defect; what they were matched *to* is.

I am also not proposing the fix inline. A successor must declare, before running, how the
draw is targeted so the **realised** distribution matches — which is a harder problem than
this contract assumed, because the map from draw to realised is endogenous.

## 4. The honest assessment of this preregistration

It had a **specification defect**: it conflated the fulfilment-delay draw with the observed
cycle time. That defect was invisible to me when writing it and was caught by a falsifier I
wrote for a different purpose — falsifier 4 was included to verify the parameterisation was
*implemented* as declared, and it caught that the parameterisation was *conceived* wrong.

This is the second contract-level defect in two days. The previous one was declaring a
prediction on a different scale from the acceptance rule (`8a6aa16`). Both were caught by
the instrument rather than by review, which is the system working, but the pattern is that
**my contracts are weaker than my measurements.**

## 5. What survives, and it is not nothing

* the point-mass diagnosis of `RESULTADO_AUTOTOMIA_2026-07-30.md` is **confirmed** —
  distributions eliminate the 60.5% modal spike;
* the machinery is built, falsified against a frozen block, and reusable;
* arm A's reproduction is exact, so the regression gate works across three consecutive
  contracts now.

**No constant changed, no default moved, nothing relabelled.** `ret_mean` under the shipped
defaults is unaffected.
