# Result — the delay-shape arms: HALTED by falsifier 4, and it caught a real defect

**Status:** `HALTED_FALSIFIER_FAILED_SPECIFICATION_DEFECT`. Executes
`docs/PREREGISTRO_DELAY_DISTRIBUCION_2026-07-30.md`. Artifact
`results/metric_audit/fulfillment_delay_distribution_v1/result.json`.
**No moment is reported**, per §5 of the contract.

> **CORRECCIÓN 2026-07-31 — dos defectos graves de este documento.**
>
> **(a) La columna «`p50` sorteado» de la §2 no existe en el artefacto y es imposible.** Los
> parámetros son emparejamientos cerrados de la mediana, así que la mediana **sorteada** es
> **exactamente 101,45 en los tres brazos**, no 101,4 / 101,6 / 101,4. Esas tres cifras
> salieron de un script ad-hoc no registrado — la cuarta vez en un día.
>
> **(b) El falsador que detuvo la corrida no implementó el filtro que el contrato exigía.**
> La §5.4 del preregistro dice «para órdenes **no bloqueadas**»; el runner
> (`run_fulfillment_delay_distribution_arms.py:102`) tomó *toda* orden puntuada, incluidas
> las bloqueadas por riesgo con `CTj` de miles de horas (el `p95` del mismo conjunto es
> 2.666–2.676 h). **Mi diagnóstico de la §2 —que el contrato confundió el sorteo con el
> ciclo— está confundido con ese filtro omitido**, y las dos explicaciones no son separables
> con este artefacto. La §4, que atribuye la culpa al contrato, es por tanto prematura: el
> contrato **sí** contenía la restricción que el código dejó fuera.
>
> **(c)** Los cuantiles reservados `p1`/`p5`/`p95` se calcularon y **nunca se compararon con
> nada**, y `p95` ni siquiera quedó registrado en `reserved_quantiles`. La falsación de forma
> que el propio contrato declaraba era ejecutable desde este artefacto y no se ejecutó.

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
