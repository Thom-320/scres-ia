# Preregistro — G2: ¿un umbral duro genera una prima que un clásico no coma?

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_g2_autotomy_threshold.py`.
Segundo generador del programa, y **el último con prior razonable en el carril WRAP**.

## Por qué G2, y qué aprendimos de G1 que cambia la pregunta

G1 salió positivo —Cobb-Douglas tiene máximo estrictamente interior, y la ablación con `c_i = 0`
lo destruye en las seis celdas— **pero la prima no apareció**: en esa superficie el **spline gana
a las dos redes**. La lección es precisa:

> **No basta con que la superficie sea no lineal. La prima exige una no linealidad que los
> métodos clásicos no capturen ya** — y una curvatura suave en una variable no lo es.

Un **umbral duro** es otra cosa. La rama de autotomía de Garrido (`CTj ≤ LT`, peso 1,0, la de
mayor peso de su métrica) está **muerta** en nuestro modelo: `GARRIDO_FULFILLMENT_DELAY_HOURS = 54
> LT = 48`, así que **0 de 416 pedidos** pueden cruzarla. Encenderla mete una **discontinuidad**
en el objetivo, no una curva.

**Y una discontinuidad no se promedia contra el ruido**: un pedido cruza el umbral o no lo cruza.
Ésa es la razón estructural por la que G2 es el mejor candidato que queda.

## Lo que ya está medido y no hay que repetir

El brazo **`FDB`** existe y está caracterizado (`docs/RESULTADO_CIERRE_AUTOTOMIA_2026-07-31.md`):
distribución de retardo con olas + banda `δ = 0,05 h`, **autotomía 0,3122 %** (contra 0,44 % de
Garrido), y **precio de fidelidad medido: 0,95 SE de `ret_mean`**. Está en el **conjunto no
dominado**. No se re-caracteriza: se **usa**.

## La hipótesis

> **G2.** Con la autotomía viva, la superficie `ρ → ReT` contiene una discontinuidad que
> **ningún** comparador clásico —lineal, interacciones, spline, árbol, **regla de umbral
> explícita**— captura, y **alguna red la supera por el SESOI de 0,05 con IC95 (t) > 0**.

## Diseño

* **Brazos**: `constant` (la física embarcada, autotomía muerta) y **`FDB`** (autotomía viva).
* **Rejilla**: buffer × turnos × `op9_rop`, 3 familias de riesgo × 2 escaladas. Semillas vírgenes.
* **Objetivo**: `ret_excel_risk_conditional` — es la métrica **que contiene el umbral**; medirlo
  sobre Cobb-Douglas no probaría nada porque CD no tiene la rama de autotomía.
* **Comparadores clásicos, y uno nuevo obligatorio**: lineal aditivo · interacciones+cuadrático ·
  spline · CART · **regla de umbral explícita** sobre el predictor más asociado al cruce. Si una
  regla de umbral iguala a la red, **no hay prima neural** — es la prueba honesta que la revisión
  externa pidió y la adopto.
* **Baseline primario, declarado AQUÍ y por principio**: `linear_interactions`. El mejor
  post-hoc se reporta al lado, etiquetado. **No repito el defecto de selección sobre test.**
* **Inferencia**: CV agrupada por semilla, intervalos **t**, y comparador de media de celda
  construido **sólo con train** como estimador del margen disponible.

## Los diagnósticos del umbral que la revisión externa exigió, y que van ANTES del veredicto

| diagnóstico | por qué es obligatorio |
|---|---|
| **frecuencia** de eventos `CTj ≤ 48` | una discontinuidad que casi nunca se cruza no puede pagar |
| **balance** entre los dos lados | un 0,3 % contra 99,7 % es un problema desbalanceado, no una no linealidad aprovechable |
| **ruido del etiquetado** | si el cruce es casi aleatorio dado `ρ`, no hay función que aprender |

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_autotomy_is_actually_alive` | **la premisa**: con `FDB` la fracción de autotomía debe ser > 0 y con el brazo embarcado exactamente 0. Si no, no hay umbral encendido |
| `f2_threshold_is_crossed_often_enough` | con una frecuencia ínfima el umbral es ruido, no estructura; el número se reporta pase o falle |
| `f3_a_threshold_rule_is_among_the_comparators` | sin ella, una «prima neural» podría ser sólo la red descubriendo un `if` |
| `f4_primary_baseline_declared_before` | la selección sobre test fue un defecto real de la corrida CD; aquí el primario está fijado en este documento |
| `f5_folds_grouped_by_seed` | una semilla compartida infla todos los `R²` |
| `f6_fidelity_price_is_disclosed` | `FDB` **empeora** `ret_mean` en 0,95 SE; usarlo sin decirlo sería vender física conveniente |
| `f7_seeds_are_virgin` | escaneo real de artefactos sellados |

## Regla de lectura, fijada de antemano

* **Alguna red ≥ SESOI sobre el primario, IC95 (t) > 0, Y por encima de la regla de umbral** →
  `NEURAL_PREMIUM_FROM_DISCONTINUITY`. **Sería la primera prima neural del proyecto.**
* **La regla de umbral iguala o supera a las redes** → `THRESHOLD_RULE_SUFFICES`: la
  discontinuidad es real y **un `if` la captura**. Resultado con contenido y, para el paper, casi
  tan valioso como el positivo.
* **Nadie supera al primario** → `DISCONTINUITY_INSUFFICIENT`. Con G1 ya cerrado, quedarían
  **dos de los cuatro generadores descartados constructivamente**.

**Lo que no autoriza:** entrenar control. G2 es predicción. El control sigue exigiendo el gate de
headroom completo, que sigue cerrado.

**Y una nota de alcance que me impongo:** `H_regime` en este carril es `~1e-4`. Aunque G2 diese
prima **predictiva**, no habría headroom **de control**, y el paper debe mantener esas dos cosas
separadas.
