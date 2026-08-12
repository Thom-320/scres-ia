# Preregistro — R1: la restricción de cola DENTRO del objetivo de selección

**Fecha:** 2026-08-12 · **Escrito ANTES de correr la re-selección.**
**Grado:** desarrollo. Replay declarado del bloque de ajuste 7420001–7420048
(`USED_DEVELOPMENT_NOT_VIRGIN`). **Cero semillas nuevas.**
**No es un rescate de Program O.** O está cerrado, es inmutable y tiene `second_rescue_forbidden`.
Esto **no puede promoverlo**; puede, como mucho, justificar pedirle al PI un experimento nuevo.

---

## 1. El defecto que este experimento apunta, leído del código

`scripts/screen_program_o_hobs_fit.py:186-240` selecciona la política observable así:

```python
admissible = all(deltas[key].mean() >= -1e-12 for key in HIGHER_KEYS) and ...
selected = min(eligible, key=lambda i: (-mean_ret_visible, -ration_ret_visible, -worst_product_fill, ...))
```

Dos cosas, y las dos importan:

1. la admisibilidad se evalúa sobre **medias** de los deltas de guardarraíl;
2. entre las admisibles se **maximiza la media** de `ret_visible`, con los guardarraíles sólo como
   criterios de desempate.

La validación correctiva murió después en `ret_visible_cvar10` bajo **LCB simultáneo**
(−0,008578 / −0,015507), con los estimadores puntuales **positivos**. Es decir: **la selección
nunca comprobó la cantidad que la mató.**

Y el oráculo sí puede satisfacerla: imponer el vector completo de guardarraíles *por tapa* cuesta
**0,81 %** del headroom bruto (0,15275 → 0,15151, `validation_custody_verdict_v1.json`). En el techo
la restricción es casi gratis. A la política nunca se le pidió.

## 2. La pregunta, y por qué es exacta y no aproximada

La clase de políticas declarada es **finita**: 4 `POLICY_IDS` × 4 acciones iniciales = **16
configuraciones**, y sus resultados están íntegramente en las matrices de calendario en disco. Por
tanto esto **no es un entrenamiento**: es una **enumeración exhaustiva** de la clase.

> **¿Contiene la clase de políticas declarada un miembro que satisfaga la restricción de cola con su
> propia inferencia, y que conserve la ventaja media?**

## 3. Las dos reglas de selección, ambas declaradas aquí

**`S_mean`** — la que se envió, reproducida tal cual: admisible por medias, después máximo de
`mean_ret_visible`, desempates en el orden original.

**`S_cvar`** — la nueva, **lexicográfica sobre la restricción que mató a O**: entre las admisibles,
exigir primero `LCB95(Δ ret_visible_cvar10) ≥ 0` por tapas pareadas, y **sólo entre las que pasan**,
maximizar `mean_ret_visible`. No hay λ que ajustar, así que no hay nada que sintonizar después de ver
el resultado.

Se reportan las 16 configuraciones con su par (Δmedia, ΔCVaR10) y sus LCB, para que la elección sea
auditable y no una caja negra.

## 4. Falsadores

* **p1_reproduzco_la_seleccion_enviada** — `S_mean` debe elegir `belief_extreme_v1` con
  `initial_action = 2`, que es lo que el ajuste sellado congeló
  (`.../artifacts/fit/result.json` → `selected_config`). *Puede fallar*, y si falla no estoy
  reproduciendo el pipeline y nada de lo demás vale.
* **p2_la_clase_contiene_una_politica_factible_en_cola** — existe al menos una configuración
  admisible con `LCB95(Δ CVaR10) ≥ 0`. **Puede fallar**, y es el desenlace más informativo: si
  ninguna de las 16 pasa, la cola **no es controlable dentro de la clase declarada**, y la familia
  O/Q cierra **por clase de política**, no por un guardarraíl mal puesto.
* **p3_la_politica_factible_conserva_la_media** — para la elegida por `S_cvar`,
  `LCB95(Δ ret_visible)` > 0. *Puede fallar:* si la ventaja media estaba comprada en la cola
  inferior, aquí se ve, y la conclusión es que **la ventaja era la concentración**.
* **p4_las_dos_reglas_pueden_diferir** — control. Si `S_mean` y `S_cvar` eligen la misma
  configuración en las cuatro celdas, el experimento no mide nada y hay que decirlo.
* **custody** — replay declarado del bloque de ajuste. Cero semillas nuevas.

## 5. Reglas de decisión, escritas antes

| resultado | veredicto |
|---|---|
| p1 falla | `BLOCKED_CANNOT_REPRODUCE_THE_SHIPPED_SELECTION` |
| p2 falla | `THE_POLICY_CLASS_CONTAINS_NO_TAIL_FEASIBLE_MEMBER` |
| p2 pasa, p3 falla | `TAIL_FEASIBLE_BUT_THE_MEAN_ADVANTAGE_WAS_THE_TAIL` |
| p2 y p3 pasan | `A_TAIL_FEASIBLE_POLICY_EXISTS_IN_THE_DECLARED_CLASS` |

**El mejor de esos cuatro desenlaces NO es un resultado confirmatorio.** Es grado desarrollo sobre
el bloque de ajuste, y su única consecuencia legítima es justificar la petición al PI de un bloque
virgen con contrato nuevo. Cualquier evaluación sobre el bloque de validación 7430001–7430048 sería
un **segundo rescate**, que está prohibido.

## 6. Lo que este preregistro NO permite

Cambiar la clase de políticas, añadir configuraciones, introducir un λ ajustable, tocar la definición
de admisibilidad, ni evaluar sobre tapas de validación. Tampoco presentar un `A_TAIL_FEASIBLE_POLICY`
como conversión observable segura: eso exige semillas vírgenes que hoy el registro prohíbe.
