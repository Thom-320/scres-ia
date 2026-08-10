# Preregistro — confirmación de la Puerta B en tapas frescas

**Fecha:** 2026-08-09. **Congelado antes de abrir una sola semilla.**
Contrato marco: `docs/CONTRATO_PROGRAMA_N_PRIMA_NEURAL_2026-08-09.md`.
Desarrollo que se confirma: `results/program_n/gate_b_cd_surface/result.json` =
`SURFACE_PREMIUM_CAPTURED`.
**Rol:** `PROSPECTIVE_CONFIRMATION_ON_A_VIRGIN_BLOCK`.

## 1. La hipótesis, escrita antes y sin escapatoria

> **Un KAN correctamente ajustado supera al mejor baseline clásico declarado de antemano
> (`linear_interactions`) en R² fuera de fold sobre `R_cobb_douglas`, con
> `media ≥ 0,05` y `IC95 excluyendo cero`.**

En desarrollo dio **+0,1140 [+0,0614, +0,1665]**. Aquí se repite el procedimiento **entero** sobre
tapas que nunca se han visto.

## 2. Qué se congela, que es todo menos las semillas

**Idéntico al desarrollo, byte a byte en el código:** mismo runner
(`scripts/run_program_n_gate_b_v1.py`), mismo objetivo, misma calibración por fold, mismos ocho
brazos, misma rejilla de ocho puntos, mismas cinco semillas de inicialización, mismo `SESOI = 0,05`,
mismo baseline primario, mismos 5 folds agrupados por semilla, mismo horizonte de 52 semanas.

**Cambian sólo los valores de las semillas:** `9400001–9400008`, ocho como en desarrollo, para que
cualquier diferencia sea **la tapa** y no el diseño.

**Se re-selecciona el hiperparámetro por fold sobre la validación interna de las tapas nuevas.** Es
deliberado: la afirmación es sobre **el procedimiento** —«un KAN correctamente ajustado»— no sobre
un modelo concreto. Congelar los hiperparámetros de desarrollo probaría algo más débil y distinto.

## 3. Puerta de un solo sentido

El bloque `9400001–9400008` es virgen y queda **consumido por esta corrida**. **Si el instrumento
resulta defectuoso al correr, el bloque queda quemado igualmente**: no hay reejecución, no hay
reajuste, no hay «lo corrijo y lo vuelvo a correr sobre las mismas semillas».

Por eso el runner ya está probado: corrió en pre-vuelo y en desarrollo sin tocar este bloque.

## 4. La regla de lectura, en orden

1. **Primero la reproducción.** `f2` exige que los brazos clásicos —código intacto— caigan dentro de
   `0,02` de sus valores de desarrollo. Si falla, las tapas nuevas producen una superficie distinta,
   la comparación no es la misma y el veredicto es `BLOCKED_INSTRUMENT`. **Nada más se lee.**

   Nota declarada: en desarrollo la reproducción fue a `4,9e-05` porque eran **las mismas** tapas.
   Aquí son otras, así que se espera una desviación real; `0,02` es la tolerancia que ya estaba
   escrita en el runner y **no se relaja** ahora.

2. Con `f2` en verde, decide `f5`: `LCB95(KAN − linear_interactions) > 0` y `media ≥ 0,05`.
   * pasa → **`SURFACE_PREMIUM_CONFIRMED_ON_FRESH_TAPES`**;
   * no pasa → **`SURFACE_PREMIUM_DID_NOT_CONFIRM`**, y el resultado de desarrollo queda
     **superado por replicación fallida**, exactamente como el techo clarividente del 2026-08-08.

3. `f4` sin `f5` → `NETWORKS_REACH_THE_LINEAR_BUT_DO_NOT_BEAT_IT_ON_FRESH_TAPES`.

**No hay rama que diga «casi».** El SESOI no se mueve después de ver el intervalo.

## 5. Lo que esta confirmación NO convierte en autorización

Sigue siendo **predicción, no control**. El contrato de E\* lo dice sin ambigüedad:
`r2_is_not_a_control_gate: true`. Un R² confirmado **no autoriza ningún aprendiz de control**, y no
se citará como si lo hiciera.

Lo que sí hace: dar grado confirmatorio a la única prima neural del proyecto, y entregar a la
Puerta C el aproximador de valor que un planner amortizado necesita.

## 6. Lo que queda pendiente aunque esto confirme

Una sola superficie. Si el KAN gana también sobre `ret_excel` —la métrica legada, reportada nunca
como objetivo de entrenamiento— la afirmación pasa a ser sobre **el método**. Con una sola
superficie es sobre **esta superficie**, y así se escribirá.
