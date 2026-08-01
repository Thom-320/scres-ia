# Preregistro — Fase 2: medir el headroom con **Cobb-Douglas**, no con ReT

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_cobb_douglas_headroom_v1.py`.

## Por qué esta medición existe

Toda la campaña de headroom —veinte y pico de experimentos, `H_regime ≈ 1e-4` en superficie,
buffers, nodos nuevos, mezcla de regímenes, contención aguas abajo, observables, contención en el
cuello— se ha medido sobre **ReT**.

Y hoy quedó medido que **ReT premia abandonar a un reclamante**: elige el reparto que entrega el
50 % de las raciones sobre el que entrega el 80 %, y en el cuello elige el calendario que deja un
producto al **18,5 %** de servicio. La preferencia sobrevive a quitar la censura
(`full_ledger`) y a acotar la cola (`clipped`), así que está **en el constructo**.

**Cobb-Douglas no tiene esa preferencia**: en la misma barrida su óptimo es el reparto
**equilibrado** en los dos regímenes (`results/metric_audit/abandonment_v1/result.json`).

> **Entonces `H_regime ≈ 1e-4` en veinte experimentos puede ser una propiedad de la MÉTRICA y no
> de la cadena.** Esta corrida lo decide.

## La hipótesis

> **H.** Con `R_cobb_douglas` como métrica primaria, `H_regime` sobre el mismo barrido de reparto
> **difiere materialmente** del `H_regime` medido con ReT.

Dos salidas, ambas informativas y ambas comprometidas de antemano:

* **`H_regime` bajo Cobb-Douglas ≥ 0,01 con `LCB95 > 0`** → **hay headroom real**, y la campaña
  anterior estaba mirando por el instrumento equivocado. Reabre la escalera de políticas sobre
  una métrica sana.
* **`H_regime` bajo Cobb-Douglas sigue siendo ≈ 1e-4** → el «cuándo NO entrenar» queda
  establecido sobre una métrica que **no** premia el abandono, y deja de ser vulnerable a la
  objeción de que se midió con una métrica rota. **Es una conclusión mucho más fuerte que la de
  esta mañana.**

## Diseño

* **Palanca**: `cssu_allocation_a`, nueve niveles en `[0,1 … 0,9]`, no fungible.
* **Regímenes (6)**: {`R2r`, `R1r+R2r`} × {base, `R23` ×3 frecuencia, ×3 frecuencia + ×2 impacto}
  — **los mismos** de la Fase 1A, para que la comparación sea del instrumento y no del diseño.
* **Semillas**: `5 600 001…` vírgenes, CRN entre celdas.
* **Métricas**: `R_cobb_douglas` **primaria**; `ret_excel_risk_conditional` y `flow_fill_rate` al
  lado, **en la misma corrida y con la misma cadencia**, que es la única forma de comparar
  instrumentos sin confundirlos con el diseño.
* **Conjunto de comparación de `κ̇`**: **todas** las celdas (régimen × reparto) a la vez. `κ̇` es
  set-relativo por definición de Garrido, así que el conjunto tiene que ser exactamente el que se
  compara.
* **Cadencia**: paso diario (lo exige el `CobbDouglasRecorder`). ReT es dependiente de la
  cadencia, así que sus valores aquí **no** son comparables contra los artefactos con `run()` —
  sólo **entre celdas de esta corrida**, que es lo que `H_regime` necesita.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_cd_and_ret_disagree_on_the_argmax` | si ambas eligen el mismo reparto, no hay nada que decidir y la premisa de la corrida cae |
| `f2_kappa_dot_set_is_every_cell` | `κ̇` sobre un subconjunto respondería otra pregunta |
| `f3_exponents_are_re_derived_not_copied` | copiar sus coeficientes importaría su escala, no su método |
| `f4_H_regime_is_non_negative` | `mean[max] ≥ max[mean]`; un negativo sería bug de agregación |
| `f5_cd_actually_varies` | si `R` es plana entre repartos, `H_regime` mide ruido — y su rango medido es estrecho (1,1 %), así que **este falsador es el que más riesgo tiene de fallar** |
| `f6_same_regimes_as_fase_1a` | comparar contra otro diseño confundiría instrumento con diseño |
| `f7_seeds_are_virgin` | reutilizar semillas invalidaría la confirmación |

**`f5` merece una nota**: el rango medido de `R_cobb_douglas` entre repartos fue **0,5467–0,5532**,
un **1,1 %** relativo. Ordena bien pero discrimina poco. Si `f5` falla, la conclusión no es «no
hay headroom» sino **«Cobb-Douglas no tiene resolución suficiente para esta pregunta»**, que es
un resultado distinto y hay que decirlo así.

## Regla de lectura

Se reporta `H_regime` bajo las **tres** métricas, la misma corrida, la misma cadencia, lado a
lado. **No elijo la métrica después de ver cuál da mejor número** — el criterio es el de arriba,
y `R_cobb_douglas` es la primaria por estar ya escrito aquí.
