# Result — la siembra de `R⁰` por `R14` no es el mecanismo: los tres brazos coinciden

**Status:** `PREREGISTERED_SHAPE_PREDICTION_REFUTED`. Artefacto
`results/metric_audit/r14_seed_arms_v1/result.json`, sellado. Raíces 3.400.001–12.

## 1. La predicción fuerte falla, y falla de forma total

| brazo | `RPj/CTj` p50 | p95 | **p50 con `CTj > 500`** | n largas |
|---|---:|---:|---:|---:|
| **Garrido** | — | **≈ 0,20** | **satura ~400 h** | — |
| A `pending_min` | 1,000 | 1,000 | **1,000** | 1.030 |
| N sin siembra | 1,000 | 1,000 | **1,000** | 1.030 |
| E instante real | 1,000 | 1,000 | **1,000** | 1.030 |

Criterio declarado: mediana `RPj/CTj` **< 0,60** para `CTj > 500`. Los tres dan **1,000
exacto**. **No hay saturación en ninguno.**

## 2. Y los momentos son indistinguibles

| momento (R1r) | A | N | E |
|---|---:|---:|---:|
| `rpj_mean` | **410,18** | **410,19** | **410,17** |
| `rpj_p95` | 2.362,20 | 2.362,20 | 2.362,20 |
| `ret_mean` `d_k` | 1,25 | 1,25 | 1,26 |

`f4` pasa técnicamente —los tres valores difieren— pero la diferencia es **0,02 sobre 410**,
la quinta cifra significativa. **El eje es real y su efecto es nulo.**

## 3. Qué refuta esto

**La compuerta `R14` NO es lo que hace `RPj = CTj`.** Quitarla del todo (`N`) no mueve nada.
Mi diagnóstico de `8482047` —que `R⁰` cayera en `OPTj` *por la compuerta `R14`*— está
**refutado**: la distancia p50 = 0,00 h que medí era cierta, pero `R14` no es quien la causa.

La causa tiene que ser **la densidad de riesgo en general**: con `R11` tocando el 75,4% de las
órdenes y 66,7% colocadas con un riesgo ya activo, **siempre hay algún origen en o junto a
`OPTj`**, venga de donde venga. Por eso `RPj/CTj = 1,000` incluso en el p95 y en las órdenes
largas.

## 4. Y devuelve la respuesta a algo ya medido

Si el problema es que **cualquier** riesgo denso siembra `R⁰` al inicio, entonces lo que hace
falta no es cambiar *qué riesgo* siembra, sino **restringir cuáles son admisibles**. Eso es
exactamente `causal_exposure`, y ya está medido (`6192460`):

| | `rpj_p95` nivel | `d_k` | `d_k` SE apareada |
|---|---:|---:|---:|
| `des_events` | 2.533 | 19,95 | 19,95 |
| **`causal_exposure`** | **672** | 39,15 | **2,07** |

**3,8× de mejora en nivel**, y residuo real 2,07 una vez descontado el efecto de denominador.
Este contrato apuntaba al término equivocado; el correcto ya lo teníamos medido.

## 5. Lo que sí queda de este contrato

* **`N` no poda de más — no poda nada.** Predije que `N` sería peor que `E` por exceso de
  poda; la realidad es que ninguno de los dos hace nada. La predicción era del signo correcto
  sobre un eje sin efecto.
* **`f2` y `f3` pasan**: ningún `RPj > CTj`, ninguna orden bajo `LT`, y en `E` todo origen de
  `R14` coincide con un evento real de `risk_events`. La implementación es correcta; lo que
  falla es la hipótesis.
* **`f5` pasa** — primera vez en cuatro corridas que el conjunto es `epsilon`-estable. Y es
  estable porque **los tres brazos son el mismo punto**.

## 6. Estado

Nada implementado, ningún default movido. `r14_r0_seed_mode` queda como opción medida y sin
efecto; conservarla o retirarla es decisión de mantenimiento, no de modelado.

**El siguiente paso ya no es una hipótesis nueva:** es cerrar la brecha de `f3` en
`causal_exposure` —lo único que ha movido `RPj` en toda la sesión— y entender la
inestabilidad de `epsilon` en los cruces que sí discriminan.
