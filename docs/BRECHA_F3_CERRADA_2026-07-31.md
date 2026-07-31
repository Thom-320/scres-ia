# La brecha de `f3` cerrada: 561 → 0, y cuesta el doble de error en `ret_mean`

**Status:** `DEFECT_CLOSED_COST_MEASURED_NOT_ADOPTED`. Nada adoptado, default intacto.

## 1. El defecto, medido y no adivinado

Bajo `causal_exposure`, **182 órdenes por semilla** llevaban **solo `R14`** con su ref en
**912,0** contra un `OPTj` de **1.080,0** — fuera de su ventana. El clamp empujaba `R⁰` a
`OPTj` y `RPj` salía igual a `CTj` desde un riesgo que **nunca se manifestó en el intervalo
de la orden**, que es exactamente lo que excluye la línea 2 del Algoritmo 2.

Ejemplo real: `OPTj = 1.080,0`, `OATj = 1.224,0`, `CTj = 144,0`, `RPj = 144,0`, y el origen
implícito `OATj − RPj = 1.080,0` = `OPTj` exacto.

## 2. Tres intentos fallidos antes del bueno, y por qué

| intento | qué hice | resultado |
|---|---|---|
| 1 | `quantity_risk_onset_scope = "order_window"` sobre `ref_start` | **0 cambios**, y `ret_mean` 0,38 → 1,79. Revertido |
| 2 | cerrar además el respaldo a `OPTj` | **0 cambios** |
| 3 | `causal_quantity_gate = "in_window"` como compuerta de entrada | **0 cambios** |

El intento 3 falló por una razón que solo se ve midiendo: **la compuerta y la selección se
contradecían**. La compuerta preguntaba si existía *algún* ref en ventana —y pasaba— mientras
`ref_start` seguía tomando el **mínimo sobre todos**, que era el de 912,0. Dos líneas mías en
desacuerdo.

**El arreglo son cinco líneas:** cuando la compuerta está activa, `ref_start` toma el mínimo
**sobre los refs en ventana**, no sobre todos.

## 3. El resultado, y su precio

| brazo | `rpj_mean` | `d_k` | `rpj_p95` | `d_k` | **`ret_mean` `d_k`** | **`f3`** |
|---|---:|---:|---:|---:|---:|---:|
| Garrido | 193,7 | — | 456,5 | — | — | — |
| causal + `always` | 167,4 | 3,23 | 672,0 | 38,09 | **1,95** | **561** |
| **causal + `in_window`** | 163,8 | 3,68 | 672,0 | 38,09 | **4,08** | **0** |

**La brecha se cierra por completo.** Y **`ret_mean` se dobla**, de 1,95 a 4,08.

La causa es directa: esas 561 órdenes pierden su `RPj`, salen de la rama de recuperación y
caen en la de riesgo-sin-recuperación, lo que cambia su `ReT`.

**No es adoptable** bajo la regla de aceptación: `ret_mean` empeora muy por encima de
`EPSILON`, y `ret_mean` tiene veto.

## 4. Qué significa

El defecto **era real y ahora es cerrable**, con el coste medido. Eso convierte una
incertidumbre en un precio: hasta ahora no sabíamos si la brecha de `f3` invalidaba el 2,07 de
residuo real de `causal_exposure`; ahora sabemos que **cerrarla no mueve `rpj_p95` en
absoluto** (672 en ambos) y que lo que mueve es `ret_mean`, a peor.

Así que el **2,07 se sostiene**: no dependía de las órdenes con atribución fuera de ventana.

## 5. Estado

`causal_quantity_gate` queda como opción con `"always"` por defecto — el comportamiento
embarcado — y solo actúa bajo `causal_exposure`. **El default es bit-idéntico**; 43 tests en
verde.

**No adopto nada.** El precio está medido y la decisión de pagarlo o no es de proyecto.
