# Result — el cruce: la regla de SE apareada funcionó, y el residuo de `RPj` es 2,07, no 39

**Status:** `PREREGISTERED_NO_ARM_ADOPTABLE_TRADEOFF_FRONTIER_MEASURED`. Artefacto
`results/metric_audit/link_x_attribution_v1/result.json`, sellado. Raíces 3.200.001–12.

## 1. La regla declarada en §3 hizo su trabajo

R1r, `nivel / d_k / d_k de SE apareada`:

| momento | A | C | L | **LC** |
|---|---|---|---|---|
| `ret_mean` | 0,01 / **1,58** / 1,58 | 0,01 / 1,74 / 1,74 | 0,01 / **0,29** / 0,29 | 0,01 / 0,44 / 0,44 |
| `rpj_mean` | 417,5 / 18,90 / 18,90 | **171,0** / 2,63 / **1,92** | 400,3 / 15,58 / 17,45 | **171,1** / 2,64 / **1,91** |
| **`rpj_p95`** | 2533,1 / **19,95** / 19,95 | **672,0** / **38,09** / **2,07** | 2458,0 / 21,04 / 19,23 | **678,0** / **39,15** / **2,13** |
| `autotomy_share` | 11,20 | 11,20 | 11,20 | 11,20 |
| `ret_above_one_share` | 3,90 | 3,90 | 3,90 | 3,90 |

**`rpj_p95`: el `d_k` empeora de 19,95 a 38,09 mientras el `d_k` de SE apareada cae de 19,95
a 2,07.** El empeoramiento es **enteramente del denominador**: el numerador mejoró ~10×
(2.533 → 672 contra una referencia de 456,5).

Veredicto pre-declarado: **`RESIDUO_MAS_CIERTO_NO_MAS_GRANDE`**.

Haber firmado esa regla **antes** de ver los números es lo que permite decirlo sin que suene
a racionalización. Si la hubiera inventado ahora, sería exactamente eso.

Y en `rpj_mean` **los dos** mejoran (18,90 → 2,63 y → 1,92), así que ahí no hay ambigüedad
ninguna: `causal_exposure` lo arregla.

## 2. Falsadores

| falsador | resultado |
|---|---|
| f2 ninguna orden bajo `LT` | **PASA** |
| **f4 `L` y `LC` difieren en `rpj_p95`** | **PASA** — 2.458,0 contra 678,0 |
| f3 `RPj` causal exige bloqueo físico | **FALLA** — 2.555 en `C`, 1.932 en `LC` |
| f5 conjunto `epsilon`-estable | **FALLA** |

**`f4` pasa, y era el que más me importaba**: el mismo eje fue ignorado en silencio bajo
`op9_linked` dos veces antes (`fulfillment_transit_mode`, luego `δ`). Esta vez sí llega.

**`f3` falla y es un hallazgo:** bajo `causal_exposure`, **2.555 órdenes reciben `RPj > 0` sin
un bloqueo físico registrado**. La atribución causal no es tan estricta como su nombre
promete — probablemente por las ramas de riesgo por cantidad y R24, que atribuyen sin
intervalo de bloqueo. Eso **acota lo que se puede afirmar** del 2,07: es correcto como
diagnóstico de denominador, pero la atribución causal misma tiene una fuga que hay que cerrar
antes de adoptarla.

## 3. Ningún brazo es adoptable — como predije

Predije no-adopción en §4 del contrato. Se cumple, por `f3`, `f5` y porque `rpj` sigue
empeorando en `d_k`.

Conjuntos no dominados: **R1r `[A, C, L, LC]`** (no discrimina) y **R2r `[L]`** (sí, y elige
`L`). Que R2r seleccione `L` y R1r no discrimine es información sobre las familias, no ruido.

## 4. La frontera de compensación, que es el resultado publicable

El contrato §6 previó este desenlace y pidió reportarlo así:

* **`L` es lo mejor para `ret_mean`** — `d_k` **0,29**, el mejor número de la sesión sobre el
  endpoint del manuscrito, contra 1,58 del statu quo.
* **`C` es lo mejor para `RPj`** — nivel de `rpj_p95` **672** contra 2.533, y residuo real
  **2,07** una vez descontado el efecto de denominador.
* **`LC` no combina lo mejor de ambos**: `ret_mean` 0,44 (peor que `L`) y `rpj_p95` 678
  (igual que `C`). **Los dos ejes no se componen**; el enlace no aporta sobre la atribución
  causal en `RPj`, y la atribución causal cuesta 0,15 de `d_k` en `ret_mean`.

Eso **es** la respuesta cuantitativa: cada arreglo cuesta el otro, y el coste está medido.

## 5. Estado

Nada implementado, ningún default movido. Lo que queda antes de poder adoptar: cerrar la fuga
de `f3` en `causal_exposure`, y entender la inestabilidad de `epsilon`, que ha fallado en las
tres últimas corridas.
