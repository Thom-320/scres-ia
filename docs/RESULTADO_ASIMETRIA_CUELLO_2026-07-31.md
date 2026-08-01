# Resultado — la asimetría **sí** crea headroom, y el headroom **es el defecto de métrica**

**Artefacto:** `results/sensitivity/bottleneck_asymmetry_v1/result.json` (sello
`2129b3508435cab5…`, `HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION`) · **los seis falsadores
PASAN** · 51.200 episodios, 256 calendarios enumerados exactamente · preregistro
`docs/PREREGISTRO_ASIMETRIA_CUELLO_2026-07-31.md`, commiteado antes de correr.

## 1. La hipótesis se confirma: `H_PI` crece con la asimetría

| `dominant_share` | 0,5 *(simétrico — la condición de la Fase 1A)* | 0,6 | 0,7 | 0,8 | 0,9 |
|---|---:|---:|---:|---:|---:|
| **`H_PI`** | **0,001169** | 0,001379 | 0,001553 | 0,002064 | **0,002094** |

**Monótona.** El mínimo está exactamente en el caso simétrico. La hipótesis que dejó la Fase 1A
—*la contención crea headroom sólo si los reclamantes son asimétricos*— **es correcta en
dirección**.

Y el control causal vuelve a dar el mismo número: con **sustitución completa** (recurso fungible)
`H_PI = **0,000000**`. Es la **tercera vez hoy**, en tres montajes distintos, que el nulo fungible
sale exactamente cero. El mecanismo está establecido.

## 2. Y aun así el veredicto es negativo — porque el guardarraíl se disparó

El preregistro decía, literalmente: *«si `worst_product_fill` **cae** mientras `ret_excel` sube,
el headroom es la métrica premiando el abandono de un producto; en ese caso **no lo reporto como
headroom**»*.

    worst_product_fill en el MEJOR calendario   0,1850
    worst_product_fill en el PEOR calendario    0,5511

**El calendario que maximiza la resiliencia deja un producto al 18,5 % de servicio.** El que la
minimiza lo deja al 55 %. Otra vez, y ahora **en el cuello de botella, con otra topología y otro
instrumento**.

`HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION`.

## 3. Lo que esto significa junto con todo lo demás de hoy

Tres montajes independientes, la misma firma:

| montaje | contención | asimetría | `H_PI`/`H_regime` | nulo fungible | servicio en el óptimo |
|---|---|---|---|---:|---|
| Fase 1A — aguas abajo | sí | no | 1,5e-04 | **0,000000** | **50 %** de fill |
| Fase 1A′ — **en el cuello** | sí | **sí** | 2,1e-03 | **0,000000** | **18,5 %** de `worst_product_fill` |
| auditoría de métrica | — | — | — | — | ReT elige 0,1; **Cobb-Douglas elige 0,5** |

> **Todo el «headroom» que ReT ofrece en esta cadena es su preferencia por abandonar a un
> reclamante.** Cuanto más asimétricos son los reclamantes, más headroom aparente — porque hay
> más que abandonar.

Eso explica de golpe por qué la asimetría «funciona» y por qué el resultado sigue siendo
negativo: **la asimetría no crea una decisión, crea una víctima más barata.**

## 4. La pregunta que esto deja, y es la siguiente

Toda la campaña ha medido headroom sobre **ReT**. Si ReT premia el abandono, entonces
`H_regime ≈ 1e-4` en veinte experimentos puede ser **la métrica**, no la cadena.

**Cobb-Douglas ordena bien** (`results/metric_audit/abandonment_v1/result.json`: su óptimo es el
reparto equilibrado en los dos regímenes, mientras ReT elige el extremo). Así que la medición que
falta es directa:

> **Repetir `H_PI` y `H_regime` con Cobb-Douglas como métrica primaria.**
>
> * si el óptimo de Cobb-Douglas **sí** se mueve con el régimen → **hay headroom real**, y toda
>   la campaña anterior estaba mirando por el instrumento equivocado;
> * si **no** se mueve → el «cuándo NO entrenar» queda establecido sobre una métrica **sana**, y
>   deja de ser vulnerable a la objeción de que medimos con una métrica rota.

Las dos salidas son publicables. La segunda es mucho más fuerte de lo que teníamos esta mañana.

## 5. Lo que NO afirma

* **No** rescata Program O: estimando distinto (`H_PI` en función de la asimetría, no `H_obs`),
  semillas vírgenes disjuntas de sus bloques quemados, sin aprendiz, con preregistro propio.
  `f6` lo declara y es verificable.
* **No** dice que `H_PI = 0,0021` sea headroom aprovechable ni aunque el servicio no cayera:
  sigue **5× bajo la barra** de 0,01.
