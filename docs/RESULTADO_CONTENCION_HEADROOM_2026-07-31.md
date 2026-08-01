# Resultado — Fase 1A: la contención **muerde**, y aun así no abre la puerta

**Artefacto:** `results/sensitivity/contention_headroom_v1/result.json` (sello
`b54ee3225043630b…`, `CONTENTION_DOES_NOT_OPEN_THE_DOOR`) · **los seis falsadores PASAN** ·
5.184 episodios · preregistro `docs/PREREGISTRO_CONTENCION_HEADROOM_2026-07-31.md`, commiteado
antes de correr (`dd34aea`).

## 1. Antes del resultado: lo que el código llevaba escondido

`reallocate_unused` estaba **cableado a `True`** en el punto de llamada. Las dos CSSU llevaban
toda la vida del proyecto corriendo en la condición **fungible** — que es *exactamente* la
condición bajo la que Program O midió **cero**. Y el reparto sólo admitía tres puntos
(`0,25 / 0,50 / 0,75`), un límite de resolución que un test codificaba como si fuera física.

Ninguna de las dos cosas venía de la tesis. Eran nuestras.

## 2. `H_regime` sobre `ret_excel_risk_conditional`

| celda | `H_regime` | IC95 |
|---|---:|---|
| `FIFO_PARTIAL` **no fungible** | **0,000153** | [0,000000, 0,000374] |
| `R24_AGE_PARTIAL` **no fungible** | 0,000141 | [0,000000, 0,000403] |
| `R24_AGE_PARTIAL` fungible | 0,000068 | [0,000002, 0,000222] |
| `SPT_FULL` **no fungible** | 0,000041 | [0,000004, 0,000696] |
| `FIFO_PARTIAL` fungible | 0,000031 | [0,000000, 0,000206] |
| `SPT_FULL` fungible | **0,000000** | [0,000000, 0,000000] |

**El mecanismo se reproduce en las tres reglas de servicio**: no fungible > fungible, siempre. Y
`SPT_FULL` fungible da **exactamente 0,000000**, que es el nulo de Program O **al dígito**.

**Pero la magnitud no llega**: el máximo es `1,5e-04`, **~65× bajo la barra de 0,01**.

## 3. Y no es que la disputa fuera tibia

`f2` mide lo que la cuota dura **renuncia** a ceder: **113.632 raciones por episodio** en el
brazo no fungible, contra **0,0** exactas en el fungible. Sobre una entrega anual de ~677.750,
eso es **~17 % del flujo sacrificado** por mantener las cuotas.

> **Ése es el hallazgo incómodo: se puede tirar el 17 % del flujo a la basura por la contención
> y el valor de conocer el régimen sigue siendo 1,5e-04.** La disputa es severa y real; lo que
> no aparece es la **decisión**.

La escalada también funcionó (`f3`: eventos R23 de 1,09 → 3,31 por episodio), y la palanca mueve
el sistema (`f1`: dispersión 2,5e-03 entre repartos, un orden de magnitud sobre el headroom).

## 4. La lectura, y por qué es informativa en vez de un simple «no»

El preregistro fijó qué significaría este desenlace, y lo sostengo: **el mismo mecanismo que dio
0,1515 en Program O da 1,5e-04 aquí**, así que la causa **no** es la ausencia de disputa. Las dos
diferencias entre los dos montajes son concretas y comprobables:

1. **Program O disputaba EN el cuello de botella** (Op5–Op7, margen 2,6 %). Aquí la disputa está
   **aguas abajo** de él: el reparto reordena una cantidad que la línea de ensamblaje ya
   determinó.
2. **Sus dos reclamantes eran ASIMÉTRICOS** (dos productos distintos, con demandas distintas).
   Los nuestros son **simétricos por construcción**: el destino de cada pedido se asigna por hash
   50/50 (`stable_cssu_destination`), así que la demanda esperada de A y B es idéntica **en todos
   los regímenes**. Un reparto óptimo que no depende del régimen es exactamente lo que un
   `H_regime` ≈ 0 describe.

**Hipótesis que esto deja, y que es la siguiente prueba, no una conclusión:** la contención crea
headroom sólo cuando los reclamantes son **asimétricos de una forma que varía con el estado**.
Contención sin asimetría es un reparto fijo con más pasos.

El artefacto `v1_1` añade `argmax_by_regime` —qué reparto gana en cada régimen— para que
«el óptimo es invariante» se pueda **leer** en vez de inferirse. Mismo código, mismas semillas,
mismos `H_regime`; sólo suma el diagnóstico.

## 5. Lo que NO afirma

Nada sobre la Fase 1B (expedición) ni la 1C (autotomía), que atacan otras dos causas. Y nada
sobre Program O: aquel contrato prohíbe rescates, y esto no es uno — es física nueva, semillas
vírgenes y su propio preregistro.
