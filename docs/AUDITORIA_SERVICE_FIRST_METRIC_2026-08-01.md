# Auditoría de `service_first_resilience_v1` — se sostiene en estructura, **no en el instrumento**

Módulo: `supply_chain/service_first_metric.py` · contrato
`docs/PREREGISTRO_METRICA_SERVICE_FIRST_2026-08-01.md`. **No se edita nada aquí**: el contrato
está congelado, así que esto es una auditoría con propuesta de **sucesor `v2`**.

## Veredicto en una línea

**La forma es correcta y la elegiría igual. El primer componente mide otra cosa de la que dice
medir**, y por esa vía reabre exactamente el agujero que la métrica venía a cerrar.

## 1. Lo que está bien, y no es poco

**Es lexicográfica, no una suma ponderada.** Ésa es la decisión estructural correcta: una suma
habría exigido inventar un tipo de cambio entre servicio y resiliencia, y **cualquier tipo de
cambio es ajustable al resultado que uno quiera**. El orden lexicográfico no lo es.

**Degrada ReT a desempate**, que es donde debe estar una métrica que ya sabemos explotable.
**Se declara estimando, no recompensa**, y prohíbe colapsarla en un escalar sin registro.

**Y funciona en el caso para el que se diseñó.** Sobre el barrido de contención, semilla
5.200.001, `R2r`:

| reparto | perdidos | fill | **gate** | ReT acotado |
|---|---:|---:|---:|---:|
| 0,1 | 80 | 0,494 | 0 | 0,3978 |
| 0,3 | 5 | 0,764 | 0 | 0,3544 |
| **0,5** | **0** | **0,851** | **1** | **0,0047** |
| 0,7 | 12 | 0,744 | 0 | 0,2917 |
| 0,9 | 75 | 0,513 | 0 | 0,3559 |

**Gana el reparto 0,5 — el que tiene el ReT MÁS BAJO de los cinco.** Invierte exactamente el
ranking de `ret_excel`. Eso es lo que se pedía.

## 2. El defecto que la invalida como objetivo: `lost_orders` no mide abandono

`BACKORDER_QUEUE_CAP = 60` (`config.py:45`), y en desbordamiento *«the last order in the list is
removed and labelled as lost»*. Es decir:

> **`lost_orders` se dispara SÓLO cuando la cola de pedidos pendientes desborda los 60.**
> **Es un proxy del desbordamiento de cola, no del abandono.**

La columna que lo delata está en la misma corrida — pedidos **ni servidos ni marcados como
perdidos**, permanentemente pendientes al horizonte:

| reparto | 0,1 | 0,3 | **0,5** | 0,7 | 0,9 |
|---|---:|---:|---:|---:|---:|
| pendientes para siempre | 60 | 60 | **41** | 60 | 60 |

Cuatro de los cinco están **clavados en el cap**. El ganador está por debajo.

**La consecuencia es un exploit mecánico:** una política que mantenga su cola en 60 o menos
**abandona hasta 60 unidades indefinidamente y registra CERO pérdidas**, pasando el gate con
puntuación perfecta. Es **la misma forma** del agujero de `ret_excel` —demanda no servida que no
cuenta como fallo— por una puerta distinta.

Aquí no muerde, porque el ganador tiene además los menos pendientes. Pero el propósito declarado
de esta métrica es **ser segura como objetivo**, y un agente entrenado sobre ella buscaría
precisamente el borde del cap.

## 3. Segundo defecto, menor pero hay que decirlo: dos de sus cuatro componentes están muertos

Medido: **ningún par de políticas empata en (gate, fill)**, así que
`-backorder_qty_final` y `ret_excel_visible_clipped_0_1` **nunca desempatan nada**. Con `fill`
continuo, los empates son de medida cero.

**En la práctica la métrica es de dos componentes: (gate de pérdida, fill rate).** Eso es
perfectamente honesto — pero el docstring promete cuatro niveles de discriminación y el paper no
puede decir que este endpoint «integra ReT». No lo integra: lo lleva de adorno.

## 4. Sobre la circularidad, que es lo que el PI preguntó

**No está ajustada al resultado, pero no puede usarse como evidencia de lo que la motivó.**

* El principio —*una cadena militar no puede abandonar una unidad, luego ninguna ganancia en otra
  dimensión compensa un pedido perdido*— es una afirmación **de dominio**, independiente de
  nuestros datos. Eso la salva.
* La forma lexicográfica es lo que impide el ajuste: **no hay parámetro que tocar.**
* **Pero** se eligió sabiendo qué políticas ordena y cómo. Por tanto es un **endpoint normativo
  estipulado**, jamás evidencia de que abandonar sea malo. Usarla para «redescubrir» el defecto de
  `ret_excel` sería circular, y el manuscrito debe decir explícitamente que es una **decisión
  declarada**, no un hallazgo.

## 5. El sucesor propuesto — `service_first_resilience_v2`

Un solo cambio en el primer componente, de binario a continuo y sobre la cantidad correcta:

    componente 1:  −(cantidad NO SERVIDA) / (cantidad demandada)

donde *no servida* = perdida **+ permanentemente pendiente al horizonte**. Con eso:

* **se cierra el exploit del cap**: quedarse en 59 pendientes deja de ser gratis;
* **desaparece el acantilado**: perder un pedido ya no equivale a perder mil, ni al revés;
* **el ranking deseado se conserva** — con la misma corrida, no servidos totales por reparto:
  `0,1 → 140 · 0,3 → 65 · **0,5 → 41** · 0,7 → 72 · 0,9 → 135`. **0,5 sigue ganando**, ahora por
  un margen continuo en vez de por un umbral.

Y un test que el módulo hoy no tiene, escrito para poder fallar:

> **una política que convierta pedidos perdidos en pedidos permanentemente pendientes no debe
> mejorar su clave.** Bajo `v1` la mejora; bajo `v2` no. Es la validación del falsador haciéndolo
> fallar contra el defecto, que es el patrón que ya nos salvó dos veces esta semana.

## 6. Qué hacer con `v1`

**Nada retroactivo.** El contrato está congelado y su resultado —`NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST`
en la expedición— **no cambia de signo** por esto: el defecto haría a la métrica *más permisiva*
con el abandono, y aun así no encontró headroom. El sucesor es para lo que venga, no para
reabrir lo cerrado.
