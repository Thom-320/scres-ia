# Auditoría de `service_first_resilience_v1` — se sostiene en estructura, **no en el instrumento**

**RECONCILIADO 2026-08-01** tras revisión externa: el §5 describía un sucesor que luego descarté,
y la tabla venía de un script ad-hoc. Artefacto sellado ahora:
`results/metric_audit/service_first_v2/result.json` (sello `0e37fe2faa3fd695…`, seis falsadores
PASA, 30 episodios por `scripts/run_service_first_v2_audit.py`).

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
perdidos**, **abiertos al horizonte**:

| reparto | 0,1 | 0,3 | **0,5** | 0,7 | 0,9 |
|---|---:|---:|---:|---:|---:|
| abiertos al horizonte | 60,0 | 59,5 | **42,8** | 58,7 | 60,0 |

Cuatro de los cinco están **clavados en el cap**. El ganador está por debajo.

**La consecuencia es un exploit mecánico:** una política que mantenga su cola en 60 o menos
**deja hasta 60 pedidos abiertos al horizonte y registra CERO pérdidas**, pasando el gate con
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

## 5. El sucesor implementado — `service_first_resilience_v2`

**Primero, el arreglo que descarté, porque es el obvio y es incorrecto.** Propuse
`−(cantidad no servida)/(demandada)`. Resulta ser **exactamente `1 − flow_fill_rate`**, así que
habría colapsado los componentes 1 y 2 en uno solo. Queda escrito para que nadie lo reintente.

**Lo que distingue ABANDONO de simplemente poco fill es dónde cae el déficit**: el abandono lo
concentra en un reclamante. Así que el componente líder de `v2` es **el fill del peor
reclamante** — continuo, no gameable por el cap de cola, no implicado por el fill agregado, y
degenera al fill agregado cuando hay un solo reclamante, que es correcto porque abandonar a un
reclamante no está definido entonces.

    v2 = ( worst_claimant_fill , flow_fill_rate , −backorder_qty_final , ReT acotado )

**El test que lo acompaña es el exploit escrito como aserción** —*convertir pérdidas en pedidos
abiertos no debe mejorar la clave*— **y lleva `v1` como control que DEBE dejarse engañar**. Sin el
control, el test podría pasar vacuamente para ambos. `tests/test_service_first_metric_v2.py`, 5/5.

**Estado honesto de `v2`: es PROSPECTIVO.** Está implementado y con test, pero **no tiene
preregistro propio ni ha sido el endpoint de ninguna corrida sellada**. No debe citarse como
métrica en uso.

## 6. Qué hacer con `v1`

**Nada retroactivo.** El contrato está congelado y su resultado —`NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST`
en la expedición— **no cambia de signo** por esto: el defecto haría a la métrica *más permisiva*
con el abandono, y aun así no encontró headroom. El sucesor es para lo que venga, no para
reabrir lo cerrado.
