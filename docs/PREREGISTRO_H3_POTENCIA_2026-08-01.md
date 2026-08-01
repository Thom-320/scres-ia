# Preregistro — corrida de potencia para H3′

**Escrito y commiteado ANTES de correr.** Instruido por el PI tras leer
`docs/RESULTADO_H1_H3_V2_2026-08-01.md`, donde `H3′` quedó **no sostenida** con el intervalo
cruzando el cero y el orden monótono en la dirección esperada.

## El cálculo de potencia, hecho y declarado

Con `n = 12` réplicas:

    diferencia memoria vs reinicio  +7,86
    IC95                            [−14,72, +28,31]   →  semiancho 21,5   →  SE ≈ 10,98

Para que `LCB95 > 0` con el efecto observado hace falta `1,96·SE < 7,86`, es decir `SE < 4,01`:

> **n ≈ 90 réplicas.**

Se fija **`n = 120`**, que da margen sobre el límite y deja `LCB95 ≈ +1,6` si el efecto real
coincide con el estimado puntual.

## Lo que NO se hace, y por qué importa

**No se amplía el bloque de 12 ya visto.** Extender una muestra tras mirar su intervalo es
peeking, y el estimado puntual que motivó este cálculo saldría del mismo dato que luego lo
confirma. Se abre un **bloque nuevo y virgen** de 120 réplicas con semillas `6 000 001…6 000 120`,
disjuntas de todas las anteriores. Las 12 originales quedan como la corrida **exploratoria** que
motivó el cálculo, y se reportan como tal.

**No se mueve la regla de lectura.** Sigue siendo `LCB95 > 0` sobre la diferencia pareada por
réplica de la varianza del coste de búsqueda entre los seis contextos, exactamente como en
`docs/PREREGISTRO_H1_H3_V2_2026-08-01.md`.

## Reparto de cómputo, declarado por la regla de enrutado

El coste domina en la superficie: 288 configuraciones × 6 contextos = **1.728 episodios por
réplica**, ~126 s en el M1 Pro y ~504 s en el VPS (medido: el VPS es ~4× más lento en carga
monoproceso).

| pool | réplicas | semillas |
|---|---:|---|
| **local** (M1 Pro) | 90 | `6 000 001 … 6 000 090` |
| **VPS** `ovh-agent-lab` | 30 | `6 000 091 … 6 000 120` |

Rebanadas **disjuntas y en paralelo**, ~3,4 h de reloj en vez de 4,2 h sólo en local. Se fusionan
por concatenación de réplicas, que es válido porque cada réplica es independiente y lleva su
propia semilla CRN.

## Falsadores de la fusión

| falsador | por qué puede fallar |
|---|---|
| `f_merge_seeds_are_disjoint` | un solapamiento haría que dos «réplicas independientes» fueran la misma |
| `f_merge_contexts_and_budget_match` | fusionar corridas con distinto presupuesto o distinto orden de contextos mezclaría dos experimentos |
| `f_merge_source_is_identical` | rebanadas producidas por versiones distintas del runner no son el mismo experimento; se compara el hash del script |

## Regla de lectura

* **`LCB95 > 0`** en memoria vs reinicio → **`H3′` sostenida**.
* **`LCB95 ≤ 0`** con `n = 120` → **`H3′` refutada con potencia suficiente**, y eso es un
  resultado más fuerte que el «no sostenida» actual: acotaría el efecto por debajo del umbral que
  el propio cálculo de potencia declara detectable.

Ambas salidas son publicables. La segunda cierra la hipótesis en vez de dejarla abierta.
