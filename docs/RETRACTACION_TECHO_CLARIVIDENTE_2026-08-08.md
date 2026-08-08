# El techo clarividente no replicó en el bloque virgen — retractación

**Fecha:** 2026-08-08 · **Artefacto que decide:** `results/expanded_signal_search/result.json`
(sello `4607522c500b7894…`) · **Autorización:** `docs/AUTORIZACION_PI_BLOQUE_8700001_2026-08-08.md`
(sha `b109034f…`) · **Diseño congelado en:** `b9115292`, antes de tocar una semilla.

## 1. El número

Sobre 48 semillas vírgenes (`8700001–8700048`, 24 entrenamiento / 24 test), 27 calendarios,
λ = 0,35, mismo entorno y misma clase de opciones:

| | doce tapes reutilizadas | 48 semillas vírgenes |
|---|---|---|
| hueco clarividente | +0,045103 | **+0,024054** |
| media del nulo de interacción | — | **+0,026641** |
| p95 del nulo | — | +0,032964 |
| p | 0,0132 | **0,7482** |

**El hueco medido es menor que la media de su propio nulo.** No es que quede corto de
significación: está por debajo de lo que produce la permutación de residuos sobre el modelo
aditivo. Es exactamente el sesgo de Jensen — un mínimo sobre 27 opciones ruidosas es un mínimo
sesgado a la baja — y nada más.

## 2. Qué queda retractado

`results/ceiling_null_diagnostic/result.json` (`CEILING_SURVIVES_THE_PERMUTATION_NULL`,
p = 0,0132, sello `ab9348cd…`) queda **superado por replicación fallida**. El artefacto **se
conserva y se etiqueta; no se borra**. No estaba mal ejecutado: su nulo era el corregido, el de
interacción, y con doce tapes daba p = 0,0132. Lo que dice el bloque nuevo es que **doce tapes no
bastaban para separar el techo del ruido de selección**, y el p pequeño era una tapa fina.

`results/signal_search/result.json` (`NO_PREFIX_SIGNAL_CAPTURES_THE_CEILING_IN_THIS_DESIGN`) queda
**vacío de objeto**: buscaba una señal que capturase un techo que ahora no se sostiene. Su negativo
no se refuerza ni se debilita; deja de ser una afirmación sobre nada.

## 3. Qué NO se lee

La regla de lectura estaba fijada en la §4 de la autorización, en orden, antes de correr:

> **Primero el techo.** Si el hueco clarividente **no** supera su nulo de interacción en el bloque
> nuevo, […] **todo lo de abajo se detiene** […]. Nada sobre señales se lee en ese caso.

Por tanto **la tabla de 26 rasgos×mapas del artefacto no se interpreta**. Se conserva en el sello
por custodia, y se declara aquí que **no se leyó**: el mejor candidato aparente
(`prefix_events_R23|k1`, ganancia 0,007454) tiene `LCB95 = −0,003512` y Holm p = 0,260, y aun si
fuera limpio sería la cuota de un techo inexistente. `f5_ceiling_replicates_on_the_new_block`
**FALLA**, y ésa es la única lectura.

## 4. Por qué esto es un resultado y no una pérdida

Cuatro intentos de conversión fallaron esta semana contra este techo: la regla de backlog, la fase
sola, fase + desviación, y la búsqueda de 13 rasgos. **Los cuatro estaban persiguiendo un
artefacto de doce tapes.** El coste de no haber abierto semillas nuevas antes fue esos cuatro
intentos; el coste de abrirlas ahora fue un bloque.

Esto también reordena la lectura del negativo anterior. `NO_PREFIX_SIGNAL_...` se archivó con la
divulgación honesta de que seis tapes de entrenamiento no distinguen «no hay señal» de «no hay
potencia». La respuesta resulta ser una tercera: **no había techo que capturar**.

## 5. Estado del bloque

`8700001–8700048` pasa a `BURNED_CONFIRMATION_COMPLETE` en `research/seed_custody_registry.json`. Un
sucesor necesita otro bloque y otra autorización del PI. **No hay reejecución sobre estas
semillas**, ni siquiera con un instrumento mejor: la puerta era de un solo sentido y ya se cruzó.

## 6. Lo que sigue en pie

El espacio de decisión **sí** está priced y **sí** tiene estructura: 21 niveles distintos, 6 puntos
no dominados, y el óptimo se mueve con λ (22 → 18 → 0 semanas). Eso lo estableció el gate de
precio y **no** depende del techo retractado. Lo que no hay es evidencia de que el óptimo se mueva
**con el estado dentro de una λ fija** — que es lo único que un aprendiz podría vender.
