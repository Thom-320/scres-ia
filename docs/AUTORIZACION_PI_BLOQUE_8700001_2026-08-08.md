# Autorización del PI — apertura del bloque virgen 8700001–8700048

**El PI autorizó la apertura el 2026-08-08.** El registro está en
`new_seed_opening: False` con estado `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, así
que esto es una **excepción del PI**, la segunda del proyecto tras
`AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md`.

**Bloque:** `8700001–8700048`, 48 semillas. Verificado antes de escribir: **cero colisiones** con
el registro y **cero** con las semillas de cualquier artefacto sellado.

## 1. La regla que hace que esto valga algo

**Una semilla virgen es una puerta de un solo sentido.** El diseño de abajo queda congelado por
este documento y su commit. Si el instrumento resulta defectuoso al correr, **el bloque queda
quemado** y un sucesor necesita otro bloque distinto: no hay reejecución, no hay reajuste, no hay
«lo corrijo y lo vuelvo a correr sobre las mismas semillas».

Ésa es la única razón por la que un bloque virgen vale más que uno reutilizado, y hoy he tenido que
bloquear cuatro instrumentos por defectos encontrados **después** de correr. Por eso el pre-vuelo
de `supply_chain/falsifiers.py` corre **sobre semillas ya quemadas** antes de tocar éstas.

## 2. Por qué se abre

`results/signal_search/result.json` devolvió `NO_PREFIX_SIGNAL_CAPTURES_THE_CEILING_IN_THIS_DESIGN`
y su propia divulgación `d2` nombra el límite: **seis tapes de entrenamiento** soportan un vecino
más cercano sobre **un** rasgo y nada más rico. El negativo no descarta una señal que un diseño
mayor encontraría, y con doce tapes reutilizadas no hay forma de saberlo.

El techo que se busca capturar es real: `results/ceiling_null_diagnostic` lo pasa contra su nulo de
interacción con `p = 0,0132`. Pero eso también se midió sobre las mismas doce tapes, así que **lo
primero que hace este bloque es volver a medirlo**.

## 3. El diseño, congelado

**Entorno**, idéntico al del gate de precio y sin un parámetro nuevo: demanda estacional
`garrido_seasonal_v1`, `strategic_buffer_release_mode = "immediate"`, lead time 336 h. **`λ = 0,35`
como titular**, banda 0,275–0,500 al lado.

**Semillas:** 24 de entrenamiento (`8700001–8700024`) y 24 de test (`8700025–8700048`). La
partición se fija aquí, por orden, y no se re-baraja.

**Clase de calendarios:** las mismas 27 opciones enumeradas `(inicio, K)`. No se amplía, para que
el techo sea comparable con el de doce tapes.

**Rasgos:** los mismos 13 estadísticos de **prefijo** en la semana 4. **La lista se congela aquí**;
no se añade ninguno después de ver nada.

**Mapas:** vecino más cercano con `k = 1` y con `k = 3` sobre un rasgo. Dos familias × 13 rasgos =
**26 tests**, con **Holm sobre los 26**. `k = 3` entra ahora porque 24 tapes de entrenamiento lo
soportan y seis no lo soportaban.

**Placebo:** el mismo mapa sobre el rasgo barajado, que conserva el mecanismo y destruye la señal.

## 4. Las reglas de lectura, en orden y fijadas de antemano

**Primero el techo.** Si el hueco clarividente **no** supera su nulo de interacción en el bloque
nuevo, entonces el techo de doce tapes era un artefacto de doce tapes, **todo lo de abajo se
detiene**, y el veredicto es `CEILING_DID_NOT_REPLICATE`. Nada sobre señales se lee en ese caso.

**Sólo si replica**, la búsqueda de señal:

* algún `(rasgo, mapa)` con `LCB95 > 0`, Holm `p < 0,05` **y** batiendo al placebo →
  `SIGNAL_FOUND`, con su cuota del techo;
* ninguno → `NO_PREFIX_SIGNAL_CAPTURES_THE_CEILING_AT_N48`, que con 24 tapes de entrenamiento **sí**
  es una afirmación con fuerza, a diferencia de la de seis;
* cualquier falsador caído → `BLOCKED_INSTRUMENT`, y **el bloque queda quemado igualmente**.

**No hay rama que diga «casi».** Y `OBSERVABLE_SIGNAL_IS_WORSE_THAN_THE_FIXED_SCHEDULE` sigue
siendo admisible: cinco de trece rasgos ya lo fueron con doce tapes.

## 5. Alcance y precio de fidelidad

La ruta de liberación y el lead time de 336 h son **extensiones nuestras sin evento fuente** —la
tesis repone periódicamente (p.107) y sus 48 h (p.111) son entrega al usuario—, y el precio del
buffer es una asunción declarada en unidades del endpoint, no una tasa monetaria. **Nada de lo que
salga de aquí se presenta como reproducción de Garrido-Ríos (2017).**

Esto **no** autoriza entrenar un aprendiz. Autoriza medir si existe una señal que un aprendiz
pudiera usar.
