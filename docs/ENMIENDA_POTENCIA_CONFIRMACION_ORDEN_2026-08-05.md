
# Enmienda — potencia de la confirmación recalculada bajo el orden contractual

**Artefacto:** `results/custody/grid_transfer_confirmation_repower/result.json` ·
**Runner:** `scripts/repower_grid_transfer_confirmation_v1.py` ·
**Contrato padre:** `docs/PREREGISTRO_CONFIRMACION_TRANSFERENCIA_REJILLA_2026-08-05.md` §4.

**No abre semillas, no adjudica, no autoriza.** El registro sigue en `new_seed_opening: false`.

## 1. Por qué

El §4 del contrato dimensionó el bloque virgen con la SD pareada de
`results/grid_transfer_v2/result.json`, que se calculó con la carrera **ordenada alfabéticamente
por nombre de directorio** en vez del orden que el propio contrato declara. Para un brazo que
arrastra estado **el orden de contextos ES la carrera**, así que la SD que fijó `n = 60` se midió
sobre una carrera que ningún contrato declaró.

## 2. Qué se mueve y qué no

**`δ* = 0,015` NO se mueve.** Se fijó antes de reservar el bloque y se mantiene. Dejarlo derivar
hacia la media corregida volvería el cálculo **circular**: el tamaño quedaría elegido para detectar
exactamente el efecto ya observado, y sería adecuado por construcción. **Sólo se re-mide la SD,
porque sólo la SD estaba mal medida.**

## 3. El resultado

| | media `δ_M` | **SD pareada** | potencia a `n = 60` | `n` para 0,86 |
|---|---:|---:|---:|---:|
| orden alfabético (superado) | +0,036547 | 0,042154 | 0,867 | 59 |
| **orden contractual** | +0,030497 | **0,027527** | **0,995** | **26** |

**El defecto de orden estaba inflando la varianza en un 53 %**, que es lo esperable: una carrera
inconsistente añade ruido al estado retenido sin añadir señal.

`n = 60` **sigue siendo suficiente** —potencia 0,995 contra un objetivo de 0,86—, así que la
reserva no queda invalidada. Y bajo la SD corregida **bastarían 26 semillas**.

## 4. La decisión que esto NO toma

Encoger el bloque de 60 a 26 **ahorraría 34 semillas vírgenes**, pero es un cambio de diseño
posterior a ver los datos de desarrollo, y esa clase de ajuste tiene su propio riesgo aunque vaya
en la dirección conservadora. **Es decisión del PI**, y este documento sólo pone las dos cifras
sobre la mesa.

Lo que no cambia en ningún caso: **hace falta una autorización explícita del PI** antes de tocar
el bloque. La que existe en el registro se la concedió a sí mismo un agente, y el inventario
central sigue declarándose incompleto.
