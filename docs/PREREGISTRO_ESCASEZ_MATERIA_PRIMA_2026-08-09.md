# Preregistro — almacenamiento finito aguas arriba, con el nivel fijado por dominio

**Fecha:** 2026-08-09. **Congelado antes de escribir el runner.**
**Rol:** `DEVELOPMENT_BOUNDARY_MAP_NO_LEARNER_AUTHORIZED`.

## 1. Por qué hay escasez y por qué NO la fijo donde el aprendiz gane

El port de Program V al DES completo cerró en `STOP_NO_PHYSICAL_HEADROOM_IN_THE_FULL_DES` con los
seis contrastes en **exactamente cero**. La razón está medida: la cadena termina con **4,09M de
unidades crudas en mano contra 133.076 consumidas por semana — 215 días de suministro**. La materia
prima no es la restricción activa, así que una decisión que sólo la mueve no puede mover el
servicio.

**El riesgo evidente de «hacer escasa la materia prima» es la ingeniería del resultado:** apretar
hasta que el mecanismo pague y llamar a eso un hallazgo. Este preregistro lo bloquea de tres formas.

## 2. La justificación de dominio, que es de la fuente

**Garrido-Ríos (2017) declara capacidad de almacenamiento ILIMITADA en WDC/AL/SB como una
simplificación explícita del modelo**, no como un hecho de la MFSC. Un depósito militar real tiene
un techo físico. Eliminar esa simplificación es exactamente el tipo de extensión que la propia
tesis invita, y es la **única** razón por la que hay una tapa aquí.

**El nivel se expresa en la unidad en que la doctrina logística dimensiona un depósito: días de
suministro.** No en unidades, no en un porcentaje del consumo del episodio, y desde luego no en el
punto donde una política empieza a ganar.

## 3. El barrido, enumerado aquí y cerrado

| celda | días de suministro | por qué está |
|---|---|---|
| `unlimited` | ∞ | **control inerte**: debe reproducir el port, con los seis contrastes en cero |
| `d180` | 180 | por encima de los 215 días observados: no debería atar |
| `d90` | 90 | reserva estratégica amplia |
| `d60` | 60 | dimensionamiento de depósito convencional |
| `d30` | 30 | operación ajustada |
| `d14` | 14 | escasez severa |

`cap = días × 19.011 unidades/día`, con el consumo medido en el DES congelado **antes** de este
documento y citado arriba. La celda `unlimited` es el control que decide si el instrumento sirve.

## 4. Las tres protecciones contra el resultado a medida

1. **Se reporta el barrido completo**, no la celda que más convenga. La respuesta es una frontera:
   *a partir de qué días de suministro la materia prima se vuelve la restricción activa*, no un
   número.
2. **El control inerte debe salir plano.** Si `unlimited` muestra headroom, el instrumento lo
   fabricó y **nada más se lee**.
3. **Una tapa que ata no es un hallazgo.** Cualquier celda donde el mejor constante ya pierda
   servicio de forma masiva está midiendo una cadena rota, no una decisión. Por eso se reporta el
   servicio del mejor constante en cada celda **junto a** cualquier headroom.

## 5. Qué se mide

Las mismas trece políticas y las mismas tapes de Program V, **importadas**. Endpoint: fill rate de
teatro. Contrastes: `H_priv`, `H_obs`, `H_ret`, y retenido contra los placebos retardado y barajado.

**Una predicción externa que este barrido puede falsar.** Un informe de Program W —cuyos commits no
están publicados y que por tanto no puedo verificar— reporta que bajo escasez el `H_ret` es
**exactamente 0**, porque los yields observados a 24–72 h revelan el estado demasiado pronto y
reiniciar Bayes equivale a retener historia. Si aquí `H_ret` sale positivo, esa lectura es falsa;
si sale cero, queda corroborada de forma independiente. **Lo escribo antes de correr.**

## 6. Falsadores

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_unlimited_cell_is_flat` | los seis contrastes en cero con capacidad infinita | si no, el instrumento fabrica headroom |
| `f2_cap_binds_when_tight` | unidades bloqueadas > 0 en las celdas ajustadas | una tapa que nunca ata no es escasez |
| `f3_blocked_is_never_destroyed` | bloqueado = ofrecido − admitido, y la masa cierra | el defecto retractado ayer |
| `f4_same_tape_same_risks` | recuento de pedidos idéntico entre políticas | la tapa no debe consumir RNG |
| `f5_best_constant_still_serves` | el mejor constante sirve ≥ 0,50 en la celda que se lea | headroom sobre una cadena rota no es headroom |
| `f6_H_priv_material` | `LCB95 ≥ 0,02` en alguna celda | puede fallar en las seis |
| `f7_H_ret_positive` | `LCB95 > 0` retenido menos reset | **la predicción de Program W dice que fallará** |

## 7. Reglas de lectura

1. Si `f1` o `f3` fallan → `BLOCKED_INSTRUMENT`, nada más se lee.
2. La frontera se lee sobre las seis celdas, con `f5` filtrando las que midan una cadena rota.
3. `H_priv` positivo **no** autoriza aprendiz: hace falta además `H_obs`, y después `H_ret` sobre
   su propia ablación. Los tres, en ese orden.
4. Sin `f7`, el veredicto es `SCARCITY_MAKES_HEADROOM_PHYSICAL_BUT_HISTORY_ADDS_NOTHING`, y ése es
   un resultado publicable: dice **dónde** vive el headroom y **por qué la memoria no lo captura**.

Semillas de desarrollo ya quemadas `8600001–8600060`. **No se abre bloque virgen.**
