# Preregistro — presupuesto compartido y caducidad, con la celda fiel a la tesis como control

**Fecha:** 2026-08-08. **Congelado antes de escribir una línea del simulador.**

## 1. Por qué este contrato y no otro ajuste del anterior

El sucesor conservativo cerró con `STATIC_TRADE_OFF_ONLY__NO_SEQUENTIAL_HEADROOM`: hueco
clarividente máximo **+0,000403** contra una barra de 0,01, con física que conserva masa. Y dejó el
mecanismo causal nombrado: con lead time de 336 h las ventanas cortas no llegan a inyectar nada, y
como el stock temprano permanece y se consume normalmente, **empezar en la semana 0 casi siempre
domina**. No hay razón física para no preposicionar todo al principio, así que no hay decisión
secuencial que tomar.

Este contrato añade **exactamente las dos razones físicas que faltaban**, y nada más:

* **Presupuesto compartido** — los tres nodos (`op3_rm`, `op5_rm`, `op9_rations`) compiten por una
  misma bolsa por periodo. Gastar en uno **precluye** el otro. Es el mecanismo de contención que es
  el único donde este proyecto ha medido headroom material, con el nulo fungible en exactamente 0.
* **Caducidad** — el stock estratégico tiene una vida útil; lo que la excede se retira y **ya estaba
  pagado**. Preposicionar temprano deja de ser gratis.

## 2. El precio de fidelidad, dicho antes y no después

**La tesis es explícita: la ración es no perecedera a tres años.** Una vida útil corta **contradice
la fuente**, y ese error ya nos costó un lane: Program K fue reclasificado como
EXPLORATORIO/CONTESTADO precisamente por asumir dos semanas de caducidad contra una ración de tres
años.

Por tanto **la caducidad no se asume: se barre**, y el extremo fiel a la tesis es la celda de
control. Con un horizonte de 26 semanas, una vida útil de **156 semanas es inerte por construcción**
— nada puede caducar dentro del horizonte — así que esa celda debe reproducir el resultado de hoy.
**Si no lo reproduce, el instrumento está roto y nada más se lee.**

El presupuesto compartido es igualmente **nuestra extensión declarada**: la tesis no modela una
bolsa de aprovisionamiento común entre nodos. Ninguna de las dos es una tasa monetaria y ninguna
viene de Garrido-Ríos (2017).

## 3. El mapa de frontera, 2 × 2, fijado aquí

| celda | vida útil | presupuesto/periodo | qué se espera |
|---|---|---|---|
| **CONTROL_FIEL** | 156 sem (inerte) | ilimitado | reproduce hoy: sin headroom secuencial |
| **SÓLO_PRESUPUESTO** | 156 sem (inerte) | ajustado | contención sin caducidad |
| **SÓLO_CADUCIDAD** | 8 sem | ilimitado | caducidad sin contención |
| **AMBOS** | 8 sem | ajustado | las dos razones a la vez |

**Las dos celdas de un solo mecanismo son las que hacen esto un experimento y no una demostración.**
Si el headroom sólo aparece en AMBOS, el resultado es una interacción. Si aparece con una sola, la
otra es decorado. Y si aparece en CONTROL_FIEL, no hemos medido lo que creemos.

«Ajustado» se fija **antes de ver resultados** como la bolsa que impide preposicionar los tres nodos
a la vez: `presupuesto = 0,5 × (suma de los tres objetivos máximos) / 26`, sin arrastre.

## 4. Cómo se mide

* **Endpoint:** `L*` adimensional, el mismo del gate anterior, sin cambios.
* **Acción:** el contrato por nodo que ya existe, `Box([op3, op5, op9, turno])`.
* **Comparadores, en orden y todos antes de cualquier aprendiz:** mejor postura fija (rejilla),
  regla de umbral sobre la señal observable, y **el clarividente por tape** como techo.
* **Control fijo elegido sólo en train**, evaluado en test, con bloques disjuntos.
* **Semillas de desarrollo ya quemadas** (`8600001–8600024`). **No se abre ningún bloque virgen.**

## 5. Falsadores, y por qué cada uno puede fallar

| id | exige | por qué puede fallar |
|---|---|---|
| `f1_faithful_cell_reproduces_today` | en CONTROL_FIEL el hueco < 0,01 | si aparece headroom donde la caducidad es inerte y el presupuesto ilimitado, el instrumento lo fabricó |
| `f2_shelf_life_expires_nothing_when_inert` | unidades caducadas = 0 con 156 sem | una vida útil inerte que retira stock es un defecto de implementación |
| `f3_budget_binds_when_tight` | gasto ≤ presupuesto en todo periodo, y la restricción se activa | un presupuesto que nunca ata no es contención |
| `f4_expiry_costs_what_it_removed` | lo caducado sale del contenedor **y** sigue contado como gastado | si caducar devolviera presupuesto, preposicionar volvería a ser gratis |
| `f5_mass_conserves` | residual de masa < 1e-6 relativo | el defecto que retractamos hoy destruía stock en silencio |
| `f6_clairvoyant_gap_is_material` | `LCB95 ≥ 0,01` en alguna celda | **puede fallar en las cuatro**, y ése sería el resultado |
| `f7_gap_survives_the_interaction_null` | `p < 0,05` contra el nulo aditivo con residuos permutados | un mínimo sobre muchas opciones ruidosas está sesgado a la baja; el techo de doce tapes ya murió por esto |
| `f8_observable_rule_is_measured` | se reporta la conversión de una regla observable | un techo clarividente sin intento de conversión no autoriza nada |

## 6. Reglas de lectura, en orden

1. **Primero el control fiel.** Si `f1` o `f2` fallan → `BLOCKED_INSTRUMENT` y **nada más se lee**.
2. Si el control se comporta, se lee el mapa 2 × 2 completo, incluidas las celdas de un solo
   mecanismo.
3. `f6` **y** `f7` deben pasar juntos para declarar headroom. `f6` solo, sin nulo, no es nada — es
   exactamente el error que mató el techo de hoy.
4. Sin `f6`+`f7`, el veredicto es `NO_SEQUENTIAL_HEADROOM_UNDER_BUDGET_AND_EXPIRY`, y **eso cierra
   la última razón física que teníamos para esperar valor de memoria en esta familia**.

**Esto no autoriza entrenar nada.** Autoriza medir si existe algo que un aprendiz pudiera usar.
