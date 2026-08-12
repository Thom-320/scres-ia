# Enmienda — el nombre defendible al lado del sellado, y qué lenguaje queda prohibido

**Fecha:** 2026-08-12 · **Origen:** barrido sobre 264 `result.json` + cinco revisiones externas
**No se renombra ni se edita ningún artefacto sellado.** Esta tabla es la autoridad de lectura:
el `claim_status` sellado conserva la historia; la columna «defendible» es lo que puede escribirse
en un manuscrito.

---

## 1. El patrón, ya con ocho casos

Un sustantivo concede una propiedad que el experimento no midió. Después el nombre se convierte en
premisa del siguiente runner, de la siguiente reunión y del abstract.

| sellado | qué mide en realidad | defendible en un manuscrito |
|---|---|---|
| `GSA_CONFIRMED_ON_VIRGIN_BLOCK` | `all_passed: false`, dos falsadores en rojo; uno dice que el estimador **no puede** devolver headroom no positivo | `GSA_ONE_BIT_CALENDAR_CHOICE__ESTIMATOR_COULD_NOT_FAIL` |
| `PERFECT_SUBSTITUTES_EVERYWHERE_ON_THE_SCREENED_GRID` | `both == buffer_only` en 18/18, pero `shifts_only == buffer_only` sólo en **8/18** | `ONE_WAY_REDUNDANCY_ONLY` |
| `train_cell_mean_comparator` como «techo» | comparador de medias de celda, superado por brazos neuronales en las **4** corridas de Gate B | `TRAIN_CELL_MEAN_COMPARATOR` |
| `strong_mpc` | `paced_policy(α,β,γ)`: 0 evaluaciones de candidato, 0 llamadas al simulador en 320 decisiones | `PACED_LINEAR_RULE` |
| `H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE` | coste de un fixture de cronometraje; `engineering_only`, `learner_trained: false` | `PLANNER_LATENCY_MEASURED__NO_QUALITY_MEASURED` |
| `belief_mpc_policy` / `oracle_model_mpc_policy` | filtro + `_myopic_split`, **un periodo, sin horizonte** | `first_order_belief_myopic` / `true_model_belief_myopic` |
| `SENSITIVITY_PREMIUM_HOLDS_ON_LEGACY_SURFACE` | `all_passed: false`, f4 falla; la re-adjudicación retira la prima | `NO_NEURAL_PREMIUM_ON_LEGACY_RET_SURFACE` |
| `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` | bate al *lookahead*, pero `ucb1_transfer` (0,045023) la bate a ella (0,052033) | `NEURON_BEATS_LOOKAHEAD_NOT_THE_LADDER` |
| `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | corridas hasta el 1 % del óptimo en una búsqueda externa sobre una superficie precomputada | `RETAINED_SEARCH_STATE_REDUCES_RUNS_TO_OPTIMUM` |
| `H2_SUPPORTED_LEARNING_CURVE` | OLS sobre **6** posiciones ordinales, por réplica; **22/120** pendientes negativas; reanálisis | `MEMORY_ADVANTAGE_GROWS_ACROSS_ORDERED_CONTEXTS__DEVELOPMENT` |

## 2. Lenguaje prohibido, y su sustituto

| prohibido | por qué | se dice |
|---|---|---|
| «bloque virgen» | el registro **se declara incompleto**; la ausencia de colisión no prueba virginidad | «bloque sin colisión conocida en el registro disponible» |
| «el mejor comparador no neuronal» | se probó una lista finita | «el mejor comparador no neuronal **probado** de la clase preregistrada» |
| «techo», «cota», «óptimo» | sólo si un falsador comprueba que ningún brazo lo supera | «comparador», y se nombra la familia que acota |
| «MPC», «planificador», «experto» | sólo con evaluaciones de candidato o llamadas al simulador **contadas** | «regla», «filtro», «política miope» |
| «confirmado» | sólo con grado derivado y custodia limpia | «prospectivo, sin colisión conocida» |
| «el efecto Alzheimer de la cadena» | se midió coste de reinicio de un **buscador**, no olvido organizacional | «coste de reiniciar el estado de búsqueda» |
| «el OFAT de la tesis» | es una reimplementación dentro del mismo experimento | «el brazo OFAT, reimplementado con el mismo presupuesto» |
| «prima neural» a secas | son cuatro estimandos distintos | «prima de **predicción**» / «de **decisión**» / «de **amortización**» / «de **portador**» |

## 3. Las dos reglas nuevas

**R1 — el nombre necesita su falsador.** Nada se llama techo, cota, óptimo, experto, MPC,
planificador, confirmado ni virgen sin un falsador que compruebe esa propiedad concreta y que pueda
fallar.

**R2 — un número citado debe ser reconstruible.** Todo número que aparezca en un documento debe
poder señalarse al campo exacto del artefacto que lo produce. Si no se puede señalar el campo, no se
cita.

R2 nace de que cité `+0,0136 [+0,0124]` durante dos días, lo llevé hasta un contrato congelado y
hasta una respuesta a un revisor externo, y **no aparece en ningún `result.json` del árbol**.
Ninguno de los falsadores existentes podía detectarlo: todos miran corridas, ninguno mira
documentos.

## 4. Lo que esto NO hace

No cambia ningún resultado. Los números de Gate B siguen siendo `+0,1081 [+0,0601, +0,1561]` y
`+0,1487 [+0,1069, +0,1905]`, y su grado corregido es `PROSPECTIVE`. Lo que cambia es qué se puede
escribir alrededor de ellos.
