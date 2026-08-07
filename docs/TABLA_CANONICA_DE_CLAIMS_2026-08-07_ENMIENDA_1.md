# Enmienda 1 a la tabla canónica de claims — 7 de agosto de 2026

Sucede a `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, que **no se edita**. Todo lo de abajo se
midió después de congelarla.

---

## 1. `H2` adjudicada — las cuatro hipótesis del borrador tienen veredicto

`results/manuscript/h2_learning_curve/result.json` (sello `74b75141241ba763…`),
**`H2_SUPPORTED_LEARNING_CURVE`**, cinco falsadores pasan, 120 réplicas, normalizador de prefijo,
cero semillas.

**Estimando: la pendiente, no el nivel.** Una ventaja grande pero plana sostiene `H4`, que ya
estaba medida; `H2` exige que la ventaja **crezca** con el número de disrupciones sucesivas.

| contexto (ordinal) | ventaja media (reinicio − memoria) |
|---|---:|
| 1 · `R1r` | +0,00000 |
| 2 · `R2r` | +0,28275 |
| 3 · `R1r+R2r` | +0,19052 |
| 4 · `R1r\|esc` | +0,22111 |
| 5 · `R2r\|esc` | +0,31709 |
| 6 · `R1r+R2r\|esc` | +0,26869 |

**Pendiente primaria +0,042201 [+0,034664, +0,049922].**
**Control nulo (aleatorio − OFAT): −0,005088 [−0,015570, +0,005658]** — cruza cero, así que la
tendencia **no** es la dificultad creciente de los contextos escalados. El cero exacto del primer
contexto es estructural: sin nada que arrastrar, memoria y reinicio son el mismo brazo.

### Estado de las cuatro

| | redacción original | reformulación declarada |
|---|---|---|
| **H1** recuperación | **SOSTENIDA** +126,0 h [+98,4, +154,5], por **absorción** (875/960 vs 755/960) | H1′ servicio perdido: SOSTENIDA |
| **H2** curva de aprendizaje | **SOSTENIDA** pendiente +0,0422 [+0,0347, +0,0499] | — |
| **H3** varianza entre intensidades | **NO SOSTENIDA** — signo contrario, IC cruza cero, con estimando presente | H3′ varianza del coste de búsqueda: SOSTENIDA |
| **H4** dependencia de `L_{t−1}` | **medida** +0,06070 [+0,04556] | — |

**Tres de cuatro se sostienen.** No se redondea a cuatro.

---

## 2. `H4` — la cifra del borrador está **prohibida**

El borrador v.0 cita **`+7,90 corridas [+6,88, +8,93]`**. Es `memory_vs_reset` sobre
`runs_to_within_1pct` bajo el normalizador **oráculo**, que ve la superficie no ejecutada. Está en
la lista de retiradas del congelamiento.

| | prohibido | canónico |
|---|---|---|
| primaria | — | **AUC +0,06070 [+0,04556, +0,08020]** |
| secundaria, censurada | 7,90 · 5,43 | **5,83** corridas [+4,44, +7,31], siempre etiquetada |

Es sustitución obligatoria, no preferencia editorial. Es fácil pasarla por alto porque el borrador
cita el número **sin nombrar su normalizador**.

---

## 3. `neuron_memory` vs `ofat_transfer` — **prohibido escribir «excluye el cero»**

`results/ofat_lcb_reconciliation/result.json` (sello `a35bb6ec721d6838…`),
**`OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE`**, B = 50.000, 40 semillas de remuestreo.

Los dos artefactos sellados puntúan **arreglos idénticos** (`f1` PASA) y dan signos opuestos:
`−2,761e−05` en `search_ladder_v2_ordered`, `+3,565e−05` en `search_ladder_v5`, con media común
`+0,01071`. **La cota inferior sale positiva en el 65 % de las semillas.** No es el dato; es el
sorteo.

> Se cita así: *indistinguibles en AUC de arrepentimiento (media +0,01071; la cota inferior cae a
> ambos lados del cero según el remuestreo).* Y se citan **las dos** cotas selladas.

**Corrección que me corresponde:** una auditoría externa citó `−0,0000276` y yo dije que se
equivocaba. Estaba citando el otro artefacto sellado. El equivocado fui yo.

Por contraste, `neuron_memory` vs `ucb1_transfer` es **estable**: la cota es positiva en el **0 %**
de las semillas. «La neurona no bate a UCB1 con transferencia» es robusto.

---

## 4. Lane nueva: `gsa_resilience_only` **CALIFICA**

`results/gsa_resilience_only/result.json` (sello `759c2955cccf4062…`), cinco falsadores, 600
cintas, cero semillas. H_obs positivo con cero excluido en **tres bloques independientes**
(+0,01307 / +0,01136 / +0,01001), **η 0,78–0,91**, y **+0,069…+0,073 sobre un placebo desinformado
que la corrida histórica no tenía**. `f2` descarta que la ganancia se compre atendiendo menos
(correlaciones −0,180 / −0,170 / +0,099).

Coste distributivo, reportado y no suavizado: el peor CSSU pierde **−0,14 / −0,125 / −0,1225** de
fill. Se registra como **decisión del PI** (2026-08-07: la medida es la resiliencia), no como
hallazgo del runner. **Desarrollo**: no autoriza entrenar, autoriza preregistrar.

---

## 5. Paso 3 — alcance estrechado y DDMRP adjudicado

`results/step3_expressiveness/result.json`: el contrato agregado tiene **un solo reclamante**
(141 pedidos, `cssu_destination = None`, sin atributo de producto), así que `worst_product_fill`
**es** `flow_fill_rate` y el guardarraíl preregistrado **no es expresable**. `NO_STRUCTURED_CONTROLLER_CONVERTS`
vale **sólo** como *«en el contrato agregado de un solo reclamante, puntuado con
`ret_excel_full_ledger`»*, y **no pasó** el screen que su propio preregistro definió.

**DDMRP queda adjudicado con `results/buffer_saturation_diagnostic/`**: ×10 por encima de la
referencia mueve la métrica **exactamente 0,000000** con `saturated_upward: true` en los tres nodos,
mientras bajar sí duele. La postura proyectada es equivalente en métrica a la sin proyectar. **En
esta cadena DDMRP degenera a una constante en el techo, y el techo cae en una región plana.** El
negativo es adjudicable; un positivo necesitaría una cadena cuyo techo esté por encima de su
objetivo.

---

## 6. El arnés de Track B **no es determinista** — y mi sospecha queda refutada

`results/determinism_diagnostic/result.json`, **`DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT`**:

| configuración | delta entre réplicas |
|---|---:|
| 8 envs, hilos por defecto | 0,472 |
| 8 envs, **hilos = 1** | 1,620 |
| **1 env**, hilos = 1 | 1,539 |

Ni el hilado de torch ni los subprocesos. La configuración `C` usa `DummyVecEnv`, que **sí** siembra,
y tampoco reproduce. **La sospecha preregistrada —`make_vec` descartando su `seed`— está
refutada** y así se reporta.

Consecuencias vinculantes: la semilla **no es unidad de réplica** en este arnés; toda lógica
pareada por semilla es nula; y **la prima neural de `track_b_v1` (+1,44 a +2,18) cae dentro de una
banda de ±2,4**, sin intervalo en su artefacto. No es «sin confirmar»: **el instrumento no la
resuelve**, y ninguna cantidad de semillas lo arregla.

---

## 7. Un artefacto descartado, y por qué

El primer intento de `split_v1` produjo **1.296 filas estáticas en `split_v1` y 18 dinámicas
—`ddmrp`, `replay_mpc`, `greedy_pi`— en `aggregate`**: en macOS `ProcessPoolExecutor` hace *spawn*,
el worker reimporta el módulo y el global vuelve a su valor por defecto. Se detectó sólo porque la
topología se escribe en cada fila.

**Los dos shards se borraron en vez de reinterpretarse.** La topología viaja ahora en el payload y
el runner **se niega a escribir un artefacto mixto**. Los cuatro shards corren de nuevo.
