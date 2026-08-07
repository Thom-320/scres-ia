# Enmienda 1 al claim freeze Q1/Q2 — 7 de agosto de 2026

**Predecesor:** `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` (commit `550a253`)
**Motivo:** una quinta auditoría externa revisó dos censos repo-wide y encontró tres huecos que el
freeze no cubría. Verifiqué cada afirmación contra los artefactos; las cinco comprobables son
correctas. Esta enmienda **no edita el predecesor** — lo supersede fila por fila, según su §8.

**Filas superseded:** §1 (frase del claim), §6 (argumento del NO-GO), §0 (alcance del inventario).
**Filas intactas:** §2, §3, §4, §5, §7 (salvo el ítem 1, reformulado abajo).

---

## E1 · El claim de Q1 se parte en dos niveles

### Lo que decía el predecesor (§1), y por qué se supersede

> «…el componente que reproduce el atributo *history-dependent* del SCL no es una familia de
> aproximadores, sino la conservación de estado de búsqueda entre contextos.»

La frase es correcta **en desarrollo** y demasiado ancha como titular. Un lector que solo lea el
claim se lleva «la retención está confirmada». No lo está: lo confirmado prospectivamente es un
carrier concreto, y **la neurona falla el contrafactual fuerte** que ese carrier pasa.

### Claim de Q1, versión congelada

> **En desarrollo, sobre un benchmark de 15 métodos, los seis brazos que conservan estado ocupan los
> seis primeros puestos, y ninguna clase de aproximador explica ese ordenamiento. En la única
> confirmación prospectiva vigente, sin embargo, no toda memoria transfiere: sólo un carrier de
> estadísticas factorizadas por nivel (`ucb1_transfer`) supera a la vez el arranque en frío y un
> replay state-blind de sus propias marginales. La memoria neuronal estudiada supera el arranque en
> frío y **falla** ese segundo contrafactual.**

Etiquetas:

```
Q1_DEVELOPMENT      RETENTION_DOMINATES_THE_LADDER          (A1, tapes quemados, no adjudica)
Q1_CONFIRMED        GRID_TRANSFER_CONFIRMED__UCB1           (A2, bloque reservado)
Q1_REFUTED_LOCALLY  NEURAL_MEMORY_FAILS_MARGINAL_REPLAY     (A2, UCB95 −0,00484)
```

La formulación publicable, que es más fuerte que «memoria > arquitectura»:

> **El estado retenido es necesario en el contexto de búsqueda evaluado, pero no suficiente: su
> estructura y la regla de decisión que lo consume determinan si la transferencia sobrevive a un
> contrafactual más exigente.**

### §7, ítem 1 — reformulado

Sustituir *«`ucb1_transfer` transfiere…»* por:

1. En transferencia prospectiva 288 → 4.608, `ucb1_transfer` es el **único** de cuatro familias que
   supera cold start **y** su replay marginal state-blind. Las otras tres —incluida `neuron`—
   superan cold start y pierden contra su propio replay marginal (A2, n=60).

Las tablas numéricas de §1 del predecesor no cambian: ya contenían este contraste. Lo que cambia es
que **deja de ser un detalle de la tabla y pasa a ser el titular**.

---

## E2 · Evidencia off-HEAD que el freeze omitía

El predecesor sólo catalogó artefactos de la rama actual. Un canon que omite una confirmación
positiva en otra rama no es un canon. Se añaden tres, con su frontera de claim.

### A10 · `garrido_h2_h3_confirmation_v1` — confirmación positiva, sin claim de aprendiz

Rama `codex/paper-b-retained-v5` (`4d446d3`).

| fichero | sha256[:16] |
|---|---|
| `contracts/garrido_h2_h3_confirmation_v1.json` | `1d3c80bd48feac4c` |
| `results/garrido_h2_h3_confirmation_v1/result.json` | `bc375d3021b64d10` |
| `results/garrido_h2_h3_confirmation_v1/completion_receipt.json` | `d4305bcf6bf5209d` |

`completion_receipt`: `global_confirmation_pass: true`, `confirmation_roots_opened: true`,
**`development_roots_opened: false`**, cuatro SHA de shards fuente. El `result_sha256` del recibo
coincide con el hash del propio `result.json` — la cadena de custodia cierra.

**Frontera, tomada del propio contrato:**

```
learners_authorized            = false
architectural_claim_authorized = false
```

> **Permite afirmar:** efectos direccionales de buffers y turnos en R1r/R2r/R3, seis paneles, Holm.
> **No permite afirmar:** nada sobre aprendices, arquitecturas ni imitación de SCL.

Es un activo de **validación física / reproducción de la tesis**. No fortalece ni contradice el null
neural, y agruparlo con la transferencia mezclaría dos preguntas. Etiqueta:
`CONFIRMED_RESOURCE_INTERVENTION_EFFECT · NO_LEARNER_CLAIM · NO_ARCHITECTURE_CLAIM`.

### A11 · `q_r1/successor_confirmation_v1` — ejecución prospectiva con STOP

Rama `codex/q-r1-retained-belief-discovery` (`fb94b1d`),
`results/q_r1/successor_confirmation_v1/adjudication.json`, sha `c643aac67ff5712f`.

```
verdict                              = STOP_REPAIRED_Q_R1_NO_RETAINED_INFORMATION_PASS
incremental_learned_residual_established = false
historical_splice_can_rescue             = false
learner_training_authorized              = false
```

Por estimando:

| estimando | `visible_pass` | `complete_cohort_pass` | `mechanism_pass` | `pass_retained_information_value` |
|---|---|---|---|---|
| `prefix_natural_replanning` | True | True | **False** | **False** |
| `sustained_control` | True | True | **False** | **False** |

> **Q-R1 tuvo una ejecución prospectiva; no produjo una confirmación positiva de valor retenido.**

Etiqueta: `PROSPECTIVE_EXECUTION_COMPLETED · COMPOUND_VERDICT_STOP · NO_POSITIVE_CONFIRMATORY_CLAIM`.

### A12 · `g3_obs_conversion_v2` — desarrollo no promotable

`results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json` (commit `167cd31`),
sha `951f5d7c38f11f6a`, status `SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE`.

La corrida usó el **bloque de semillas del v2** (`7800001–7800140`) y el **margen `lost_orders` del
v2** (0,5), pero fue sellada contra el contrato **legacy**, que declara otro bloque
(`5200001–5200016`) y otro margen (0,25). No conforma con ninguno de los dos.

`prohibited_claims`, literal del artefacto — los cinco:

1. que la corrida se ejecutó o selló bajo el contrato v2;
2. que es plenamente confirmatoria bajo el contrato legacy;
3. que es una confirmación v2 virgen o independiente;
4. que el runner original ejecutó el falsador f2 completo del v2;
5. **que este artefacto confirma una prima neural.**

`permitted_claims` permite citarlo **como evidencia de desarrollo con esta limitación de alcance**.

> **No está bloqueado por papeleo.** El contrato define qué resultado habría contado, sobre qué
> bloque y bajo qué márgenes. Sin conformidad no se sabe si el análisis corresponde a una hipótesis
> congelada o a un híbrido accidental. Se repara con un experimento prospectivo nuevo, no cambiando
> una ruta en un JSON.

---

## E3 · El NO-GO de C1 se refuerza con un argumento que el freeze no tenía

El predecesor (§6) fundó el NO-GO en que **el estimando no está definido** (`worst_product_fill`
nunca persistido). Sigue siendo el bloqueador principal. Se añaden dos hechos de custodia:

**El único bloque registrado como nunca abierto no está disponible, y además no serviría.**

```
contracts/g3a_asymmetric_claimants_v2.json  (sha 20952a3bff3c7b5b)
  development_block: {start: 7700001, end: 7700120, status: RESERVED_NOT_OPENED}
```

Dos razones independientes:

1. está **reservado para G3a**, no es una bolsa genérica de 120 semillas;
2. está declarado `development_block` — **no es un bloque confirmatorio**. Aunque se liberara, abrirlo
   para C1 no produciría una confirmación.

**Y el inventario no permite declarar bloques nuevos.**

```
research/seed_custody_registry.json  (sha 5e6eb180c37e6803)
  status = BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED
```

Lo escaso no son las semillas. Es un bloque **predeclarado, auditado contra todas las ramas, libre
de colisiones conocidas y abierto bajo autorización**. Con 94 ramas sin inventariar, no se sabe qué
rangos ya se gastaron.

`C1_VIRGIN_BLOCK = NO-GO` se mantiene, ahora por tres vías independientes: estimando indefinido,
bloque inexistente, inventario incompleto. Las precondiciones C1-A…C1-I del predecesor siguen
vigentes sin cambios.

---

## E4 · Reglas de higiene que esta enmienda congela

### R1 · Tipo de corrida ≠ veredicto científico

```
run_role: CONFIRMATION            NO implica claim confirmado
claim_status: PROSPECTIVE_...     NO implica resultado positivo
```

A11 es el caso testigo: `PROSPECTIVE_CONFIRMATION` conviviendo con
`pass_retained_information_value: false`. Ningún censo puede contar por `run_role` ni por
`claim_status`. **La unidad de conteo es la adjudicación terminal**, y si un artefacto no la tiene,
no cuenta.

### R2 · La unidad de deduplicación no es `result.json`

Es la tupla:

```
(contract_sha256, execution_commit, seed_or_tape_block, estimand, endpoint)
```

Cuatrocientos intervalos positivos derivados de dos familias de tapes son contabilidad, no
replicación. Los intervalos de `ALZHEIMER_EFFECT_*`, `H1_SUPPORTED` y `H3_PRIME_SUSTAINED` no son
evidencias independientes por tener nombres de fichero distintos.

### R3 · Ninguna cifra de deriva se mantiene a mano

`main` vs rama: el freeze dijo 790, la tabla canónica dice 786, la auditoría dijo 793, el valor real
al escribir esto es **794/8**. Cualquier cifra escrita a mano nace caducada. Se cita el SHA de la
rama y `main`; la deriva se calcula al leer, o no se cita.

### R4 · Un guardarraíl no se retira después de ver quién gana

Si Garrido responde que el servicio promedio es doctrinalmente aceptable, eso **motiva un contrato
nuevo**; no rescata retroactivamente un brazo que ganó abandonando reclamantes. El guardarraíl se
congeló antes de ver los resultados. Retirarlo ahora convertiría la validación de dominio en una
herramienta de selección.

### R5 · Más folds no son más mundos

Pasar de 5 a 10 unidades mueve el crítico t de 2,776 a 2,262: **−18,5 %**, no la mitad. Y sólo baja
la anchura de verdad si las unidades nuevas son **replicaciones independientes**, no particiones
correlacionadas de la misma superficie. Un intervalo que no cierra con validación cruzada necesita
datos nuevos y un preregistro nuevo.

---

## E5 · Cierres

### CVaR — negativo informativo, cerrado

`results/citable_risk_attitudes/result.json` (sha `91b0e887c6982db8`):
`H_regime_cvar = 0,048226736392119496`, **idéntico en las tres α** — con seis regímenes todas
colapsan al peor. Contra la barra congelada de `0,05` está **por debajo**. Las actitudes de aversión
al riesgo mueven el headroom hacia cero, no hacia la barra.

No es una campaña a un empujón de distancia. Es un resultado negativo informativo, y `0,0482` no se
convierte retrospectivamente en `0,05` por una respuesta posterior de Garrido. Una respuesta suya
sobre doctrina de peor teatro **define un estimando prospectivo nuevo**; no rescata éste.

### Líneas que no se persiguen hasta después del manuscrito

| línea | razón |
|---|---|
| CVaR | negativo bajo barra congelada |
| CD premium «con más folds» | R5: requiere unidades independientes |
| DMLPA `nhead4` / `1layer` | optimizar arquitectura antes de que exista prima neural estable |
| G3c dwell | la estimación primaria tiene signo negativo |
| B5 / B7 vía retirada de fairness | R4: sólo contrato nuevo, nunca promoción retroactiva |
| `event_triggered_env` | no refutado, pero sustancialmente socavado por la invarianza del óptimo |

### `PROMISING_LANES_REGISTRY.md`

1.016 líneas append-only que terminan antes de `track_b_nonneural`, `step3_pooled` y la tabla
canónica. **Se conserva como registro histórico y deja de intervenir en decisiones.** Su mezcla de
resultados vigentes, superseded y señales de dos semillas es un mecanismo eficiente para resucitar
claims muertos. La instrucción permanente de «nunca perder un lane» se cumple archivándolo, no
consultándolo.

---

## E6 · Orden de trabajo

1. **El manuscrito.** Sus artefactos ya están congelados con SHA; no espera al registro.
2. **`research/evidence_registry.jsonl`** en paralelo — una fila por experimento científico, clave
   R2, con `terminal_adjudication`, `evidence_grade`, `claim_boundary`, `superseded_by`,
   `promotable`. De ahí se generan la tabla canónica, el inventario de semillas y la tabla de
   supersesiones. Es la infraestructura correcta; no es prerrequisito de escribir.
3. **Importar A10 y A11 al canon** con sus fronteras, sin fusionar sus ramas.
4. **Instrumento del paso 3:** fill por producto, endpoint obligatorio, actuador DDMRP, tests con
   producto sacrificado. La estimación de ~5 h cubre `worst_product_fill`; **DDMRP costará más**.
5. **Cinco preguntas a Garrido**, no catorce: `sumBt`; si existe piso vinculante por peor
   producto/teatro o la doctrina se evalúa por resiliencia promedio; capacidad y tiempo físico de
   Op11; deadlines permanentes con autoridad de admisión/abandono (R09); dos rutas reales + flota
   finita + aviso predespacho (R03, licencia de dominio para Program L).
6. **Program L** después del manuscrito y oracle-first. Es el mejor candidato *nuevo*, no un claim
   cercano: le faltan validación de dominio, runner, liveness, gate de headroom, igualdad de
   recursos y frontera clásica antes de cualquier aprendiz.

---

## Custodia

Esta enmienda es datada y no se edita en sitio. Una corrección se emite como
`…_ENMIENDA_2.md`, declarando qué fila supersede y por qué.

Sigue vigente la regla del predecesor: **una cifra sin ruta y SHA no entra en el manuscrito, en una
diapositiva, ni en una reunión con Garrido.**
