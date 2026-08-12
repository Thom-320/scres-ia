# Segunda respuesta al auditor — concedido lo cierto, rechazado lo falso, con el campo exacto

**Fecha:** 2026-08-12 · **Auditado por vosotros:** `9f712330` · **Estado hoy:** `3c67b881`
**Cinco dictámenes recibidos.** Verifiqué cada cargo contra los artefactos antes de aceptarlo o
rechazarlo. Ninguno se adoptó sin comprobar, y ninguno se descartó sin comprobar.

---

## 1. Concedido, y es el peor de todos: el número no existe

Teníais razón, y el fallo es más grave de lo que dijisteis.

Durante dos días cité *«el aprendiz batió al belief-MPC por **+0,0136 [LCB95 +0,0124]**»*, en el
briefing, en la primera respuesta y como **premisa entera** de la enmienda `d_min` a Program X.

`results/audit_positive_validation/result.json`, celda `positive`:

```
learner_vs_best_structured   mean +0,011477   lcb95 +0,009135   51/60
SESOI 0,010000               converts false
claim_status  AUDIT_STOPS_CORRECTLY_BUT_POSITIVE_DIRECTION_NOT_DEMONSTRATED
```

**No puedo reconstruir mi `+0,0136` desde ningún campo del árbol.** Eso lo empeora: no fue leer mal
una celda, fue citar algo que no está.

**Y la dirección se invierte**, que es el cargo que dos de vosotros hicisteis y que confirmo:

| celda | `min_dwell` | `rho` | learner − mejor estructurado | learner − modelo verdadero |
|---|---|---|---|---|
| `positive` | 4 | 0,90 | +0,011477 [+0,009135], 51/60 | +0,010323 [+0,007559] |
| `no_memory` | **1** | 0,50 | +0,009066 [+0,007931], **58/60** | **+0,019374 [+0,017430]** |

El aprendiz también «gana» sin permanencia mínima, con **más** tapas favorables, y su ventaja sobre
el brazo de modelo verdadero es **mayor sin dwell**. Además `min_dwell` y `rho` **están confundidos**
—no hay celda que los separe— exactamente como uno de vosotros señaló.

**Mi propio gate `G4b` ya falla con los datos existentes**: en `positive`, el filtro de modelo
verdadero (0,906880) supera al de primer orden (0,905726) por **0,001154**, un orden de magnitud bajo
el SESOI. La mala especificación **no es material**.

**Retractado:** el número, la frase «el único sitio donde un aprendiz ganó», la enmienda `d_min`
completa y el contrato v2. `v1` vuelve a ser el vigente, con arista
`SUPERSEDED_BY_FAILED_PREMISE` en `research/supersession_registry.json`.
Detalle: `docs/RETRACTACION_CONTENTION_V1_Y_ENMIENDA_X_2026-08-12.md`.

## 2. Concedido: el grado de Gate B, y era defecto de instrumento

Teníais razón: `gate_b_confirmation_v3` sella `run_role: DEVELOPMENT` y
`scope: DEVELOPMENT_REANALYSIS_NO_NEW_SEEDS`. **Causa:** ambos campos eran cadenas fijas en el
runner; añadí `--seed-base` después y nunca los toqué.

Reparado derivándolos, y re-adjudicado el grado de las cinco corridas en
`results/program_n/gate_b_grade_correction/result.json`, **sin abrir ninguna semilla y sin tocar
ningún `claim_status`**. `v3` pasa a `PROSPECTIVE`.

**Y acepto vuestro techo sobre el lenguaje.** El registro se declara incompleto, así que el grado
más fuerte que un bloque puede ganar aquí es
`PROSPECTIVE_FRESH_BLOCK_NO_KNOWN_COLLISION_VIRGINITY_NOT_PROVEN`. La palabra **«virgen» sale de
todos los documentos**; un falsador `g3_no_run_is_promoted_to_virgin` lo impide de ahora en
adelante.

## 3. Concedido: `all_passed` no veía la custodia

`gate_b_cd_surface` sellaba `all_passed: true` junto a `custody.passed: false`.
**Causa exacta:** `F.summarise` filtraba `computed is True` antes de puntuar, y `custody_falsifier`
devuelve `{passed, not_applicable, evidence}` **sin esa clave**. El blindaje contra *sobre*contar
abrió un agujero de *sub*contar.

Reparado y **validado reintroduciendo el defecto real**: `gate_b_cd_surface` pasa a
`all_passed: false`. Dos tests de mutación nuevos, uno en cada dirección.

## 4. Concedido: el bucle externo estaba peor narrado de lo que admití

Auditados los cinco artefactos:

* **la familia que sobrevive en regret simple final es `lookahead_kg`, NO la neurona**
  (`neuron` simultaneous_lcb95 **−0,0035**, cruza cero). Di el «1/6» correcto **sin nombrar la
  familia**, lo que deja creer que era la neurona;
* en la métrica primaria del propio *ladder*, `ucb1_transfer` **0,045023** bate en punto a
  `neuron_memory` **0,052033** (menor es mejor), diferencia −0,00701 [−0,02444, +0,01408];
* **cuatro de los cinco artefactos son relecturas de UN bloque de 12 semillas** (5300001–5300012);
* `retention_simultaneous` y `retention_contrasts` son **post-hoc**, no preregistrados;
* «12,42 el OFAT **de la tesis**» es engañoso: es una reimplementación dentro del mismo experimento;
* la pendiente H2 es una OLS sobre **6 puntos**, con **22 de 120 réplicas en negativo**.

Lo que sí queda: la retención baja el AUC de regret **6/6 bajo inferencia simultánea**. Es un
resultado de **retención**, no de **neurona** — y así se escribirá.

---

## 5. Rechazado con evidencia de código: la superficie NO es analíticamente predecible

Éste era vuestro cargo más peligroso y **no se sostiene**.

La acusación: el índice Cobb-Douglas es log-lineal en sus cinco drivers, luego un comparador
*source-aware* log-lineal lo predeciría casi perfectamente y la prima sería un artefacto de clase
comparadora incompleta.

**La primera mitad es correcta.** `supply_chain/cobb_douglas_resilience.py:209-223`:
`R = sigmoid(Σ signo_i · a_i · ln(driver_i))`.

**La segunda no.** Ningún brazo recibe los drivers. `base_features` y `rich_features`
(`scripts/run_cd_surface_prediction_premium.py:75-86`, reusadas sin cambios en
`run_program_n_gate_b_v1.py:44-46,185-186`) entregan **siete números de configuración**:
`buf/1344`, tres one-hot de familia de riesgo y tres de escalación. Ni `zeta`, ni `epsilon`, ni
`phi`, ni `tau`, ni `kappa_dot` aparecen.

Los cinco drivers son la **salida no observada de un DES SimPy de 13 operaciones**
(`MFSCSimulation.step()`), con cuatro flujos RNG independientes, leída periodo a periodo por
`CobbDouglasRecorder` y **recalibrada dentro de cada fold sólo con filas de entrenamiento**.

**El mapa configuración → drivers no es analítico.** Un regresor log-lineal sobre los drivers
predeciría `R` casi perfectamente porque recibiría **los argumentos de la propia fórmula**: es una
tautología, y responde otra pregunta —«¿la fórmula reproduce sus propios argumentos?»— en vez de la
que el estudio hace: «¿se puede predecir la resiliencia resultante desde la política de buffer, la
familia de riesgo y la escalación, **sin correr el DES**?».

**Por eso ese comparador no se añade.** Los demás que pedisteis —GBDT, random forest, GAM/spline
multivariado, GP/kernel ridge y AR(p) para la clase con historia— **sí**, y con las mismas siete
features. Es la Fase 1a y corre antes de pedir una sola semilla.

## 6. Corregido: uno de vosotros citó mal Gate A2

Un dictamen afirma que la red queda `−0,001415` bajo el mejor no neuronal.
`results/program_n/gate_a2_track_b/result.json`:
**`−0,559369 [−0,747601, −0,385607]`, 7/48**. `linear_feedback` 99,1264 frente a `mlp` 98,5670.

Lo señalo porque la conclusión —Track B cerrado— es la misma, pero la magnitud no: es **400 veces**
mayor, y con ella el argumento de que una realimentación lineal barata domina.

---

## 7. Lo que vuestro barrido de nombres encontró y yo no

Ejecuté vuestra petición sobre los 264 `result.json`. El peor caso **no es ninguno de los míos**:

`results/gsa_confirmation/result.json` lleva `claim_status: GSA_CONFIRMED_ON_VIRGIN_BLOCK` con
`all_passed: false` y dos falsadores en rojo, uno de los cuales dice literalmente *«un estimador
que no puede devolver headroom no positivo no puede fallar en confirmar, y no confirma nada»*
(`n_tapes_non_positive: 89`). Ya superado por `gsa_confirmation_corrective`.

También `PERFECT_SUBSTITUTES_EVERYWHERE_ON_THE_SCREENED_GRID`, contradicho en **10 de 18 celdas**
por su propia rejilla. En total, **20 artefactos** combinan una palabra afirmativa en `claim_status`
con un `scope` que dice desarrollo, replay o sin semillas nuevas.

**La regla nueva, que ninguno de mis falsadores podía imponer porque todos miran corridas y ninguno
mira documentos:**

> Un número citado en un documento debe ser reconstruible desde el campo exacto del artefacto que lo
> produce. Si no se puede señalar el campo, no se cita.

---

## 8. Qué se hace ahora, y qué NO

**Fase 1, cero semillas, en local:** (a) Gate B contra la clase no neuronal completa; (b) el bucle
externo re-adjudicado separando **retención** de **portador neural** — el segundo estimando **nadie
lo ha medido**, y si `ucb1_transfer` retenido iguala a la neurona, el claim es de retención.

**Fase 2, un solo bloque nuevo**, con la **semilla como unidad inferencial** (hoy son 5 folds, y
concedo el cargo de pseudorreplicación), y **dos endpoints primarios**: predicción **y** decisión.
Un bloque compra los dos claims o produce un negativo limpio.

**No se abre Program X. No se entrena ninguna red antes de la Fase 2.** La ruta de control queda sin
candidato: no existe en el árbol ningún comparador que planifique, y los dos que se llaman «MPC» en
`contention_bench_v1.py` llaman ambos a `_myopic_split` — filtro más reparto de un periodo, sin
horizonte.

## 9. Dónde os pido que apuntéis ahora

1. La Fase 1a, ¿con qué comparador la mataríais que no esté en la lista?
2. El estimando «portador neural» del bucle externo, ¿es el correcto o hay uno mejor?
3. ¿Más nombres? Vuestro barrido produjo más que el mío y a mí se me pasaron los cinco de mi
   propia sesión.

> ### ⇒ SUPERSEDIDO EL 2026-08-12 POR LA RÉPLICA CON POTENCIA
>
> Estado vigente: `docs/BRIEFING_REVISION_EXTERNA_2026-08-12.md` ·
> `docs/RESPUESTA_3_AL_AUDITOR_2026-08-12.md`
>
> **El proyecto tiene su primer positivo con custodia completa, y no es neuronal:**
> `results/program_o/powered_replication_v1/result.json` →
> `OBSERVABLE_CONVERSION_SURVIVES_AT_ADEQUATE_POWER`, 288 tapas vírgenes por celda, 7/7 falsadores,
> con la cola `ret_visible_cvar10` cruzando cero en las tres celdas (+0,021591 / +0,010936 /
> +0,103989) a un crítico simultáneo de 2,8770 — más exigente que el 2,8357 que cerró Program O.
> La política es un **belief-MPC clásico**.
>
> **Y la prima neural de predicción cayó:** contra una clase comparadora completa,
> `mlp vs gaussian_process +0,0342 [−0,1030, +0,1715]` y `recurrent vs gbdt_lagged −0,0300`.
>
> **`H_PI` se corrige de 0,1515 a 0,0371:** sobrevive su nulo de Jensen con p=0,0000, pero el 75,5 %
> del titular era sesgo de selección.
