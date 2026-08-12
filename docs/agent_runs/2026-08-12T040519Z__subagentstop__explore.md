# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `Explore`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a972b852f975b2a57`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T04:05:19.234693+00:00`
- **commit**: `1573b9bc`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a972b852f975b2a57.jsonl`

## Final response

He auditado los cinco `result.json` completos. Resumen ejecutivo primero, luego el detalle exigido artefacto por artefacto, y al final las cuatro preguntas puntuales A–D.

## Hallazgo central antes de entrar en detalle

Toda la cadena de "bucle externo" — excepto `garrido_meta_learner` y la mitad de `h2_learning_curve` — se apoya en **un único bloque de 12 semillas** (`5300001`–`5300012`), abierto como "virgen" el 2026-08-01 en `garrido_meta_learner`/`garrido_meta_learner_v2`, y reutilizado como *replay* declarado ("`garrido_q2_des288`") en `search_ladder_v5`, y de ahí reanalizado otra vez en `retention_contrasts` y `retention_simultaneous`. Es decir: **tres de los cuatro artefactos citados en la fila "bucle externo" del briefing no corren simulación nueva; son relecturas estadísticas del mismo experimento de 12 semillas.**

---

## 1. `results/garrido_meta_learner/result.json`

- `claim_status`: `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`
- `run_role`: **ausente** (campo no existe en este esquema, más antiguo que los demás)
- `scope`: **ausente**
- `contract_path`: `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md`, preregistrado el 31-07, corrido el 2026-08-01T03:19:35Z
- **Semillas**: nuevas/vírgenes en el momento — `f6_seeds_are_virgin: passed`, bloque `5300001`–`5300012` (12 semillas), `elapsed_seconds: 1515.94` (~25 min de simulación real, no reanálisis).
- **Estimando**: no hay campo `estimand` explícito; según el preregistro (línea 55): *"corridas hasta quedar dentro del 1% del mejor"* — para cada (brazo, semilla, contexto) se cuenta en qué paso del presupuesto de 24 corridas el mejor-hasta-ahora entra dentro del 1% del óptimo real de esa celda (288 configuraciones, `ret_excel_risk_conditional`), y se promedia sobre 12 semillas × 6 contextos = 72 celdas.
- **Números exactos** (`runs_to_within_1pct`):
  - `neuron_memory`: 7.236111111111112 → **7.24**
  - `neuron_reset`: 13.541666666666666 → **13.54**
  - `ofat`: 12.416666666666666 → **12.42**
  - `random`: 19.541666666666668
  - `alzheimer_effect_runs_saved_by_memory` (reset − memory): mean 6.3056, IC95 [5.1806, 7.4861]
  - `memory_vs_ofat`: mean 5.1806, IC95 [3.5278, 6.6389]
  - `memory_vs_random`: mean 12.3056, IC95 [10.5556, 14.1528]

## 2. `results/manuscript/h2_learning_curve/result.json`

- `claim_status`: `H2_SUPPORTED_LEARNING_CURVE`
- `run_role`: **ausente**; `scope`: `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION`
- **Semillas**: ninguna nueva. Es reanálisis de dos artefactos ya sellados: `garrido_meta_learner_h3power_h3_contract_local_v2` (90 réplicas, semillas 6000001–6000090) y `..._vps_v2` (30 réplicas, 6000091–6000120) → n=120. **Nótese: es un bloque de semillas distinto** al 5300001–5300012 usado por los otros tres artefactos.
- **Estimando exacto**: *"OLS slope of (reset AUC − memory AUC) against the context ordinal 1..6, per replicate"* — es decir, una regresión lineal por cada una de las 120 réplicas, sobre solo **6 puntos x** (la posición ordinal 1–6 de los contextos fijos `R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`, recorridos en ese orden pre-declarado, no en el tiempo).
- **Números exactos**: `primary_slope`: mean **+0.04220147575193482**, IC95 [+0.03466393985914079, +0.04992205677530306], n=120. Nulo propio (`random − ofat`): mean **−0.00508806**, IC95 [−0.01557, +0.00566].
- El propio artefacto advierte: *"A large but FLAT advantage supports H4, not H2"* — es decir, el diseño reconoce que una pendiente positiva **no distingue automáticamente** entre "la red aprende a través de contextos" (H2) y "hay una ventaja constante contaminada por el orden" (H4); el falsador f3 compara contra el par nulo (random-ofat, que no retiene nada) para descartar que la pendiente sea solo el orden de los contextos.

## 3. `results/retention_simultaneous/result.json`

- `claim_status`: `RETENTION_SURVIVES_SIMULTANEOUS_INFERENCE_6_OF_6_ON_AUC_BUT_1_OF_6_ON_FINAL_SIMPLE_REGRET`
- `run_role`: `REPLAY_REANALYSIS`; `scope`: `DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION`
- `registration_status`: `POST_HOC_MULTIPLICITY_REPAIR_REQUESTED_BY_REVIEW_NOT_PREREGISTERED` (`preregistration: null`)
- **Semillas**: ninguna nueva de simulación. `source`/`reference_path` = `results/search_ladder_v5/result.json` (mismas 12 semillas 5300001–5300012, ya replay). Sí usa 40 semillas nuevas de *bootstrap* (20260806–20260845), pero eso es solo para probar la sensibilidad del intervalo, no para generar más datos de simulación.
- **Estimando**: *"paired per-seed AUC(reset) − AUC(retained) per family, with simultaneous inference across the six families as one inferential family"* — endpoint primario `auc_regret_norm` (AUC de regret), endpoint secundario "final simple regret at budget 24".
- **Números exactos**:
  - AUC (primario), bajo criterio simultáneo (`c_simultaneous=2.591`, `simultaneous_lcb95 > 0`): **6 de 6 familias** (gp_ei, lookahead_kg, neuron, ofat, thompson, ucb1) tienen `simultaneous_lcb95` positivo.
  - Regret simple final (secundario), mismo criterio simultáneo: **solo 1 de 6** tiene `simultaneous_lcb95 > 0`: **`lookahead_kg`**, con `simultaneous_lcb95 = 0.00245`, `mean = 0.026765`. Las otras cinco (incluida `neuron`, el protagonista central del claim) cruzan cero: `neuron` `simultaneous_lcb95 = −0.0035`, `mean = 0.016822`.
  - **Matiz importante**: bajo el criterio Holm marginal (no simultáneo, menos estricto), 4 de 6 familias rechazan la nula en "final" (gp_ei, lookahead_kg, neuron, ofat); solo bajo el criterio simultáneo más duro baja a 1/6. El "1/6" del `claim_status` es el número correcto pero corresponde específicamente al criterio simultáneo, más estricto que Holm solo.
  - `supersedes_for_multiplicity`: `results/retention_contrasts/result.json` — este artefacto es la corrección posterior de ese, pedida por revisión.

## 4. `results/retention_contrasts/result.json`

- `claim_status`: `RETENTION_LOWERS_REGRET_IN_6_OF_6_FAMILIES`
- `run_role`: `REPLAY_REANALYSIS`; `scope`: `DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION`
- `registration_status`: `POST_HOC_REANALYSIS_OF_A_SEALED_ARTIFACT_NOT_PREREGISTERED`
- **Semillas**: ninguna nueva; `source` = `results/search_ladder_v5/result.json`, mismas 12 semillas replay.
- **Estimando**: *"paired per-seed AUC(reset) − AUC(retained), positive means retention helps"* (solo marginal, familia por familia, sin corrección de multiplicidad — de ahí que `retention_simultaneous` lo suceda).
- **Números exactos** (mean, IC95, `seeds_favouring_retention` de 12):
  - `gp_ei`: 0.02271 [0.01276, 0.03410], 12/12
  - `lookahead_kg`: 0.03461 [0.02610, 0.04315], 12/12
  - `neuron`: 0.06070 [0.04568, 0.07953], 12/12
  - `ofat`: 0.03750 [0.02920, 0.04675], 12/12
  - `thompson`: 0.01985 [0.01022, 0.02956], 10/12
  - `ucb1`: 0.05153 [0.03583, 0.06593], 11/12
- Esto es lo que el nombre `6_OF_6_FAMILIES` mide con precisión: 6/6 con `lcb95 marginal > 0` (no simultáneo).

## 5. `results/search_ladder_v5/result.json`

- `claim_status`: `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`
- `run_role`: `CACHE_ANALYSIS`; `scope`: `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`
- **Semillas**: `f7_no_fresh_seeds` → `status: "DECLARED_REPLAY"`, `replay_of: "garrido_q2_des288"`, mismas 12 semillas 5300001–5300012, **explícitamente no vírgenes**. Los 11 brazos de `search_ladder_v4` se reproducen bit a bit (`max_drift: 0.0`); se añaden 4 brazos nuevos (`lookahead_kg`, `lookahead_kg_transfer`, `thompson`, `thompson_transfer`) calculados sobre la misma cinta replayada.
- **Métrica**: `primary_metric: "auc_regret_norm"` (AUC de regret; **menor es mejor**, `oracle=0.0` es el mejor posible).

---

## Respuestas a las preguntas A–D del revisor

### A. "7,24 con memoria vs 13,54 reseteada vs 12,42 OFAT de la tesis"

Los tres números vienen **exactamente** de `results/garrido_meta_learner/result.json`, campo `runs_to_within_1pct`, definido en `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md` como "corridas hasta quedar dentro del 1% del mejor" (óptimo real de cada una de las 288 configuraciones), promediado sobre 12 semillas × 6 contextos = 72 celdas (7.236111×72=521, 13.541666×72=975, 12.416666×72=894 — enteros exactos, consistente).

**Precisión que falta en el briefing**: "12,42 el OFAT de la tesis" no es un número tomado de la tesis publicada de Garrido (2017); es el brazo `ofat` **reimplementado dentro de este mismo experimento**, corrido con el mismo presupuesto/semillas/superficie de 288 configuraciones que los brazos de la neurona (confirmado por `f2_ofat_is_really_one_factor_at_a_time`: cambia exactamente una coordenada por paso, replicando el *diseño* OFAT descrito en la tesis, no reproduciendo sus resultados numéricos originales). Es una comparación justa dentro del mismo tablero, pero la frase "de la tesis" puede leerse como si fuera un dato citado del documento original de Garrido, y no lo es.

### B. "6/6 en AUC, 1/6 en simple regret final"

Ambos números son correctos y provienen de `results/retention_simultaneous/result.json`, bajo el criterio **simultáneo** (no Holm marginal). La familia que sí sobrevive en regret simple final es **`lookahead_kg`** (`simultaneous_lcb95 = 0.00245`, apenas por encima de cero), **no** `neuron` — la familia central del relato del "efecto Alzheimer" (`neuron`) de hecho **no sobrevive** en ese endpoint secundario bajo inferencia simultánea (`simultaneous_lcb95 = −0.0035`, cruza cero). El briefing (`docs/BRIEFING_REVISION_EXTERNA_2026-08-10.md`, línea 38) reporta bien el "1/6" pero no dice cuál familia es, dejando la impresión implícita de que sigue siendo la neurona.

### C. `ucb1_transfer` vs `neuron_memory` en `search_ladder_v5`

Confirmado. Métrica: `mean_auc_regret` (`auc_regret_norm`), **menor es mejor**. Valores exactos:
- `ucb1_transfer`: **0.04502300698358852**
- `neuron_memory`: **0.05203274760040456**

`ucb1_transfer` tiene mejor (menor) punto estimado que `neuron_memory` — una diferencia relativa de ~13.4%. Esto se confirma en `ranking_best_first`, que lista `ucb1_transfer` en 1er lugar y `neuron_memory` en 2º, y en `vs_neuron_memory.ucb1_transfer`: mean **−0.00701** (negativo = `ucb1_transfer` mejor que `neuron_memory`), pero **no significativo**: IC95 [−0.02444, +0.01408], cruza cero, n=12.

El `claim_status` (`THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`) es técnicamente correcto en lo que afirma — la neurona bate con IC95 > 0 a `lookahead_kg` (mean 0.0628, IC95 [0.0433,0.0808]) y a `lookahead_kg_transfer` (mean 0.0281, IC95 [0.0128,0.0419]) — pero el nombre invita a leer "la neurona aguanta la búsqueda" como "la neurona es el mejor brazo del ladder", y no lo es en punto estimado: un método clásico con memoria (`ucb1_transfer`, ya existente desde `v4`, no un brazo nuevo de esta corrida) tiene mejor regret medio, aunque sin diferencia estadísticamente distinguible de cero. El briefing (línea 39, "la neurona aguanta contra búsqueda con anticipación") no menciona esta comparación con `ucb1_transfer` en absoluto.

### D. Pendiente H2 +0.0422 [+0.0347, +0.0499]

Confirmado en `results/manuscript/h2_learning_curve/result.json`, campo `primary_slope`: mean 0.04220147575193482, IC95 [0.03466393985914079, 0.04992205677530306], n=120.

Es una regresión OLS de (AUC-reset − AUC-memoria) por réplica **contra la posición ordinal 1–6** de una secuencia de **exactamente 6 contextos fijos** (`R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`, en ese orden pre-declarado en `context_order`), no contra tiempo real ni contra número de corridas. Cada una de las 120 réplicas produce una pendiente propia sobre solo 6 puntos x; el IC reportado resume esas 120 pendientes. El propio artefacto reconoce la fragilidad de este diseño con el falsador `f3_the_order_confound_is_absorbed`, que compara contra un par nulo sin memoria (random−ofat) para verificar que la pendiente no sea simplemente el efecto de que los últimos 3 contextos estén "escalados" (mayor frecuencia), y con `f4_the_slope_can_be_negative`, que documenta 22 de 120 réplicas con pendiente negativa (rango observado: −0.0564 a +0.1478).

---

## Otras discrepancias de nombre-vs-contenido que vale la pena señalar

1. **`garrido_meta_learner`** carece de los campos `run_role`/`scope` que sí tienen los otros cuatro — es un esquema más temprano (2026-08-01), sin la taxonomía de gobernanza (`REPLAY_REANALYSIS`, `CACHE_ANALYSIS`, etc.) que el proyecto adoptó después. Vale la pena que el revisor sepa que este artefacto no se etiqueta a sí mismo bajo esas categorías porque no existían aún, no porque sea una confirmación "de primera clase" respecto a los demás.
2. El mismo bloque de 12 semillas (`5300001`–`5300012`) fue abierto como "virgen" en `garrido_meta_learner` (03:19 UTC) y reabierto/reusado en `garrido_meta_learner_v2` (05:29 UTC, mismo día) — ambos con idéntico `claim_status`. Esto no invalida `garrido_meta_learner` por sí solo, pero confirma que ese bloque de 12 semillas es el que después se congela como `garrido_q2_des288` y se replay-ea en `search_ladder_v2` a `v5`, `retention_contrasts` y `retention_simultaneous` — cinco artefactos adicionales derivados de la misma corrida original de 12 semillas.
3. `search_ladder_v5.claim_status` ("la neurona aguanta") no menciona que un brazo clásico preexistente (`ucb1_transfer`) tiene mejor punto estimado en la métrica primaria del propio artefacto (ítem C arriba).
4. `retention_simultaneous.claim_status` da el "1/6" correcto pero no nombra la familia (`lookahead_kg`) ni aclara que no es `neuron` (ítem B arriba).

## Raw payload

```json
{
 "agent_id": "a972b852f975b2a57",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a972b852f975b2a57.jsonl",
 "agent_type": "Explore",
 "background_tasks": [
  {
   "agent_type": "Explore",
   "description": "Sweep for naming inflation",
   "id": "a0d03a92a8a9fd002",
   "status": "running",
   "type": "subagent"
  },
  {
   "agent_type": "Explore",
   "description": "Outer-loop evidence audit",
   "id": "a972b852f975b2a57",
   "status": "running",
   "type": "subagent"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "xhigh"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "He auditado los cinco `result.json` completos. Resumen ejecutivo primero, luego el detalle exigido artefacto por artefacto, y al final las cuatro preguntas puntuales A–D.\n\n## Hallazgo central antes de entrar en detalle\n\nToda la cadena de \"bucle externo\" — excepto `garrido_meta_learner` y la mitad de `h2_learning_curve` — se apoya en **un único bloque de 12 semillas** (`5300001`–`5300012`), abierto como \"virgen\" el 2026-08-01 en `garrido_meta_learner`/`garrido_meta_learner_v2`, y reutilizado como *replay* declarado (\"`garrido_q2_des288`\") en `search_ladder_v5`, y de ahí reanalizado otra vez en `retention_contrasts` y `retention_simultaneous`. Es decir: **tres de los cuatro artefactos citados en la fila \"bucle externo\" del briefing no corren simulación nueva; son relecturas estadísticas del mismo experimento de 12 semillas.**\n\n---\n\n## 1. `results/garrido_meta_learner/result.json`\n\n- `claim_status`: `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`\n- `run_role`: **ausente** (campo no existe en este esquema, más antiguo que los demás)\n- `scope`: **ausente**\n- `contract_path`: `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md`, preregistrado el 31-07, corrido el 2026-08-01T03:19:35Z\n- **Semillas**: nuevas/vírgenes en el momento — `f6_seeds_are_virgin: passed`, bloque `5300001`–`5300012` (12 semillas), `elapsed_seconds: 1515.94` (~25 min de simulación real, no reanálisis).\n- **Estimando**: no hay campo `estimand` explícito; según el preregistro (línea 55): *\"corridas hasta quedar dentro del 1% del mejor\"* — para cada (brazo, semilla, contexto) se cuenta en qué paso del presupuesto de 24 corridas el mejor-hasta-ahora entra dentro del 1% del óptimo real de esa celda (288 configuraciones, `ret_excel_risk_conditional`), y se promedia sobre 12 semillas × 6 contextos = 72 celdas.\n- **Números exactos** (`runs_to_within_1pct`):\n  - `neuron_memory`: 7.236111111111112 → **7.24**\n  - `neuron_reset`: 13.541666666666666 → **13.54**\n  - `ofat`: 12.416666666666666 → **12.42**\n  - `random`: 19.541666666666668\n  - `alzheimer_effect_runs_saved_by_memory` (reset − memory): mean 6.3056, IC95 [5.1806, 7.4861]\n  - `memory_vs_ofat`: mean 5.1806, IC95 [3.5278, 6.6389]\n  - `memory_vs_random`: mean 12.3056, IC95 [10.5556, 14.1528]\n\n## 2. `results/manuscript/h2_learning_curve/result.json`\n\n- `claim_status`: `H2_SUPPORTED_LEARNING_CURVE`\n- `run_role`: **ausente**; `scope`: `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION`\n- **Semillas**: ninguna nueva. Es reanálisis de dos artefactos ya sellados: `garrido_meta_learner_h3power_h3_contract_local_v2` (90 réplicas, semillas 6000001–6000090) y `..._vps_v2` (30 réplicas, 6000091–6000120) → n=120. **Nótese: es un bloque de semillas distinto** al 5300001–5300012 usado por los otros tres artefactos.\n- **Estimando exacto**: *\"OLS slope of (reset AUC − memory AUC) against the context ordinal 1..6, per replicate\"* — es decir, una regresión lineal por cada una de las 120 réplicas, sobre solo **6 puntos x** (la posición ordinal 1–6 de los contextos fijos `R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`, recorridos en ese orden pre-declarado, no en el tiempo).\n- **Números exactos**: `primary_slope`: mean **+0.04220147575193482**, IC95 [+0.03466393985914079, +0.04992205677530306], n=120. Nulo propio (`random − ofat`): mean **−0.00508806**, IC95 [−0.01557, +0.00566].\n- El propio artefacto advierte: *\"A large but FLAT advantage supports H4, not H2\"* — es decir, el diseño reconoce que una pendiente positiva **no distingue automáticamente** entre \"la red aprende a través de contextos\" (H2) y \"hay una ventaja constante contaminada por el orden\" (H4); el falsador f3 compara contra el par nulo (random-ofat, que no retiene nada) para descartar que la pendiente sea solo el orden de los contextos.\n\n## 3. `results/retention_simultaneous/result.json`\n\n- `claim_status`: `RETENTION_SURVIVES_SIMULTANEOUS_INFERENCE_6_OF_6_ON_AUC_BUT_1_OF_6_ON_FINAL_SIMPLE_REGRET`\n- `run_role`: `REPLAY_REANALYSIS`; `scope`: `DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION`\n- `registration_status`: `POST_HOC_MULTIPLICITY_REPAIR_REQUESTED_BY_REVIEW_NOT_PREREGISTERED` (`preregistration: null`)\n- **Semillas**: ninguna nueva de simulación. `source`/`reference_path` = `results/search_ladder_v5/result.json` (mismas 12 semillas 5300001–5300012, ya replay). Sí usa 40 semillas nuevas de *bootstrap* (20260806–20260845), pero eso es solo para probar la sensibilidad del intervalo, no para generar más datos de simulación.\n- **Estimando**: *\"paired per-seed AUC(reset) − AUC(retained) per family, with simultaneous inference across the six families as one inferential family\"* — endpoint primario `auc_regret_norm` (AUC de regret), endpoint secundario \"final simple regret at budget 24\".\n- **Números exactos**:\n  - AUC (primario), bajo criterio simultáneo (`c_simultaneous=2.591`, `simultaneous_lcb95 > 0`): **6 de 6 familias** (gp_ei, lookahead_kg, neuron, ofat, thompson, ucb1) tienen `simultaneous_lcb95` positivo.\n  - Regret simple final (secundario), mismo criterio simultáneo: **solo 1 de 6** tiene `simultaneous_lcb95 > 0`: **`lookahead_kg`**, con `simultaneous_lcb95 = 0.00245`, `mean = 0.026765`. Las otras cinco (incluida `neuron`, el protagonista central del claim) cruzan cero: `neuron` `simultaneous_lcb95 = −0.0035`, `mean = 0.016822`.\n  - **Matiz importante**: bajo el criterio Holm marginal (no simultáneo, menos estricto), 4 de 6 familias rechazan la nula en \"final\" (gp_ei, lookahead_kg, neuron, ofat); solo bajo el criterio simultáneo más duro baja a 1/6. El \"1/6\" del `claim_status` es el número correcto pero corresponde específicamente al criterio simultáneo, más estricto que Holm solo.\n  - `supersedes_for_multiplicity`: `results/retention_contrasts/result.json` — este artefacto es la corrección posterior de ese, pedida por revisión.\n\n## 4. `results/retention_contrasts/result.json`\n\n- `claim_status`: `RETENTION_LOWERS_REGRET_IN_6_OF_6_FAMILIES`\n- `run_role`: `REPLAY_REANALYSIS`; `scope`: `DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION`\n- `registration_status`: `POST_HOC_REANALYSIS_OF_A_SEALED_ARTIFACT_NOT_PREREGISTERED`\n- **Semillas**: ninguna nueva; `source` = `results/search_ladder_v5/result.json`, mismas 12 semillas replay.\n- **Estimando**: *\"paired per-seed AUC(reset) − AUC(retained), positive means retention helps\"* (solo marginal, familia por familia, sin corrección de multiplicidad — de ahí que `retention_simultaneous` lo suceda).\n- **Números exactos** (mean, IC95, `seeds_favouring_retention` de 12):\n  - `gp_ei`: 0.02271 [0.01276, 0.03410], 12/12\n  - `lookahead_kg`: 0.03461 [0.02610, 0.04315], 12/12\n  - `neuron`: 0.06070 [0.04568, 0.07953], 12/12\n  - `ofat`: 0.03750 [0.02920, 0.04675], 12/12\n  - `thompson`: 0.01985 [0.01022, 0.02956], 10/12\n  - `ucb1`: 0.05153 [0.03583, 0.06593], 11/12\n- Esto es lo que el nombre `6_OF_6_FAMILIES` mide con precisión: 6/6 con `lcb95 marginal > 0` (no simultáneo).\n\n## 5. `results/search_ladder_v5/result.json`\n\n- `claim_status`: `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`\n- `run_role`: `CACHE_ANALYSIS`; `scope`: `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`\n- **Semillas**: `f7_no_fresh_seeds` → `status: \"DECLARED_REPLAY\"`, `replay_of: \"garrido_q2_des288\"`, mismas 12 semillas 5300001–5300012, **explícitamente no vírgenes**. Los 11 brazos de `search_ladder_v4` se reproducen bit a bit (`max_drift: 0.0`); se añaden 4 brazos nuevos (`lookahead_kg`, `lookahead_kg_transfer`, `thompson`, `thompson_transfer`) calculados sobre la misma cinta replayada.\n- **Métrica**: `primary_metric: \"auc_regret_norm\"` (AUC de regret; **menor es mejor**, `oracle=0.0` es el mejor posible).\n\n---\n\n## Respuestas a las preguntas A–D del revisor\n\n### A. \"7,24 con memoria vs 13,54 reseteada vs 12,42 OFAT de la tesis\"\n\nLos tres números vienen **exactamente** de `results/garrido_meta_learner/result.json`, campo `runs_to_within_1pct`, definido en `docs/PREREGISTRO_META_APRENDIZ_2026-07-31.md` como \"corridas hasta quedar dentro del 1% del mejor\" (óptimo real de cada una de las 288 configuraciones), promediado sobre 12 semillas × 6 contextos = 72 celdas (7.236111×72=521, 13.541666×72=975, 12.416666×72=894 — enteros exactos, consistente).\n\n**Precisión que falta en el briefing**: \"12,42 el OFAT de la tesis\" no es un número tomado de la tesis publicada de Garrido (2017); es el brazo `ofat` **reimplementado dentro de este mismo experimento**, corrido con el mismo presupuesto/semillas/superficie de 288 configuraciones que los brazos de la neurona (confirmado por `f2_ofat_is_really_one_factor_at_a_time`: cambia exactamente una coordenada por paso, replicando el *diseño* OFAT descrito en la tesis, no reproduciendo sus resultados numéricos originales). Es una comparación justa dentro del mismo tablero, pero la frase \"de la tesis\" puede leerse como si fuera un dato citado del documento original de Garrido, y no lo es.\n\n### B. \"6/6 en AUC, 1/6 en simple regret final\"\n\nAmbos números son correctos y provienen de `results/retention_simultaneous/result.json`, bajo el criterio **simultáneo** (no Holm marginal). La familia que sí sobrevive en regret simple final es **`lookahead_kg`** (`simultaneous_lcb95 = 0.00245`, apenas por encima de cero), **no** `neuron` — la familia central del relato del \"efecto Alzheimer\" (`neuron`) de hecho **no sobrevive** en ese endpoint secundario bajo inferencia simultánea (`simultaneous_lcb95 = −0.0035`, cruza cero). El briefing (`docs/BRIEFING_REVISION_EXTERNA_2026-08-10.md`, línea 38) reporta bien el \"1/6\" pero no dice cuál familia es, dejando la impresión implícita de que sigue siendo la neurona.\n\n### C. `ucb1_transfer` vs `neuron_memory` en `search_ladder_v5`\n\nConfirmado. Métrica: `mean_auc_regret` (`auc_regret_norm`), **menor es mejor**. Valores exactos:\n- `ucb1_transfer`: **0.04502300698358852**\n- `neuron_memory`: **0.05203274760040456**\n\n`ucb1_transfer` tiene mejor (menor) punto estimado que `neuron_memory` — una diferencia relativa de ~13.4%. Esto se confirma en `ranking_best_first`, que lista `ucb1_transfer` en 1er lugar y `neuron_memory` en 2º, y en `vs_neuron_memory.ucb1_transfer`: mean **−0.00701** (negativo = `ucb1_transfer` mejor que `neuron_memory`), pero **no significativo**: IC95 [−0.02444, +0.01408], cruza cero, n=12.\n\nEl `claim_status` (`THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`) es técnicamente correcto en lo que afirma — la neurona bate con IC95 > 0 a `lookahead_kg` (mean 0.0628, IC95 [0.0433,0.0808]) y a `lookahead_kg_transfer` (mean 0.0281, IC95 [0.0128,0.0419]) — pero el nombre invita a leer \"la neurona aguanta la búsqueda\" como \"la neurona es el mejor brazo del ladder\", y no lo es en punto estimado: un método clásico con memoria (`ucb1_transfer`, ya existente desde `v4`, no un brazo nuevo de esta corrida) tiene mejor regret medio, aunque sin diferencia estadísticamente distinguible de cero. El briefing (línea 39, \"la neurona aguanta contra búsqueda con anticipación\") no menciona esta comparación con `ucb1_transfer` en absoluto.\n\n### D. Pendiente H2 +0.0422 [+0.0347, +0.0499]\n\nConfirmado en `results/manuscript/h2_learning_curve/result.json`, campo `primary_slope`: mean 0.04220147575193482, IC95 [0.03466393985914079, 0.04992205677530306], n=120.\n\nEs una regresión OLS de (AUC-reset − AUC-memoria) por réplica **contra la posición ordinal 1–6** de una secuencia de **exactamente 6 contextos fijos** (`R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`, en ese orden pre-declarado en `context_order`), no contra tiempo real ni contra número de corridas. Cada una de las 120 réplicas produce una pendiente propia sobre solo 6 puntos x; el IC reportado resume esas 120 pendientes. El propio artefacto reconoce la fragilidad de este diseño con el falsador `f3_the_order_confound_is_absorbed`, que compara contra un par nulo sin memoria (random−ofat) para verificar que la pendiente no sea simplemente el efecto de que los últimos 3 contextos estén \"escalados\" (mayor frecuencia), y con `f4_the_slope_can_be_negative`, que documenta 22 de 120 réplicas con pendiente negativa (rango observado: −0.0564 a +0.1478).\n\n---\n\n## Otras discrepancias de nombre-vs-contenido que vale la pena señalar\n\n1. **`garrido_meta_learner`** carece de los campos `run_role`/`scope` que sí tienen los otros cuatro — es un esquema más temprano (2026-08-01), sin la taxonomía de gobernanza (`REPLAY_REANALYSIS`, `CACHE_ANALYSIS`, etc.) que el proyecto adoptó después. Vale la pena que el revisor sepa que este artefacto no se etiqueta a sí mismo bajo esas categorías porque no existían aún, no porque sea una confirmación \"de primera clase\" respecto a los demás.\n2. El mismo bloque de 12 semillas (`5300001`–`5300012`) fue abierto como \"virgen\" en `garrido_meta_learner` (03:19 UTC) y reabierto/reusado en `garrido_meta_learner_v2` (05:29 UTC, mismo día) — ambos con idéntico `claim_status`. Esto no invalida `garrido_meta_learner` por sí solo, pero confirma que ese bloque de 12 semillas es el que después se congela como `garrido_q2_des288` y se replay-ea en `search_ladder_v2` a `v5`, `retention_contrasts` y `retention_simultaneous` — cinco artefactos adicionales derivados de la misma corrida original de 12 semillas.\n3. `search_ladder_v5.claim_status` (\"la neurona aguanta\") no menciona que un brazo clásico preexistente (`ucb1_transfer`) tiene mejor punto estimado en la métrica primaria del propio artefacto (ítem C arriba).\n4. `retention_simultaneous.claim_status` da el \"1/6\" correcto pero no nombra la familia (`lookahead_kg`) ni aclara que no es `neuron` (ítem B arriba).",
 "permission_mode": "plan",
 "prompt_id": "8da6c998-d92d-4cbe-9173-2ca354bc53b9",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
