# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a88d4726f5e87c7ce`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-10T03:41:12.728910+00:00`
- **commit**: `5744f4b8`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

El veredicto que imprime es `SENSITIVITY_PREMIUM_HOLDS_ON_LEGACY_SURFACE`. **La etiqueta sobrepasa lo que el dato dice, y el defecto es del conjunto de falsadores de la Puerta B.**

## Lo que salió

```
linear_lagged               +0.9316   <- clasico, encabeza todo
recurrent                   +0.8963
train_cell_mean_comparator  +0.8943
tree                        +0.8611   <- clasico
kan_tuned                   +0.8582
linear_interactions         +0.7906   <- baseline primario
mlp_tuned                   +0.7208

contra el primario:   kan +0.0676 [+0.0120, +0.1233]  PASA
                     tree +0.0705 [+0.0201, +0.1209]  PASA
                      mlp -0.0698 [-0.1581, +0.0186]  no
```

**Un árbol de regresión pasa el mismo criterio que el KAN, y por un margen mayor.** `f5` sólo compara contra el baseline primario, así que declara «prima» cuando lo que hay es que `linear_interactions` no es el mejor modelo no neuronal de esta superficie. Pareado:

```
kan - tree            -0.0029 [-0.0839, +0.0782]   empate
kan - linear_lagged   -0.0734 [-0.0958, -0.0510]   pierde
mlp - tree            -0.1403 [-0.2738, -0.0067]   pierde
```

La regla del contrato marco es explícita: **la red debe batir al mejor comparador NO neuronal**, no a la constante ni al primario. Bajo esa regla, en `ret_excel` **no hay prima neural**. La Puerta A2 sí implementa ese comparador (`vs_best_nonneural`) y por eso pudo matar a la Puerta A; la Puerta B nunca lo implementó, y esta sensibilidad es lo que lo destapa. Es un defecto mío de instrumento, heredado del preregistro de B.

`f4` falla, coherente con eso: `min(mlp, kan) − linear = −0,0698`.

## Qué sobrevive a las dos superficies, y qué no

**Sobrevive:** una red bate al lineal primario en ambas. Nada más.

**No sobrevive la arquitectura.** Cobb-Douglas v3: MLP 0,7975 > KAN 0,7805 > árbol 0,6890. `ret_excel`: árbol 0,8611 > KAN 0,8582 > MLP 0,7208. El orden se invierte por completo — ya lo hacía entre bloques dentro de Cobb-Douglas, y ahora también entre métricas.

**No sobrevive el brazo recurrente.** En Cobb-Douglas batía a `linear_lagged` (+0,1487 [+0,1069]); aquí **pierde** (−0,0353 [−0,1030, +0,0324]), y `linear_lagged` encabeza la tabla entera. La Fig. 5 como predictor gana en la métrica principal y no gana en la legada.

## Lo que esto significa para el claim

La sensibilidad hizo su trabajo, y el resultado **no es el que esperaba**: la afirmación de método no queda sostenida tal cual. Lo honesto es que **la prima de predicción es específica del endpoint Cobb-Douglas**, y en la superficie legada el margen sobre el primario existe pero lo captura igual de bien un modelo clásico.

Eso no la destruye. Cobb-Douglas es el endpoint primario declarado y `ret_excel` es sensibilidad legada por una razón medida — premia el abandono. Pero la redacción ya no puede decir «la red captura el margen»; tiene que decir **«sobre el índice Cobb-Douglas, una red captura un margen que el lineal deja; sobre la métrica legada ese margen lo captura también un árbol»**.

**No parcheo el artefacto sellado.** Lo que corresponde es enmendar el preregistro de B para añadir el comparador `best_nonneural` que A2 ya tiene, y re-adjudicar las tres corridas existentes contra él —sin abrir semillas, porque los `per_fold` ya están guardados—. Eso puede tumbar también el `SURFACE_PREMIUM_CAPTURED` de la v3: allí el MLP (0,7975) sí supera a `linear_lagged` (0,7520) y al árbol (0,6890), así que probablemente aguante, pero no lo afirmo hasta calcularlo pareado. Dime si lo hago.

## Raw payload

```json
{
 "agent_id": "a88d4726f5e87c7ce",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a88d4726f5e87c7ce.jsonl",
 "agent_type": "",
 "background_tasks": [
  {
   "command": "until ! pgrep -f run_seasonal_r2_headroom_gate_v2 >/dev/null; do sleep 15; done; echo listo; tail -50 /tmp/gate2.log",
   "description": "Wait for full v2 run",
   "id": "bgrvcph0j",
   "status": "running",
   "type": "shell"
  },
  {
   "command": "until ! pgrep -f run_seasonal_r2_headroom_gate_v2 >/dev/null; do sleep 15; done; echo terminado; tail -46 /tmp/gate2.log",
   "description": "Wait for v2 and show verdict",
   "id": "bclqbgfvh",
   "status": "running",
   "type": "shell"
  },
  {
   "command": "until grep -qE \"INERT|NOT_INERT\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/bwgse5ac9.output 2>/dev/null; do sleep 20; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/bwgse5ac9.output",
   "description": "until grep -qE \"INERT|NOT_INERT\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/bwgse5ac9.output 2>/dev/null; do sleep 20; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/bwgse5ac9.output",
   "id": "b5puo813u",
   "status": "running",
   "type": "shell"
  },
  {
   "command": "until grep -qE \"veredicto|Error|Traceback\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b1j8i524a.output 2>/dev/null; do sleep 30; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b1j8i524a.output",
   "description": "until grep -qE \"veredicto|Error|Traceback\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b1j8i524a.output 2>/dev/null; do sleep 30; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b1j8i524a.output",
   "id": "bbq6czqzq",
   "status": "running",
   "type": "shell"
  },
  {
   "command": "until grep -qE \"veredicto|Error|Traceback\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b10qy3t79.output 2>/dev/null; do sleep 20; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b10qy3t79.output",
   "description": "until grep -qE \"veredicto|Error|Traceback\" <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b10qy3t79.output 2>/dev/null; do sleep 20; done; cat <TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/tasks/b10qy3t79.output",
   "id": "brif8ya8q",
   "status": "running",
   "type": "shell"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "medium"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "hazlo",
 "permission_mode": "bypassPermissions",
 "prompt_id": "a402028a-abc3-46cd-a3d8-06777dbdc350",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
