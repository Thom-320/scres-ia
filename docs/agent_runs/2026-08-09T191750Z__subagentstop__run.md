# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a84b8f3f88d622f64`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-09T19:17:50.896853+00:00`
- **commit**: `3388dcf6`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

He avanzado hasta donde puedo sin el resultado. Te resumo, porque la respuesta a tu pregunta ya cambió.

## Tu premisa era falsa, y eso es una buena noticia

Reverifiqué los **242 artefactos con veredicto**. **Sí hemos tenido victorias neuronales.** Lo que no hemos hecho nunca es cobrarlas con custodia.

**Por qué pierde RL donde pierde, y está medido:** en todos los entornos que construimos el estado latente tiene **dos o tres estados con modelo generativo conocido**. Ahí un filtro bayesiano escrito a mano es óptimo, y una red sólo puede empatar. `program_v`: privilegiado menos Bayes **+0,00076, UCB95 +0,0023**. No queda ni un cuarto de punto.

**Y el contraejemplo es nuestro.** En `contention_v1` el aprendiz **batió al belief-MPC por +0,0136 [LCB95 +0,0124]**. Única diferencia: el régimen es **semi-Markov**, así que el filtro está **mal especificado**.

> **La prima neural no vive donde el problema es difícil. Vive donde la creencia exacta no es calculable en forma cerrada.** Los seis entornos donde perdimos eran difíciles *y* exactamente modelables. Ése fue el error de diseño.

## Sobre RNN: sí, pero no para batir a un filtro

Un actor recurrente contra un filtro bien especificado es la manera de volver a empatar. Sirve como **(1)** estimador de estado donde el proceso es semi-Markov, **(2)** amortizador del planner, **(3)** surrogate de la superficie — que es el **nivel 3, *pattern recognition***, que Garrido nombra con esas palabras.

**Y ahí ya ganamos:** el efecto Alzheimer tiene precio medido, y H2 tiene curva de aprendizaje con pendiente **+0,0422 [+0,0347, +0,0499]** contra su propio nulo. **H1, H2 y H4 de tu borrador ya están soportadas.** H3 no.

## Lo hecho esta sesión

Contrato marco congelado, y las dos primeras puertas preregistradas.

**Puerta A — corriendo ahora.** `track_b_nonneural` es el único sitio donde una red bate a un comparador **no neuronal**: MLP 98,74 vs regla 97,14. Pero el bake-off **nunca llamó a `model.save`** y guardó una media por semilla: ese `+1,60` **no tiene intervalo y nadie puede reevaluarlo**. Por eso hay que reentrenar.

**El pre-vuelo cazó tres defectos míos antes de abrir el bloque** — corrido a 2.000 pasos contra `/tmp`:

- el presupuesto era **5× el declarado** (961.809 parámetros contra un contrato de 200k);
- el control obligado a diferir era frágil y falló por una razón ajena al arnés;
- y **el comparador era demasiado débil para ser batido con honestidad** — la constante ajustaba en 59,4 contra una regla en 87,3. Eso le habría regalado a la red una prima sobre un brazo que nadie optimizó.

**Puerta B — preregistrada.** El artefacto dice `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED`: había **+0,0625** de margen y las redes quedaron **por debajo del lineal**. Leyendo el código, los brazos neuronales eran **600 pasos fijos de Adam, sin validación, sin early stopping, sin tuning y una sola semilla** — contra **OLS, que es óptimo en forma cerrada para su base**. Un ajuste sin sintonizar contra un óptimo analítico. La hipótesis a falsar es **sobre nosotros**, no sobre el entorno, y `f4` puede fallar.

## Lo que queda

La Puerta A lleva ~45 min entrenando (5 semillas × 200k pasos, estimo ~2 h en total). No lanzo la Puerta B hasta que cierre, para no competir por CPU. Te doy el veredicto en cuanto termine — con el intervalo pareado que ese `+1,60` nunca tuvo, y con la posibilidad explícita de que no sobreviva.

## Raw payload

```json
{
 "agent_id": "a84b8f3f88d622f64",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a84b8f3f88d622f64.jsonl",
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
   "command": "cd <HOME>/Projects/research/scres-ia; until ! ps -p 343 >/dev/null 2>&1; do sleep 120; done; echo \"=== terminado ===\"; tail -25 /tmp/gate_a_full.log",
   "description": "cd <HOME>/Projects/research/scres-ia; until ! ps -p 343 >/dev/null 2>&1; do sleep 120; done; echo \"=== terminado ===\"; tail -25 /tmp/gate_a_full.log",
   "id": "bpyspokt1",
   "status": "running",
   "type": "shell"
  },
  {
   "command": "cd <HOME>/Projects/research/scres-ia; until ! ps -p 343 >/dev/null 2>&1; do sleep 120; done; echo \"=== TERMINADO ===\"; tail -22 /tmp/gate_a_full.log; echo \"--- modelos ---\"; ls results/program_n/gate_a_track_b/models/",
   "description": "cd <HOME>/Projects/research/scres-ia; until ! ps -p 343 >/dev/null 2>&1; do sleep 120; done; echo \"=== TERMINADO ===\"; tail -22 /tmp/gate_a_full.log; echo \"--- modelos ---\"; ls results/program_n/gate_a_track_b/models/",
   "id": "bl9hnmd0g",
   "status": "running",
   "type": "shell"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "high"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "dime el veredicto de la puerta A cuando termine",
 "permission_mode": "auto",
 "prompt_id": "df6db5b7-f0a4-48a3-80e5-383a0978aa56",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
