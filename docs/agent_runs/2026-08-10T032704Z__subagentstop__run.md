# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a26f24661f540d97d`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-10T03:27:04.068968+00:00`
- **commit**: `2c5ad72e`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

Las dos cerraron, y dan en direcciones opuestas.

## Puerta B v3 — `SURFACE_PREMIUM_CAPTURED`, 7/7 falsadores

Bloque 9600001–9600008. **`f2` reparado pasa, y no por construcción** — el control positivo dispara.

```
mlp_tuned      +0.1081 [+0.0601, +0.1561]   PASA
kan_tuned      +0.0911 [-0.0082, +0.1904]   no
tree           -0.0004 [-0.1022, +0.1013]   no
spline_buffer  -0.0082 [-0.0536, +0.0372]   no
recurrent vs linear_lagged  +0.1487 [+0.1069, +0.1905]
```

**Lo que replica es «una red bate al lineal», no «el KAN lo bate».** En el bloque anterior fue el KAN quien pasaba (+0,0650) y el MLP quien perdía (−0,0127); aquí es exactamente al revés. Los dos bloques son tapas distintas del mismo diseño y la identidad del brazo ganador se da la vuelta. Cualquier redacción que diga «el KAN captura el margen» está sobreajustada al bloque — la afirmación que aguanta es la de familia.

Y hay algo que el propio artefacto delata: `train_cell_mean_comparator` (0,7547), que llamamos techo, **queda por debajo del MLP (0,7975) y del recurrente (0,9007)**. `f1` pasa, pero el nombre está mal: acota su familia, no la superficie. Lo corrijo en la redacción antes de que se cite como techo.

El brazo recurrente ahora sí separa de cero contra `linear_lagged`, su comparador correcto. Ésa es la Fig. 5 de Garrido **como predictor**, batiendo a un modelo clásico con el mismo conjunto de información.

## Puerta A2 — `NO_QUALITY_PREMIUM_AGAINST_THE_WIDENED_CLASS`

Y aquí la Puerta A se cae, que es para lo que A2 existía.

```
linear_feedback  99.127   <- el comparador nuevo
mlp              98.567
threshold_rule   98.095
ewma_rule        98.025
constant_best    98.016

mlp vs mejor no-neuronal   -0.559 [-0.748, -0.386]   7/48 tapas favorables
mlp vs threshold_rule      +0.472 [+0.275, +0.658]   37/48
```

**La prima de A era real contra la clase que teníamos y muere contra la clase que ensanchamos.** El MLP bate a la regla de umbral y a los dos placebos de historia — barajada y congelada —, así que la memoria hace algo. Pero una **realimentación lineal**, cuatro líneas de código, lo bate a él. `f4` falla y es dato, no instrumento.

De paso: el `+1,60` sobre la regla que traía la Puerta A se reduce a **+0,47** al parear por semilla con custodia. Parte de aquel titular era el emparejamiento que faltaba.

**Una salvedad de instrumento**, que verifiqué en vez de darla por buena: el falsador de custodia marca `COLLISION`, y la única entrada en conflicto es la reserva del propio bloque, con `consumed_by` apuntando a este mismo `result.json`. Es autorreferencia, no reutilización. El defecto es del chequeo, que debería excluir su propia reserva; no lo parcheo sobre la marcha para no tocar un artefacto recién sellado.

## Dónde queda esto

La prima neural de **control** queda cerrada también en Track B, que era la última candidata de calidad del repositorio. La prima de **predicción** sobrevive y ahora tiene confirmación limpia en bloque virgen.

Eso deja el claim exactamente donde tu propia evidencia lo venía empujando: **nivel 3 de Garrido, reconocimiento de patrón en el bucle externo** — donde ya ganábamos con el efecto Alzheimer y la curva H2, y ahora con una superficie donde la red bate al lineal y el recurrente bate a su comparador con la misma información.

Lo que falta para que sea una afirmación de método y no de una corrida: **la segunda superficie con `ret_excel` como sensibilidad**. Es lo único pendiente de la Puerta B y no necesita bloque nuevo. Dime si lo lanzo.

## Raw payload

```json
{
 "agent_id": "a26f24661f540d97d",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a26f24661f540d97d.jsonl",
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
 "last_assistant_message": "lánzalo",
 "permission_mode": "bypassPermissions",
 "prompt_id": "a097cf01-679c-473c-9ae6-90c210d698dc",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
