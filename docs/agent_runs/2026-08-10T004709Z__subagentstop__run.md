# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a132434b5746bc940`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-10T00:47:09.922312+00:00`
- **commit**: `23a796db`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

**Puerta B: `SURFACE_PREMIUM_CAPTURED`. Siete de siete falsadores, ninguno fallido.** Es la primera prima neural del proyecto que sobrevive a su propio protocolo.

## Lo que la hace legible: los clásicos reproducen a 4,9e-05

```
recurrent                   +0.8981   (conjunto de información MÁS RICO)
kan_tuned                   +0.7446   (antes +0.6019)
mlp_tuned                   +0.6952   (antes +0.5841)
train_cell_mean_comparator  +0.6931   (antes +0.6931)
linear_interactions         +0.6306   (antes +0.6306)   ← baseline primario
```

Mismo objetivo, mismos folds, mismas ocho semillas, misma calibración. **Lo único que cambió fue el ajuste de las redes** — y eso es `f2`, lo que hace que el resto signifique algo.

| contra el baseline primario | media | IC95 | ¿pasa? |
|---|---|---|---|
| **`kan_tuned`** | **+0,1140** | **[+0,0614, +0,1665]** | **sí** |
| `mlp_tuned` | +0,0646 | [−0,0883, +0,2174] | no |

## H_B era cierta, y el problema éramos nosotros

El KAN pasó de **estar por debajo del lineal a estar por encima del techo**. Lo que cambió: estandarización sobre train, validación interna con parada temprana, rejilla declarada idéntica y cinco semillas promediadas. El predecesor tenía **600 pasos fijos, una semilla, sin validación** contra **OLS en forma cerrada**.

**No era el entorno. Era un ajuste sin sintonizar contra un óptimo analítico.**

## Y esto responde tu pregunta sobre KAN mejor que un sí o un no

Ya teníamos `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP`. Ahora, en **predicción**, el KAN gana y el MLP no llega a significación:

> **El KAN pierde como controlador de búsqueda y gana como reconocedor de patrones.**

Es exactamente donde la Fig. 3 de Garrido sitúa el problema — **nivel 3, *pattern recognition***. La respuesta a su Q1 pasa a ser **«KAN sí, para la tarea que usted identificó; no para la otra»**, con las dos mitades medidas. Y el KAN es **estable entre folds** (0,711–0,774) mientras el MLP oscila (0,459–0,790): por eso el MLP tiene media positiva e intervalo que cruza cero.

## Dos cosas que no autoriza, y las dejo escritas

**Es predicción, no control.** Que una red prediga mejor la superficie no implica que un controlador neuronal mejore el servicio. **No autoriza ningún aprendiz de control** — eso lo decide la Puerta C.

**El «techo» no era un techo.** El KAN lo supera (0,7446 > 0,6931) porque generaliza entre celdas donde una media por celda no puede. Sigue siendo referencia útil; **no es cota superior** y no se citará como tal.

El brazo recurrente llega a 0,8981 **pero ve la resiliencia de la configuración anterior**, que ningún otro ve. Por eso se juzga sólo contra `linear_lagged`: **+0,2053 [+0,1051, +0,3056]**. Compararlo con el lineal sería medir el conjunto de información, no la arquitectura.

## A2 lanzada

El pre-vuelo confirmó el arreglo: **`f3` ahora pasa**. La clase no neuronal se amplía a cuatro brazos, el mejor se elige **sobre el bloque de ajuste** y su validez se comprueba contra un **suelo absoluto**.

Y `f5` es la que te importa para lo de RNN: la red debe batir a **`ewma_rule`**, un filtro exponencial que usa historia **con orden**. En la Puerta A la red batió al placebo congelado pero no al barajado. Si aquí sólo empata con el EWMA, la lectura es `PREMIUM_IS_CAPACITY_NOT_MEMORY` — y ésa sería la respuesta empírica a si necesitamos recurrencia en el control.

Ya se ve la saturación otra vez en el pre-vuelo: constante 98,29 contra umbral 98,29. Te doy el veredicto de A2 cuando cierre, en unas tres horas.

## Raw payload

```json
{
 "agent_id": "a132434b5746bc940",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a132434b5746bc940.jsonl",
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
  "level": "high"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "dime el veredicto de A2 cuando termine",
 "permission_mode": "bypassPermissions",
 "prompt_id": "bc844973-27d2-4649-9de6-8ad9c7b3bd94",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
