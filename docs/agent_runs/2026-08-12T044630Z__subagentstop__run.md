# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a3759fcc15d743087`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T04:46:30.605736+00:00`
- **commit**: `6811ac9f`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

## Fase 1a: la prima de predicción no sobrevive. Es el resultado del día

```
mlp_tuned  vs gaussian_process  +0.0342 [-0.1030, +0.1715]   no
kan_tuned  vs gaussian_process  +0.0172 [-0.1783, +0.2128]   no
recurrent  vs gbdt_lagged       -0.0300 [-0.1113, +0.0513]   no
```

Contra `linear_interactions` el MLP daba **+0,1081 [+0,0601, +0,1561]** y el recurrente **+0,1487 [+0,1069, +0,1905]**. Contra una clase que incluye un proceso gaussiano, **no pasa ninguno** — y `gbdt_lagged` (0,9306) es sencillamente **mejor** que el brazo recurrente (0,9007).

Mismas siete features, mismos folds, mismas semillas, mismo criterio congelado. **Lo único que cambió fue la clase comparadora**, y costó 15 minutos de portátil.

**El número era real y su interpretación no.** Medía que `linear_interactions` no es el mejor modelo clásico de esta superficie — una afirmación sobre el *baseline*, no sobre la red. Las cinco revisiones dijeron que la clase era demasiado estrecha para sostener las palabras «mejor comparador no neuronal». Tenían razón.

El artefacto guarda `premium_under_primary_only: true` junto a `readjudicated_premium: false`. **La diferencia entre esos dos campos es el hallazgo.**

### Y estuve a punto de repetir el defecto que documenté esta mañana

La re-adjudicación imprimía `SURVIVES` mientras un proceso gaussiano batía a todos los brazos neuronales — porque su cadena de veredicto leía `gate_b_confirmation_v3`, la corrida confirmatoria, cuyo `per_fold` sólo tiene la clase estrecha. **Un veredicto gobernado por el artefacto conveniente en vez de por la evidencia más amplia**: la quinta instancia del patrón, escrita por mí, horas después de escribir la regla contra él.

Corregido: el estado lo gobierna la clase más ancha efectivamente ajustada, y el artefacto registra qué corrida lo gobernó. Veredicto real: **`SURFACE_PREMIUM_SURVIVES_THE_NARROW_CLASS_ONLY_NOT_THE_WIDENED_ONE`**.

### Dónde queda el proyecto, con las dos fases cerradas

| estimando | veredicto |
|---|---|
| prima de **predicción** vs clase completa | **cae** — un GP y un GBDT con lag baten a las redes |
| **retención** en el bucle externo | **vive** — +0,0607, 6/6 familias con inferencia simultánea |
| **portador neural** en el bucle externo | **empate con el clásico delante** — −0,0070, `ucb1_transfer` primero |
| **control** | cerrado en todas partes |
| **amortización** | cerrada, sin experto que califique |

Es la cuarta fila de la tabla que preregistré: *cae 1a, sobrevive 1b*. **No nos quedamos sin claim**, pero el claim cambia de dueño: lo que responde a Garrido es **retener estado de búsqueda reduce el coste de redescubrimiento**, y el portador que mejor lo hace es **UCB1, no una red**.

Ese sigue siendo un resultado publicable y contesta su pregunta 1 — sólo que la respuesta es «no hace falta una red», y eso hay que decirlo con esas palabras.

**No lanzo la Fase 2.** Su premisa era que algo sobreviviera a la Fase 1 en predicción, y no sobrevivió. Antes de gastar un bloque nuevo conviene que decidas si el paper es el negativo con la retención como positivo, o si quieres que ataque primero la Fase 3 —surrogate orientado a decisión con optimizador exacto—, que ahora tiene un rival claro y duro: el GBDT con lag.

Commits `e11be022` y `6811ac9f`, empujados.

## Raw payload

```json
{
 "agent_id": "a3759fcc15d743087",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a3759fcc15d743087.jsonl",
 "agent_type": "",
 "background_tasks": [],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "high"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "ataca la fase 3",
 "permission_mode": "auto",
 "prompt_id": "3983416a-f0a6-4651-afa1-26c3a8b5e5ba",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
