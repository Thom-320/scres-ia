# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a204d19aa9ebe67c6`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-09T20:51:45.782933+00:00`
- **commit**: `f6cdbf9e`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

**Veredicto: `NO_VALID_NONNEURAL_COMPARATOR`. Y la red ganó.**

Las dos cosas a la vez, y el orden importa: la regla congelada dice que si `f3` falla, el veredicto es ése **aunque la red gane**. Ganó, y el veredicto se queda.

## Los números

```
mlp                   98.4641      threshold_rule   97.9942
mlp_frozen_history    98.2060      constant_best    97.9696
mlp_shuffled_history  98.0975      untrained_net    72.0266
```

| contraste | media | IC95 | tapes |
|---|---|---|---|
| **mlp − regla** | **+0,4699** | **[+0,2372, +0,7024]** | 36/48 |
| mlp − constante | +0,4945 | [+0,2646, +0,7273] | 36/48 |
| mlp − placebo **congelado** | +0,2118 | [+0,0035, +0,4199] | 33/48 |
| mlp − placebo **barajado** | +0,1033 | **[−0,1670, +0,3718]** | 26/48 |
| regla − constante | +0,0246 | **[−0,0302, +0,0754]** | 32/48 |

Seis de siete falsadores en verde. Presupuesto 215.889 parámetros (error 7,9 %), bloques disjuntos, control obligado a diferir. **Falla `f3`**: la regla no bate a la constante.

## `f3` falló por lo contrario de lo que vigilaba, y el defecto es mío

`f3` existía para impedir un **comparador de paja**. Aquí pasó lo opuesto: **la constante está saturada**. Tras el refuerzo que yo mismo introduje después del pre-vuelo, la constante ajusta en **98,21** y la regla en **98,30** — la versión adaptativa no tiene nada que mejorar.

Escribí un falsador que **no distingue «el comparador es de paja» de «el comparador es tan bueno que su versión adaptativa no añade nada»**. Eso es un defecto de mi diseño.

**No reetiqueto el artefacto.** Reescribir la regla después de ver un +0,47 que me gusta es exactamente el mecanismo que llevo todo el día desmontando en el trabajo de otros.

## Dos hallazgos que valen sin depender del veredicto

**El `+1,60` original no se reproduce.** Contra un comparador bien ajustado la ventaja es **un tercio**: +0,49. Y la diferencia no la causó la red — la causó **lo mal ajustado que estaba el comparador**: la constante de esta corrida (97,97) **bate a la regla de aquélla** (97,14).

**La memoria explica parte, no todo.** La red bate a su placebo **congelado**, pero contra el **barajado el intervalo cruza cero**. Tener historia ayuda; **que esté ordenada no está demostrado**. Eso apunta a `PREMIUM_IS_CAPACITY_NOT_MEMORY`, que es la distinción que los dos placebos existían para hacer — y una respuesta distinta, no peor, a la Q1 de Garrido.

## Lo que aprendimos del entorno

**En `track_b_v1`, una constante bien buscada casi agota la clase no neuronal que sabemos escribir.** Ése es el hallazgo estructural, y explica por qué la prima parecía tan grande antes: no había con qué compararla.

El bloque `9200001–9200120` queda quemado. El sucesor necesita bloque nuevo, `f3` sustituido por «el mejor no-neuronal es `max(constante, regla, …)`» con validez contra un **suelo absoluto**, y una familia estructurada de verdad — umbral sobre un estadístico más rico o control por creencia.

¿Lanzo la Puerta B, que ya está preregistrada y no compite por semillas, o prefieres primero el sucesor de A con el falsador arreglado?

## Raw payload

```json
{
 "agent_id": "a204d19aa9ebe67c6",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a204d19aa9ebe67c6.jsonl",
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
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "high"
 },
 "hook_event_name": "SubagentStop",
 "last_assistant_message": "corre la puerta B",
 "permission_mode": "auto",
 "prompt_id": "e356cc79-43ba-4c31-b331-27940cc39315",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
