# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a9c5c01130b11153d`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-10T01:28:27.927787+00:00`
- **commit**: `62c95043`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

Lanzada. Ahora tus dos preguntas, que son las importantes.

## Qué es Submission A

**Es la sumisión del primer paper** — el paquete de Program Q a C&IE. Se convirtió en la llave de todo lo demás porque varios contratos escribieron «no ejecutar ciencia nueva hasta que Submission A tenga recibo».

Y el propio repositorio ya había diagnosticado que esa llave no protege nada científico. `docs/DONDE_PODEMOS_SER_LAXOS_2026-08-02.md` §5:

> *«el bloqueo real de Submission A es **editorial y humano**, no experimental»*

Por eso levantarla es coherente y no un atajo. **Levanté sólo eso.** Lo que sigue en pie, y lo dejé escrito para que no se lea como aflojar en general: `H_compute`, la no-inferioridad en servicio y recursos, `r2_is_not_a_control_gate`, la escalera con la red al final, y `LCB95(Delta_obs) ≥ SESOI` para cualquier afirmación de calidad.

**La Puerta C queda autorizada sólo en su carril de amortización** — el único que el contrato permite sin residual de calidad. Y anoté el precio: si Garrido objeta el constructo después, ese resultado nace con una objeción de alcance que no tendría si hubiéramos esperado.

## Qué es «control», y por qué está lejos

Distingo tres cosas que el borrador mezcla:

**Predicción** — acertar el valor de resiliencia de una configuración. Es la Puerta B, es el nivel 3 de Garrido, y es donde el KAN gana.

**Control** — que una política **cambie las acciones dentro del episodio** según lo que observa, y que eso **mejore el servicio real**. El estimando es:

```
Delta_obs = V(política observable) − V(mejor controlador estructurado)
```

**No es «la red predice mejor».** Es «la red, decidiendo, sirve más raciones que la mejor regla que sepamos escribir». Y ahí está cerrado en todas partes: `STRUCTURED_CONTROL_SUFFICES`, `THRESHOLD_RULE_SUFFICES`, privilegiado−Bayes `+0,00076`.

**Amortización** — igual calidad, menos coste online. Es lo que la Puerta C mide, y no exige `Delta_obs > 0`.

## Cómo lo lograríamos

La respuesta está en nuestro propio dato y es una sola frase: **`Delta_obs > 0` exige que la creencia exacta no sea calculable**. Donde el estado latente tiene tres estados y modelo conocido, un filtro escrito a mano es óptimo y la red empata — eso es todo lo que hemos construido.

Nuestro único contraejemplo lo dice: en `contention_v1`, con régimen **semi-Markov**, el aprendiz batió al belief-MPC por **+0,0136 [+0,0124]**. Ahí el filtro está **mal especificado por construcción**.

Así que la receta, tomada de lo que cada programa ya demostró:

- **estado latente que un filtro de primer orden no capture** — semi-Markov, duración oculta, o suficientes fuentes correlacionadas para que el posterior exacto sea intratable;
- **contención por recurso no fungible** — el único mecanismo con headroom material (`H_PI = 0,1515`, nulo fungible exactamente 0);
- **objetivo vectorial con restricción de cola** — la ReT de Garrido es nativamente un vector de cuatro componentes, y Program Q ganó al open-loop y **cayó exactamente por el guardrail de peor producto**. Escribir a mano una regla que optimice cuatro componentes con restricción es difícil; es la ventaja estructural que RL sí tiene;
- **planificación cara online**, que ya está certificada.

Y la parte incómoda: **eso ya no es la MFSC de Garrido**. Va en el brazo `mfsc_stressed`, con su precio de fidelidad medido, y ningún resultado suyo se presenta como reproducción de la tesis.

**Lo que no funcionará, y ya lo probamos cuatro veces hoy:** hacer el problema más difícil sin hacer la creencia intratable. Buffer con precio, presupuesto, caducidad, escasez de almacén — los cuatro cerraron en cero exacto.

Te doy el veredicto de A2 y el de la confirmación B en cuanto cierren.

## Raw payload

```json
{
 "agent_id": "a9c5c01130b11153d",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-a9c5c01130b11153d.jsonl",
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
 "last_assistant_message": "dime los veredictos de A2 y la confirmación B",
 "permission_mode": "bypassPermissions",
 "prompt_id": "1e55f003-50d1-48e0-9dc1-dc489dc31f5b",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
