# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `af7b011d3edad243c`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-10T04:11:55.794603+00:00`
- **commit**: `2add435b`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`

## Final response

## Puerta C, etapa C0: `NO_QUALIFYING_EXPERT`

Cerró **antes de entrenar nada**, que es exactamente lo que su regla de parada exige. Y cerró por un hallazgo que no esperaba.

```
k3_strong_mpc
  planifica   False   (0 evaluaciones de candidato, 0 llamadas al simulador, sobre 320 decisiones)
  caro        False   (razon experto/red 0.051x -- es 20x MAS BARATO)
  mejor       True    (ret_order +0.01242 [+0.00546, +0.01928] vs budgeted_sS)

estar_direct_des_mpc
  planifica   True    (192 llamadas al DES)
  caro        True    (razon experto/red 44.359x)
  mejor       False   (NUNCA MEDIDO)
```

**El brazo que llamamos `strong_mpc` no es un MPC.** Es `paced_policy(α, β, γ)` — la misma regla en forma cerrada que el brazo `inventory_paced`, con α liberado. Su propio docstring dice *«no latent state»*. Lo instrumenté sobre 320 decisiones reales y no evalúa un solo candidato ni llama una sola vez al simulador.

Es el mismo defecto que el «techo»: **un nombre puesto por el papel esperado, sin una medición que lo respalde**. Escribí `c1` justamente porque acabábamos de cometerlo, y disparó.

Lo que **no** se cae: su ventaja de calidad es real. `+0,01242 [+0,00546, +0,01928]` sobre 300 tapas selladas, con `lost` no-inferior y recurso exacto. Lo que se cae es llamarlo planificador — y con ello, que haya algo que amortizar. `Δ_amortización` sale **negativo por construcción**: la regla cuesta 1,67e-07 s por decisión y la red que la imitaría cuesta 3,29e-06 s.

**El otro candidato sí planifica** —192 llamadas al DES, 44.359× la latencia de la red— pero su preflight es `engineering_only` con `learner_trained: false` y **no lo compara contra ninguna heurística**. No es que se midiera y perdiera: **nunca se midió**. Amortizarlo sería comprar un desconocido.

Los dos falsadores de calidad y coste fallan, `c4` pasa —los candidatos reciben veredictos distintos, así que la auditoría discrimina— y el artefacto lleva `training_authorized_by_this_run: false`.

Control de cordura que corrí de paso: el replay del brazo K3 con la política instrumentada da `ret_order` 0,93437 sobre 40 tapas contra 0,94095 sellado sobre 300. Coherente con el submuestreo, sin sorpresas.

### Qué queda

C1 **no se autoriza**. Lo que abriría la Puerta C es un experimento distinto y acotado: **medir la calidad del `DirectDESMPC` contra la mejor heurística barata** en el DES extendido. Si gana, tiene las dos condiciones —caro y mejor— y C1 se construye tal como está preregistrado. Si pierde, la amortización queda cerrada con evidencia en vez de con una ausencia.

Ese experimento no es la Puerta C; es su prerrequisito, y no lo lanzo sin que lo decidas porque es el único de esta serie con coste de cómputo real.

## Raw payload

```json
{
 "agent_id": "af7b011d3edad243c",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/agent-af7b011d3edad243c.jsonl",
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
 "last_assistant_message": "mide la calidad del DirectDESMPC contra la heurística",
 "permission_mode": "bypassPermissions",
 "prompt_id": "af43b35e-3522-497b-ad43-808557381242",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
