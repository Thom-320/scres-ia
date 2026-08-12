# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `workflow-subagent`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a1b8bc2e5d649bec2`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T05:29:54.095194+00:00`
- **commit**: `b68bcf19`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a1b8bc2e5d649bec2.jsonl`

## Final response

Auditoría completada. Todos los números verificados contra los campos exactos, y las cifras titulares de DRA-2b y F reconstruidas de forma independiente desde los CSV del propio artefacto.

## Lo que confirmo

**DRA-2b: los dos números son correctos.**
`results/program_d/dra2b_long_horizon_calibration/verdict.json`
- `ret_delta_mean_ci95[0]` = **0.02211993672160424**, CI95 [0.017247548348180042, 0.027344305432644444]
- `service_loss_reduction_mean_ci95[0]` = **0.030752649771667553**, CI95 [0.019873060769892113, 0.040328686453635625]
- Reconstruidos a mano desde `oracle_state_summary.csv` + `resource_static_rows.csv`: 0.0221199367 y 0.0307526498. Coinciden a 10 dígitos.
- La barra es **0.05 sobre el punto** (`scripts/run_dra2b_long_horizon_gate.py:297`: `service_ci[0] >= .05 and service_ci[1] > 0`; `ci[0]` es la media, no el límite inferior). Distancia = **0.019247**. Ni siquiera el extremo superior del IC (0.04033) llega.

**Programa F: correcto.** `results/program_f/screen/verdict.json` → FSC-24 `oracle_ret_delta_ci95` = [0.022584251053516607, 0.00670892917424756, 0.04898328997154962]; `observable_conversion` es `false` en **24/24** celdas. La mejor conversión es FSC-15 = 0.47138631202220543 contra 0.50, pero además su oracle es 0.007942 < 0.01 y es celda de 1 token (inadmisible): no podía pasar por ninguna vía.

## Lo que la auditoría añade (y esto cambia la lectura de DRA-2/2b)

**1. El 3.075% de servicio es el más favorable de los nueve contrastes posibles.** La regla de dominancia de recursos admite exactamente **una** de las nueve políticas estáticas —`threshold_5000__wait_72h`— y es la **peor en ReT de las nueve** (0.695776 vs 0.699103–0.704763). Recomputado desde `resource_static_rows.csv`, la reducción de servicio del oráculo contra cada estática es:

| comparador | ΔReT | Δservicio |
|---|---:|---:|
| threshold_5000__wait_72h (el admitido) | +0.022120 | **+0.030753** |
| threshold_5000__wait_48h | +0.018793 | +0.025350 |
| threshold_1000__wait_48h (mejor ReT) | +0.013134 | **+0.008220** |

Contra la mejor estática el beneficio de servicio es **0.822%**, no 3.075%: la distancia a la barra pasa de 0.0192 a **0.0418**. DRA-2b no perdió por poco en servicio; perdió por poco *sólo contra el comparador más débil disponible*.

**2. El ReT de DRA-2b sí es robusto al comparador.** El peor de los nueve contrastes da **+0.013134**, todavía por encima de 0.01. Eso es real y publicable.

**3. El PASS de servicio de DRA-2 (5.174%) es de filo de navaja.** `results/program_d/dra2_exact_branching_calibration/resource_gate_verdict.json` → `service_loss_reduction_mean_ci95[0]` = 0.051741499975052035. Recomputado: contra `threshold_5000__wait_48h` sería **0.041190** (FALLA la barra del 5%). Esa política quedó fuera del sobre por **0.017 departures** (12.567 vs 12.550 del candidato, 0.13%). Con ese margen infinitesimal, la narrativa "DRA-2 pasó ambas puertas prácticas y sólo cayó por horizonte" se convierte en "falló servicio igual que DRA-2b".

**4. La puerta de DRA-2b replicó en un universo de cintas disjunto y nadie lo reportó.** `results/program_e/oracle_training/verdict.json` corre el mismo `DRA2B_LONG_HORIZON_PRE_TREE_GATE` sobre las 80 cintas de entrenamiento de Programa E (900001–900080, solapamiento **0** con 860001–860060), 160 estados: ReT **+0.021833** [0.017920, 0.026141], servicio **+0.033474** [0.024315, 0.042243]. Mismo veredicto, mismos dos gates fallando. El near-miss es reproducible, no un artefacto de muestra.

**5. Defecto de instrumento en el IC de DRA-2b (no invierte el veredicto).** `scripts/run_dra2b_long_horizon_gate.py:119-123` remuestrea los **120 estados i.i.d.**, y hay 2 estados por cinta sobre 60 cintas — no está clusterizado por cinta, a diferencia de D1 (`cluster_ci`) y DRA-1. DRA-2 no sufre esto (1 estado por cinta, n=60). Inflando el IC por √2 (correlación intra-cinta perfecta, el peor caso) el servicio llegaría a ~0.0443 y el ReT bajaría a ~0.0152: **ninguna de las dos puertas cambia de signo**.

**6. Programa F: el patrón estructural que el veredicto no nombra.** Las únicas 4 celdas con `tree_ret_delta` positivo (FSC-02 0.002692, FSC-10 0.004422, FSC-15 0.003744, FSC-19 0.003025) son **todas de 1 token** (3 acciones). Las 8 celdas admisibles de 2 tokens tienen las 8 el delta del árbol negativo (−0.012074 a −0.029145). La conversión observable sólo fue positiva donde el espacio de acción era mínimo. Además 7/8 admisibles pasan la barra material de 0.01 (la excepción es FSC-05, media 0.003560).

**7. Nota de instrumento en la conversión de F:** `conversion = tree_delta / oracle_ci[0]` divide un delta de **rollout episódico completo** entre un delta de **ramificación por estado a 4 semanas** — numerador y denominador no comparten horizonte, así que "0.471" no es literalmente "capturó el 47% del headroom". El STOP es robusto a esto porque la puerta también exige `tree_delta > 0` y las 8 admisibles son negativas.

## La respuesta a tu pregunta central

**En D, D1, DRA-1, DRA-2, DRA-2b, E y F no hay ni una sola lane que ganara en media y muriera en cola o en equidad.** La puerta de cola de Programa E **pasó** (`candidate_cvar10_ret` 0.4101893414151531 ≥ `baseline_cvar10_ret` 0.4101678074032772). Lo más parecido es el árbol ret-only de D1, que rompió el guardrail de pedidos perdidos en **+23.25%** [18.74%, 27.69%] contra un techo de 2% — pero también perdía en media (−0.010709), así que no es un caso de "ganó en media, murió en cola".

El único caso de "ganó en media, murió en una restricción" es **DRA-2b: ganó en ReT y murió en un endpoint co-primario de servicio** — y, como muestra el punto 1, murió más ampliamente de lo que decía el registro.

Ficheros clave: `<HOME>/Projects/research/scres-ia/results/program_d/dra2b_long_horizon_calibration/verdict.json`, `<HOME>/Projects/research/scres-ia/results/program_d/dra2_exact_branching_calibration/resource_gate_verdict.json`, `<HOME>/Projects/research/scres-ia/results/program_e/oracle_training/verdict.json`, `<HOME>/Projects/research/scres-ia/results/program_e/validation/verdict.json`, `<HOME>/Projects/research/scres-ia/results/program_f/screen/verdict.json`.

## Raw payload

```json
{
 "agent_id": "a1b8bc2e5d649bec2",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a1b8bc2e5d649bec2.jsonl",
 "agent_type": "workflow-subagent",
 "background_tasks": [
  {
   "description": "Find the lanes that came closest to a claim, verify their numbers against artifacts, and rank what is reopenable",
   "id": "wydnbmt6b",
   "name": "closest-losses-audit",
   "status": "running",
   "type": "workflow"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "xhigh"
 },
 "hook_event_name": "SubagentStop",
 "permission_mode": "auto",
 "prompt_id": "ee334d76-63aa-4489-9fa7-aac74d371f0b",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
