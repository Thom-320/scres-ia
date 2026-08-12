# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `workflow-subagent`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `ac224d0498fef0342`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T05:29:33.601148+00:00`
- **commit**: `b68bcf19`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-ac224d0498fef0342.jsonl`

## Final response

Auditoría completa. Todo lo que reporto abajo está leído de un campo concreto; doy ruta y ruta-de-claves. T y U1 **no existen en HEAD** — viven off‑HEAD en el commit `94b8dfe` (los extraje con `git show`).

## Respuesta directa a las tres preguntas que hiciste

**M — punto aislado, y el que casi salva la región no la habría salvado.** El contrato `contracts/program_m_shared_lift_reservation_v1.json` (`/risk_and_signal_design/connected_region_rule`) exige ≥3 celdas adyacentes que abarquen **dos niveles de hazard Y dos de duración**. Las seis celdas validadas son en realidad **tres físicas duplicadas**: `h_pi_per_tape` es *byte-idéntico* entre `s70` y `s85` (H_PI es clarividente, la calidad de señal no lo mueve). Las "dos celdas que pasan" son una sola. La vecina `h50_d120` falló por **0,000994** (LCB 0,009006262634792352 vs 0,01) — pero aun pasando sólo aporta un segundo hazard, no una segunda duración. La celda que aporta la segunda duración, `h75_d72`, tiene LCB **−0,006787466057508785**, a **0,016787** de la barra. `passing_connected_components: []`.

Además el screen (`hpi_screen_v1/result.json`, `cell_summaries[*].h_pi_mean`) **sí** había nominado una región válida: h50_d120 = 0,019974, h75_d72 = 0,015432, h75_d120 = 0,036245. La validación en semillas frescas (7300025–7300048) tumbó dos de tres. Es regresión a la media, no defecto.

**U1 — punto aislado y en el borde del trayecto.** Los cinco puntos del trayecto 8, celda `rho75_share90`: `+0,11851288253816857` (LCB +0,027286041002375087), y después **−0,0274 / −0,0962 / −0,0783 / −0,0458**. El punto 1 difiere del 0 sólo en `duration_impact` (1,1041 vs 1,6407) y ya cambia de signo. Su LCB (−0,09983423763100147) está a **0,1148** de la barra 0,015. Y el punto ganador es el índice 0 del trayecto, es decir un extremo: la conectividad no era establecible en una dirección por construcción. Más grave: ese punto se seleccionó de **12 candidatos sobre 3 cintas** con `classical_h_obs_loo_mean = +0,018288809667905342` y cintas [+0,0254, +0,0588, **−0,0294**]. Pasar de +0,0183 en 3 cintas a +0,1185 en 12 es un salto de 6,5× en la misma celda: hay riesgo serio de maldición del ganador, no verificado.

**T — el techo y el observable NO son el mismo estimando.** Techo `+0,022256594702943076` (LCB +0,018099955743517304) es un oráculo de selector sobre 48 cintas. El observable `+0,0023443405391502504` (LCB +0,0012526912231512386) es un **delta de un solo paso**; el propio artefacto lo dice: `"caveat": "one-step convertibility is not a closed-loop hybrid rollout"`. **No dividas uno entre otro para reportar η.** Contra la barra real del contrato (`go_to_training_gate`: "LCB95 residual value versus strongest deployable MPC at least 0.015") el observable queda a **0,0137473**.

## Lo que buscabas: ganó en media, murió en cola/equidad

**Ninguna de las seis lanes que pediste murió así.** Las dos que sí, y que aparecieron mientras auditaba a sus sucesores T/U1, son **O y Q** — y en ambas la distancia es minúscula y el asesino es la **corrección de simultaneidad**, no el efecto:

| | media (pasa) | guardarraíl | punto del guardarraíl | LCB simultáneo | barra | distancia |
|---|---|---|---|---|---|---|
| **O** rho75_share90 | +0,0985 (LCB +0,0660) | `ret_visible_cvar10` | **+0,03501669** | −0,008577578 | ≥0 | **0,008578** |
| **O** rho90_share75 | +0,0735 (LCB +0,0430) | `ret_visible_cvar10` | **+0,01953513** | −0,015506924 | ≥0 | **0,015507** |
| **Q** rho90_share90 | H_OL +0,1172 (LCB +0,1061) | `worst_product_fill` vs clásico | −0,004507798 | −0,026322146 | ≥−0,02 | **0,006322** |

En O los **tres puntos estimados de CVaR10 son positivos**. Lo que cruza cero es el borde inferior simultáneo sobre **69 estimandos** con crítico 2,8357534289190336. Cálculo mío a partir de `estimate` y `bootstrap_se` (derivado, no es un campo): con un borde por-comparación z=1,645, O rho75_share90 daría **+0,00973** y *pasaría*; rho90_share75 daría **−0,00079** y fallaría por menos de una milésima. En Q, rho75_share90 daría **−0,019738** y *pasaría* el margen −0,02 por 0,00026.

Es decir: dos programas confirmatorios murieron en cola con el punto estimado sano, y en tres de las cinco celdas la muerte la firma la multiplicidad. No estoy diciendo que haya que revocarlo — los bordes estaban preregistrados. Estoy diciendo que la distancia es de milésimas y que el mecanismo es identificable.

## Dos defectos de instrumento que debes conocer

**S se contradice consigo mismo dentro del mismo commit.** `results/program_s/s1b_execution_route_audit_v1/result.json@94b8dfe` certifica LOC_SURGE y CROSS_ECHELON_SURGE como `certified_transducer` ("certificate passes"). El diagnóstico de U1 del **20‑07** encuentra esas dos exactamente fallando: error máximo de replay **0,08571428571428573** (LOC_SURGE, campo `ret_visible_cvar10`) y **0,026651848286211866** (CROSS_ECHELON). La única máscara que la auditoría de rutas **excluyó** (PRODUCTION_QUALITY_SURGE) es la única cuyo error es pequeño: 1,3558433253764782e-05. La certificación acertó al revés. En HEAD, `results/program_s/s1_transducer_preflight_v1/result.json` sigue diciendo `PASS_S1_TRANSDUCER_PREFLIGHT_ALL_MASKS_ELIGIBLE`.

**V full DES tiene colisión de custodia.** `falsifiers.custody.evidence.status = "COLLISION"`: 24 de 60 semillas (8600001–8600024) ya estaban en un artefacto sellado y 12 en `results/demand_process/result.json`. Los seis contrastes salen **exactamente 0,0**, así que la colisión no cambia la conclusión, pero el run no es independiente.

**Dos discrepancias de cita que encontré:**
- El adjudicador de L cita el pico como `+0,005411 / LCB −0,007353`; el artefacto que él mismo referencia por SHA dice **+0,005414874419010528 / −0,007345042123380664**. Quinta decimal, pero no reconstruible tal como está escrito.
- El veredicto de T `STOP_NO_HISTORY_PREDICTIVE_INCREMENT` **contradice sus propios números**: `history_boosting.delta_vs_frozen = +0,12901033194245806` con `cluster_lcb95 = +0,0915902110922272`. El incremento predictivo existe y es grande. Lo que no existe es incremento **de decisión**: `action_change_fraction = 0,046875`. El nombre del veredicto está mal puesto y es citable en contra nuestra.

**Rutas absolutas de lo auditado:** `<HOME>/Projects/research/scres-ia/results/program_m/hpi_validation_v1/result.json`, `<HOME>/Projects/research/scres-ia/results/paper2_search/program_l_l0_extended_grid.json`, `<HOME>/Projects/research/scres-ia/results/program_v/{prelearner_gate_v1,des_port_v1}/result.json`, `<HOME>/Projects/research/scres-ia/results/program_o/fixed_clock_hobs_corrective_validation_v1/remote_run/artifacts/validation/result.json`. T/U1/Q sólo en `94b8dfe` (copias extraídas en `<TMP>/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/scratchpad/off/`). No modifiqué ningún fichero del repo.

## Raw payload

```json
{
 "agent_id": "ac224d0498fef0342",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-ac224d0498fef0342.jsonl",
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
