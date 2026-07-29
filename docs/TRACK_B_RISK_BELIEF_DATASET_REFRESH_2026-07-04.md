# Refresco del dataset de creencia de riesgo con mezcla diversa — 2026-07-04

## Qué se hizo

Corrección a mi propio plan: Sprint 1 (`outputs/experiments/track_b_risk_belief_predictor_2026-07-04/`)
**ya usaba datos fixed-RNG** (`source_dir` = `track_b_risk_event_counterfactual_fixed_rng_2026-07-04`),
no el ledger viejo pre-fix como yo había asumido al escribir el plan. Verificado directamente en
`summary.json`.

Lo que sí faltaba: solo dos políticas (`ppo_mlp`, `real_kan`) alimentaban el dataset. Corrí 6
heurísticas más (`heur_forecast_threshold`, `heur_downstream_reactive`, `heur_s1_max_downstream`,
`heur_hysteresis`, `heur_disruption_aware`, `heur_tuned`) como rollouts completos (5 seeds × 12
episodios cada una, mismo protocolo), y combiné todo en
`outputs/experiments/track_b_belief_dataset_diverse_fixed_rng_2026-07-04/` (49,920 filas de paso,
425,688 eventos de riesgo, 8 políticas en total). Re-corrí
`scripts/audit_track_b_risk_belief_predictor.py` sobre esta mezcla.

## Hallazgo metodológico (no obvio, vale la pena anotarlo)

Las features `memory_only` (`mem_weeks_since_last_Ri`, conteos, EWMA) dieron **exactamente los
mismos AUC** que antes de añadir las 6 heurísticas, hasta muchos decimales. Esto no es un error:
bajo el RNG corregido, el calendario de riesgos es independiente de la política, así que para un
mismo (seed, episodio) el par (features de memoria, etiqueta futura) es idéntico sin importar qué
política generó el rollout. Añadir más políticas a la mezcla **no diversifica** `memory_only` —
solo diversificaría correr más semillas/episodios distintos. Donde sí añade valor real es en los
feature sets que incluyen contexto operacional (`operational_no_forecast`,
`with_forecast_regime`), porque backlog/fill-rate/presión de cola sí varían por política.

## Resultado: el patrón R24 es robusto a través de 8 políticas muy distintas

Para R24 a horizonte 1 semana, `operational_no_forecast` AUC por política:

| Política | AUC |
|---|---:|
| ppo_mlp | 0.615 |
| real_kan | 0.613 |
| heur_s1_max_downstream | 0.611 |
| heur_downstream_reactive | 0.609 |
| heur_forecast_threshold | 0.608 |
| heur_disruption_aware | 0.603 |
| heur_tuned | 0.600 |
| heur_hysteresis | 0.596 |

Todas caen en un rango angosto (0.596-0.615), sin importar si la política es RL entrenado o una
heurística de mano completamente distinta. Esto refuerza la conclusión de Sprint 1: la señal
predictiva para R24 a 1-2 semanas es una propiedad del **mundo simulado** (la dinámica del riesgo
+ el estado operacional correlacionado), no un artefacto de una política en particular — un
argumento más sólido para invertir en Sprint 2 sobre R24 específicamente.

Los patrones de R11 (degenera a horizonte≥2 semanas) y R13 (AUC alto solo con base rate ~99%,
poco accionable) también se replican de forma idéntica/consistente a través de las 8 políticas —
confirmando que la limitación de esos dos riesgos no es tampoco un artefacto de la muestra de 2
políticas original.

## Siguiente paso

Con esto confirmado, seguir con Sprint 2 tal como está en el plan: construir
`MLPBeliefExtractor`/`RealKANBeliefExtractor`, targeting R24 (1-2 semanas) como tarea auxiliar
principal.
