# El nuevo contrafactual de Codex hereda el mismo defecto de RNG — confirmado empíricamente

Fecha: 2026-07-03. Corrige una afirmación central de
`docs/TRACK_B_RISK_EVENT_COUNTERFACTUAL_PRE_VERDICT_2026-07-03.md` (Codex,
`scripts/audit_track_b_risk_event_counterfactual.py`).

## La afirmación en cuestión

El documento de Codex dice (línea 15-16): *"R_reset es el mismo episodio y la misma semilla de
evaluación, **es decir, la misma trayectoria de riesgos y demanda**, pero reemplazando las
acciones..."* — es decir, asume exactamente el supuesto que
`docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md` (mío, mismo día, corrida
antes) ya había mostrado que es falso. El script nuevo usa el mismo mecanismo de empalme
(`run_policy_episode` con `reset_steps`/`reset_action`, misma `eval_seed`, continúa con la
política tras la ventana) que el auditor original — solo mejora las anclas (eventos reales por
`risk_id`) y la referencia de calma (cuartil de menor intensidad propia). No corrige, ni menciona,
el problema de fondo.

## Verificación directa (no narración — corrida yo mismo, reusando su propio código)

Cargué `ppo_mlp` seed=1 con `load_runtime`/`run_policy_episode` de su script, corrí el episodio
"full" (seed=1, episodio=1, eval_seed real), tomé el primer ancla real de R11 con ventana pre
válida (paso 16, ventana [-4,-1] → pasos 12-15, solo **4 de 104 pasos del episodio**), y corrí el
reset con esos 4 pasos sustituidos. Comparé los `sim.risk_events` de ambas corridas:

```text
full ret_excel  = 0.005442187706643586  (876 eventos de riesgo)
reset ret_excel = 0.005602454156193999  (865 eventos de riesgo)
Calendarios idénticos: False
Eventos compartidos: 683 / 876 (78%)
Primera divergencia: evento #143 de 876 (≈16% del episodio)
```

Con **solo 4 pasos alterados de 104**, el calendario de riesgos ya diverge en menos de una quinta
parte del episodio, y al final solo comparte el 78% de sus eventos con el episodio "full". El
delta de ReT Excel medido (`+0.00016` en este caso concreto) no puede atribuirse limpiamente al
cambio de comportamiento en la ventana — una parte no cuantificada, probablemente dominante dado
lo temprano de la divergencia, viene de que el "reset" literalmente vivió un episodio distinto
(menos roturas de R11, otra mezcla de riesgos) a partir de ese punto.

## Qué significa para los números ya reportados

Las 8 filas de `summary_by_policy_risk.csv` / `verdict.md` (deltas de orden 1e-5 a 1e-6, tasas
positivas 45-58%, CI95 cruzando cero en casi todos los casos) tienen el mismo problema que
encontré en mis propias Rondas 1-4: son consistentes con **ruido dominado por divergencia de
calendario de riesgo**, no con una medición causal limpia del valor de la anticipación. El
**veredicto direccional** ("no hay evidencia suficiente de prevención") coincide con todo lo demás
que hemos encontrado por vías válidas (rollouts completos de heurísticas, event-study
descriptivo), así que probablemente no está llevando a Garrido en una dirección equivocada — pero
los números específicos (deltas, tasas positivas, CI95) de esta tabla **no deben citarse como una
prueba causal**, porque el método que los produce no lo es.

## Recomendación

1. No usar `risk_event_counterfactual_pre.csv` / la tabla de `verdict.md` como evidencia causal en
   el documento para Garrido ni en el manuscrito. Puede mencionarse como intento metodológico, con
   la misma nota honesta que ya aplicamos al auditor original.
2. Este es el segundo diseño de empalme (el original de Codex + el mío + ahora este) que cae en el
   mismo defecto — confirma que **no es un problema de calidad de ancla o de referencia de calma**,
   es la arquitectura de RNG del simulador (ver
   `docs/TRACK_B_RNG_ENTANGLEMENT_FIX_INVESTIGATION_2026-07-03.md` para el fix diseñado y su costo).
   No vale la pena iterar más variantes de empalme sin ese fix.
3. La vía que sigue siendo válida y ya está corrida:
   `docs/TRACK_B_HEURISTIC_FULL_ROLLOUT_VERDICT_2026-07-03.md` (rollouts completos independientes,
   sin empalme) — apunta a la misma conclusión honesta sin este confound.
4. Avisar a Codex de este hallazgo para que no se seguan iterando variantes del contrafactual por
   ventana sin el fix de RNG.
