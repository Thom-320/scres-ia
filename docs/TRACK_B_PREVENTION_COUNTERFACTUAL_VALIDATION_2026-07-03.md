# Validación del contrafactual R_full - R_reset(w) — 2026-07-03

## Qué se intentó

`scripts/audit_track_b_prevention_mechanism.py` (Codex) clasificó PPO+MLP y Real-KAN como
"reactiva" usando `R_full - R_reset(w)` por ventana (pre/evento/post) con una política estática
fija (`s2_d1.50`) como referencia de reemplazo. Antes de aceptar esa conclusión, se construyó
`scripts/audit_track_b_prevention_sanity_check.py` para validar la REGLA de clasificación contra
tres referencias de comportamiento conocido: una estática (debe salir "sin señal"), una
heurística explícitamente preventiva (`heur_forecast_threshold`, debe salir "preventiva") y una
explícitamente reactiva (`heur_downstream_reactive`, debe salir "reactiva").

## Iteraciones y hallazgos reales

1. **Primera corrida (método original, referencia fija):** falló contra ambas heurísticas de
   referencia (ninguna clasificó como se esperaba). Solo la estática pasó.
2. **Hallazgo #1 — dilución del ReT de episodio completo.** El ReT se promedia sobre ~686
   pedidos de un episodio de 104 semanas; una ventana de ~13 semanas solo toca ~78 pedidos. Se
   implementó `windowed_ret_excel` (ReT restringido a los pedidos colocados dentro de la ventana,
   preservando el estado acumulado correcto de la fórmula). No resolvió el problema por sí solo.
3. **Hallazgo #2 — referencia de reset confundida (el más importante).** Usar una única
   política estática fija (`s2_d1.50`) como reemplazo para TODAS las políticas mezcla el efecto
   de "quitar la reacción/anticipación" con el de "cambiar a un turno/despacho que esa política
   nunca usaría". Se corrigió calculando, por política, su propio vector de acción promedio en
   períodos de calma (`event_window=="baseline"`) y usando ESO como referencia de reset. Con esta
   corrección, `heur_forecast_threshold` clasificó correctamente como preventiva (episodio
   completo, ventana post = [+1,+8]).
4. **Hallazgo #3 — inestabilidad de la ventana.** `heur_downstream_reactive` (reacciona a
   presión de cola, una señal retrasada) nunca clasificó como reactiva. Al ampliar la ventana
   post-evento de [+1,+8] a [+1,+20] semanas para darle tiempo a una señal retrasada, la
   clasificación de `heur_forecast_threshold` **se revirtió** (volvió a "sin señal"). Es decir:
   pequeños cambios de ventana voltean el signo de un delta ya de por sí diminuto (orden 1e-5
   frente a un ReT de ~0.005-0.006).

## Veredicto

**El método, tal como está, no tiene suficiente poder estadístico a esta escala (5 semillas × 12
episodios = 60 pares) para autovalidarse de forma estable.** Los deltas relevantes son
consistentemente del orden de 1e-5 a 1e-6 y cambian de signo con ajustes metodológicos menores
(ancho de ventana, dilución). Seguir ajustando parámetros hasta que las tres referencias
clasifiquen "bien" sería buscar la configuración que por azar da la respuesta esperada, no una
validación real.

**No se debe usar `R_full - R_reset(w)` todavía para afirmar que PPO+MLP o Real-KAN son
"reactivas"** (ni "preventivas"). La conclusión "reactiva" del run `full5` de Codox
(`outputs/experiments/track_b_prevention_mechanism_audit_2026-07-03_full5/`) usa la referencia de
reset fija (no corregida) y no ha sido validada contra las heurísticas de referencia — no debe
citarse como hallazgo cerrado.

## Qué sí queda en pie

- La causalidad de Granger y la correlación cruzada con rezagos (Codex,
  `lead_lag_summary.csv`/`cross_correlation_summary.csv` en el mismo bundle) no tienen este
  problema de referencia confundida — miden asociación temporal directamente sobre las series de
  acción y riesgo, sin sustituir ninguna acción. **Pero no miden valor causal en ReT**. Deben
  tratarse como evidencia descriptiva de timing, no como prueba suficiente de prevención o
  reacción útil para resiliencia.
- Los otros componentes del audit (ablación de forecast sin reentrenar, sensibilidad por
  oclusión) no dependen de este contrafactual y siguen siendo válidos tal cual.

## Recomendación

1. Pausar el refinamiento del contrafactual por ventana.
2. Presentar Granger/correlación cruzada solo como evidencia descriptiva auxiliar para la pregunta
   preventivo/reactivo. No reemplazan el test causal basado en la ReT del Excel.
3. En el documento para Garrido, dejar la pregunta preventivo/reactivo explícitamente abierta,
   no cerrada con "reactiva" — es la posición honesta dado lo encontrado aquí.
4. Si se quiere retomar el contrafactual por ventana más adelante, haría falta mucha más escala
   (varias veces más semillas/episodios) para que el ruido no domine señales de esta magnitud, o
   un rediseño que ancle la ventana por tipo de señal (forecast vs. presión de cola) en vez de un
   ancla única basada en régimen.
