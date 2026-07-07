# Heurísticas como rollouts completos independientes (vía válida) — 2026-07-03

## Qué es esto

Reemplaza el intento de `R_full - R_reset(w)` (inválido, ver
`docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md`) por la comparación honesta
que sí es válida hoy: evaluar cada heurística de referencia como su **propia política completa**,
reset fresco por episodio, mismo protocolo que las políticas entrenadas (5 seeds × 12 episodios,
`adaptive_benchmark_v2`, v7, `track_b_v1`). Sin empalme, sin anclar sobre eventos futuros — el
mismo patrón CRN-pareado que ya usa el resto del proyecto (E1-E6, no-forecast confirm).

Script: `scripts/audit_track_b_heuristic_full_rollout.py`. Artefacto:
`outputs/experiments/track_b_heuristic_full_rollout_2026-07-03/`.

## Resultado en ReT Excel (n=60 episodios por heurística)

| Política | order_ret_excel_mean | CI95 | fill_rate | cost_index |
|---|---:|---|---:|---:|
| heur_downstream_reactive | 0.004484 | [0.004450, 0.004519] | 0.0033 | 0.333 |
| heur_forecast_threshold | 0.005389 | [0.005347, 0.005430] | 0.0036 | 0.947 |
| heur_s1_max_downstream | 0.004573 | [0.004531, 0.004615] | 0.0033 | 0.333 |

Para contexto (ya verificado en documentos previos, mismo protocolo):

| Política | order_ret_excel_mean |
|---|---:|
| Mejor estática pura (s2_d1.50) | 0.005428 |
| PPO+MLP (10 seeds, canónico) | 0.005898 |
| Real-KAN (10 seeds) | 0.005938 (+9.4% vs. estática) |

**Lectura**: ninguna heurística de referencia (ni la "preventiva" por forecast, ni la "reactiva"
por presión de cola, ni el strawman de downstream siempre-máximo) supera a la mejor estática pura,
y ninguna se acerca a PPO+MLP o Real-KAN. `heur_forecast_threshold` es la más cercana pero queda
por debajo de la estática (0.005389 vs 0.005428) y a mucha distancia de PPO (0.005898) — a pesar
de tener costo casi máximo (0.947, casi siempre en S3+downstream alto). Esto confirma, de forma
independiente y sin el confound de RNG, que la ganancia de las políticas entrenadas **no** se
explica por replicar una regla simple de "reaccionar a la cola" o "reaccionar al forecast" — hace
falta algo más matizado que ninguna heurística de mano captura.

## Patrón de acción alineado a eventos reales (validación del método descriptivo)

Para riesgos frecuentes (R11/R13/R24), `heur_forecast_threshold` (diseñada para ser preventiva)
NO muestra un patrón limpio de "sube antes, baja después" — se mantiene saturada casi todo el
tiempo (~0.91-0.93) sin una tendencia clara ligada al evento. Esto es consistente con lo ya visto
en Real-KAN (también casi saturado) y es un resultado honesto, no un fallo del método: bajo
`adaptive_benchmark_v2`, el forecast está elevado gran parte del episodio, así que una regla de
umbral simple dispara casi siempre — la métrica de intensidad de acción no es sensible para
diferenciar "antes" de "después" cuando la política ya opera cerca del techo.

`heur_downstream_reactive` (diseñada para ser reactiva a presión de cola en op10/op12) tampoco
muestra un salto post-evento claro para R11/R13/R24 específicamente — probablemente porque esos
riesgos (rotura de estación, escasez de materia prima, demanda) no disparan directamente la
presión de cola en op10/op12 que esta heurística vigila; su lógica está más ligada a R22/R23
(destrucción de LOC/unidad hacia adelante). Es un recordatorio útil: el patrón esperado de una
heurística "de libro" depende de qué señal específica observa, y no todo diseño reactivo
responde a todo tipo de riesgo.

`heur_s1_max_downstream` (constante) produce `action_intensity` perfectamente plano en todas las
semanas relativas — confirma que la métrica y el pipeline de agregación funcionan correctamente
(control de sanidad).

## Veredicto

Esta comparación es metodológicamente válida (sin empalme, sin confound de RNG) y aporta dos
cosas: (1) refuerza, de forma independiente, que PPO/Real-KAN ganan por un margen que ninguna
heurística simple explica; (2) muestra que ni siquiera las heurísticas *diseñadas* para ser
claramente preventivas o reactivas producen un patrón de intensidad de acción "de libro de texto"
bajo este benchmark — la pregunta de si la política aprendida es preventiva sigue mejor
respondida por la ruta de memoria histórica + cabeza auxiliar (ya en curso) que por análisis de
intensidad de acción post-hoc, sea con empalme (inválido) o sin él (válido pero de baja
sensibilidad para este propósito específico).
