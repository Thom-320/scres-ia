# ¿KAN aprende? ¿Es preventiva o reactiva la política? — veredicto consolidado bajo RNG corregido — 2026-07-04

Consolida dos auditorías independientes, ambas ya corridas y verificadas número por número contra
sus CSV/JSON fuente (no narración): la interpretabilidad de Real-KAN y el contrafactual causal
`R_full - R_reset(w)`, este último por primera vez válido gracias al fix de RNG
(`strict_exogenous_crn=True` + `regime_rng` dedicado — ver
`docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md` y
`docs/TRACK_B_RNG_ENTANGLEMENT_FIX_INVESTIGATION_2026-07-03.md`).

## Pregunta 1: ¿Real-KAN está aprendiendo algo real?

**Sí.** Ver `docs/REAL_KAN_INTERPRETABILITY_VERDICT_2026-07-04.md` para el detalle completo.
Resumen: cargando el checkpoint fixed-RNG ya entrenado (sin reentrenar), reactivando
`save_act`/`symbolic_enabled` de pykan sobre observaciones reales de evaluación:

- La atribución por variable se concentra **2.5x por encima de lo esperado por azar** (top-10 de
  52 variables = 47.7% del peso total, vs. 19.2% uniforme).
- Las variables top son operacionalmente sensatas: presión de demanda (`rations_theatre_norm`),
  `backorder_rate`, fracción de tiempo caído, estado del régimen, probabilidad de defecto R14.
- La curva aprendida (spline) para la variable más atribuida muestra una forma de umbral-y-rampa
  clara y no lineal; variables de atribución cero (`op10_down`, `op12_down`, `op2_down`,
  `ewma_downstream_risk`) muestran curvas literalmente vacías — el método discrimina bien entre
  señal usada e ignorada, no ve patrones en todo.

## Pregunta 2: ¿es preventiva, reactiva, o ambas?

### El contrafactual ahora es válido — y da una respuesta clara: ninguna de las dos, todavía

`scripts/audit_track_b_risk_event_counterfactual.py` sobre los checkpoints fixed-RNG (PPO+MLP y
Real-KAN), 5 seeds × 12 episodios, los 7 riesgos discretos ahora válidos bajo el fix de RNG (R11,
R12, R13, R21, R22, R23, R24; R14 excluido por depender de producción real):

| Política | Riesgo | Categoría | Pares | Positivos | Delta ReT Excel medio | Lectura |
|---|---|---|---:|---:|---:|---|
| PPO+MLP | R11 | frecuente | 125 | 5/125 (4%) | +0.00000025 | sin señal |
| PPO+MLP | R12 | intermedio | 163 | 0/163 (0%) | -0.00000027 | reduce ReT |
| PPO+MLP | R13 | frecuente | 180 | 57/180 (32%) | +0.00000834 | sin señal |
| PPO+MLP | R21 | raro | 59 | 7/59 (12%) | +0.00000871 | sin señal |
| PPO+MLP | R22 | intermedio | 205 | 36/205 (18%) | +0.00000551 | sin señal |
| PPO+MLP | R23 | intermedio | 161 | 19/161 (12%) | +0.00000363 | sin señal |
| PPO+MLP | R24 | frecuente | 153 | 34/153 (22%) | +0.00000920 | sin señal |
| Real-KAN | R11 | frecuente | 125 | 3/125 (2%) | -0.00000007 | reduce ReT |
| Real-KAN | R12 | intermedio | 163 | 0/163 (0%) | 0.00000000 | sin señal |
| Real-KAN | R13 | frecuente | 180 | 19/180 (11%) | -0.00000089 | reduce ReT |
| Real-KAN | R21 | raro | 59 | 6/59 (10%) | +0.00000002 | sin señal |
| Real-KAN | R22 | intermedio | 205 | 10/205 (5%) | -0.00000025 | reduce ReT |
| Real-KAN | R23 | intermedio | 161 | 5/161 (3%) | -0.00000003 | reduce ReT |
| Real-KAN | R24 | frecuente | 153 | 11/153 (7%) | -0.00000077 | reduce ReT |

**PPO+MLP**: mean delta direccionalmente positivo en 6/7 riesgos, pero con tasas de pares
positivos muy bajas (4-32%, todas muy por debajo del 67% que se exige para llamarlo "aporta ReT").
Es decir: la media está siendo arrastrada por pocos episodios/anclas con un delta grande, mientras
que la mayoría de los pares individuales están cerca de cero o son negativos. Es la firma clásica
de ruido, no de un efecto poblacional estable.

**Real-KAN**: prácticamente plano o negativo en todos los riesgos. Esto es consistente con y
explicado por su propia calma (peor caso, cuartil de menor intensidad): incluso en su estado "más
tranquilo", `shift=0.88`, `op10=0.80`, `op12=0.97` (escala [-1,1]) — casi al techo. Compárese con
la calma de PPO+MLP: `shift=-0.64`, `op10=-0.32`, `op12=-0.40` — un rango mucho más amplio entre
calma y actividad. Real-KAN casi no tiene un estado "bajo" contra el cual contrastar uno "alto"; por
diseño del contrafactual, si la política casi no varía, hay poco margen para que un cambio de
comportamiento pre-riesgo muestre valor causal.

### Veredicto

**Ninguna de las dos arquitecturas (PPO+MLP, Real-KAN) muestra evidencia causal robusta de
comportamiento preventivo**, evaluado con el método ahora válido, a escala completa (5×12,
1,371 pares contrafactuales combinados). Esto no es una limitación del método — el mismo método
detecta perfectamente diferencias reales cuando existen (ver la validación contra heurísticas de
referencia diseñadas, `docs/TRACK_B_PREVENTION_COUNTERFACTUAL_VALIDATION_2026-07-03.md`). Es un
hallazgo real sobre lo que aprenden hoy PPO+MLP y Real-KAN bajo este contrato de recompensa y
observación: **son adaptativos y mejoran la resiliencia medida en ReT Excel, pero no anticipan por
timing.** Real-KAN además "aprende" en el sentido de la Pregunta 1 (representación interna no
trivial, splines interpretables) — pero lo que aprende es una postura de capacidad casi permanente,
no una regla de anticipación selectiva.

## Qué sigue

Confirma la dirección ya acordada con el usuario: la vía para lograr prevención real no es más
ajuste de PPO+MLP/Real-KAN tal como están, sino memoria histórica observada + una cabeza auxiliar
de predicción de riesgo, evaluada siempre con la ReT Excel final — plan ya aprobado y en marcha
(ver plan de sesión: refresco del dataset de creencia hecho, `v10` raw-features smoke corriendo/en
revisión de confound de presupuesto de entrenamiento, extractor con cabeza auxiliar como siguiente
paso).
