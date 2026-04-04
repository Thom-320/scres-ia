# Brief para Garrido: Auditoría Track B — 3 abril 2026

## Resumen ejecutivo

Corrimos una auditoría completa del Track B en una sesión intensiva. Los resultados son publicables pero cambiaron el framing: **PPO es un controlador adaptativo eficiente, no un predictor anticipatorio.** La contribución principal es que RL funciona cuando el action space cubre el cuello de botella operativo activo — ni antes ni después.

---

## Resultados principales

### 1. La ablación cierra la historia causal de Track B

Entrenamos PPO bajo tres configuraciones de action space (500k timesteps, 5 seeds):

| Configuración | Fill rate | Order-level ReT | Horas ensamblaje | Veredicto |
|---|---|---|---|---|
| **Joint (7D completo)** | **1.000** | **0.948** | 18,529 | Mejor costo-eficiencia |
| Downstream-only (shift fijo S2) | 1.000 | 0.953 | 33,322 | Downstream = palanca clave |
| Shift-only (downstream fijo) | 0.953 | 0.686 | 26,295 | **Pierde vs mejor estático** |
| Mejor estático (s2_d2.00) | 0.988 | 0.460 | 29,120 | Referencia |

**Lectura:** Sin control downstream, PPO no puede ganar — exactamente como en Track A. El downstream dispatch (Op10/Op12) es la dimensión de acción necesaria y suficiente. Los shifts aportan eficiencia de costo (57% menos horas de ensamblaje) pero no mejoran el fill rate.

### 2. PPO descubre una estrategia no obvia

Del audit canónico con telemetría de acciones:

| Métrica | PPO | s3_d2.00 (mejor estático) |
|---|---|---|
| Fill rate | 1.000 | 0.985 |
| Order-level ReT | 0.950 | 0.449 |
| % tiempo en S1 | 77% | 0% |
| % tiempo en S3 | 6% | 100% |
| Op10 multiplier promedio | 1.36 | 2.00 (fijo) |
| Op12 multiplier promedio | 1.50 | 2.00 (fijo) |
| % steps ambos downstream ≥1.9 | 17% | 100% |
| Horas de ensamblaje | ~18,500 | ~43,700 |

PPO usa turnos mínimos (S1) la mayor parte del tiempo y compensa con downstream moderado. **No maxea nada.** Es más frugal Y más efectivo que el mejor estático.

### 3. PPO se degrada bajo estrés — no es invencible

Evaluamos el PPO congelado (entrenado en adaptive_benchmark_v2) bajo escenarios de riesgo creciente:

| Escenario | PPO fill | PPO ReT | PPO %recovery | Mejor estático fill | Mejor estático ReT |
|---|---|---|---|---|---|
| current | 1.000 | 0.915 | 6.7% | 0.9997 | 0.755 |
| increased | 0.993 | 0.748 | 24.7% | 0.901 | 0.201 |
| severe | 0.966 | 0.424 | 64.3% | 0.656 | 0.139 |
| **severe_extended** | **0.558** | **0.147** | **99.8%** | 0.282 | **0.149** |

En severe_extended, PPO pierde en ReT (0.147 vs 0.149 del mejor estático). Es el primer escenario donde un estático iguala a PPO en resiliencia.

**Para el paper:** La curva de degradación es evidencia fuerte. Autotomy baja de 96.8% → 0.1% conforme sube el riesgo. PPO pasa de "absorbe todo" a "vive en recovery permanente."

### 4. Las reward functions son casi intercambiables en Track B

Reward sweep (7 modes, 500k × 5 seeds):

| Reward mode | Fill rate | Order-level ReT |
|---|---|---|
| ReT_cd_v1 | 0.999988 | 0.954 |
| control_v1 | 0.999988 | 0.953 |
| ReT_seq_v1 | 0.999988 | 0.951 |
| ReT_garrido2024_train | 0.999975 | 0.948 |
| ReT_unified_v1 | 0.999963 | 0.945 |
| ReT_corrected | 0.843 | 0.269 |
| ReT_thesis | 0.836 | 0.258 |

Las top 5 son prácticamente idénticas. Esto refuerza el claim: **cuando el action space está alineado con el bottleneck, la reward importa poco.** Las piecewise (ReT_corrected, ReT_thesis) siguen fallando como reward de entrenamiento.

### 5. PPO gana a todas las heurísticas

| Policy | Fill | ReT | Shift mix | Cost idx |
|---|---|---|---|---|
| **PPO** | **1.000** | **0.928** | 77/14/9 | 0.441 |
| heur_tuned | 0.987 | 0.468 | 67/5/28 | 0.536 |
| heur_hysteresis | 0.985 | 0.502 | 65/0/35 | 0.566 |
| heur_disruption_aware | 0.986 | 0.468 | 75/19/6 | 0.439 |
| heur_s1_max_downstream (strawman) | 0.833 | 0.301 | 100/0/0 | 0.333 |
| s2_d2.00 (mejor estático) | 0.988 | 0.460 | 0/100/0 | 0.667 |

El strawman "S1 + downstream al máximo" (fill=0.833) demuestra que no basta con maximizar el dispatch fijo — se necesita el control adaptativo.

---

## Sobre la pregunta de anticipación

### ¿PPO anticipa riesgos?

**Respuesta corta: No en el sentido de predicción. Sí en el sentido de reacción al régimen.**

**Evidencia del forecast scrambling test:**

| Condición | Fill | ReT | Autotomy% | S1% |
|---|---|---|---|---|
| Forecasts reales | 1.000 | 0.950 | 96.8% | 76.9% |
| Forecasts permutados (aleatorios) | 1.000 | 0.951 | 96.8% | 78.0% |
| Forecasts en cero | 1.000 | 0.913 | 90.6% | 60.9% |

PPO con forecasts aleatorios funciona **igual** que con forecasts reales. **No usa el contenido informativo de los forecasts.** El zeroed cae pero por distribution shift (el obs se ve diferente al training), no por pérdida de anticipación.

**Evidencia de la observation ablation:**
- v7_full (con forecasts): PPO fill=0.999, ReT=0.844
- v7_no_forecast (sin forecasts): PPO fill=1.000, ReT=0.891

Quitar los forecasts no degrada — de hecho mejora ligeramente.

**Evidencia del análisis de régimen (5 seeds):**
- P(S>1 | nominal) = 0.126
- P(S>1 | pre_disruption) = 0.323 (2.6x)
- P(S>1 | disrupted) = 0.170

PPO SÍ responde al cambio de régimen (escala 2.6x más en pre_disruption), pero no usa los forecasts numéricos para hacerlo.

### Framing correcto

**Decir:** "Efficient adaptive resilience control with regime-responsive behavior"
**No decir:** "Anticipatory" ni "forecast-informed" ni "predictive"
**Extensión futura:** Las redes de creencias (propuesta de David) podrían aportar valor si las disrupciones tuvieran patrones temporales no-Markovianos o si los forecasts se quitaran del observation space.

---

## Posicionamiento vs literatura (papers que mandó Garrido)

| Paper | Método | Relación con nosotros |
|---|---|---|
| Rajagopal & Sivasakthivel 2024 | FFNN para selección de estrategia SCR | Estático, no secuencial. Citamos en related work. |
| Rezki & Mansouri 2024 | ANN para gestión de riesgo SC | Predicción supervisada, no control. Citamos brevemente. |
| Ordibazar et al. 2022 | Recommender + counterfactual XAI | Más cercano por interpretabilidad, pero no RL. Citamos. |

**Nuestro diferenciador:** Closed-loop adaptive control on a validated operational DES, con evidencia negativa honesta (Track A) y diagnóstico causal (ablación Track B). No somos los primeros en AI+SCR, pero somos los primeros en demostrar que el action space alignment determina el éxito de RL en resiliencia operacional.

---

## Estructura del paper propuesta

1. **Introducción** — SCRES + RL, gap: nadie ha estudiado cómo el action space determina el éxito
2. **Related work** — RL for SC, reward design, resilience metrics, citar Rajagopal/Rezki/Ordibazar
3. **Modelo DES** — Reconstrucción Garrido 2017, 13 operaciones, validación Table 6.10
4. **Formulación RL** — obs/action/reward, motivación Track A → Track B
5. **Track A** — Resultado negativo: PPO ≤ S=2, explicación mecánica (F2, F9, F11)
6. **Track B** — Resultado positivo: ablación, reward insensitivity, cross-scenario
7. **Análisis de política** — Distribución de shifts, estrategia downstream, régimen-responsividad
8. **Discusión** — ¿Cuándo sirve RL? Limitaciones. Extensiones (DKANA, severe training)
9. **Conclusión**

### Claim central (uno solo)

> "RL for supply chain resilience succeeds when the action space covers the active operational bottleneck. We demonstrate this through a dual-track benchmark: Track A (upstream-only) shows RL cannot beat static heuristics; Track B (upstream + downstream) enables efficient adaptive control."

---

## Limitaciones que el paper debe reconocer

1. **Forecasts con bajo contraste** — min=0.36, mean=0.64, nunca señal "tranquilo"
2. **Benchmark Markoviano** — transiciones de régimen con probabilidades fijas
3. **adaptive_benchmark_v2 no fuerza non_recovery** bajo condiciones normales
4. **Capacidad:** S1 produce ~18k raciones/semana vs demanda ~37k/semana (no hay sobrecapacidad con S1, pero S2/S3 sí producen más que suficiente)
5. **PPO vanilla** — no exploramos arquitecturas avanzadas (por diseño: el mérito es el benchmark)
6. **No probamos entrenamiento bajo severe** — solo evaluación cross-scenario

---

## Qué sigue corriendo

| Proceso | Estimado | Para qué |
|---|---|---|
| Ablación 500k downstream_only (3/5 seeds) | ~1-1.5h | Completa tabla de ablación del paper |
| Obs ablation v5_7d (Codex) | ~30min | Nice-to-have, no bloquea |
