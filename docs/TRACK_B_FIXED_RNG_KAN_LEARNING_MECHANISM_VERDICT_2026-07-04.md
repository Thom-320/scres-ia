# Track B fixed-RNG: que aprende Real-KAN y si es preventivo/reactivo — 2026-07-04

## Pregunta

Queremos separar tres afirmaciones:

1. **Aprendizaje de resultado:** si Real-KAN aprende una politica que mejora la resiliencia ReT Excel.
2. **Aprendizaje de timing:** si Real-KAN cambia sus acciones antes o despues de riesgos reales.
3. **Prevencion causal:** si las acciones antes del riesgo aportan valor en ReT Excel.

La metrica principal sigue siendo `order_ret_excel_mean`, la formula ReT del Excel de Garrido aplicada a nivel de orden.

## Artefactos usados

- Politicas fixed-RNG:
  - `outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/`
  - `outputs/experiments/track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03/`
- Event-study descriptivo fixed-RNG:
  - `outputs/experiments/track_b_fixed_rng_risk_event_ledger_2026-07-04/`
- Contrafactual pre-riesgo quick fixed-RNG:
  - `outputs/experiments/track_b_fixed_rng_risk_event_counterfactual_frequent_quick_2026-07-04/`

Nota: se excluye R14 de la lectura causal principal porque sus defectos dependen de la produccion real. La auditoria causal se enfoca primero en riesgos discretos exogenos frecuentes: R11, R13 y R24.

## 1. Real-KAN si aprende

Real-KAN gana en ReT Excel bajo fixed-RNG:

| Politica | ReT Excel | Costo |
|---|---:|---:|
| Real-KAN fixed-RNG | 0.00594620 | 1.000 |
| PPO+MLP fixed-RNG | 0.00592064 | 0.700 |
| Mejor estatica del bundle | 0.00542514 | 1.000 |

Real-KAN supera a PPO+MLP en 5/5 seeds:

| Seed | Real-KAN | PPO+MLP | Delta |
|---:|---:|---:|---:|
| 1 | 0.00594504 | 0.00590254 | +0.00004250 |
| 2 | 0.00594702 | 0.00590989 | +0.00003713 |
| 3 | 0.00595563 | 0.00594749 | +0.00000814 |
| 4 | 0.00593981 | 0.00591905 | +0.00002076 |
| 5 | 0.00594350 | 0.00592423 | +0.00001927 |

Conclusion: **si hay aprendizaje funcional**. Real-KAN no solo corre; aprende una politica que mejora ReT Excel frente a PPO+MLP y frente a estaticas.

## 2. Que parece aprender Real-KAN

La evidencia actual apunta a que Real-KAN aprende una postura de **alta capacidad casi permanente**:

| Politica | Intensidad media | Desv. std. | Shift medio | Op10 medio | Op12 medio |
|---|---:|---:|---:|---:|---:|
| PPO+MLP | 0.372 | 0.266 | 0.568 | 1.364 | 1.285 |
| Real-KAN | 0.999 | 0.008 | 1.000 | 2.000 | 1.998 |

Donde la intensidad resume turno, despacho Op10 y despacho Op12. En Real-KAN:

- `shift_norm` esta en 1.0 casi siempre: turno alto.
- `op10_multiplier` esta en 2.0.
- `op12_multiplier` esta casi siempre en 2.0.
- La variacion de accion es minima.

Lectura: Real-KAN aprende que, en este benchmark de estres, la mejor forma de maximizar ReT Excel es operar casi al techo de capacidad. Eso explica su mayor resiliencia y tambien su mayor costo.

## 3. Preventivo o reactivo

### Event-study descriptivo

Al alinear acciones contra eventos reales R11/R13/R24, Real-KAN no muestra un salto fuerte ni antes ni despues, porque ya esta cerca del maximo desde el inicio.

Ejemplo de intensidad promedio:

| Politica | Riesgo | Pre-evento | Evento | Post-evento | Lectura |
|---|---|---:|---:|---:|---|
| Real-KAN | R11 | 0.981 | 0.981 | 0.983 | casi saturado |
| Real-KAN | R13 | 0.981 | 0.982 | 0.983 | casi saturado |
| Real-KAN | R24 | 0.982 | 0.981 | 0.983 | casi saturado |
| PPO+MLP | R11 | 0.352 | 0.352 | 0.335 | variable |
| PPO+MLP | R13 | 0.343 | 0.342 | 0.325 | variable |
| PPO+MLP | R24 | 0.351 | 0.353 | 0.331 | variable |

Esto sugiere que **Real-KAN no es reactivo en timing**: no espera a que aparezca backlog o riesgo para subir capacidad. Pero tampoco demuestra prevencion especifica por riesgo: simplemente ya opera alto.

### Contrafactual pre-riesgo fixed-RNG

El smoke contrafactual reemplaza las acciones aprendidas en semanas pre-riesgo por la accion de calma de la misma politica y mide:

`R_full - R_reset(pre-risk, risk_id)`

Primer resultado quick, riesgos frecuentes:

| Politica | Riesgo | Pares | Positivos | Delta ReT medio | Lectura |
|---|---|---:|---:|---:|---|
| PPO+MLP | R11 | 39 | 4/39 | -0.00000012 | sin prevencion |
| PPO+MLP | R13 | 54 | 17/54 | +0.00000585 | senal positiva debil |
| PPO+MLP | R24 | 45 | 10/45 | +0.00000910 | senal positiva debil |
| Real-KAN | R11 | 39 | 0/39 | -0.00000046 | sin prevencion |
| Real-KAN | R13 | 54 | 8/54 | -0.00000060 | sin prevencion |
| Real-KAN | R24 | 45 | 6/45 | +0.00000020 | sin senal clara |

Conclusion causal preliminar:

- **Real-KAN no muestra evidencia de prevencion especifica por riesgo** en este smoke.
- **PPO+MLP muestra una senal pre-riesgo pequena en R13/R24**, pero todavia debil: los deltas son pequenos y la proporcion de pares positivos no es alta.

## Veredicto

Real-KAN aprende, pero lo que aprende actualmente parece ser:

> una politica robusta/intensiva de maxima capacidad que aumenta la resiliencia ReT Excel, no una politica claramente preventiva o reactiva por timing.

Clasificacion actual:

| Politica | Aprende ReT | Preventiva | Reactiva | Mejor descripcion |
|---|---|---|---|---|
| PPO+MLP | si | debil/no concluyente | parcialmente adaptativa | eficiente y sensible al estado |
| Real-KAN | si | no demostrado | no claramente reactiva | robusta/intensiva, casi always-on |

Para Garrido, la frase correcta seria:

> KAN si aporta una mejora pequena pero consistente en resiliencia pura. Sin embargo, en esta version el mecanismo no parece ser anticipacion fina del riesgo, sino una politica de alta capacidad permanente. La prevencion real sigue siendo una frontera de trabajo: requiere memoria historica de riesgos frecuentes y/o una cabeza auxiliar de prediccion, evaluada finalmente con ReT Excel.

## Siguiente paso recomendado

Si queremos buscar comportamiento preventivo real, no basta con KAN tal como esta. El siguiente experimento debe agregar:

1. memoria historica de riesgos frecuentes (`weeks_since_last_Ri`, conteos 8w/26w, EWMA por riesgo);
2. una cabeza auxiliar que prediga ocurrencia de R11/R13/R24 a 1-4 semanas;
3. entrenamiento PPO+MLP y PPO+Real-KAN con esa creencia interna;
4. auditoria final con ReT Excel y contrafactual fixed-RNG por riesgo.

Ese experimento si probaria la idea biologica: aprender una creencia interna de que ciertos riesgos frecuentes suelen venir pronto y actuar antes.
