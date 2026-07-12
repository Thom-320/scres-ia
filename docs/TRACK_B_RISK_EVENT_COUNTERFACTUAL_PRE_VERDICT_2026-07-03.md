# Track B risk-event counterfactual audit: pre-risk value in ReT Excel

> **Estado actualizado:** artefacto exploratorio, no evidencia causal final. Despues de ejecutar este auditor se encontro
> un problema estructural de RNG bajo `adaptive_benchmark_v2`: al sustituir acciones dentro de un episodio, puede cambiar
> la trayectoria futura de regimen/riesgo. Ver `docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md`.
> Por eso los deltas de este documento no deben usarse para afirmar prevencion ni ausencia de prevencion. La alternativa
> valida actual es evaluar politicas completas independientes; ver
> `docs/TRACK_B_HEURISTIC_FULL_ROLLOUT_VERDICT_2026-07-03.md`.

Fecha: 2026-07-03

## Pregunta

Garrido quiere saber si el agente **se prepara antes** de los riesgos frecuentes o si principalmente **reacciona** despues.
El event-study anterior mostro patrones de accion alrededor de eventos reales, pero era descriptivo. Este auditor intentaba
hacer el siguiente paso causal:

```text
R_full - R_reset(pre-risk, risk_id)
```

`R_full` es el ReT Excel final del episodio con la politica congelada. `R_reset` pretendia ser el mismo episodio y la misma
semilla de evaluacion, reemplazando las acciones de las semanas -4 a -1 antes de un riesgo real por la **calma propia de
esa politica**. No se reentrena nada. El problema descubierto despues es que esa sustitucion no conserva necesariamente la
misma trayectoria futura de regimen/riesgo, asi que el empalme deja de ser un contrafactual limpio.

Importante: el resultado usado aqui es `ret_excel` de `compute_episode_metrics`, la metrica Garrido/Excel del episodio.
No se usa `order_level_ret_mean` como criterio causal.

## Artefactos

```text
scripts/audit_track_b_risk_event_counterfactual.py
outputs/experiments/track_b_risk_event_counterfactual_2026-07-03/
```

Archivos principales:

- `risk_event_counterfactual_pre.csv`: pares contrafactuales por politica, riesgo, seed, episodio y ancla.
- `summary_by_policy_risk.csv`: resumen de deltas ReT Excel.
- `calm_action_by_policy.csv`: calma empirica usada para cada politica.
- `risk_event_ledger.csv`: eventos reales del DES usados como anclas.
- `verdict.md`: resumen generado por el runner.

## Diseno

- Politicas: PPO+MLP y Real-KAN.
- Escala: 5 seeds x 12 episodios.
- Riesgos auditados: R11, R13, R24 y R14.
- Ventana: semanas -4 a -1 antes del inicio real del riesgo.
- Maximo: 4 anclas por riesgo/episodio, para evitar que riesgos muy frecuentes reemplacen casi todo el episodio.
- Comparacion prevista: misma semilla de evaluacion y solo cambio de accion en la ventana auditada. El hallazgo posterior
  de RNG muestra que la trayectoria futura de riesgos no queda garantizada, por eso el documento queda exploratorio.

## Calma propia de cada politica

Como R11/R14 son tan frecuentes que no queda una zona limpia lejos de riesgos, el runner usa el cuartil de menor intensidad
de accion de cada politica como su estado de calma.

| Politica | Fuente de calma | Filas | shift | op10 | op12 |
|---|---|---:|---:|---:|---:|
| PPO+MLP | cuartil de menor intensidad propia | 1,560 | -0.164 | -0.118 | -0.117 |
| Real-KAN | cuartil de menor intensidad propia | 1,560 | 0.591 | 0.914 | 0.978 |

Esto confirma algo que ya sabiamos: incluso la "calma" de Real-KAN es intensiva; su politica opera cerca del techo de
recursos. Por eso Real-KAN puede ganar en resiliencia marginal, pero no es una politica austera.

## Resultados

Intervalos CI95 son aproximados por error estandar normal sobre pares contrafactuales.

| Politica | Riesgo | Pares | Positivos | Delta ReT Excel medio | CI95 aprox. | Lectura |
|---|---:|---:|---:|---:|---:|---|
| PPO+MLP | R11 | 122 | 54/122 | -0.00000331 | [-0.00001644, +0.00000982] | sin senal preventiva causal |
| PPO+MLP | R13 | 180 | 92/180 | -0.00001105 | [-0.00002360, +0.00000151] | sin senal preventiva causal |
| PPO+MLP | R14 | 180 | 86/180 | -0.00000652 | [-0.00001813, +0.00000509] | sin senal preventiva causal |
| PPO+MLP | R24 | 166 | 76/166 | -0.00000846 | [-0.00002019, +0.00000327] | sin senal preventiva causal |
| Real-KAN | R11 | 122 | 62/122 | +0.00000587 | [-0.00000591, +0.00001765] | senal positiva pequena, no concluyente |
| Real-KAN | R13 | 180 | 95/180 | +0.00000389 | [-0.00000946, +0.00001725] | senal positiva pequena, no concluyente |
| Real-KAN | R14 | 180 | 99/180 | +0.00000337 | [-0.00000866, +0.00001540] | senal positiva pequena, no concluyente |
| Real-KAN | R24 | 162 | 78/162 | -0.00000453 | [-0.00001796, +0.00000890] | sin senal preventiva causal |

## Veredicto

**No usar esta tabla como veredicto causal.**

Como diagnostico exploratorio, la tabla tampoco mostraba una senal preventiva estable. Pero la razon metodologica principal
es mas fuerte: el diseno `R_full - R_reset(w)` no es confiable bajo el RNG actual del benchmark adaptativo. La evidencia
vigente queda asi:

1. El event-study descriptivo sugiere que PPO+MLP es mayormente reactivo, con una senal pre-riesgo pequena.
2. Las heuristicas evaluadas como politicas completas independientes no explican la ganancia de PPO+MLP/Real-KAN.
3. El comportamiento preventivo sigue abierto; no debe declararse demostrado con este empalme.

Para Garrido: podemos decir que ya tenemos una politica adaptativa que mejora la resiliencia, y que estamos midiendo la
prevencion con eventos reales de Garrido. Pero, por ahora, el comportamiento preventivo no esta demostrado. La ruta mas
prometedora para lograrlo sigue siendo la arquitectura de creencia:

```text
PPO/Real-KAN + memoria historica de riesgos frecuentes + cabeza auxiliar de prediccion + evaluacion final en ReT Excel
```

Eso permitiria entrenar una representacion interna tipo: "este riesgo frecuente suele venir pronto", y luego verificar si
esa creencia produce acciones pre-riesgo que aumenten ReT Excel.
