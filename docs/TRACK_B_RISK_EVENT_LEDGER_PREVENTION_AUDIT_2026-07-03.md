# Track B risk-event ledger audit: first prevention-oriented event study

Fecha: 2026-07-03

## Pregunta

Garrido quiere saber si la politica aprendida se comporta como:

- **preventiva:** prepara la cadena antes de que llegue un riesgo;
- **reactiva:** actua cuando ya aparece deterioro operativo;
- **mixta:** combina preposicionamiento moderado y respuesta posterior.

El primer auditor preventivo usaba un ancla difusa de regimen/forecast. Este nuevo paso ancla el analisis en eventos
reales del DES: `sim.risk_events`, con `risk_id`, inicio, duracion y operacion afectada.

## Artefacto

Directorio:

```text
outputs/experiments/track_b_risk_event_ledger_2026-07-03/
```

Archivos:

- `risk_event_ledger.csv`: todos los eventos de riesgo capturados por politica/seed/episodio.
- `risk_frequency_check.csv`: conteo observado por riesgo.
- `risk_event_aligned_action_study.csv`: accion promedio por categoria de riesgo y semana relativa.
- `risk_event_aligned_by_risk_study.csv`: accion promedio por `risk_id` y semana relativa.
- `risk_event_action_summary_by_category.csv`: resumen pre/event/post por categoria.
- `risk_event_action_summary_by_risk.csv`: resumen pre/event/post por riesgo.

Politicas auditadas:

- `ppo_mlp`
- `real_kan`

Escala:

- 5 seeds x 12 episodios x 2 politicas.
- Evaluacion de politicas congeladas; no hay reentrenamiento.

## Conteo de eventos

| Riesgo | Categoria | Eventos observados | Eventos/año observados | Esperado/año R1 |
|---|---:|---:|---:|---:|
| R11 | frecuente | 15,647 | 65.37 | 103.67 |
| R13 | frecuente | 9,149 | 38.23 | 14.40 |
| R14 | frecuente_tasa | 74,454 | 311.08 | n/a |
| R24 | frecuente | 4,721 | 19.72 | 26.03 |
| R12 | intermedio | 306 | 1.28 | 2.18 |
| R22 | intermedio | 819 | 3.42 | 4.34 |
| R23 | intermedio | 335 | 1.40 | 2.17 |
| R21 | raro | 129 | 0.54 | 1.09 |
| R3 | raro | 16 | 0.07 | 0.11 |

Interpretacion: la auditoria ahora tiene miles de anclas para riesgos frecuentes. Eso corrige el problema anterior de
tener una sola ancla por episodio. R3 sigue siendo demasiado raro para usarlo como prueba principal de prevencion.

Nota sobre R14: aparece como muchos eventos tipo defecto/tasa. Debe analizarse separado (`frecuente_tasa`) porque puede
dominar numericamente las categorias discretas.

## Event-study por categoria

La intensidad de accion resume `shift`, `op10_q` y `op12_q` en una escala comparable. Para cada evento, se compara:

- `pre`: semanas -4 a -1 antes del evento;
- `event`: semana 0;
- `post`: semanas +1 a +4 despues del evento.

| Politica | Categoria | Pre -4:-1 | Evento 0 | Post +1:+4 | Delta evento-pre | Delta post-pre |
|---|---:|---:|---:|---:|---:|---:|
| PPO+MLP | frecuente | 0.2336 | 0.2346 | 0.2352 | +0.0010 | +0.0016 |
| PPO+MLP | frecuente_tasa | 0.2351 | 0.2356 | 0.2350 | +0.0005 | -0.0000 |
| PPO+MLP | intermedio | 0.2272 | 0.2243 | 0.2212 | -0.0030 | -0.0060 |
| PPO+MLP | raro | 0.2173 | 0.2309 | 0.2362 | +0.0136 | +0.0190 |
| Real-KAN | frecuente | 0.9693 | 0.9695 | 0.9697 | +0.0002 | +0.0004 |
| Real-KAN | frecuente_tasa | 0.9697 | 0.9693 | 0.9687 | -0.0004 | -0.0010 |
| Real-KAN | intermedio | 0.9703 | 0.9696 | 0.9704 | -0.0006 | +0.0001 |
| Real-KAN | raro | 0.9630 | 0.9641 | 0.9656 | +0.0011 | +0.0026 |

## Event-study por riesgos frecuentes

| Politica | Riesgo | Pre -4:-1 | Evento 0 | Post +1:+4 | Delta evento-pre | Delta post-pre |
|---|---:|---:|---:|---:|---:|---:|
| PPO+MLP | R11 | 0.2330 | 0.2343 | 0.2347 | +0.0012 | +0.0017 |
| PPO+MLP | R13 | 0.2344 | 0.2343 | 0.2367 | -0.0001 | +0.0023 |
| PPO+MLP | R24 | 0.2337 | 0.2361 | 0.2340 | +0.0024 | +0.0002 |
| PPO+MLP | R14 | 0.2351 | 0.2356 | 0.2350 | +0.0005 | -0.0000 |
| Real-KAN | R11 | 0.9692 | 0.9690 | 0.9696 | -0.0001 | +0.0004 |
| Real-KAN | R13 | 0.9696 | 0.9702 | 0.9700 | +0.0006 | +0.0003 |
| Real-KAN | R24 | 0.9691 | 0.9694 | 0.9697 | +0.0002 | +0.0005 |
| Real-KAN | R14 | 0.9697 | 0.9693 | 0.9687 | -0.0004 | -0.0010 |

## Lectura honesta

Este auditor ya es mucho mejor que el ancla de regimen/forecast porque usa eventos reales y miles de observaciones para
riesgos frecuentes. Pero todavia **no prueba prevencion causal**.

Lo que muestra hoy:

1. **PPO+MLP muestra un perfil mayormente reactivo, con una senal pre-riesgo pequena.** Hay una subida ligera antes del
   evento (`0.2332` a `0.2337` en semanas -4 a -1 para el agregado de riesgos frecuentes), pero la subida mayor aparece
   despues del evento (`0.2346` a `0.2363` de semana 0 a +4 en la lectura por buckets). Esto es compatible con una mezcla
   adaptativa, no con prevencion limpia.
2. **Real-KAN opera casi saturado todo el tiempo.** Su accion promedio esta alrededor de `0.969`; por eso no se ve un
   patron de timing preventivo claro. La ganancia de Real-KAN parece venir de una postura operativa intensiva, no de una
   anticipacion selectiva demostrada.
3. **Los riesgos raros muestran deltas mayores en PPO+MLP, pero tienen muy pocas observaciones.** No deben usarse como
   prueba de prevencion.
4. **R11 y R14 requieren cautela por solapamiento.** R11 ocurre aproximadamente una vez por semana en este benchmark; por
   eso, el "pre" de un evento puede coincidir con el "post" de otro evento anterior. La senal pre-riesgo podria ser
   anticipacion real o recuperacion residual de un riesgo cercano. R14 es frecuente, pero se comporta como tasa/defecto;
   mezclarlo con eventos
   discretos distorsiona las categorias.

## Veredicto

**Estado: auditor implementado, clasificacion preventiva todavia abierta.**

Ahora si podemos decirle a Garrido que ya no estamos mirando solo regimenes abstractos: estamos alineando las acciones a
los riesgos reales de Garrido. El primer resultado sugiere que las politicas actuales son adaptativas y robustas, pero no
demuestra todavia un comportamiento preventivo fuerte ante R11/R13/R24.

## Siguiente paso tecnico

Este siguiente paso fue implementado, pero luego quedo marcado como exploratorio/no causal por el hallazgo de RNG:

```text
outputs/experiments/track_b_risk_event_counterfactual_2026-07-03/
docs/TRACK_B_RISK_EVENT_COUNTERFACTUAL_PRE_VERDICT_2026-07-03.md
docs/TRACK_B_COUNTERFACTUAL_RNG_ENTANGLEMENT_FINDING_2026-07-03.md
```

La prueba causal mide valor en ReT Excel:

```text
R_full - R_reset(pre-risk, risk_id)
```

usando:

- eventos reales por `risk_id`;
- calma propia de cada politica;
- riesgos frecuentes primero;
- anclas muestreadas por evento, para evitar que R11/R14 reemplacen casi todo el episodio.

Sin embargo, el empalme `R_full - R_reset(w)` no debe usarse como evidencia causal final bajo el benchmark adaptativo
actual, porque cambiar acciones dentro del episodio puede cambiar la trayectoria futura de regimen/riesgo. La alternativa
valida ya corrida es comparar politicas completas independientes:

```text
docs/TRACK_B_HEURISTIC_FULL_ROLLOUT_VERDICT_2026-07-03.md
```

En paralelo, la ruta mas prometedora sigue siendo construir la arquitectura de creencia:

```text
PPO/Real-KAN + memoria historica de riesgos frecuentes + cabeza auxiliar de prediccion + evaluacion final en ReT Excel
```

Esta auditoria nos da exactamente las anclas necesarias para entrenar y evaluar esa idea.
