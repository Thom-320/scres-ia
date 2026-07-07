# Ruta para convertir adaptación reactiva en prevención

Fecha: 2026-07-03

## Decisión de encuadre

El resultado actual de Paper 1 debe seguir formulado como **aprendizaje adaptativo**: la política aprende a mover
recursos y despacho cuando el espacio de acción toca el cuello de botella operativo. Todavía no debemos afirmar que la
política es preventiva, porque la auditoría preventiva/reactiva aún no pasa controles básicos contra heurísticas de
referencia.

La meta siguiente es más ambiciosa: convertir una política adaptativa/reactiva en una política **preventiva o mixta**,
capaz de prepararse ante riesgos frecuentes antes de que aparezcan backlog, pérdida de servicio o congestión aguas
abajo.

## Qué significa prevención en este modelo

- **Reactiva:** la política espera evidencia de daño operativo: backlog, caída del fill rate, retrasos o presión
  downstream. Luego aumenta turnos o despacho para recuperar resiliencia.
- **Preventiva:** la política actúa antes del daño visible, usando historia de riesgos, recurrencia o forecast, y esas
  acciones previas aumentan la ReT final calculada con la ecuación de Excel de Garrido.
- **Mixta:** la política prepara moderadamente antes de riesgos comunes y responde con más fuerza cuando el daño ya se
  materializa.

El criterio de valor debe seguir siendo la **ReT del Excel de Garrido**. Métricas como fill rate, backlog o presión
downstream pueden servir para explicar el mecanismo, pero no reemplazan la métrica principal.

## Estado actual

- Track B confirma aprendizaje adaptativo: PPO+MLP y Real-KAN superan la grilla estática bajo el mismo protocolo.
- Real-KAN aporta novedad e interpretabilidad, pero con más costo operativo.
- Las pruebas con historia pura no han demostrado ventaja: MLP con historia, DMLPA/CAN y RecurrentPPO/LSTM no superan
  al PPO+MLP canónico bajo el protocolo actual.
- El forecast existe en la observación y puede servir como sonda de prevención, pero es polémico: si no aporta ventaja
  clara, conviene dejarlo fuera del resultado principal y usarlo solo para estudiar prevención.
- El reentrenamiento `v7_no_forecast` aterrizó después de definir este roadmap: conserva una ventaja positiva sobre la
  estática bajo `adaptive_benchmark_v2` (`ReT=0.005668` vs `0.005214`, delta `+0.000454`) y usa menor costo promedio
  (`assembly_cost_index≈0.645`). Queda por debajo del PPO+MLP canónico con forecast, pero confirma que el resultado no
  depende exclusivamente de leer el forecast. Esto refuerza el encuadre: forecast como sonda de prevención, no como
  requisito para que Track B aprenda.
- El audit inicial `R_full - R_reset(w)` no está listo para concluir prevención/reactividad: el uso de una acción fija
  global como reset falló contra heurísticas conocidas. Debe reemplazarse por referencias de calma específicas por
  política y por anclas separadas para prevención y reacción.
- Tras ampliar y ajustar ventanas, el contrafactual por ventana siguió siendo inestable: los deltas en ReT son del
  orden de `1e-5` y cambian de signo con decisiones metodológicas pequeñas. Por ahora se pausa como prueba concluyente.
- Granger/correlación cruzada pueden mostrarse como timing descriptivo, pero no como prueba de que una acción mejoró la
  resiliencia. Para Garrido, la prueba dura sigue siendo ReT Excel.

## Cómo añadir capacidad preventiva

### 1. Clasificar riesgos por frecuencia e impacto

No tiene sentido exigir predicción de black swans. La prevención debe evaluarse primero en riesgos frecuentes y de menor
impacto, donde el agente realmente puede aprender patrones recurrentes.

Propuesta:

- Estimar frecuencia, duración e impacto por riesgo.
- Separar riesgos en: frecuentes-aprendibles, operacionales moderados, severos raros.
- Entrenar/evaluar prevención primero sobre los frecuentes-aprendibles.
- Usar riesgos raros solo como stress test de robustez, no como prueba principal de anticipación.

En la parametrización actual de Garrido, esto tiene sentido empírico:

- **Muy frecuentes/aprendibles:** R11, R14 y R24. En el nivel actual, R11 ocurre cada ~84.5 h en esperanza
  (~104 eventos/año), R24 cada ~336.5 h (~26 eventos/año), y R14 genera muchos defectos por turno. En niveles
  aumentado/severo, estas repeticiones suben mucho más.
- **Intermedios:** R22/R23/R21, según régimen. Pueden servir como segunda etapa.
- **Raros/no-aprendibles como prevención:** R3. Es el black swan; no debe ser la prueba principal de anticipación.

Tambien queda confirmado que Track B no vive solamente en `adaptive_benchmark_v2`: la matriz E3 ya evaluo la politica
congelada bajo los niveles nativos de Garrido (`current`, `increased`, `severe`) y gano 5/6 celdas contra comparadores
estaticos livianos. En h104, las dos celdas defendibles para Paper 1 (`current` e `increased`) tambien pasaron una
frontera densa de 147 politicas:

- `current/h104`: PPO `0.005648` vs mejor estatica densa `0.005405`, delta `+0.000244`.
- `increased/h104`: PPO `0.003660` vs mejor estatica densa `0.003037`, delta `+0.000623`.

Esto justifica usar los riesgos nativos de Garrido para la auditoria preventiva: no son una extrapolacion decorativa,
sino un escenario donde la politica ya mostro margen real.

Estado técnico actualizado: el primer `risk_event_ledger.csv` ya fue instrumentado y corrido para PPO+MLP y Real-KAN
con 5 seeds x 12 episodios. Ver
`docs/TRACK_B_RISK_EVENT_LEDGER_PREVENTION_AUDIT_2026-07-03.md`. La auditoría ahora puede anclar acciones a eventos
reales (`risk_id`, inicio/fin, duración y operación afectada), no solo a régimen/forecast. El resultado inicial muestra
miles de anclas para R11/R13/R24/R14, pero todavía no prueba prevención causal: falta medir valor en ReT Excel con
contrafactuales pre-riesgo por `risk_id`.

### 2. Dar memoria de riesgo sin dar un oráculo

En vez de depender de un forecast privilegiado, construir señales históricas observables:

- semanas desde el último evento de cada riesgo;
- tasa empírica en las últimas 8 y 26 semanas;
- EWMA de ocurrencia reciente;
- duración/magnitud del último evento;
- familia o nodo afectado;
- tiempo desde la recuperación.

Estas señales permiten que el agente aprenda recurrencia sin recibir directamente el futuro.

### 3. Añadir una cabeza auxiliar de predicción de riesgo

La arquitectura más defendible para prevención no es simplemente “más historia”. Es una política que aprende una creencia
interna de riesgo:

```text
observación + historia -> encoder compartido -> política PPO
                                      -> cabeza auxiliar: P(R_i en 1/2/4/8 semanas)
                                      -> cabeza auxiliar: horas esperadas de disrupción
```

La pérdida sería:

```text
L = L_PPO + lambda_1 * BCE(riesgo futuro) + lambda_2 * MSE(horas de disrupción)
```

Esto responde directamente a Garrido/David: el agente no solo reacciona, sino que aprende una representación predictiva
de riesgos comunes.

### 4. Usar recompensa de resiliencia futura, sin cambiar la métrica final

La idea de “recompensa que calcule resiliencia futura” ya existía parcialmente:

- `FutureCreditRewardWrapper` implementa `ReT_excel_delta_bootstrap` y `ReT_excel_terminal_shaped` en una línea antigua.
- `control_v1_pbrs` existe y está probado, pero no es todavía una recompensa futura de Excel-ReT para Track B.

Estado: **propuesta activa, no implementada para Track B 8D**.

La versión correcta para Paper 1/Track B sería:

- entrenamiento: recompensa con valor futuro aproximado de ReT;
- evaluación: siempre ReT final del Excel de Garrido;
- aceptación: solo cuenta si mejora ReT y pasa contrafactuales pre-riesgo.

### 5. Auditar prevención con contrafactuales válidos

El auditor ideal debe medir valor causal, no solo correlación:

```text
R_full - R_reset(pre)
```

donde `R_full` es la ReT final de la política aprendida, y `R_reset(pre)` es la ReT si reemplazamos las acciones
pre-riesgo por la acción de calma propia de esa política.

Regla:

- Si `R_full - R_reset(pre) > 0` y la acción sube antes del daño, hay evidencia de prevención.
- Si el valor aparece solo después de backlog/fill-rate bajo, hay evidencia reactiva.
- Si aparecen ambos, la política es mixta.

El reset no debe ser una acción estática universal. Debe ser la acción de calma de cada política; de lo contrario, se
mezclan dos efectos y la auditoría falla.

Estado actual: este método queda **pausado** como clasificador preventivo/reactivo. No pasó los sanity-checks de forma
estable con 5 semillas x 12 episodios. Puede retomarse con más escala o con anclas separadas por mecanismo, pero no debe
usarse hoy para etiquetar PPO+MLP o Real-KAN como preventivos/reactivos.

## Próximo experimento recomendado

1. Usar el resultado no-forecast como evidencia de robustez: el forecast no es indispensable para ganar, pero puede
   quedarse como sonda de prevención.
2. Usar el `risk_event_ledger.csv` ya generado para diseñar contrafactuales pre/post por `risk_id`, empezando por
   R11/R24/R13/R14 y dejando R21/R3 como stress tests.
3. Implementar un Track B `v8_risk_memory` con memoria histórica por riesgo frecuente.
4. Añadir una cabeza auxiliar de predicción de riesgo común.
5. Entrenar PPO+MLP y Real-KAN bajo el mismo contrato 8D.
6. Evaluar con ReT Excel y auditar:
   - mejora vs grilla estática;
   - pérdida al enmascarar/shufflear memoria de riesgo;
   - `R_full - R_reset(pre)` por riesgo frecuente;
   - `R_full - R_reset(post)` para recuperación;
   - costo operativo.

Paso barato antes de reentrenar: extender el auditor para estratificar los episodios ya evaluados por familias de
riesgo frecuentes/intermedias/raras. Si las señales de acción aparecen sólo en riesgos frecuentes, eso apoyaría la
hipótesis de aprendizaje de recurrencia; si no aparecen ni ahí, primero hay que construir mejores features de memoria
antes de prometer prevención.

## Mensaje para Garrido

El paper ya demuestra aprendizaje adaptativo. La siguiente frontera científica es demostrar aprendizaje preventivo:
que el agente no solo reacciona cuando la cadena se deteriora, sino que aprende la recurrencia de riesgos comunes y se
prepara antes. Esa afirmación requiere una política con memoria de riesgo y una auditoría causal basada en la ReT del
Excel; todavía no debe prometerse como resultado cerrado.
