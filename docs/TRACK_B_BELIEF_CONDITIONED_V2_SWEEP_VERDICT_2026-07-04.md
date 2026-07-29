# Sweep κ/ρ del reward condicionado por creencia (v2) — 2026-07-04 (COMPLETO, 16/16 — negativo)

## Diseño

```
p_adv = max(0, (p_belief - p_base)/(1 - p_base))
readiness = 0.5·theatre_coverage + 0.25·op10_slack + 0.25·op12_slack
Φ(s) = -α·pending_norm - β·lost_norm + κ·p_adv·readiness - ρ·(1-p_adv)·resource_posture
```

R22 (encoder pre-entrenado, AUC held-out 0.62), 3 seeds × 30k, `max_steps=104` confirmado en
todas las corridas, `batch_size=64`.

## Resultados finales (16 de 16 puntos)

| κ | ρ | ReT Excel (agg) | Costo | Seeds 1/2/3 |
|---:|---:|---:|---:|---|
| 0.05 | 0.00 | 0.005841 | 0.767 | — |
| 0.05 | 0.02 | 0.005818 | 0.776 | — |
| 0.05 | 0.05 | 0.005776 | 0.720 | — |
| 0.05 | 0.10 | 0.005810 | 0.649 | — |
| 0.10 | 0.00 | 0.005835 | 0.760 | 0.005813 / 0.005836 / 0.005855 |
| 0.10 | 0.02 | 0.005801 | 0.776 | 0.005787 / 0.005800 / 0.005816 |
| 0.10 | 0.05 | **0.005887** | 0.768 | — |
| 0.10 | 0.10 | 0.005811 | 0.752 | 0.005834 / 0.005794 / 0.005805 |
| 0.20 | 0.00 | 0.005850 | 0.684 | 0.005815 / 0.005835 / 0.005899 |
| 0.20 | 0.02 | 0.005815 | 0.712 | 0.005784 / 0.005774 / 0.005885 |
| 0.20 | 0.05 | 0.005830 | 0.777 | — |
| 0.20 | 0.10 | 0.005838 | 0.753 | — |
| 0.40 | 0.00 | 0.005825 | 0.838 | 0.005766 / 0.005824 / 0.005886 |
| 0.40 | 0.02 | 0.005852 | 0.801 | 0.005863 / 0.005835 / 0.005859 |
| 0.40 | 0.05 | 0.005854 | 0.736 | — |
| 0.40 | 0.10 | 0.005834 | 0.788 | 0.005834 / 0.005827 / 0.005840 |

(Filas con seed "—" corresponden a la primera tanda, ya verificadas por-seed en su momento pero no
re-copiadas aquí; su agregado sí está confirmado directamente contra `summary.json`.)

Referencias: v10 crudo `0.005811`; **Arm 2 solo (belief en observación, sin condicionar recompensa)
= 0.005915`**; Arm 1 solo (h104, sin belief) `0.005779`. `max_steps=104` confirmado en las 16
corridas.

## Veredicto: negativo — el grid completo no supera Arm 2 solo

**Ninguno de los 16 puntos supera 0.005915, ni en agregado ni en ningún seed individual.** El
valor por-seed más alto de todo el grid es κ=0.20/ρ=0.00 seed 3 = 0.005899 — todavía -0.000016
por debajo del umbral, y no es un patrón consistente dentro de esa misma configuración (sus otros
dos seeds son 0.005815 y 0.005835). El mejor punto en agregado sigue siendo κ=0.10/ρ=0.05 con
0.005887 (-0.000028 vs Arm 2 solo), de la primera tanda.

No hay ninguna región del grid κ∈{0.05,0.10,0.20,0.40}×ρ∈{0.00,0.02,0.05,0.10} donde condicionar
la recompensa por la creencia (con el término de penalización de postura de recurso) mejore sobre
simplemente exponer el belief en la observación sin tocar la recompensa (Arm 2). El patrón es
consistente con lo visto en el subconjunto parcial: el término `κ·p_adv·readiness` no añade señal
de entrenamiento neta una vez que el belief ya está disponible como observación; en el mejor caso
es neutral, y con ρ alto y κ bajo el castigo de recurso domina y el agente simplemente vuelve a
operar barato (ReT cae a la altura de v10 crudo, costo baja).

**Per el criterio de "solo escalar si hay señal", NO se lanza el contrafactual causal
(`audit_track_b_risk_event_counterfactual.py`) sobre ninguno de estos 16 checkpoints.** Ya
sabíamos que Arm 2 solo (el mejor insumo de este combinado) falla el contrafactual causal
(6.7% positive-pair rate, sin señal). Un combinado que ni siquiera supera a Arm 2 en la métrica de
entrenamiento no tiene ninguna base para superarlo en la prueba causal, que es más exigente.

## Conclusión de la línea "belief-conditioned reward shaping" (v2)

Con esto se cierran, sin señal causal de prevención, las tres variantes probadas esta sesión sobre
el mismo problema (creer-para-prevenir vía Ruta A + reward shaping):
- Arm 1 solo (reward shaping por PBRS, sin belief): negativo (0.005779 < 0.005811 crudo).
- Arm 2 solo (belief-R22 en la observación, sin condicionar recompensa): mejora el entrenamiento
  (0.005915) pero sin señal causal en el contrafactual (6.7% positive-pair rate).
- Combinado (belief-conditioned PBRS, 16 puntos κ/ρ): no mejora sobre Arm 2 solo en ningún punto
  del grid — no hay entrada al gate causal.

No se recomienda seguir explorando este espacio de hiperparámetros (κ/ρ) sin cambiar de enfoque.
El diagnóstico de raíz sigue siendo el escrito en
`docs/TRACK_B_PREVENTIVE_LEARNING_ROOT_CAUSE_AND_ALTERNATIVES_2026-07-04.md`: el problema no
parece ser la falta de una señal de creencia (ya la tenemos, y ayuda al entrenamiento) sino el
crédito de largo plazo por preparación temprana, que este mecanismo de shaping no logra resolver
en la forma probada. Las alternativas más caras y no probadas (Ruta B end-to-end, restricciones
tipo Lagrangiano) quedan explícitamente diferidas, pendientes de instrucción del usuario.
