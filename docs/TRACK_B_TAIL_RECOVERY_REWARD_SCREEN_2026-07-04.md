# Track B tail/recovery reward screen — 2026-07-04

## Pregunta

Despues del gate oraculo vimos que el ReT Excel medio casi no se mueve, pero
que las metricas de cola y recuperacion si se mueven: `ret_excel_cvar05`,
`ret_excel_rolling_4w_min`, `service_loss_auc_per_order`, `ttr_mean` y
`ttr_p95`.

El siguiente paso logico fue no probar otra arquitectura, sino preguntar si una
recompensa de entrenamiento que mire cola/recuperacion puede hacer que PPO
compre esa resiliencia sin destruir la metrica primaria de Garrido.

## Protocolo

Todos los brazos usan:

- PPO+MLP.
- `observation_version=v7`.
- `action_contract=track_b_v1`.
- `risk_level=adaptive_benchmark_v2`.
- `max_steps=104`.
- Seeds `1..3`.
- `train_timesteps=30000`.
- `eval_episodes=8`.
- `n_steps=1024`, `batch_size=64`.
- Track B fixed-RNG (`strict_exogenous_crn=True` en el entorno).

Comparador directo:

- `outputs/experiments/track_b_v7_calibration_3seed_30k_2026-07-04/`
  (`control_v1`, misma escala).

Brazos probados:

- `ReT_tail_v2`: reward de entrenamiento orientado a servicio/recuperacion,
  backlog containment y costo.
- `ReT_excel_plus_cvar` con `alpha=0.2`: ReT Excel incremental menos penalidad
  de cola de service loss.
- `control_v2`: reward operacional densa con servicio/recurso mas explicitos.

Artefactos:

- `outputs/experiments/track_b_tail_reward_screen_ret_tail_v2_3seed_30k_2026-07-04/`
- `outputs/experiments/track_b_tail_reward_screen_excel_cvar_a02_3seed_30k_2026-07-04/`
- `outputs/experiments/track_b_tail_reward_screen_control_v2_3seed_30k_2026-07-04/`
- Resumen: `outputs/experiments/track_b_tail_reward_screen_summary_2026-07-04/`

## Resultado

| Reward de entrenamiento | ReT Excel | CVaR05 | Peor 4w | Service-loss AUC/order | TTR mean | TTR p95 | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_v1` baseline | 0.005843 | 0.001912 | 0.002210 | 166614.2 | 96.83 | 175.74 | 0.734 |
| `ReT_tail_v2` | **0.005886** | **0.002244** | **0.002403** | **120931.0** | **93.23** | **159.33** | **0.646** |
| `ReT_excel_plus_cvar` alpha=0.2 | 0.005739 | 0.001304 | 0.001882 | 345599.4 | 109.97 | 236.13 | 0.756 |
| `control_v2` | 0.005826 | 0.001843 | 0.002170 | 184772.9 | 99.02 | 188.76 | 0.640 |

`ReT_tail_v2` mejora todos los indicadores relevantes frente al baseline de la
misma escala:

- ReT Excel: `+0.0000427` (`+0.73%`).
- CVaR05: `+0.0003317` (`+17.35%`).
- Peor ventana de 4 semanas: `+0.0001935` (`+8.76%`).
- Service-loss AUC/order: `-45,683` (`-27.42%`).
- TTR mean: `-3.60 h` (`-3.72%`).
- TTR p95: `-16.41 h` (`-9.34%`).
- Costo: `-0.088` (`-11.99%`).

`ReT_excel_plus_cvar` y `control_v2` no pasan el gate: ambas degradan ReT Excel
y las metricas de cola/recuperacion frente al baseline de la misma escala.

## Lectura

Este es el primer resultado de la rama preventiva que muestra una senal limpia
en la direccion correcta sin cambiar arquitectura:

1. conserva y mejora la metrica primaria de Garrido,
2. mejora cola/recuperacion,
3. baja costo,
4. usa el contrato canonico v7/8D/fixed-RNG.

Todavia no prueba prevencion causal. Prueba algo mas basico y necesario: que si
premiamos explicitamente la recuperacion y la cola, PPO puede comprar esa forma
de resiliencia. Por eso si amerita escalar.

## Siguiente paso lanzado

Se lanzo confirmatoria:

`outputs/experiments/track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04/`

Protocolo:

- PPO+MLP.
- `ReT_tail_v2`.
- `v7`, `track_b_v1`, `adaptive_benchmark_v2`.
- Seeds `1..5`.
- `60000` pasos.
- `12` episodios de evaluacion.
- `h104`.

Sesion:

`tmux track_b_ret_tail_v2_confirm_5seed_60k`.

Regla de decision: si `ReT_tail_v2` conserva la mejora en ReT Excel y mantiene
mejora en CVaR05 / peor 4w / service-loss AUC / TTR a escala 5x60k, entonces
se justifica correr el analisis preventivo/reactivo sobre ese checkpoint. Si no
se sostiene, la rama se cierra como smoke positivo no confirmatorio.

## Confirmatoria 5 seeds x 60k

La confirmatoria terminó en:

`outputs/experiments/track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04/`

Comparador canónico fixed-RNG:

`outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/`

| Reward | ReT Excel | CVaR05 | Peor 4w | Service-loss AUC/order | TTR mean | TTR p95 | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_v1` fixed-RNG | 0.005921 | 0.002284 | 0.002013 | 94427.3 | 91.97 | 157.16 | 0.700 |
| `ReT_tail_v2` confirm | **0.005932** | **0.002334** | **0.002091** | **87953.0** | **91.57** | **153.96** | 0.722 |

Deltas medios:

- ReT Excel: `+0.0000116`.
- CVaR05: `+0.0000504`.
- Peor ventana 4w: `+0.0000774`.
- Service-loss AUC/order: `-6474`.
- TTR mean: `-0.40h`.
- TTR p95: `-3.20h`.
- Costo: `+0.022`.

Signos por seed:

| Métrica | Signos favorables |
|---|---:|
| ReT Excel | 3/5 |
| CVaR05 | 4/5 |
| Peor 4w | 4/5 |
| Service-loss AUC/order | 3/5 |
| TTR mean | 4/5 |
| TTR p95 | 3/5 |
| Costo | 2/5 |

Lectura confirmatoria: `ReT_tail_v2` sostiene una mejora modesta pero consistente en cola y
recuperación, con una pequeña mejora en ReT Excel medio. El costo ya no baja como en el smoke; sube
ligeramente. Por tanto, esto no reemplaza automáticamente al spine `control_v1`, pero sí es la
primera variante que justifica una auditoría preventivo/reactiva posterior: ya hay algo real que
explicar.

Siguiente paso: correr el `R_full - R_reset(pre-risk)` fixed-RNG sobre el checkpoint
`ReT_tail_v2`, empezando por R22/R24. Si la señal aparece antes del evento y con tasa de pares
positivos razonable, se puede hablar de prevención de cola/recuperación; si no, el hallazgo queda
como mejor recompensa adaptativa de recuperación, no como prevención.

### Counterfactual lanzado

Se lanzó:

`outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_2026-07-04/`

Comando conceptual:

```text
audit_track_b_risk_event_counterfactual.py
  --policies ppo_mlp
  --ppo-bundles outputs/experiments/track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04
  --reward-mode ReT_tail_v2
  --target-risks R22 R24
  --seeds 1 2 3 4 5
  --eval-episodes 12
  --max-events-per-risk-episode 8
```

Sesión:

`tmux track_b_ret_tail_v2_cf_r22_r24`.

### Resultado del counterfactual

Artefacto:

`outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_2026-07-04/`

| Política | Riesgo | Pares | Positivos | Delta ReT Excel medio | Cobertura reset | Lectura |
|---|---|---:|---:|---:|---:|---|
| PPO+MLP `ReT_tail_v2` | R22 | 374 | 35/374 | +0.00000340 | 0.037 | sin señal causal clara |
| PPO+MLP `ReT_tail_v2` | R24 | 393 | 41/393 | +0.00000349 | 0.037 | sin señal causal clara |

El delta medio es positivo, pero demasiado pequeño y con una tasa de pares positivos muy baja
(`9.4%` y `10.4%`). El resultado no cumple el criterio mínimo para llamar a la política
preventiva. La mejora confirmada de `ReT_tail_v2` debe interpretarse como **mejor recompensa
adaptativa de cola/recuperación**, no como aprendizaje preventivo robusto.

## Veredicto final de esta rama

- `ReT_tail_v2` sí compra resiliencia de cola/recuperación:
  CVaR05, peor ventana 4w, service-loss AUC y TTR mejoran frente a `control_v1`.
- La mejora sobre ReT Excel medio es real pero pequeña.
- El costo sube ligeramente en la confirmatoria 5x60k.
- El counterfactual pre-riesgo R22/R24 no muestra prevención causal robusta.

Por disciplina de parada, no se lanzan más variantes esta noche dentro de esta familia. Si se
retoma prevención, tendría que hacerse con una reformulación explícita del objetivo del paper:
prevenir como **mejorar cola/TTR/service-loss**, no como subir de forma material el ReT Excel
promedio.

## Extension 10 seeds en marcha

Como la señal confirmatoria 5x60k fue positiva pero moderada, el siguiente paso por la rama con
señal fue extender a seeds `6..10` bajo el mismo protocolo, **incluyendo el comparador `control_v1`**
para que la comparación 10-seed sea pareada y no mezcle escalas.

Corridas lanzadas el 2026-07-05:

- `outputs/experiments/track_b_control_v1_fixed_rng_extension_6_10_60k_2026-07-05/`
- `outputs/experiments/track_b_ret_tail_v2_extension_6_10_60k_2026-07-05/`

Protocolo:

- PPO+MLP.
- `v7`, `track_b_v1`, `adaptive_benchmark_v2`.
- Seeds `6 7 8 9 10`.
- `60000` pasos.
- `12` episodios de evaluacion.
- `h104`.
- `n_steps=1024`, `batch_size=64`.

Sesiones:

- `tmux track_b_control_v1_fixed_rng_6_10_60k`
- `tmux track_b_ret_tail_v2_6_10_60k`

Regla de decision 10-seed: si `ReT_tail_v2` mantiene mejora en cola/recuperacion
(`CVaR05`, peor ventana 4w, `service_loss_auc`, `TTR`) sin degradar materialmente ReT Excel medio,
se promueve como sidecar de **resiliencia de cola/recuperacion**. Si la señal desaparece en seeds
6..10, queda como resultado 5-seed prometedor pero no robusto.

## Extension 10 seeds — resultado

Las extensiones seeds `6..10` terminaron y se fusionaron con seeds `1..5`.

Artefactos nuevos:

- `outputs/experiments/track_b_control_v1_fixed_rng_extension_6_10_60k_2026-07-05/`
- `outputs/experiments/track_b_ret_tail_v2_extension_6_10_60k_2026-07-05/`
- `outputs/experiments/track_b_ret_tail_v2_10seed_merged_summary_2026-07-05/`

Resultado agregado 10 seeds:

| Reward | ReT Excel | CVaR05 | Peor 4w | Service-loss AUC/order | TTR mean | TTR p95 | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_v1` fixed-RNG 10seed | 0.005913 | 0.002207 | 0.001821 | 98725.8 | 92.52 | 161.15 | 0.680 |
| `ReT_tail_v2` 10seed | **0.005929** | **0.002289** | **0.001908** | **87043.7** | **91.78** | **155.96** | **0.662** |

Deltas pareados 10 seeds:

| Métrica | Delta medio | CI95 | Signos favorables |
|---|---:|---:|---:|
| ReT Excel | +0.00001604 | [-0.00000036, +0.00003244] | 7/10 |
| CVaR05 | +0.00008242 | [-0.00001371, +0.00017855] | 8/10 |
| Peor 4w | +0.00008700 | [+0.00000702, +0.00016697] | 8/10 |
| Service-loss AUC/order | -11682.0 | [-24121.0, +757.0] | 7/10 |
| TTR mean | -0.7411h | [-1.5955, +0.1133] | 8/10 |
| TTR p95 | -5.1945h | [-9.5558, -0.8332] | 7/10 |
| Costo | -0.01784 | [-0.11624, +0.08055] | 6/10 |

Lectura 10-seed: la señal de `ReT_tail_v2` **se sostiene direccionalmente**, pero no como win
confirmatorio limpio. Todas las direcciones principales son favorables, y dos métricas cierran cero
con CI95 t(9): peor ventana 4w y TTR p95. La métrica primaria, ReT Excel, mejora en media y en
7/10 seeds, pero su CI95 toca cero. CVaR05, service-loss AUC y TTR mean también quedan como señales
favorables pero no cerradas.

Promoción recomendada: `ReT_tail_v2` no reemplaza el spine principal de Q1 y no debe citarse como
win confirmado en el manuscrito. Sí merece quedar como **candidate sidecar** de resiliencia de
cola/recuperación. La narrativa correcta no es "aprendió prevención"; es "el diseño de recompensa
apunta a una mejora direccional de cola/recuperación, con dos métricas cerrando cero a 10 seeds,
pero el ReT Excel primario todavía requiere 15-20 seeds si queremos confirmación limpia".

Para cerrar la pregunta preventiva, se lanzó una extensión del counterfactual pre-riesgo sobre
seeds `6..10`:

`outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_6_10_2026-07-05/`

Si esa extensión vuelve a mostrar tasa de pares positivos baja, queda cerrado como adaptativo
tail/recovery, no preventivo.

## Counterfactual 10 seeds — cierre preventivo

La extensión seeds `6..10` del counterfactual terminó y se fusionó con seeds `1..5`.

Artefacto:

`outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_10seed_summary_2026-07-05/`

| Riesgo | Pares | Pares positivos | Tasa positiva | Delta ReT Excel medio | Lectura |
|---|---:|---:|---:|---:|---|
| R22 | 766 | 78 | 0.102 | +0.00000331 | sin señal causal clara |
| R24 | 789 | 82 | 0.104 | +0.00000336 | sin señal causal clara |

Veredicto mecanístico final: incluso con el reward `ReT_tail_v2`, las acciones pre-riesgo
R22/R24 no muestran una contribución causal estable. El delta medio es positivo, pero está en el
orden de `3e-6` y sólo alrededor de 10% de los pares son positivos. Por tanto, la mejora 10-seed
de `ReT_tail_v2` se debe tratar como **aprendizaje adaptativo de recuperación/cola**, no como
prevención.

## Veredicto final 10-seed

`ReT_tail_v2` queda como candidate sidecar de resiliencia de cola/recuperación:

- mejora ReT Excel medio de forma pequeña y direccional, pero el CI95 toca cero;
- mejora peor ventana 4w y TTR p95 con CI95 favorable;
- service-loss AUC y TTR mean se mueven en dirección favorable, pero todavía tocan cero;
- mejora CVaR05 en 8/10 seeds, aunque el CI95 roza cero;
- no demuestra prevención causal pre-riesgo;
- no reemplaza el spine `control_v1` del paper Q1;
- se ejecutó la extensión a 15 seeds para intentar cerrar el claim primario; el resultado final se
  reporta abajo.

## Extension 15 seeds — cierre de promoción

Se extendió a seeds `11..15` para comprobar si el CI del ReT Excel primario cerraba.

Artefactos:

- `outputs/experiments/track_b_control_v1_fixed_rng_extension_11_15_60k_2026-07-05/`
- `outputs/experiments/track_b_ret_tail_v2_extension_11_15_60k_2026-07-05/`
- `outputs/experiments/track_b_ret_tail_v2_15seed_merged_summary_2026-07-05/`

Resultado agregado 15 seeds:

| Reward | ReT Excel | CVaR05 | Peor 4w | Service-loss AUC/order | TTR mean | TTR p95 | Costo |
|---|---:|---:|---:|---:|---:|---:|---:|
| `control_v1` fixed-RNG 15seed | 0.005902 | 0.002123 | 0.001761 | 100832.9 | 93.43 | 165.36 | 0.662 |
| `ReT_tail_v2` 15seed | 0.005910 | 0.002168 | 0.001813 | 95539.2 | 93.12 | 162.75 | 0.651 |

Deltas pareados 15 seeds, CI95 t(14):

| Métrica | Delta medio | CI95 | Signos favorables |
|---|---:|---:|---:|
| ReT Excel | +0.00000831 | [-0.00000444, +0.00002106] | 9/15 |
| CVaR05 | +0.00004487 | [-0.00002516, +0.00011490] | 10/15 |
| Peor 4w | +0.00005231 | [-0.00001280, +0.00011742] | 10/15 |
| Service-loss AUC/order | -5293.6 | [-15433.7, +4846.4] | 9/15 |
| TTR mean | -0.3064h | [-1.0004, +0.3877] | 10/15 |
| TTR p95 | -2.6163h | [-6.4763, +1.2437] | 10/15 |
| Costo | -0.01067 | [-0.07464, +0.05331] | 8/15 |

Lectura final: la extensión 15-seed **no cerró** el ReT Excel primario ni las métricas secundarias.
La dirección sigue siendo favorable en la mayoría de seeds, pero el efecto promedio se redujo y
todas las CI95 cruzan cero. Por tanto, `ReT_tail_v2` baja de "candidate sidecar prometedor" a
**idea secundaria no confirmada**. Es útil metodológicamente para explicar que una recompensa
orientada a cola puede mover la dirección correcta, pero no debe entrar al manuscrito como claim
empírico salvo como trabajo futuro o apéndice exploratorio.
