# Por qué no emerge prevención, y qué alternativas quedan — 2026-07-04

Actualiza `docs/PREVENTIVE_LEARNING_ROADMAP_2026-07-03.md` y
`docs/RESEARCH_PROPOSALS_REGISTRY_2026-06-28.md` con lo aprendido en la sesión del
2026-07-03/04: el contrafactual ya es válido (fix de RNG) y se probaron memoria cruda,
belief-escalar y Ruta A (encoder pre-entrenado) en ambas arquitecturas. Ninguna mostró prevención
causal. Ver `docs/TRACK_B_OVERNIGHT_PREVENTIVE_LEARNING_FINAL_VERDICT_2026-07-04.md` para el
detalle experimental. Este documento responde la pregunta de fondo: **¿por qué?**, y **¿qué falta
probar?**

## El marco correcto (ya estaba en el registro, ahora con evidencia que lo confirma)

`docs/RESEARCH_PROPOSALS_REGISTRY_2026-06-28.md` ya lo planteaba así:

> `R_t = f(decision frontier, observability, resource pricing, shock recurrence)`

Un agente necesita las cuatro cosas a la vez para ser preventivo. Ya verificamos tres:

1. **Frontera de decisión**: el contrato de 8D puede representar "prepararse antes" (subir turno,
   subir despacho downstream antes de que llegue el daño). ✓ Existe.
2. **Observabilidad**: `v10` ya tiene memoria histórica real de R24 (`weeks_since_last`,
   conteos 8/26 semanas, EWMA), con AUC held-out de 0.62-0.77 — información genuina, no ruido. ✓
   Existe.
3. **Recurrencia del riesgo**: R24 es lo bastante frecuente y aprendible (confirmado en el
   predictor supervisado). ✓ Existe.
4. **Precio del recurso** — **este es el que probablemente falta**, y es la explicación más
   consistente con todo lo que vimos esta noche.

## Por qué el precio del recurso es sospechoso número uno

Tres piezas de evidencia, independientes entre sí, apuntan al mismo lugar:

**A. El mecanismo real de Ruta A no fue anticipación, fue aversión al riesgo permanente.**
Verificamos que Ruta A (PPO+MLP) casi elimina el turno S1 (0.08% del tiempo vs. 6-9% en los
comparadores) y opera con S2 como default. Es un cambio de **postura basal**, no de **timing**
puntual. Esto es exactamente lo que un agente *reward-maximizador* haría si el precio de "estar
en modo barato quedó peor parado" es mayor que el ahorro de operar barato — sin necesitar
predecir NADA sobre el futuro. El agente no necesitó su cabeza de creencia para llegar aquí; solo
necesitó que operar en S1 resultara, en promedio, más caro que quedarse en S2.

**B. Ya se documentó exactamente el mismo problema en Track A, con un mecanismo distinto.**
`docs/EXCEL_REWARD_PREPOSITIONING_AUDIT_2026-06-27.md` (2026-06-27, antes de esta sesión): PPO
intentando aprender la decisión de preposicionamiento inicial (antes del episodio) colapsaba a
políticas constantes o inestables entre semillas — **"the first action has a long delayed-credit
problem"**. Esto es Track A, acción discreta, un mecanismo de preposicionamiento totalmente
distinto al de Track B — y aun así, el mismo síntoma: PPO vainilla no logra asignar crédito
correctamente a una acción temprana cuyo beneficio se materializa muchas semanas después.

**C. El contrafactual mismo mide exactamente esto y da cero.** `R_full - R_reset(pre-riesgo)`
pregunta: "¿la acción de esta ventana específica, antes de este evento específico, aportó ReT?"
Si el crédito de esa acción no fluye correctamente hacia atrás durante el entrenamiento (por el
descuento `gamma=0.99` combinado con GAE y una ventana de rollout de 1024 pasos ~ 6 episodios),
PPO nunca tuvo un incentivo de gradiente claro para aprenderla — y el resultado (8.9% de pares
positivos, incluso con una representación de creencia genuina) es exactamente lo que se esperaría.

## Qué necesitaría un agente para ser preventivo (no solo tener la información)

No basta con que la información exista en la observación. Hacen falta, además:

1. **Que el diseño de recompensa premie "actuar antes" de forma distinguible de "reaccionar
   rápido después"** — hoy, si reaccionar rápido ya recupera casi todo el ReT perdido, no hay
   incentivo marginal para prepararse antes. La ganancia de prevención tiene que ser
   *recompensada explícitamente*, no asumida.
2. **Que el crédito de una acción temprana llegue de vuelta al gradiente de política sin
   diluirse en 15-20+ semanas de descuento** — esto requiere o bien recompensa moldeada
   (potential-based shaping) que dé señal INMEDIATA por reducir el riesgo futuro esperado, o una
   estructura de política que separe la decisión de preparación de la decisión semana a semana.
3. **Que la representación predictiva se mantenga "con forma de predicción" durante todo el
   entrenamiento**, no solo al inicializar — Ruta A dio una buena inicialización, pero nada
   durante el fine-tuning de PPO obliga a la red a seguir usando esa estructura para predecir; el
   agente la reaprovechó para otra cosa (postura basal general).
4. **Que el riesgo elegido sea uno donde reaccionar rápido sea genuinamente peor que prepararse**
   — si no lo es, no hay ReT que ganar con prevención, sin importar qué tan bien entrenemos.

## Alternativas, ordenadas por qué tan directamente atacan el punto 4 (precio del recurso) y 1-2 (crédito)

### 1. Recompensa moldeada por ReT futura (potential-based shaping) — la más directa

Ya existe media implementación: `FutureCreditRewardWrapper`
(`scripts/run_cf20_learning_repair.py`), con `ReT_excel_delta_bootstrap` y
`ReT_excel_terminal_shaped`. Nunca se portó al contrato Track B 8D
(`docs/PREVENTIVE_LEARNING_ROADMAP_2026-07-03.md` ya lo marcaba como "propuesta activa, no
implementada"). La idea: `r'_t = r_t + γ·Φ(s_{t+1}) - Φ(s_t)`, con `Φ` construido sobre la
creencia de riesgo (p.ej. `-α·P(R24 pronto)·costo_esperado_de_no_prepararse`). Esto da señal
INMEDIATA (no 15 semanas después) por reducir el riesgo futuro esperado — ataca directamente el
problema de crédito diluido. La evaluación final sigue siendo ReT Excel sin moldear, como siempre.
**Costo: medio. Ya hay 50% del código.**

### 2. RL restringido/Lagrangiano — ataca el precio del recurso directamente

Ya está en el registro (P1), con dos referencias reales y verificadas
(Sabal Bermúdez/del Rio Chanona/Tsay 2023, Hasturk et al. 2025):
`max E[ReT] sujeto a E[recurso] ≤ B`. En vez de que cada semana de "modo caro" compita
directamente con la ganancia de ReT en el mismo escalar (como hoy), el agente puede gastar
recursos con libertad DENTRO de un presupuesto por episodio — quitando la razón por la que hoy
"prepararse antes" tiene que ganarle a "quedarse barato" en cada paso individual. **Costo:
medio-alto (nueva formulación de recompensa + multiplicador de Lagrange que ajustar).**

### 3. Descomponer la política en preposicionamiento + reactiva semanal (P3 del registro)

`π = π_init(a_0|z_0) · ∏ π_weekly(a_t|s_t,h_t)`. En vez de pedirle a un solo PPO que aprenda
"prepárate antes" y "reacciona después" con el mismo crédito de gradiente de 104 semanas, dale a
la decisión de preparación su propio componente entrenable con una señal más directa (p.ej.
optimizada contra la superficie estática, un problema mucho más tratable que el crédito RL
completo). Esto es exactamente lo que `EXCEL_REWARD_PREPOSITIONING_AUDIT_2026-06-27.md` recomendó
después de ver que PPO vainilla no estabilizaba la decisión inicial en Track A. **Costo:
alto (cambio estructural de política, no solo de reward/observación).**

### 4. Ruta B — pérdida auxiliar continua durante el entrenamiento de PPO

Ya la teníamos identificada como la opción de mayor riesgo técnico
(`docs/TRACK_B_RISK_BELIEF_AUX_HEAD_IMPLEMENTATION_PLAN_2026-07-03.md`), pero ahora tenemos una
razón concreta para intentarla: Ruta A demostró que el pre-entrenamiento por sí solo no basta
porque nada mantiene la representación "con forma de predicción" durante el fine-tuning. Con
`L = L_PPO + λ1·BCE(riesgo futuro) + λ2·MSE(horas)` activa durante TODO el entrenamiento, la
representación no puede "olvidarse" de predecir. **Costo: alto (tocar el loop interno de SB3,
riesgo de bugs sutiles, ya señalado en el propio plan original).**

### 5. Elegir un riesgo donde reaccionar tarde sea genuinamente peor

R24 se eligió porque tenía la señal supervisada más limpia (AUC 0.62), no porque sea el riesgo
donde la prevención vale más. Vale la pena revisar R22/R23 (destrucción de LOC/unidad adelantada)
— si su tiempo de recuperación es más largo o su daño más difícil de revertir reactivamente, ahí
sí podría haber ReT real que ganar con anticipación, aunque la señal supervisada sea más débil por
ser más raros. **Costo: bajo — es solo redefinir el objetivo de la cabeza auxiliar y volver a
correr el pipeline de Ruta A ya construido.**

### 6. Intervención de entorno: encarecer/retrasar la reacción a propósito

Si se hace la respuesta reactiva deliberadamente más lenta o más cara (p.ej. un lag artificial
al cambiar turno o despacho), y el agente SIGUE sin anticipar, eso confirmaría que el problema es
de crédito/recompensa, no de capacidad. Si el agente SÍ empieza a anticipar bajo esa presión, eso
confirma que la anticipación es aprendible cuando de verdad conviene. **Costo: medio (cambio de
entorno, pero no de arquitectura ni de pipeline de entrenamiento).**

### 7. Planeación explícita basada en modelo (MPC / lookahead)

En vez de esperar que la anticipación emerja de un escalar de recompensa vía RL libre de modelo,
usar un modelo de dinámica de riesgo aprendido + optimización explícita a varios pasos (estilo
control predictivo). Codifica "mira adelante y actúa ahora para prevenir" directamente, en vez de
esperar que emerja. **Costo: muy alto — arquitectura distinta, no incremental sobre lo ya
construido.**

## Recomendación de orden

Dado el patrón de esta noche (representación predictiva ya construida y funcionando, pero sin
incentivo para usarla como anticipación), el orden de mayor-probabilidad-de-éxito-por-costo es:

**1 (recompensa moldeada) → 5 (cambiar el riesgo objetivo, barato y en paralelo) → 2 (RL
restringido) → 3/4 (cambios estructurales, más caros) → 6 → 7.**

1 y 5 se pueden probar juntos sin conflicto (mismo pipeline, cambia el `Φ` y el `risk_id`
objetivo). Ninguno de los dos requiere tocar SB3 por dentro ni cambiar la arquitectura de
política — son los que más aprovechan lo ya construido esta noche (dataset v10, extractor de
creencia, script de entrenamiento con encoder trasplantado).

## Actualización de ejecución — 2026-07-04

Después de revisar este diagnóstico con el usuario, se implementaron los dos brazos de menor costo
que atacan directamente el problema:

1. **Arm 1 — future-credit reward shaping para Track B 8D.** Se portó el wrapper a
   `scripts/track_b_future_credit_reward.py` y se agregó
   `scripts/run_track_b_future_credit_sidecar.py`. El primer artefacto completado
   (`outputs/experiments/track_b_future_credit_ppo_3seed_30k_2026-07-04/`) confirma que el código
   corre, pero quedó en `max_steps=260`, por lo que no es comparable directamente con los smokes
   preventivos `h104`. Resultado exploratorio: `order_ret_excel_mean=0.0059067`, costo `0.650`.
   Se relanzó la versión corregida `h104` en
   `outputs/experiments/track_b_future_credit_ppo_3seed_30k_h104_2026-07-04/`.
2. **Arm 2 — Ruta A retargeted a R22.** Se reutilizó el pipeline de encoder pre-entrenado,
   cambiando el objetivo de predicción de R24 a R22. El entrenamiento comparable `h104`
   (`outputs/experiments/track_b_belief_encoder_ppo_r22_3seed_30k_2026-07-04/`) sí tiene señal:
   `order_ret_excel_mean=0.0059149`, costo `0.698`, mejor que Ruta A-R24 (`0.0059024`) y muy cerca
   del PPO+MLP fixed-RNG `v7` a 60k (`0.0059206`) usando sólo 30k pasos. Se lanzó el contrafactual
   pre-R22 en
   `outputs/experiments/track_b_belief_encoder_ppo_r22_counterfactual_r22_2026-07-04/`.
   Resultado: **no hay prevención causal robusta**. El delta pre-R22 fue apenas
   `+0.00000065`, con `12/150` pares positivos (`8%`). La mejora del brazo R22 es real como
   entrenamiento/representación, pero no viene de acciones pre-R22 con valor estable en ReT Excel.

La regla de avance queda igual: R22 queda cerrado como mejora de representación/operación, **no**
como prevención. Si future-credit `h104` no mejora contra el PPO+MLP `v7` al mismo presupuesto,
no se escala. Si future-credit sí pasa, el siguiente brazo lógico es auditar su contrafactual
pre-riesgo y sólo después considerar combinarlo con memoria/belief.

3. **Arm 1 + Arm 2 combinado — belief-conditioned PBRS v2.** Se probó la variante mejorada que
   condiciona el potencial por exceso de riesgo sobre tasa base (`p_adv`) y penaliza postura cara
   en calma (`rho`):

   ```text
   Phi = -alpha*pending - beta*lost
         + kappa*p_adv*readiness
         - rho*(1-p_adv)*resource_posture
   ```

   Artefactos: barrido completo
   `k ∈ {0.05, 0.10, 0.20, 0.40}` × `rho ∈ {0, 0.02, 0.05, 0.10}` bajo
   `outputs/experiments/track_b_belief_conditioned_v2_k*_r*_2026-07-04/`.
   Mejor resultado: `k010_r005`, `order_ret_excel_mean=0.005887`, costo `0.768`.
   La mejor variante `k=0.05` fue `k005_r000`, `order_ret_excel_mean=0.005841`, costo `0.767`;
   la variante de menor costo con ReT razonable fue `k020_r000`, `order_ret_excel_mean=0.005850`,
   costo `0.684`. Ninguna alcanza a Ruta A R22.
   Esto supera `v10` crudo (`0.005811`), pero queda por debajo de Ruta A R22 (`0.0059149`) y del
   PPO+MLP `v7` fixed-RNG 60k (`0.0059206`). **No pasa el filtro de entrenamiento suficiente; no
   se lanza contrafactual.**

Resultado acumulado antes del gate oráculo: ya probamos memoria cruda, belief escalar, Ruta A R24,
Ruta A R22, future-credit general y belief-conditioned PBRS v2, incluyendo un pequeño barrido local
`kappa/rho`. Varias ramas mejoran ReT, pero ninguna demuestra prevención causal ni supera de forma
convincente al spine eficiente PPO+MLP `v7`. En ese punto, las alternativas que quedaban parecían
estructurales: RL restringido/presupuesto de recursos, política descompuesta
preposicionamiento+reactiva, o Ruta B con pérdida auxiliar end-to-end.

4. **Arm 6 — intervención de entorno: reacción retardada por inercia de turnos.**
   Se activó el soporte existente de `surge_inertia` en Track B para hacer que la respuesta
   reactiva sea menos instantánea: el turno efectivo sólo puede subir gradualmente
   (`surge_ramp_per_step=1`), aunque la demovilización sigue siendo inmediata. Esto no cambia la
   métrica primaria (`order_ret_excel_mean`) ni la frontera de acción `track_b_v1`; sólo cambia la
   dinámica operacional que convierte la acción deseada en capacidad efectiva. El objetivo es
   diagnóstico: si la política empieza a anticipar bajo este entorno, entonces la prevención era
   aprendible pero no necesaria en el entorno original; si sigue sin anticipar, el problema apunta
   con más fuerza a crédito temporal/recompensa/política.

   Corrida:
   `outputs/experiments/track_b_surge_inertia_r22_ruta_a_3seed_30k_2026-07-04/`
   con Ruta A R22, `v10`, `adaptive_benchmark_v2`, `control_v1`, `h104`, seeds `1..3`,
   `30k` pasos, `8` episodios de evaluación, `batch_size=64`, `surge_budget_hours=4032`.

   Resultado: `order_ret_excel_mean=0.005845`, CI95 `[0.005736, 0.005953]`, costo `0.410`,
   semillas `0.005886 / 0.005735 / 0.005914`. Mejora sobre `v10` crudo (`0.005810698`) y
   reduce mucho el costo, pero queda por debajo de Ruta A R22 sin inercia (`0.005914858`) y del
   PPO+MLP `v7` fixed-RNG 60k (`0.005920640`).

   Veredicto: no pasa el filtro para lanzar contrafactual. La reacción retardada genera un
   trade-off eficiente de bajo costo, pero no desbloquea la señal preventiva que buscábamos.
   Arm 6 queda cerrado sin `R_full - R_reset(pre-R22)`.

5. **Gate oráculo de techo preventivo.** Para decidir si valía la pena seguir con cambios
   estructurales caros, se corrió un replay oráculo sobre el checkpoint canónico PPO+MLP `v7`
   fixed-RNG: el oráculo conoce el calendario real futuro de R22/R24 y fuerza boosts privilegiados
   en `shift`, `op10_q` y `op12_q` durante ventanas de `1, 2, 4, 8` semanas antes de los eventos.

   Artefacto:
   `outputs/experiments/track_b_oracle_prevention_ceiling_2026-07-04/`.
   Veredicto:
   `docs/TRACK_B_ORACLE_PREVENTION_CEILING_VERDICT_2026-07-04.md`.

   Línea base replay: `0.005920543949378895`, coincidente con el canónico
   `0.005920640377903427` (diferencia `9.6e-8`). Mejor resultado del grid:
   R24/R22+R24 con `L=8`, `boost=1.0`, `delta=+0.000021`, costo `0.997`. R22 solo:
   `delta=+0.000006`, costo `0.836`. El umbral de señal era `+0.0004`.

   Conclusión revisada: ni siquiera futuro perfecto y preparación privilegiada compran ReT Excel
   medio de forma material bajo `control_v1`. Sin embargo, el análisis posterior del mismo oráculo
   sí mostró mejoras consistentes en métricas de cola/recuperación (`service_loss_auc`, `ttr`,
   `cvar05`, `rolling_4w_min`). Por tanto, RL restringido, política de dos niveles y Ruta B dejan
   de ser prioridades **si el objetivo sigue siendo ReT Excel medio**; pero podrían tener sentido
   si la pregunta preventiva se redefine explícitamente como reducción de cola/tiempo de
   recuperación.

6. **Event-conditioned resilience purchase.** Para convertir esa intuición en una métrica
   interpretable, se agregó:

   `docs/TRACK_B_EVENT_RESILIENCE_PURCHASE_VERDICT_2026-07-04.md`

   Resultado: frente a una regla barata (`heur_disruption_aware`), PPO+MLP sí compra resiliencia
   local alrededor de R22/R24: mejora continuidad local, evita backorders y reduce backlog. Frente
   a PPO+MLP, Real-KAN compra mejoras locales pequeñas con mucha más intensidad. Esto deja el
   framing final más preciso: Q1 queda como aprendizaje adaptativo/operacional; la prevención
   pura no se ve en ReT Excel medio, pero la próxima métrica a probar si se reabre prevención es
   cola/recuperación por costo, no otra arquitectura sobre la misma recompensa.

7. **Gate directo de recompensa tail/recovery.** Se ejecutó el siguiente paso lógico: probar si una
   recompensa que premie cola/recuperación puede mover las métricas donde el oráculo sí mostró
   techo, sin cambiar arquitectura.

   Veredicto dedicado:
   `docs/TRACK_B_TAIL_RECOVERY_REWARD_SCREEN_2026-07-04.md`.

   Protocolo corto común: PPO+MLP, `v7`, `track_b_v1`, `adaptive_benchmark_v2`, `h104`, seeds
   `1..3`, `30k` pasos, `8` episodios, `batch_size=64`, fixed-RNG. Comparador:
   `control_v1` en la misma escala (`outputs/experiments/track_b_v7_calibration_3seed_30k_2026-07-04/`).

   Resultado:

   | Reward | ReT Excel | CVaR05 | Peor 4w | Service-loss AUC/order | TTR mean | TTR p95 | Costo |
   |---|---:|---:|---:|---:|---:|---:|---:|
   | `control_v1` | 0.005843 | 0.001912 | 0.002210 | 166614.2 | 96.83 | 175.74 | 0.734 |
   | `ReT_tail_v2` | **0.005886** | **0.002244** | **0.002403** | **120931.0** | **93.23** | **159.33** | **0.646** |
   | `ReT_excel_plus_cvar` alpha=0.2 | 0.005739 | 0.001304 | 0.001882 | 345599.4 | 109.97 | 236.13 | 0.756 |
   | `control_v2` | 0.005826 | 0.001843 | 0.002170 | 184772.9 | 99.02 | 188.76 | 0.640 |

   Lectura: `ReT_tail_v2` es la primera rama de prevención/recompensa que mejora simultáneamente
   ReT Excel, cola (`CVaR05`, peor ventana 4w), recuperación (`TTR`) y costo. Esto todavía no
   demuestra prevención causal, pero sí demuestra que PPO puede comprar la forma de resiliencia que
   el promedio ReT Excel no estaba premiando con suficiente fuerza.

   Siguiente paso ya lanzado: confirmatoria `ReT_tail_v2` 5 seeds × 60k × 12 episodios en
   `outputs/experiments/track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04/`
   (`tmux track_b_ret_tail_v2_confirm_5seed_60k`). Si se sostiene, recién ahí se justifica correr
   el análisis preventivo/reactivo sobre ese checkpoint. Si no se sostiene, se cierra como smoke
   positivo no confirmatorio.

   Actualización confirmatoria: `ReT_tail_v2` sí se sostuvo parcialmente a escala 5x60k. Frente a
   `control_v1` fixed-RNG 5x60k, mejora ReT Excel (`0.005921 -> 0.005932`), CVaR05
   (`0.002284 -> 0.002334`), peor ventana 4w (`0.002013 -> 0.002091`), service-loss AUC/order
   (`94427 -> 87953`) y TTR p95 (`157.16h -> 153.96h`). El costo sube ligeramente
   (`0.700 -> 0.722`).

   Se corrió entonces el contrafactual fixed-RNG `R_full - R_reset(pre-risk)` para R22/R24:
   `outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_2026-07-04/`.
   Resultado: R22 `+0.00000340` con `35/374` pares positivos; R24 `+0.00000349` con `41/393`
   pares positivos. Deltas positivos pero diminutos y tasas positivas demasiado bajas. Veredicto:
   **sin prevención causal robusta**.

   Conclusión final de esta rama: `ReT_tail_v2` es un reward prometedor para resiliencia de
   cola/recuperación, pero no demuestra aprendizaje preventivo. Por disciplina de parada, no se
   lanzan más variantes dentro de esta familia sin una decisión explícita de cambiar el objetivo
   del paper hacia cola/TTR/service-loss.

   Actualización 10-seed: se completó la extensión pareada seeds `6..10` para `control_v1` y
   `ReT_tail_v2`, fusionando después seeds `1..10`. Resultado 10-seed:

   | Métrica | Delta `ReT_tail_v2 - control_v1` | CI95 | Signos favorables |
   |---|---:|---:|---:|
   | ReT Excel | +0.00001604 | [-0.00000036, +0.00003244] | 7/10 |
   | CVaR05 | +0.00008242 | [-0.00001371, +0.00017855] | 8/10 |
   | Peor 4w | +0.00008700 | [+0.00000702, +0.00016697] | 8/10 |
   | Service-loss AUC/order | -11682.0 | [-24121.0, +757.0] | 7/10 |
   | TTR mean | -0.7411h | [-1.5955, +0.1133] | 8/10 |
   | TTR p95 | -5.1945h | [-9.5558, -0.8332] | 7/10 |
   | Costo | -0.01784 | [-0.11624, +0.08055] | 6/10 |

   Lectura honesta: sólo peor ventana 4w y TTR p95 cierran cero limpiamente a 10 seeds. ReT Excel,
   CVaR05, service-loss AUC y TTR mean se mueven en la dirección correcta, pero todavía tocan cero
   con CI95 t(9). Por tanto, `ReT_tail_v2` queda como **candidate sidecar**, no claim confirmado.

   Counterfactual 10-seed pre-riesgo sobre R22/R24:
   R22 `+0.00000331`, `78/766` positivos; R24 `+0.00000336`, `82/789` positivos. Por tanto,
   la señal 10-seed queda cerrada como **adaptativa tail/recovery**, no preventiva causal.

   Actualización final n=15: se extendió a seeds `11..15` para intentar cerrar el claim primario.
   El resultado no promovió la rama. Deltas pareados 15 seeds, CI95 t(14): ReT Excel `+0.000008`
   `[-0.000004,+0.000021]`; CVaR05 `+0.000045` `[-0.000025,+0.000115]`; peor 4w `+0.000052`
   `[-0.000013,+0.000117]`; service-loss AUC/order `-5294` `[-15434,+4846]`; TTR mean
   `-0.306h` `[-1.000,+0.388]`; TTR p95 `-2.616h` `[-6.476,+1.244]`; costo `-0.011`
   `[-0.075,+0.053]`. Ninguna métrica cierra cero a n=15. Cierre: evidencia exploratoria de
   dirección favorable, no claim confirmatorio; no seguir escalando sin una razón de manuscrito
   explícita.
