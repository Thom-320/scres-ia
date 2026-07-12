# Arm 1 (reward shaping) y Arm 2 (retarget a R22), probados por separado — 2026-07-04

## Arm 1 — Reward shaping (`ReT_excel_terminal_shaped`, v7, sin belief)

**Corrección 2026-07-04:** el primer número `0.005907` correspondía a un run que quedó en
`max_steps=260`, no al horizonte preventivo comparable `h104`. Sirve como prueba de plomería, pero
no debe compararse contra los smokes `h104`. El re-run comparable `h104` ya aterrizó y cambia el
veredicto.

### Run comparable `h104`

Artefacto:
`outputs/experiments/track_b_future_credit_ppo_3seed_30k_h104_2026-07-04/`

| Métrica | Valor |
|---|---:|
| ReT Excel (3 seeds × 30k, h104) | **0.005779** |
| CI95 | [0.005688, 0.005870] |
| Costo | 0.665 |

Comparado con v10 crudo (`0.005811`) y Ruta A R22 (`0.005915`), este brazo **no pasa** el primer
filtro de entrenamiento. No se escala ni se audita causalmente bajo esta forma general de
`Phi(s)`.

### Run no comparable `h260` (solo evidencia de plomería)

| Métrica | Valor |
|---|---:|
| ReT Excel (3 seeds × 30k, h260) | **0.005907** |
| CI95 | [0.005823, 0.005990] |
| Costo | 0.650 |
| Por semilla | 0.005822 / 0.005955 / 0.005943 |

Este resultado no debe usarse como evidencia de mejora comparable; el horizonte era distinto.

## Arm 2 — Retarget a R22 (Ruta A, mismo encoder pero prediciendo R22 en vez de R24)

| Métrica | Valor |
|---|---:|
| ReT Excel (3 seeds × 30k) | **0.005915** |
| CI95 | [0.005881, 0.005948] |
| Costo | 0.698 |
| Por semilla | 0.005925 / 0.005881 / 0.005938 |

Comparado con v10 crudo: 0.005811 → **+0.000104 (+1.79%)**. Comparado con Ruta A apuntando a R24:
0.005902 → **+0.000013**, una mejora marginal. Consistente en las 3 semillas.

**Contrafactual `R_full - R_reset(pre-R22)`**: 150 pares, **10 positivos (6.7%)**, delta medio
**+0.00000040** (esencialmente cero). Sin señal causal.

## Lectura

Los brazos aislados separan dos conclusiones:

- **Arm 1 general (`Phi` sin belief) no pasa el filtro comparable `h104`**. El run `h260` era
  alentador, pero no comparable.
- **Arm 2 confirma que el problema no era "R24 es mal objetivo"** — cambiar a un riesgo con
  downtime obligatorio (R22) mejora entrenamiento, pero tampoco desbloquea causalidad pre-riesgo.
  La explicación de fondo sigue siendo el diseño de crédito/recompensa y el precio del recurso,
  no solo la elección del riesgo.

## Un matiz importante sobre Arm 1 antes de descartar reward shaping del todo

El `Φ(s)` de Arm 1 es una función de calidad de estado **general** (backlog pendiente, órdenes
perdidas, cobertura de `rations_theatre`) — **no está condicionada a la creencia de riesgo**. Es
decir, premia "estar siempre preparado", el mismo mecanismo que ya vimos en Ruta A (casi eliminar
S1), no específicamente "prepárate más justo antes de un riesgo previsto". Esto explica por qué
el run no comparable podía verse bien sin aislar timing preventivo: tal como está construido,
no distingue "antes de un riesgo" de "en general".

**Esto no descarta la hipótesis de reward shaping — descarta esta versión específica, sin
condicionar por creencia.** La combinación natural que queda sin probar es
`Φ(s) = ... + κ·P_belief(riesgo_pronto)·cobertura` — una recompensa que premie la cobertura
*más* cuando la creencia predice un riesgo próximo, no siempre por igual. Esa sí sería una prueba
limpia de "combinación" en el sentido que preguntaste.

## Estado

Arm 1 general queda cerrado negativamente en `h104`. Arm 2 R22 queda como mejora de
representación/entrenamiento, no prevención causal. La única variante de reward shaping que todavía
tiene sentido probar es la versión condicionada por creencia:
`Φ(s) = ... + κ·P_belief(riesgo_pronto)·cobertura`, idealmente con las mejoras de calibración y
precio de recurso documentadas en el diagnóstico raíz.

## Variante combinada v2 — belief-conditioned PBRS con `p_adv` y penalización de calma

Se probó la versión mejorada de la combinación Arm 1 + Arm 2:

```text
p_adv = max(0, (p_belief - p_base) / (1 - p_base))
Phi = -alpha*pending - beta*lost
      + kappa*p_adv*readiness
      - rho*(1-p_adv)*resource_posture
```

Esta versión corrige el defecto conceptual de la primera combinación: no usa probabilidad cruda,
sino exceso sobre tasa base (`p_adv`), y penaliza postura cara cuando no hay alerta (`rho`).

| Variante | ReT Excel | CI95 | Costo | Lectura |
|---|---:|---:|---:|---|
| `k005_r000` | 0.005841 | [0.005787, 0.005895] | 0.767 | no pasa |
| `k005_r002` | 0.005818 | [0.005747, 0.005890] | 0.776 | no pasa |
| `k005_r005` | 0.005776 | [0.005729, 0.005823] | 0.720 | no pasa |
| `k005_r010` | 0.005810 | [0.005728, 0.005892] | 0.649 | no pasa |
| `k010_r000` | 0.005835 | [0.005811, 0.005859] | 0.760 | no pasa |
| `k010_r002` | 0.005801 | [0.005784, 0.005818] | 0.776 | no pasa |
| `k010_r005` | 0.005887 | [0.005868, 0.005907] | 0.768 | mejor v2, pero no pasa |
| `k010_r010` | 0.005811 | [0.005787, 0.005835] | 0.752 | no pasa |
| `k020_r000` | 0.005850 | [0.005800, 0.005900] | 0.684 | no pasa |
| `k020_r002` | 0.005815 | [0.005745, 0.005884] | 0.712 | no pasa |
| `k020_r005` | 0.005830 | [0.005765, 0.005895] | 0.777 | no pasa |
| `k020_r010` | 0.005838 | [0.005833, 0.005842] | 0.753 | no pasa |
| `k040_r000` | 0.005825 | [0.005757, 0.005893] | 0.838 | no pasa |
| `k040_r002` | 0.005852 | [0.005835, 0.005870] | 0.801 | no pasa |
| `k040_r005` | 0.005854 | [0.005819, 0.005888] | 0.736 | no pasa |
| `k040_r010` | 0.005834 | [0.005826, 0.005841] | 0.788 | no pasa |

Comparadores directos:

- `v10` crudo PPO: `0.005811`.
- Ruta A R22: `0.005914858`.
- PPO+MLP `v7` fixed-RNG 60k: `0.005920640`.
- Future-credit general `h104`: `0.005779`.

Veredicto: **la versión combinada v2 mejora sobre `v10` crudo, pero no supera el brazo Ruta A R22
ni el PPO+MLP `v7` fixed-RNG.** Por disciplina de parada, no se lanza contrafactual causal: sin
señal de entrenamiento suficiente, no hay razón para gastar cómputo probando prevención.

Actualización final del barrido: se completaron las 16 combinaciones `kappa/rho` previstas
(`k ∈ {0.05, 0.10, 0.20, 0.40}`, `rho ∈ {0, 0.02, 0.05, 0.10}`). La mejor sigue siendo
`k010_r005` (`0.005887`), todavía por debajo de Ruta A R22 (`0.005914858`) y del PPO+MLP `v7`
fixed-RNG (`0.005920640`). Algunas variantes reducen costo (`k005_r010`, `k020_r000`), pero no
retienen suficiente ReT. Se cierra esta familia de `Phi` sin contrafactual causal.

## Arm 6 — diagnóstico con reacción retardada (`surge_inertia`)

Después de cerrar Arm 1, Arm 2 y la combinación PBRS v2, se abrió el diagnóstico de entorno que
quedaba en el plan: hacer que la reacción tardía sea menos instantánea. Track B ya tenía soporte
de `surge_inertia`; se conectó a los runners de entrenamiento/auditoría y se lanzó Ruta A R22 con:

- `observation_version=v10`
- `action_contract=track_b_v1`
- `risk_level=adaptive_benchmark_v2`
- `reward_mode=control_v1`
- `max_steps=104`
- seeds `1..3`
- `train_timesteps=30000`
- `eval_episodes=8`
- `surge_inertia=true`
- `surge_ramp_per_step=1`
- `surge_budget_hours=4032`

Artefacto:
`outputs/experiments/track_b_surge_inertia_r22_ruta_a_3seed_30k_2026-07-04/`.

Este brazo no pretende ser headline. Es una prueba diagnóstica: si retrasar la reacción hace que
aparezca señal preventiva, entonces el entorno original permitía recuperar demasiado rápido y no
premiaba anticipar. Si no aparece, la siguiente explicación plausible ya no es "falta presión para
prevenir", sino crédito temporal/política/recompensa más estructural.

### Resultado

| Métrica | Valor |
|---|---:|
| ReT Excel (`order_ret_excel_mean`) | **0.005845** |
| CI95 | [0.005736, 0.005953] |
| Costo | **0.410** |
| Por semilla | 0.005886 / 0.005735 / 0.005914 |

Comparadores:

- `v10` crudo PPO: `0.005810698`.
- Ruta A R22 sin inercia: `0.005914858`.
- PPO+MLP `v7` fixed-RNG 60k: `0.005920640`.
- Future-credit general `h104`: `0.005779012`.

Lectura: el agente sí aprende bajo reacción retardada y mantiene una mejora clara sobre las
estáticas del mismo entorno, con un costo mucho menor. Pero no iguala Ruta A R22 ni el spine
`v7`; cae alrededor de `-0.000070` frente a Ruta A R22. Por la regla preregistrada, esto **no
pasa el filtro de entrenamiento suficiente** para lanzar `R_full - R_reset(pre-R22)`.

Veredicto: Arm 6 no demuestra que "hacer lenta la reacción" desbloquee prevención. El resultado
sí muestra un trade-off interesante de bajo costo, pero no el comportamiento preventivo que
buscábamos. Se cierra sin contrafactual causal.

## Follow-up: recompensa tail/recovery (`ReT_tail_v2`)

Después del gate oráculo revisado, la hipótesis cambió: quizá no faltaba arquitectura ni forecast,
sino una recompensa que premie explícitamente la resiliencia de cola/recuperación que ReT Excel
medio diluye.

Se probó un gate corto con PPO+MLP, `v7`, `track_b_v1`, `adaptive_benchmark_v2`, seeds `1..3`,
`30k` pasos y `8` episodios, comparando `control_v1` contra tres rewards:

- `ReT_tail_v2`
- `ReT_excel_plus_cvar` (`alpha=0.2`)
- `control_v2`

Veredicto completo:
`docs/TRACK_B_TAIL_RECOVERY_REWARD_SCREEN_2026-07-04.md`.

Resultado clave: `ReT_tail_v2` sí pasó el gate corto. Frente a `control_v1` de la misma escala:

- ReT Excel: `0.005843 -> 0.005886`.
- CVaR05: `0.001912 -> 0.002244`.
- Peor ventana 4w: `0.002210 -> 0.002403`.
- Service-loss AUC/order: `166614 -> 120931`.
- TTR mean: `96.83h -> 93.23h`.
- TTR p95: `175.74h -> 159.33h`.
- Costo: `0.734 -> 0.646`.

`ReT_excel_plus_cvar` y `control_v2` no pasaron: degradan ReT Excel y las métricas de
cola/recuperación en este gate.

Lectura: `ReT_tail_v2` no prueba prevención causal todavía, pero sí es la primera señal limpia de
que el agente puede comprar resiliencia de cola/recuperación bajo un reward alineado con esa
pregunta. Por eso se lanzó confirmatoria 5 seeds × 60k:

`outputs/experiments/track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04/`

Sesión activa:
`tmux track_b_ret_tail_v2_confirm_5seed_60k`.
