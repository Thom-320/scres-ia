# Track B no-forecast fixed-RNG final verdict — 2026-07-05

## Veredicto

La versión **sin forecast explícito** queda confirmada como una columna vertebral
muy defendible para el paper: bajo el protocolo fixed-RNG final, quitar los dos
campos de forecast (`risk_forecast_48h_norm`, `risk_forecast_168h_norm`) no
produce una pérdida medible de la resiliencia principal ReT Excel.

La lectura correcta no es "no-forecast gana"; la lectura correcta es más útil
para reviewers:

> La ventaja de Track B no depende de darle al agente un forecast privilegiado.
> El modelo mantiene esencialmente la misma ReT Excel sin esos campos.

## Artefactos

No-forecast fixed-RNG, 15 seeds:

- `outputs/experiments/track_b_no_forecast_fixed_rng_1_5_60k_2026-07-05/v7_no_forecast/`
- `outputs/experiments/track_b_no_forecast_fixed_rng_6_10_60k_2026-07-05/v7_no_forecast/`
- `outputs/experiments/track_b_no_forecast_fixed_rng_11_15_60k_2026-07-05/v7_no_forecast/`

Full-v7 fixed-RNG comparator, 15 seeds:

- `outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/`
- `outputs/experiments/track_b_control_v1_fixed_rng_extension_6_10_60k_2026-07-05/`
- `outputs/experiments/track_b_control_v1_fixed_rng_extension_11_15_60k_2026-07-05/`

Merged comparison:

- `outputs/experiments/track_b_no_forecast_fixed_rng_final_15seed_2026-07-05/`

Helper:

- `scripts/summarize_track_b_no_forecast_final.py`

## Protocolo verificado

Para los tres bloques no-forecast:

- Observation base: `v7`
- Wrapper: `ForecastMaskWrapper`, que pone en cero solo:
  - `risk_forecast_48h_norm`
  - `risk_forecast_168h_norm`
- Action contract: `track_b_v1`
- Reward: `control_v1`
- Risk: `adaptive_benchmark_v2`
- Seeds: `1..15`, en bloques `1..5`, `6..10`, `11..15`
- Train timesteps: `60000`
- Eval episodes: `12`
- Horizon: `max_steps=104`
- PPO: `n_steps=1024`, `batch_size=64`
- Código Track B actual: `strict_exogenous_crn=True` en `MFSCSimulation`

Nota importante: esta corrida quita **forecast**, no el one-hot de régimen. La
pregunta "¿qué pasa si quitamos régimen + forecast?" se cubre por la claim C16
y por la ablation `v7_no_regime_forecast`; esta corrida responde
específicamente la crítica de forecast privilegiado / lectura adelantada del
simulador.

## Resultados agregados

| Variante | Seeds | ReT Excel mean | CI95 | Costo mean | CI95 costo |
|---|---:|---:|---:|---:|---:|
| Full-v7 | 15 | 0.00590177 | [0.00588871, 0.00591483] | 0.66173 | [0.61415, 0.70932] |
| No-forecast | 15 | 0.00590391 | [0.00588441, 0.00592341] | 0.68497 | [0.64330, 0.72664] |

Comparación pareada no-forecast menos full-v7:

| Métrica | Delta medio | CI95 | Pares favorables |
|---|---:|---:|---:|
| ReT Excel | +0.00000214 | [-0.00001255, +0.00001683] | 9/15 |
| ReT Excel CVaR05 | +0.00001499 | [-0.00006391, +0.00009389] | 8/15 |
| Worst 4w ReT | +0.00000602 | [-0.00007244, +0.00008448] | 9/15 |
| Service-loss AUC/order | -136.21 | [-12864.04, +12591.63] | 8/15 |
| TTR mean | +0.011h | [-0.897, +0.920] | 9/15 |
| TTR p95 | -0.630h | [-5.841, +4.581] | 11/15 |
| Cost index | +0.02324 | [-0.03248, +0.07895] | 5/15 |

## Lectura

El ReT Excel primario queda estadísticamente indistinguible entre full-v7 y
no-forecast. El delta medio es incluso ligeramente favorable a no-forecast, pero
el CI cruza cero; por rigor, no se debe presentar como una mejora.

Esto sí es suficiente para una decisión de encuadre:

- Usar **no-forecast** como versión principal es más defendible ante reviewers.
- Mantiene la misma resiliencia Excel dentro del ruido estadístico.
- Evita la objeción más incómoda: que el agente "lee el futuro" mediante campos
  de forecast generados por el simulador.
- La versión full-v7 puede quedar como ablation/sonda de prevención, no como
  columna vertebral.

## Implicación para Garrido

Mensaje sugerido:

> "Probamos la versión sin forecast explícito. El resultado principal de
> resiliencia se mantiene: no hay pérdida medible en ReT Excel a 15 seeds. Por
> eso, para el paper, podemos presentar una versión más conservadora: el agente
> aprende usando el estado operacional de la cadena, no una predicción
> privilegiada del simulador."

## Implicación para el paper

Recomendación:

1. Mantener la métrica primaria como ReT Excel de Garrido.
2. Reportar el resultado principal como mejora vs. la grilla estática densa, no
   como comparación contra la resiliencia absoluta de la tesis.
3. Usar no-forecast como el spine si el manuscrito necesita la versión más
   reviewer-safe.
4. Mencionar full-v7 y `v7_no_regime_forecast` como auditorías de observación,
   no como el resultado principal.

