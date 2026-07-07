# Track B fixed-RNG confirm verdict — 2026-07-03

## Estado

Las corridas confirmatorias de **PPO+MLP** y **Real-KAN** para Track B 8D con fixed RNG terminaron correctamente en el VPS.

Artefactos:

`outputs/experiments/track_b_fixed_rng_confirm_5seed_60k_2026-07-03/`

`outputs/experiments/track_b_real_kan_fixed_rng_confirm_5seed_60k_2026-07-03/`

Protocolo verificado:

- Seeds: 1-5
- Entrenamiento: 60,000 timesteps por seed
- Evaluación: 12 episodios por seed
- Horizonte: 104 pasos semanales
- Observación: `v7`
- Acción: `track_b_v1` / 8D
- Reward de entrenamiento: `control_v1`
- Riesgo: `adaptive_benchmark_v2`
- Código VPS: `strict_exogenous_crn=True` activo en Track B y `regime_rng` separado para el controlador adaptativo.

## Resultado principal PPO+MLP

La métrica principal aquí es **`order_ret_excel_mean`**, la fórmula ReT del Excel de Garrido aplicada a nivel de órdenes.

| Política | ReT Excel | CI95 | Costo |
|---|---:|---:|---:|
| PPO+MLP fixed-RNG | 0.00592064 | [0.00590559, 0.00593569] | 0.700 |
| Mejor estática del bundle (`s3_d1.50`) | 0.00542514 | [0.00541351, 0.00543676] | 1.000 |
| Mejor no-PPO del bundle (`heur_disruption_aware`) | 0.00543225 | [0.00541912, 0.00544538] | 0.454 |

Comparación:

- PPO+MLP vs mejor estática del bundle: **+0.00049550 ReT Excel** (**+9.13%**).
- PPO+MLP vs mejor no-PPO del bundle: **+0.00048839 ReT Excel** (**+8.99%**).
- PPO+MLP fixed-RNG vs ancla PPO+MLP canónica previa aproximada (0.0058977): **+0.00002294** (**+0.39%**).

## Lectura

El fix de RNG **no rompe el resultado de Track B**. Al contrario, PPO+MLP mantiene una ventaja clara sobre las políticas estáticas/heurísticas evaluadas en el mismo bundle y queda levemente por encima del ancla canónica anterior.

Este resultado todavía **no reemplaza** por sí solo la comparación contra la frontera densa de 147 celdas, porque este bundle usa el comparador smoke/ligero. La lectura correcta es:

> Track B sobrevive el fix de RNG bajo el protocolo confirmatorio PPO+MLP 5 seeds x 60k; el siguiente nivel de rigor, si se decide, es reconstruir la frontera densa bajo fixed-RNG.

## Resultado Real-KAN

Real-KAN también terminó bajo el mismo protocolo y debe compararse con la misma métrica **`order_ret_excel_mean`**, no con `order_level_ret_mean`.

| Política | ReT Excel | CI95 | Costo |
|---|---:|---:|---:|
| Real-KAN fixed-RNG | 0.00594620 | [0.00594103, 0.00595137] | 1.000 |
| PPO+MLP fixed-RNG | 0.00592064 | [0.00590559, 0.00593569] | 0.700 |
| Mejor estática del bundle (`s3_d1.50`) | 0.00542514 | [0.00541351, 0.00543676] | 1.000 |

Comparación:

- Real-KAN vs PPO+MLP fixed-RNG: **+0.00002556 ReT Excel** (**+0.43%**), positivo en **5/5 seeds**.
- Real-KAN vs mejor estática del bundle: **+0.00052106 ReT Excel** (**+9.60%**).
- PPO+MLP vs mejor estática del bundle: **+0.00049550 ReT Excel** (**+9.13%**).

Per-seed Real-KAN minus PPO+MLP deltas:

| Seed | Real-KAN | PPO+MLP | Delta |
|---:|---:|---:|---:|
| 1 | 0.00594504 | 0.00590254 | +0.00004250 |
| 2 | 0.00594702 | 0.00590989 | +0.00003713 |
| 3 | 0.00595563 | 0.00594749 | +0.00000814 |
| 4 | 0.00593981 | 0.00591905 | +0.00002076 |
| 5 | 0.00594350 | 0.00592423 | +0.00001927 |

## Interpretación conjunta

El resultado queda muy limpio:

1. **PPO+MLP sigue siendo una alternativa fuerte y eficiente**: gana claramente en ReT Excel con costo medio (`0.700`) y sobrevive el fix de RNG.
2. **Real-KAN gana marginalmente en resiliencia pura**: supera a PPO+MLP en 5/5 seeds bajo fixed-RNG.
3. **El precio de Real-KAN es operacional**: llega a costo `1.000`, es decir, usa el techo de recursos/turnos del contrato.

Para el paper, la lectura defendible no es que PPO+MLP haya sido desplazado automáticamente. La lectura es:

> PPO+MLP es el spine eficiente y ya probado; Real-KAN es el sidecar arquitectónico más fuerte, con una ganancia pequeña pero consistente en ReT Excel a costa de operar al máximo recurso.
