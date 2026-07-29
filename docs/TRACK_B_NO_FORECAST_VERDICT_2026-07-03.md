# Track B no-forecast confirmatory verdict

Fecha: 2026-07-03

## Estado

**DONE.** La corrida `v7_no_forecast` termino correctamente.

Artefacto:

```text
outputs/experiments/track_b_e2_no_forecast_confirm_2026-07-03/v7_no_forecast/
```

Archivos verificados:

- `summary.json`
- `policy_summary.csv`
- `seed_metrics.csv`
- `comparison_table.csv`
- `episode_metrics.csv`

El `tmux` `track_b_no_forecast_confirm` ya no existe y no hay proceso activo asociado. Esto es una finalizacion normal,
no una falla.

## Protocolo verificado

| Campo | Valor |
|---|---|
| Observacion base | `v7` |
| Ablacion | `v7_no_forecast` |
| Forecast removido | `risk_forecast_48h_norm`, `risk_forecast_168h_norm` |
| Accion | `track_b_v1` 8D |
| Seeds | 1-5 |
| Training | 60k timesteps/seed |
| Eval | 12 episodios/seed |
| Horizonte | h104 |
| Reward entrenamiento | `control_v1` |
| Riesgo | `adaptive_benchmark_v2` |
| Modelo | PPO+MLP |

## Resultado principal: ReT Excel

Para Garrido, la lectura importante es `order_ret_excel_mean`.

| Politica | ReT Excel | Costo (`assembly_cost_index`) |
|---|---:|---:|
| PPO+MLP canónico v7 completo, 10 seeds | 0.005898 | ~0.665 |
| PPO+MLP `v7_no_forecast`, 5 seeds | 0.005895 | 0.645 |
| PPO+MLP `v7_no_regime_forecast`, 5 seeds | 0.005841 | 0.633 |
| Mejor heuristica del paquete no-forecast (`heur_tuned`) | 0.005436 | 0.592 |
| Mejor estatica pura comparable (`s2_d1.50`) | 0.005428 | 0.667 |

Comparaciones:

- `v7_no_forecast` vs PPO+MLP canonico 10-seed: `-0.0000025` ReT Excel (`-0.04%`), esencialmente empate.
- `v7_no_forecast` vs mejor heuristica del paquete: `+0.000459` (`+8.45%`).
- `v7_no_forecast` vs mejor estatica pura comparable: `+0.000467` (`+8.61%`).
- `v7_no_forecast` usa menos recurso que el canónico aproximado (`0.645` vs `~0.665`).

## Nota sobre métricas internas

El `summary.json` tambien reporta `order_level_ret_mean_mean`, que queda:

- PPO `v7_no_forecast`: `0.005668`
- Mejor estatica pura `s2_d1.50`: `0.005214`
- Delta: `+0.000454`

Ese campo es consistente con el `decision` del runner. Para comunicacion con Garrido, sin embargo, mantenemos como
principal la ReT Excel/order-level reportada en `order_ret_excel_mean`.

## Veredicto

**Forecast no es necesario para el resultado principal de Track B.**

La politica sin forecast explicito conserva practicamente toda la ReT Excel del PPO+MLP canonico y gana con claridad a
las politicas estaticas/heuristicas comparables bajo el mismo protocolo. Por tanto:

1. El headline de Paper 1 puede defenderse sin depender de una senal de forecast potencialmente privilegiada.
2. El forecast debe tratarse como **sonda de prevencion**, no como columna vertebral del resultado.
3. Si la auditoria preventiva futura demuestra accion pre-riesgo y valor positivo en ReT Excel, forecast puede volver
   como una herramienta de alerta temprana.
4. Si no demuestra prevencion, el resultado principal queda igual de defendible usando `v7_no_forecast`.

## Implicacion para el documento de Garrido

Actualizar el texto: ya no corresponde decir "hay que reentrenar una version sin forecast"; esa prueba ya se hizo. La
frase correcta es:

> Reentrenamos Track B sin los dos forecasts explicitos. El modelo conserva practicamente toda la ReT Excel y mantiene
> una mejora clara frente a estaticas. Por eso el forecast no es requisito para el resultado principal; lo dejamos como
> herramienta para estudiar prevencion.
