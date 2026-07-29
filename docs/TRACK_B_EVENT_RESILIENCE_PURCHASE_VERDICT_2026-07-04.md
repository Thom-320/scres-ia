# Métrica de "resiliencia comprada" — Track B — 2026-07-04

## Pregunta

Garrido mira principalmente resiliencia. La pregunta no es sólo si una política usa más recursos,
sino si esos recursos **compran resiliencia** alrededor de los riesgos reales. Para responderlo,
agregué una auditoría secundaria:

```text
Resiliencia comprada =
  mejora local alrededor del riesgo / aumento de postura pre-riesgo
```

La métrica primaria del paper sigue siendo `order_ret_excel_mean` (ReT Excel de Garrido) a nivel
episodio. Esta auditoría no reemplaza esa métrica: la complementa para explicar *qué se compra*
cuando la política sube shift/dispatch.

## Implementación

Script:

```text
scripts/audit_track_b_event_resilience_purchase.py
```

Artefactos:

```text
outputs/experiments/track_b_event_resilience_purchase_vs_ppo_2026-07-04/
outputs/experiments/track_b_event_resilience_purchase_vs_heur_disruption_2026-07-04/
```

Usa `step_ledger_full.csv` y `risk_event_ledger.csv` fixed-RNG, con eventos reales R22/R24. Ventana
por evento: `t=-4..+8` semanas respecto al inicio real del riesgo.

Como el ledger semanal no contiene valores ReT Excel por orden dentro de cada ventana, la compra
local de resiliencia se mide con proxies operacionales alineados con ReT:

- continuidad local de servicio post-evento: `1 - new_backorder / new_demand`;
- backorder nuevo evitado;
- AUC de backlog pendiente reducido sobre el nivel pre-evento;
- semanas de recuperación reducidas hasta volver al backlog pre-evento;
- todo pareado por el mismo `(seed, episode, eval_seed, risk_id, event_index)`.

## Lectura 1 — ¿KAN compra más resiliencia que PPO?

Baseline: `ppo_mlp`.

| Política | Riesgo | Δ intensidad pre-riesgo | Δ continuidad local | Backorder evitado | AUC backlog reducido | Tasa positiva |
|---|---|---:|---:|---:|---:|---:|
| Real-KAN | R22 | +0.642 | +0.005839 | +837 | +145 | 6.0% |
| Real-KAN | R24 | +0.626 | +0.002392 | +339 | +231 | 4.9% |

Lectura: Real-KAN compra una mejora local pequeña sobre PPO, pero pagando muchísima más postura
pre-riesgo. La tasa de eventos donde esa compra es positiva es baja. Esto coincide con el resultado
global: Real-KAN puede subir un poco ReT, pero lo hace casi al techo de recurso, no como prevención
eficiente.

## Lectura 2 — ¿PPO compra resiliencia frente a una regla barata?

Baseline: `heur_disruption_aware`.

| Política | Riesgo | Δ intensidad pre-riesgo | Δ continuidad local | Backorder evitado | AUC backlog reducido | Tasa positiva |
|---|---|---:|---:|---:|---:|---:|
| PPO+MLP | R22 | +0.171 | +0.285124 | +41,130 | +13,867 | 74.0% |
| PPO+MLP | R24 | +0.165 | +0.296248 | +42,669 | +11,205 | 72.9% |
| Real-KAN | R22 | +0.812 | +0.290964 | +41,968 | +14,012 | 96.2% |
| Real-KAN | R24 | +0.791 | +0.298640 | +43,009 | +11,437 | 93.6% |

Lectura: frente a una regla barata, PPO+MLP sí compra resiliencia local de forma clara: mejora
continuidad, evita backorders y reduce backlog alrededor de R22/R24 con una subida moderada de
postura. Real-KAN compra casi lo mismo localmente, pero gastando mucha más intensidad.

## Qué nos dice esto

Esta auditoría separa dos cosas:

1. **Sí hay compra local de resiliencia frente a heurísticas baratas.** PPO+MLP compra servicio y
   recuperación alrededor de R22/R24. Esto explica por qué el spine aprende algo real y supera
   estáticas/heurísticas.
2. **No hay techo preventivo adicional relevante por encima del PPO canónico.** Real-KAN y el
   oráculo de futuro perfecto sólo añaden mejoras pequeñas con mucho costo. Eso confirma que el
   margen de prevención bajo `control_v1` es muy bajo.

## Cómo usarlo en el paper/reunión

Frase recomendada:

> El agente sí compra resiliencia: frente a reglas baratas, mejora la continuidad local y reduce
> backorders alrededor de R22/R24. Pero esa compra es principalmente adaptativa/reactiva; incluso
> con conocimiento perfecto del futuro, la ganancia adicional por prepararse antes es prácticamente
> nula bajo la métrica ReT Excel actual.

Esto permite decir algo honesto y útil para Garrido:

- **sí aprende resiliencia operacional**;
- **sí compra resiliencia frente a reglas simples**;
- **no encontramos evidencia de prevención adicional con valor material en ReT Excel**;
- para estudiar prevención pura habría que cambiar la economía del entorno o la métrica, no sólo
  la arquitectura.
