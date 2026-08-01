# Resultado Fase 1B — expedición bajo `service_first_resilience_v1`

**Artefacto:** `results/sensitivity/expedite_headroom_v2/result.json`
**Contrato:** `docs/PREREGISTRO_EXPEDICION_HEADROOM_V2_2026-08-01.md`
**Estado:** `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST`
**Lectura:** los nueve falsadores pasan; no se autoriza MLP/PPO.

## Diseño ejecutado

Se evaluaron 32 cintas de riesgo comunes (4 regímenes × 8 semillas), 4 semillas
de calibración, 4 semillas held-out, 5 presupuestos (`0, 168, 336, 672,
1344 h`) y tres brazos por presupuesto: constante calibrada, tape-oracle y
placebo. El resultado contiene 240 episodios de evaluación. El tape-oracle
usa únicamente el score de solapamiento riesgo–pierna congelado en el contrato;
no es un óptimo del outcome ni una política desplegable.

## Gate principal

La barra era `H_PI_fill ≥ 0,01` con `LCB95 > 0`, junto con no empeorar la
compatibilidad de servicio. No se alcanza en ningún presupuesto. En el mayor
presupuesto:

| contraste tape-oracle − constante | media | LCB95 | UCB95 |
|---|---:|---:|---:|
| `flow_fill_rate` | −0,00000085 | −0,00000256 | 0,00000000 |
| `lost_orders` | 0,000 | 0,000 | 0,000 |
| `backorder_qty_final` | 0,000 | 0,000 | 0,000 |
| `ret_excel_risk_conditional` | +0,001559 | +0,000732 | +0,002387 |
| `ret_excel_visible_clipped_0_1` | +0,001850 | +0,000882 | +0,002597 |
| `service_loss_auc_ration_hours` | +383 113 | +254 879 | +538 444 |

La mejora diagnóstica de ReT no convierte la expedición en una mejora de
servicio: el fill rate no sube y el área de pérdida de servicio empeora. Por
eso la métrica service-first no promueve el brazo. El tape-oracle tampoco
supera al placebo en fill rate (`LCB95 = 0` en `B=1344`); el valor medido no
es información temporal suficiente para abrir PPO.

El patrón intermedio es consistente: ReT condicional mejora en `B=672` y
`B=1344`, pero el efecto es subcrítico en servicio; en `B=168` y `B=336`
el contraste ReT contra la constante es negativo. Esto se conserva como
sensibilidad, no como resultado de resiliencia.

## Integridad

Los falsadores `f1`–`f9` pasan: presupuesto conservado, reducción PT de 12 h
real, una reducción por invocación elegible, identidad B=0, cinta CRN común,
placebo con el mismo gasto, calendarios desplegables sin futuro, ausencia de
triunfo por abandono y semillas vírgenes.

## Decisión

Esta Fase 1B **no abre la puerta** a MLP/PPO bajo la métrica que no premia
abandonar. El resultado no demuestra que ninguna expedición imaginable pueda
ser útil; cierra esta interfaz, presupuesto, calendario y efecto físico
preregistrados. El siguiente trabajo autorizado es la Fase 4: rerunear el
meta-aprendiz corregido sobre 90 configuraciones thesis-native y, por separado,
288 configuraciones extendidas, manteniendo la frontera entre evaluación entre
corridas y control intraepisodio.
