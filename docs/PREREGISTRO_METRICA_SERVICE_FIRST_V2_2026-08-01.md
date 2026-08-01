# Preregistro — `service_first_resilience_v2` como endpoint, sellado sobre la contención

**Escrito y commiteado ANTES de correr.** Runner: `scripts/run_contention_service_first_v2.py`.
Convierte `v2` de **prospectivo** en **endpoint contratado y usado**, que es lo que le faltaba
según `docs/AUDITORIA_SERVICE_FIRST_METRIC_2026-08-01.md` §5.

## La métrica, congelada aquí

    v2 = ( worst_claimant_fill , flow_fill_rate , −backorder_qty_final , ReT acotado )

orden **lexicográfico**, mayor es mejor. `worst_claimant_fill = min_c (entregado_c / demandado_c)`;
con un solo reclamante degenera a `flow_fill_rate`, porque **abandonar a un reclamante no está
definido cuando sólo hay uno**.

**Por qué el componente líder es el peor reclamante y no la cantidad no servida.** La segunda es
exactamente `1 − flow_fill_rate` y colapsaría los dos primeros componentes en uno. Lo que
distingue **abandono** de simplemente *poco fill* es **dónde cae el déficit**: el abandono lo
concentra. Queda escrito el candidato descartado para que nadie lo reintente.

**Y por qué NO es `lost_orders`, que era el líder de `v1`:** `.lost = True` se asigna en **un
único sitio** de todo el simulador, dentro del manejador de desbordamiento de la cola de
`BACKORDER_QUEUE_CAP = 60`. Es un proxy del **desbordamiento**, no del abandono — verificado
estructuralmente en `results/metric_audit/service_first_v2/result.json` (sello `0e37fe2f…`).

## El experimento: la contención, re-medida bajo el endpoint sano

Mismo barrido que la Fase 1A —9 repartos × 6 regímenes, capacidad **no fungible**— que es donde
se midió que `ret_excel` prefiere el reparto que entrega el **50 %** de las raciones sobre el que
entrega el **80 %**. Re-medirlo con `v2` es la prueba directa de que el endpoint corrige eso, y de
si queda **headroom** una vez corregido.

* **regímenes (6)**: `{R2r, R1r+R2r}` × `{base, freq ×3, freq ×3 e impacto ×2 en R23}`
* **repartos (9)**: `0,1 … 0,9`
* **semillas**: `6 400 001…` vírgenes, verificadas contra los 314 valores que declaran los
  artefactos sellados; CRN entre celdas
* **reglas de servicio**: `FIFO_PARTIAL` (la que expone el reparto de forma continua)

## Los dos estimandos, y la limitación que los separa

**Una clave lexicográfica no admite media.** No se puede promediar una tupla ordenada, así que
`H_regime = mean_r[max_a] − max_a[mean_r]` **no está definida sobre `v2`**. Lo digo antes de
correr en vez de inventar una agregación:

1. **`argmax` por régimen bajo la clave lexicográfica completa** — bien definido, y es la pregunta
   de política: *¿el mejor reparto cambia con el régimen?*
2. **`H_regime` sobre `worst_claimant_fill`**, el componente **líder**, que sí es escalar y es el
   que decide en la práctica. Con LCB95 por bootstrap agrupado por celda.

## Falsadores

| falsador | por qué puede fallar |
|---|---|
| `f1_v2_and_ret_disagree` | si `v2` eligiera el mismo reparto que `ret_excel`, el endpoint no estaría corrigiendo nada y todo esto sobra |
| `f2_leading_component_binds` | si `worst_claimant_fill` fuese constante entre repartos, `v2` degeneraría a sus componentes inferiores y el líder sería decorativo |
| `f3_claimant_partition_exists` | sin dos reclamantes, `worst_claimant_fill` **es** `flow_fill_rate` y el experimento no prueba lo que dice |
| `f4_H_regime_is_non_negative` | `mean[max] ≥ max[mean]` por construcción; un negativo sería bug de agregación |
| `f5_seeds_are_virgin` | escaneo real de todos los artefactos sellados, no `passed: True` |
| `f6_lexicographic_key_is_not_averaged` | promediar la tupla sería inventar un tipo de cambio entre componentes, que es justo lo que el orden lexicográfico existe para evitar |

## Regla de lectura, fijada de antemano

* **El `argmax` de `v2` se mueve entre regímenes Y `H_regime` sobre el líder ≥ 0,01 con
  `LCB95 > 0`** → `SERVICE_FIRST_HEADROOM_FOUND`. Sería el **primer** headroom del proyecto bajo
  un endpoint que no premia el abandono, y autorizaría la Fase 3 sobre esta palanca.
* **El `argmax` NO se mueve** → el negativo central de la campaña **se extiende al endpoint
  sano**, que es el resultado más fuerte disponible: ya no se puede objetar «midieron con una
  métrica rota», porque esta métrica está construida precisamente para no serlo.
* **El `argmax` se mueve pero `H_regime < 0,01`** → hay dependencia del régimen sin magnitud
  aprovechable; se reporta como tal y no se titula como headroom.

**Lo que este preregistro NO autoriza:** entrenar nada. `v2` es un **endpoint normativo
estipulado** —una decisión de dominio, no un hallazgo— y no puede usarse como evidencia de que
abandonar sea malo. Eso ya estaba escrito en la auditoría y se mantiene.
