# Promoción no ratificada del contrato v2 — registro y reversión

**Fecha del hallazgo:** 2026-07-31 · **Commit que la introdujo:** `4111cbc` (2026-07-17,
«Implement Program Q architecture lab and replication gates»).

## Qué pasó

`docs/RET_EXCEL_REQUEST_SNAPSHOT_V2_CONTRACT_2026-07-14.md` se editó **in situ** tres días
después de su congelación. El cambio no fue redaccional:

| | texto congelado (2026-07-14) | texto del 2026-07-17 |
|---|---|---|
| **Status** | «implemented **provisionally**; same-time Garrido confirmation and v2 re-score **required before any Paper-2 claim**» | «implemented as the **frozen researcher-defined primary**; Garrido confirmation controls source-faithfulness language» |
| convención same-time | «Garrido/Simulink confirmation of that convention **is required** before virgin confirmation» | la alternativa `snapshot_before_events` queda disponible para resensibilización explícita |

Es decir: el documento **se promovió de provisional a primario congelado**.

## Por qué se revierte

La autoridad verificable por máquina no es el documento, sino
`research/paper2_exhaustive_search/metric_governance_audit.json`, y **sigue diciendo lo
contrario**. `scripts/verify_paper2_exhaustion.py` exige literalmente:

```
status == "CANONICAL_RET_EXCEL_REQUEST_SNAPSHOT_V2__PROVISIONAL__RESCORE_REQUIRED"
implementation_status == "..._PROVISIONAL_PENDING_SAME_TIME_GARRIDO_CONFIRMATION"
same_timestamp_authority == "PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION"
```

y esas tres aserciones **pasan hoy**. La promoción entró en la prosa y **nunca se llevó al
artefacto de gobernanza**: el documento y la atestación llevan dos semanas contradiciéndose, y
la que manda es la que ninguna de las dos partes tocó.

Además, la promoción depende de un hecho externo que **no tenemos**: la confirmación de Garrido
sobre la convención de simultaneidad. Declarar «frozen researcher-defined primary» no la
sustituye; a lo sumo la renombra.

**Acción tomada:** el documento vuelve byte a byte a su versión de `ff5e4a8`. Con eso, el pin
de `reproducibility_manifest.json` vuelve a coincidir sin re-atestar nada.

## Lo que sí es cierto del cambio del 07-17, y se conserva

El código **sí** ganó dos opciones reales en `compute_order_level_ret_excel_request_snapshot_ledger`:

* `same_time_precedence` — `events_before_snapshot` (el valor por defecto y el congelado) o
  `snapshot_before_events`, esta última **solo** para resensibilización explícita;
* `force_reconstruct` — recalcular `Bt`/`Ut` ignorando los campos nativos capturados.

Y la frontera causal del conteo pasó de la clave de fila `(j, OPTj)` al **tiempo de solicitud**,
para dejar de asumir que `j` es cronológico.

**Medido, eso no cambia ningún número.** `results/metric_audit/v2_metric_freeze_equivalence/`
compara la implementación congelada y la actual sobre la **misma** población, en dos estratos:
tal como se embarca (ruta `captured_at_request`) y con los campos nativos borrados para forzar
la rama de reconstrucción — **3.289 filas en cada estrato, 0 diferencias, max|Δ| = 0**.

Así que el cambio del 07-17 es una **reformulación robusta**, no una redefinición del endpoint.
Lo que no era admisible era **promover el estatus del contrato** por el mismo camino y sin
tocar la gobernanza.

## Qué haría falta para ratificarla

1. La confirmación de Garrido/Simulink sobre la convención de simultaneidad, **o** una decisión
   explícita del PI de declararla convención del investigador y **decirlo en el paper**.
2. Actualizar `metric_governance_audit.json` (los tres estados) y su cadena de hashes.
3. Una enmienda fechada y sellada aparte, no una edición del documento del 14.
