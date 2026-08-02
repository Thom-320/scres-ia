# G3c — cierre de los bloqueadores 1 y 3

Esta enmienda no abre semillas ni adjudica resultados. Reemplaza prospectivamente los
bloqueadores 1 y 3 de `ENMIENDA_G3C_TRES_BLOQUEADORES_2026-08-01.md`.

## Bloqueador 1

El contrato anterior mezclaba `min_dwell` y `switch_cost`. G3c v2 conserva únicamente
`min_dwell_days` con niveles `{1, 3, 7}`. `1` es el nulo legacy; `3` y `7` son tratamientos.
`switch_cost_rations` queda fuera y requiere un contrato separado.

Fuente: [PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_V2_2026-08-02.md](PREREGISTRO_G3C_ACOPLAMIENTO_TEMPORAL_V2_2026-08-02.md).

## Bloqueador 3

La identidad del nulo deja de ser una afirmación textual. Se implementa en
`supply_chain/scientific_payload.py` mediante `canonical_scientific_payload` y
`scientific_payload_sha256`. El payload incluye órdenes, eventos, acciones, ledgers y métricas;
el envelope excluye timestamps y provenance.

El test `test_null_arm_is_identical_to_the_shipped_defaults` compara el modelo legacy con el
nulo explícito en el mismo tape. Un test mutacional verifica que cambiar una métrica cambia el
hash. El hash del envelope (`self_sha256`) permanece separado y no se usa para `f1`.

## Estado resultante

```text
BLOCKER_1: RESOLVED
BLOCKER_2: RESOLVED BY ENMIENDA_G3C_MARGENES_OPERACIONALES_2026-08-02
BLOCKER_3: RESOLVED
G3c: DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT
```
