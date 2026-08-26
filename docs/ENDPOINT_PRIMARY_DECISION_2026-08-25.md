# ENDPOINT_PRIMARY_DECISION — 2026-08-25

**Decisión del PI (2026-08-25), adoptando el dictamen de ChatGPT Pro:**

## Endpoint primario del manuscrito

**`ret_excel_request_snapshot_v2`**, congelado. Es el estimand primario congelado
de ESTE paper porque los niveles 1–4 completos fueron calculados bajo él.

Esta decisión NO declara que snapshot sea «la verdadera resiliencia». Declara que
es el primario preregistrado bajo el cual se ejecutó el programa cerrado.
Cambiar ahora a full ledger obligaría a re-puntuar el programa con un endpoint
nuevo post-hoc: exactamente la clase de margin-shopping que la gobernanza prohíbe.

## Sensibilidad declarada

`ret_excel_full_ledger`, `ret_excel_clipped_0_1` y demás miembros admisibles de
la familia se publican como **sensibilidad**, incluyendo la tabla completa de
inversión de signo endpoint×bloque:

| Bloque | snapshot/clipped | full ledger |
|---|---:|---:|
| A | +0.01247 (IC excluye 0) | −0.00448 (IC excluye 0) |
| B | −0.01217 (IC excluye 0) | +0.00842 (IC excluye 0) |

Mensaje metodológico frontal: **la escalarización no solo puede ocultar la
sustitución media↔mínimo; distintas procedencias admisibles del ledger pueden
invertir el veredicto sobre la misma pareja de políticas.**

## Contrato futuro

`ret_excel_full_ledger` queda **candidato prospectivo** a primario en un contrato
nuevo, con su propio preregistro y semillas vírgenes. Eso es otra cosa distinta y
no altera este manuscrito.

## Correcciones narrativas simultáneas (obligatorias)

1. Methods: N=256, potencia conjunta 0.8755, max-t studentizado bidireccional,
   10.000 remuestras — no N=128/0.806.
2. Brazo primario learner: **RecurrentPPO**.
3. belief-MPC: miembro de la familia clásica; ganadores por celda:
   `min_cost_flow__2` / `min_cost_flow__2` / `max_pressure__0`.
4. Una sola bibliografía canónica (ver `papers/cie_submission/references_checklist.md`).
