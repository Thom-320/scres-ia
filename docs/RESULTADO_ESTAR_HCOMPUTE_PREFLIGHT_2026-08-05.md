# Resultado engineering — E* bridge y H_compute burned-only

**Fecha:** 2026-08-05
**Contrato:** `contracts/garrido_expanded_des_e_star_v2_hcompute.json`
**Estado científico:** `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`

## Alcance

Este recibo cubre únicamente ingeniería y medición de coste sobre el fixture
burned `R1r_actual_tapes.json`. No abrió semillas nuevas, no seleccionó una
arquitectura, no entrenó un learner y no constituye confirmación científica.

Program Q, Program O y `thesis_1to1` permanecen inmutables.

## Bridge source-conserving

El adaptador `EStarDESAdapter` ejecuta el DES histórico `MFSCSimulation`.
`M000` se compara contra el golden histórico; las máscaras P/U/D usan
procurement, buffers y dispatch explícitos con conservación, capacidades finitas
y lead times. Un buffer lleno bloquea la llegada; no hay derrame ni inyección
estratégica.

El smoke burned-only recorrió las ocho máscaras y produjo:

```text
claim_status: BRIDGE_SMOKE_PASS
observed_digest: feaef05c0f31e9f82091d063b004b45823694341b7dc6225d4f4341ff37fc206
raw/ration residuals: 0
strategic injection: 0
```

El falsador de mutación que omite el ledger de fuente falla, por lo que el
recibo no depende únicamente de una ejecución que se compare consigo misma.

## H_compute

El backend medido es `DirectDESMPC`, un controlador enumerativo de rollouts
directos del DES. No se presenta como MPC óptimo; su objetivo aquí es hacer
explícito y reproducible el coste de planificación que el gate pretende medir.

La configuración usa 30 mediciones calientes y 5 warm-ups por nivel:

| nivel | llamadas DES del MPC directo | p95 (s) |
|---|---:|---:|
| `S0_M000` | 6 | 0,0043 |
| `S1_P` | 12 | 0,0083 |
| `S2_U` | 6 | 0,0036 |
| `S3_PU` | 12 | 0,0082 |
| `S4_FULL` | 192 | 0,1553 |

El presupuesto firmado de llamadas es `10 × M000 = 60`. El gate de llamadas
pasa (`192 > 60`) y las llamadas aumentan en dos transiciones consecutivas
(`S2 → S3 → S4`). El gate de latencia no pasa en este hardware, porque el
presupuesto de 10% de la cadencia de 168 horas es 60.480 s.

Resultado del preflight:

```text
H_COMPUTE_PASS_NEURAL_AMORTIZATION_ELIGIBLE
```

Esto sólo autoriza preparar, después de la puerta científica, una comparación
de amortización. No autoriza datos frescos ni aprendizaje ahora.

## Artefactos canónicos

- [recibo del bridge](../results/estar_expanded_bridge_smoke_v1/result.json)
- [preflight H_compute](../results/estar_hcompute_preflight_v1/result.json)
- [contrato E* v2](../contracts/garrido_expanded_des_e_star_v2_hcompute.json)

La selección formal del endpoint, SESOI, márgenes de no inferioridad, hardware
de confirmación y recibo de Submission A o supersession explícita siguen siendo
pendientes antes de cualquier bloque virgen o learner.
