# Result — the overlap subtraction: Algorithm 1 implemented, `APj > CTj` eliminated

**Status:** `CORRECTION_APPLIED_DEFAULT_UNCHANGED`. Artifact
`results/metric_audit/delay_physical_arms_v1/result.json` (re-run, resealed).

## 1. The defect

Algorithm 1 (thesis p.68), line 3:

> `APj = ΣRcr – Σ(R1r ∩ … ∩ Rc4)`

That is the **measure of the union** of the risk-impact intervals. The shipped code
accumulated `total_disruption_hours += …` at **six sites** and never subtracted anything, so
two risks down at the same time were billed twice.

`min(total_disruption_hours, LTj)` was **masking** it. With the cap removed, `1,716` orders
in the F′ arm came out with `APj > CTj` — an autonomy period longer than the cycle that
contains it (`docs/RESULTADO_DELAY_FISICO_2026-07-31.md`, falsifier 4).

## 2. The fix

Every impact interval is now collected at all six sites, clipped to `[OPTj, OATj]`, and the
**union measure** replaces the sum. `apj_overlap_mode="sum"` retains the old arithmetic for
reproducing runs frozen under it.

| | violaciones `APj > CTj` |
|---|---:|
| `sum` (statu quo) | **145 / 597** |
| `union` (Algoritmo 1) | **0 / 597** |

Falsifier 4 now **PASSES** in the factorial, on 12 roots and both families.

## 3. Scope, measured rather than asserted

| lane | efecto |
|---|---|
| **default embarcado (`elapsed`)** | **BIT-IDÉNTICO** — 6 comparaciones, 0 discrepancias |
| carril `disruption` (no default) | **cambia**, y debe: es el único que consume `total_disruption_hours` |
| `disruption` + `apj_overlap_mode="sum"` | **reproduce lo congelado** — 4 comparaciones, 0 discrepancias |

`total_disruption_hours` alimenta solo `APj` y la rama `disruption` de `RPj`. Bajo los
defaults embarcados la autotomía es inalcanzable (`CTj = 54 > LT = 48`) y el modo es
`elapsed`, así que la unión no toca nada. **Ninguna cifra congelada se mueve**, y toda
corrida congelada bajo `disruption` reproduce fijando `apj_overlap_mode="sum"`.

El interruptor está en `calibration_stamp()` y **gatea la comparabilidad**, junto con
`autotomy_apj_cap` y `fulfillment_transit_mode`.

## 4. Lo que sigue abierto

La corrida del factorial **sigue detenida en f3**: la cadencia de flete no produce la
distribución de `CTj` (36 valores distintos por corrida contra 500 exigidos, 60,7% modal).
Eso es independiente de esto y ya está diagnosticado — la finalización de la orden ya está
sincronizada con las olas. La dispersión de Garrido viene de otro sitio.

**No adopto el brazo sin tope.** Con la unión el tope `min(total, LTj)` es **redundante por
construcción** —la unión clipada a `[OPTj, OATj]` no puede exceder `CTj`— pero decidir cuál
de los dos queda como default mueve `APj` en cuanto la autotomía sea alcanzable, y eso
pertenece al contrato del delay, no a esta corrección.
