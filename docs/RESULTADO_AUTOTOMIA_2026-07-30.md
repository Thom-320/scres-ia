# Result — the autotomy arms: not adoptable, and my direction prediction was wrong

**Status:** `PREREGISTERED_NEGATIVE_ARM_C_NOT_ADOPTED`. Executes
`docs/PREREGISTRO_AUTOTOMIA_2026-07-30.md`. Artifact
`results/metric_audit/autotomy_arms_v1/result.json`. Roots 2,500,001–12.

> **CORRECCIÓN 2026-07-31 — dos de los cuatro falsadores no podían fallar.**
>
> **Falsador 3 es aritmética, no verificación.** En B y C el delay es la constante 48,0074, y
> `supply_chain.py:2603` la asigna a toda orden servida desde stock: `min(CTj) = 48,0074`
> idénticamente, y `48,0074 > 48` hace `below_lt == 0` idénticamente. Prueba la definición del
> brazo, no la implementación. Además el contrato pedía «órdenes **no bloqueadas**» y el
> runner (`:127-131`) recorre todas.
>
> **La tolerancia nunca se ejercitó.** Con `CTj ≡ 48,0074`, `CTj − LT ≡ 0,0074 ≤ 0,01`, así
> que los tres brazos C salen bit-idénticos — como muestra la tabla. **La prueba 96/98 que el
> contrato ofrece como única justificación del ajuste declarado jamás se corrió contra nuestra
> salida.**
>
> **La regla de aceptación omite los cuatro falsadores.** `adopt_C` (`:219-221`) no incluye
> ninguno, mientras el contrato §6 los exige todos.
>
> **`suma d_k` es una escalarización que el contrato maestro prohíbe** («do NOT collapse it
> with weights»). Las menciones de §2 y §5 a la suma **no** son evidencia admisible; quedan
> como descriptivo. Ver `contracts/paper_b_v2_amendment_2026-07-31.json`.

## 1. Falsifiers

All four passed: arm A reproduced the frozen block; arm B left `autotomy_share` at exactly
0.000; the floor in B and C is 48.0074 with no order below `LT`; and no order in C carries
`APj > 0` together with `RPj > 0`.

## 2. The result

`d_k` against `fidelity_reference_v3`:

| momento | A | B | C 0.01 | C 0.05 | C 0.10 |
|---|---:|---:|---:|---:|---:|
| **R1r** `autotomy_share` | 11.2 | 11.2 | **78.5** | **78.5** | **78.5** |
| **R1r** `ret_mean` | **1.6** | 3.6 | **78.6** | **78.6** | **78.6** |
| **R1r** suma | **46.0** | 47.7 | 202.4 | 202.4 | 202.4 |
| **R2r** `autotomy_share` | 4.6 | 4.6 | **22.9** | **22.9** | **22.9** |
| **R2r** suma | 17.0 | **14.1** | 48.1 | 48.1 | 48.1 |

Raw `autotomy_share` under C is **0.659356 in R1r** against a reference of
**0.00436282** — **151.1×** too much. **`adopt_C = False`.**

> **CORRECCIÓN 2026-07-31.** Decía «165×», obtenido dividiendo por el `0,004` redondeado de
> la tabla en vez de por el valor de la referencia. El factor es 151,1.

## 3. My prediction was wrong, and in the direction

The preregistration declared, in `d_k`:

* §4.1 — *B leaves `autotomy_share` at exactly 0.000*: **correct**, 0.00000 in both families.
* §4.3 — *`ret_mean` degrades*: **correct**, and catastrophically under C (1.6 → 78.6).
* §4.2 — *C improves `d_k(autotomy_share)` in both families*: **WRONG.** It worsens it,
  11.2 → 78.5. I declared a direction and got it backwards.

Getting §4.1 and §4.3 right does not offset that. The value of declaring the direction is
that being wrong is visible, and it is.

## 4. Why — and this is the real finding

**All three tolerances give bit-identical results.** 0.01, 0.05 and 0.10 produce exactly
0.65936. The tolerance is irrelevant, and that is the diagnosis.

| | Garrido | nuestro (piso 48.0074) |
|---|---|---|
| forma de `CTj` cerca del piso | distribución continua | **masa puntual** |
| valor modal | — | 54,0 (embarcado): **60,5%**; 48,0074 (brazo C): 69,2% |
| p1 / p5 / p25 / p50 | 48.41 / 50.42 / 75.00 / 101.45 | 48.007 / 48.007 / 48.007 / 48.007 |
| filas en [48.007, 48.06] | 98 = **0.45%** | **69.2%** |

His `CTj` is a **continuous distribution starting at 48.0074**; ours is a **point mass at
the fulfilment-delay constant**. Every unblocked order in our model completes in exactly the
same time.

That settles the whole autotomy question:

* with the delay at **54**, the point mass sits above `LT` and autotomy **never** fires;
* with the delay at **48.0074**, the point mass sits inside the band and **69%** fires;
* **no constant produces 0.44%**, because 0.44% requires a distribution with a thin lower
  tail, and a constant has no tail at all.

The defect is not the value of `GARRIDO_FULFILLMENT_DELAY_HOURS`. **It is that it is a
constant.** That is a modelling gap, not a calibration error, and it was invisible from
either arm alone — arm B was included precisely to separate floor from predicate, and it
did its job.

## 5. What arm B did show, and it is small but real

Floor alone, keeping the shipped predicate: R2r's `ret_above_one_share` improves **4.0 →
0.3** and its total `d_k` **17.0 → 14.1**. R1r's `ret_mean` degrades **1.6 → 3.6**, so it is
not adoptable either, but it localizes something: part of R2r's out-of-range tail is a
consequence of the 54 h floor.

## 6. Status of the constant

`GARRIDO_FULFILLMENT_DELAY_HOURS = 54` is now documented as failing on **three** counts:

1. it was fitted in June against **one** observable (ReT magnitude), explicitly labelled
   *«provisional reproduction default, not a complete behavioral calibration»*;
2. it makes the autotomy branch **structurally unreachable** (0 of 416 orders);
3. it is a **constant where the source data is a distribution**, so no value of it can
   reproduce his order-completion behaviour near the lead time.

Point 3 is new and is the one that matters, because it says the repair is not a new number.

## 7. What changed in the code

`autotomy_predicate` and `autotomy_tolerance_hours` added, defaults `"le"` and `0.0` — the
shipped behaviour, and falsifier 1 proves arm A is unchanged. `LEAD_TIME_PROMISE = 48`
untouched. **Nothing relabelled, no constant swept into the default.**

## 8. Where this leaves us

Two open gaps, both now with named causes and neither with a proposed fix I trust:

* **`rpj_p95` / the saturation** — his `RPj` flattens near 400 h for `CTj ≥ 1,000`; four
  mechanisms proposed and refuted today;
* **`autotomy_share`** — needs the fulfilment delay to become a distribution, which is new
  physics and needs its own preregistration, with the shape constrained by his observable
  `CTj` quantiles (48.41 / 50.42 / 75.00 / 101.45 at p1/p5/p25/p50).

The second is now the better-posed of the two: the target is directly measurable from his
data, unlike the saturation, for which I still have no mechanism.

`ret_mean` under the shipped defaults is **1.6 SD (R1r) / 2.0 (R2r)** and untouched by any
of this.
