# Preregistration — repairing the ReT out-of-range tail

**Status:** `PREREGISTRATION_DRAFT_AWAITING_PI_SIGNATURE`. Nothing here is applied. The
canonical metric is unchanged and every frozen result stands as reported.

**This preregistration is written with the sign of the outcome already known**, which is
unusual and is stated up front rather than buried. `results/metric_audit/ret_repair_variants_v1/`
already measured all three candidate repairs; in R2r every one of them reverses the
frozen `NOT_SEPARATED` verdict to `MPC_AHEAD`. That is precisely why the rule has to be
fixed in writing *before* the rerun: choosing among repairs after seeing which produces a
publishable result is the Program G error, and knowing the answer in advance makes the
temptation worse, not better.

## 1. The defect being repaired

`ReT = 0.5/RPj` on the recovery branch (`supply_chain/ret_thesis.py:125`) is unbounded as
`RPj → 0`, and `RPj` accumulates **hours** while `R14`/`R24` are demand shocks measured in
**rations** (`_ret_quantity_risk_units`). An order can therefore enter recovery carrying a
large quantity indicator and a nearly null temporal `RPj`.

Measured consequence, custodied in `results/metric_audit/ret_tail_family_sweep_v1/`:

| family | scored orders | ReT > 1 | max ReT | mean inflation vs clipped |
|---|---:|---:|---:|---:|
| R1r `(0,0,336)` | 3,279 | 0 (0.00%) | 0.0712 | 1.00000x |
| R2r `(336,0,168)` | 3,108 | 7 (0.23%) | 73.9082 | 1.05784x |

And the metric is not monotone in lateness: in both families the highest-scoring order is
among the *least* delayed, while the most-delayed order scores near zero.

**Not claimed:** that R24 causes those delays. The indicator is a recorded surge
magnitude under retrospective event attribution, not an identified causal path.

## 2. The repair, and why this one

**Primary repair: clip each per-order ReT into `[0, 1]`.**

Adopted over my own earlier recommendation of *quantity → time*, on the reviewer's
argument, which I accept:

- **Clipping needs no new model.** Garrido defines R as "a numerical dimensionless index
  ranging between 0 — the lowest level — and 1 — the highest". Clipping enforces a range
  the specification already declares. It introduces no quantity that is not already
  defined.
- **Quantity → time requires a causal model that is not identified.** Crediting `RPj`
  with `CTj − LTj` whenever a quantity risk touched an order assumes that risk caused
  that lateness. §1 says we cannot assert that. It is the repair that addresses the
  *cause* rather than the symptom — which is why I preferred it — but preferring the
  more explanatory repair is not a licence to assume its premise.
- **An `RPj` floor is arbitrary without a signed temporal resolution.** The sweep shows
  the answer does not move across 0.5–24 h, which makes the floor *harmless*, not
  *justified*. `floor = 0.5 h` is numerically identical to clipping, so it adds nothing.

**Secondary, declared but not adopted:** *quantity → time* is registered as a sensitivity
to be reported alongside, never as the primary. It requires, before it may be promoted:
an identified causal path from quantity shortfall to order lateness, and a stated
conversion with units.

## 3. What will be rerun, and what will not

**Rerun under the repaired metric:** the v2 comparator arms on their existing 24 tapes,
by replay of the recorded posture sequences — no new roots, no new search. The physical
replay gate stays as-is: fill, lost, delivered, unresolved, injection and terminal stock
must reproduce to 1e-9.

**Not rerun, and not relabelled:** every frozen result keeps its reported value under the
metric it was computed with. Program Q, the H2/H3 confirmation, the buffer gate and the
90-configuration reproduction are untouched. The repaired metric is reported as a
**second column**, never as a substitute.

## 4. Declared in advance

| item | value |
|---|---|
| primary endpoint | `ret_excel_clipped_0_1`, mean over scored orders |
| comparison | MPC vs each family's true 216-posture incumbent, paired within tape |
| incumbents | R1r `(0,0,336)`, R2r `(336,0,168)`, from v2's own `result.json` |
| tapes | the existing 12 per family; no new roots opened |
| interval | paired bootstrap, 10,000 resamples, CI95 |
| decision rule | `MPC_AHEAD` iff CI95 lower bound > 0; `STATIC_AHEAD` iff upper < 0; otherwise `NOT_SEPARATED` |
| expected result | **R2r `MPC_AHEAD` at Δ ≈ +0.0120, R1r `NOT_SEPARATED` at Δ ≈ 0.000000** — stated because it is already known |
| reported alongside, always | canonical `ret_excel`, `ret_excel_full_ledger`, `R_cobb_douglas`, fill, lost, unresolved, delivered, injected |

**What would falsify the repair rather than confirm it:** if clipping changed R1r, where
there is no out-of-range order to clip. R1r must remain `NOT_SEPARATED` at Δ = 0.000000.
A repair that moves a family it cannot touch is a broken repair.

## 5. What this will and will not license

**Will license:** reporting that under a metric bounded to its own declared range, the
corrected MPC separates from the static incumbent in R2r, and does not in R1r.

**Will not license:** replacing the canonical verdict, or presenting the repaired number
as the headline. The honest shape is that **the R2r verdict is endpoint-sensitive** —
`NOT_SEPARATED` under the metric as specified, `MPC_AHEAD` under the same metric bounded
to its declared range, with the difference produced by 7 orders in 3,108.

That sensitivity is the finding. Neither number is the finding on its own.

## 6. Signature

Requires the PI's sign-off before execution. Two things need an explicit decision that is
not mine to make: whether the clipped variant may enter the manuscript as a second column,
and whether the *quantity → time* causal identification is worth commissioning at all.
