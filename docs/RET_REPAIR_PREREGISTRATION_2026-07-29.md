# Preregistration — repairing the ReT out-of-range tail

**Status:** `DEVELOPMENT_REPLAY_COMPLETE_PROSPECTIVE_CONFIRMATION_FROZEN`. Nothing here
is applied retrospectively. The canonical metric is unchanged and every frozen result
stands as reported.

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

**Implementation check — scoped to the retrospective replay only.** On the twelve
development tapes R1r contained **zero** orders above 1.0, so clipping is a mathematical
no-op there and R1r must come back at Δ = 0.000000. Movement would mean an implementation
bug, not a finding.

**CORRECTED 2026-07-29.** An earlier draft stated this as the falsifier for the whole
repair, including §7's prospective run. That is wrong. The sixteen prospective roots are
new tapes: an out-of-range tail may legitimately appear in R1r, and clipping would then
legitimately move it. That would be a *result*, not a broken repair. Reading the
narrative expectation as a decision rule would have flagged a legitimate outcome as an
instrument failure.

**The prospective adjudication is governed by the frozen contract**
(`contracts/ret_metric_repair_confirmation_v1.json`, sha `c1efdc20...`) and its declared
`PASS_MATERIAL` / directional / fail thresholds — never by the expectations written in
this document. Nothing in the contract is changed by this correction; only a false
statement about it is removed.

## 5. What this will and will not license

**Will license:** reporting that under a metric bounded to its own declared range, the
corrected MPC separates from the static incumbent in R2r, and does not in R1r.

**Will not license:** replacing the canonical verdict, or presenting the repaired number
as the headline. The honest shape is that **the R2r verdict is endpoint-sensitive** —
`NOT_SEPARATED` under the metric as specified, `MPC_AHEAD` under the same metric bounded
to its declared range, with the difference produced by 7 orders in 3,108.

That sensitivity is the finding. Neither number is the finding on its own.

## 6. Signature

Scientific execution of the separate prospective contract is authorized. Two manuscript
decisions still require the PI's explicit sign-off: whether the clipped variant may enter
as a second column, and whether the *quantity → time* causal-identification work is worth
commissioning at all.

## 7. Prospective corrective confirmation

The existing-tape replay above is a development sensitivity whose sign is already
known. A separate executable contract now freezes a genuinely prospective check:
`contracts/ret_metric_repair_confirmation_v1.json`.

- Primary: official request-snapshot ReT clipped per order to `[0,1]`.
- Mandatory sensitivity: quantity→time proxy followed by the same clip, labelled
  `DISCLOSED_PROXY_NOT_EXACT_ATTRIBUTION`.
- Comparator: the development incumbent frozen before new roots open; the best
  static posture on confirmation roots is descriptive only.
- Roots: 16 new tapes per family, with five future scenarios per MPC candidate.
- Material pass: primary LCB95 above 0.005, proxy LCB95 above zero and flow-fill
  LCB95 no worse than −0.005.
- Directional-only pass: primary and proxy LCB95 above zero but the primary does
  not clear 0.005.
- Neither outcome changes the historical endpoint or authorizes a learner/KAN.

The development signs and exact deltas are disclosed inside the contract. Scientific
execution is authorized as a separate corrective study; use in the manuscript and
promotion of the causal proxy still require PI review.

## 8. Terminal prospective outcome — 2026-07-30

The prospective contract completed with 16/16 tapes in each family, the full
216-posture domain, five future scenarios per candidate, valid completion receipts,
and all replay-prefix hashes equal.

```text
R1r  NOT_CONFIRMED
      delta = -0.00001954
      CI95  = [-0.00004940, -0.00000021]

R2r  PASS_MATERIAL_REPAIRED_MPC
      delta = +0.01247474
      CI95  = [+0.00910860, +0.01590910]
      positive tapes = 15/16
```

The mandatory proxy agrees in both families. It remains a disclosed proxy, not
identified causal attribution.

**Important correction to the development interpretation in §5:** on the new R2r
tapes, canonical and clipped ReT both favor MPC by approximately +0.0125. The
prospective run therefore confirms the preregistered bounded-endpoint MPC result,
but it does **not** reproduce the development claim that clipping itself reverses
the sign. Conversely, `ret_excel_full_ledger` favors the static incumbent and MPC
delivers fewer total rations while using substantially less strategic material.
The terminal claim is endpoint- and resource-bounded, not dominance.

Full adjudication and custody:

- `results/metric_audit/ret_metric_repair_confirmation_v1/result.json`
- `results/metric_audit/ret_metric_repair_confirmation_v1/custody.json`
- `docs/RET_METRIC_REPAIR_CONFIRMATION_V1_OUTCOME_2026-07-30.md`
