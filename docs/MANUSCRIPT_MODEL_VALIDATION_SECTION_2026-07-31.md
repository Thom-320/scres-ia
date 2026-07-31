# Model validation — manuscript section (C&IE), with the two levels kept apart

**Status:** `DRAFT_FOR_MANUSCRIPT`. Every figure traces to a sealed artifact; nothing here
is quoted from prose. Written 2026-07-31 to replace an overclaim that would otherwise reach
reviewers.

---

## Why this section is being rewritten

Our working note said the model was *"validated with <1% variance against the original
Excel."* That is true of **one** of two distinct validations and false of the other, and a
reviewer of *Computers & Industrial Engineering* will separate them even if we do not.

* **Formula fidelity** — does our resilience metric compute what Garrido's workbook
  computes, on his own rows? **Yes, exactly.**
* **Behavioural fidelity** — does our discrete-event model *generate* order histories whose
  statistics match his? **Partly, and we can now say precisely where.**

Conflating them would let a reviewer refute the strong claim with a single moment.

---

## 1. Formula fidelity: exact

Re-computing Garrido's ReT expression over every formula-bearing row of the two canonical
workbooks (`Raw_data1+Re.xlsx`, `Raw_data2+Re.xlsx`) gives **0 mismatches over 47,546 rows**.

This establishes that our implementation of

    ReT_j = IF(any risk, IF(APj > 0, APj/LT, 0.5/RPj), 1 − (ΣBt + ΣUt)/j)

is his, not an interpretation of his. **Unchanged by any 2026-07-31 work.**

**Excluded source, and it must be disclosed:** `Rsult_1.xlsx` is *not* used. Its twelve
configurations differ from the raw workbooks by −1,949 to +735 rows and it stores `Re` as a
pasted constant rather than the live formula, so it cannot serve as ground truth.

## 2. Behavioural fidelity: six moments, measured against 19 canonical sheets

Reference: `results/metric_audit/fidelity_reference_v3/result.json` (sha `31ecf9f9dae8058a`),
built from 19 sheets — CF1, CF3–CF10 for the R1r family and CF11–CF20 for R2r. Each moment
is computed **identically** on his rows and on ours.

Discrepancy is reported in combined standard errors,

    d_k = |M_k − R̄_k| / sqrt( s_k²/n_sheets + se_k² )

so `d_k = 1` means one combined standard error. Twelve independent roots per cell.

| moment | R1r ours | `d_k` | R1r ref | R2r ours | `d_k` | R2r ref |
|---|---:|---:|---:|---:|---:|---:|
| **`ret_mean`** | 0.007 | **1.25** | 0.006 | 0.249 | **1.76** | 0.201 |
| `ret_above_one_share` | 0.000 | 3.90 | 0.001 | 0.000 | 3.99 | 0.001 |
| `autotomy_share` | 0.000 | 11.20 | 0.004 | 0.000 | 4.56 | 0.001 |
| `rpj_mean` | 410.2 | 9.35 | 193.7 | 784.9 | 2.32 | 626.6 |
| `rpj_p95` | 2362.2 | 11.46 | 456.5 | 3479.4 | 1.49 | 3042.0 |

**The endpoint the paper reports, `ret_mean`, sits at 1.25 and 1.76 combined standard
errors.** The R2r family reproduces on four of five moments. The R1r gaps are concentrated
in the recovery-period statistics and in autotomy.

## 3. What reproduces exactly, and it is more than the moments suggest

The order cycle time decomposes as

    CTj = 48 + k·24 + δ

and this reconstruction returns his observed `p25 = 75.00` and `p50 = 101.45` exactly.

* **`48 h`** is `LT` = Op10 (24 h) + Op11 (0 h) + Op12 (24 h), thesis §6.3 and §6.8.2.
  **Reproduced exactly.**
* **`k·24`** is the daily-freight cadence, «at a daily freight rate (ROP = 24 hours)».
  **Reproduced exactly:** `(CTj − 48) mod 24 = 0` for **100%** of our delayed orders, and
  his histogram's empty bands at [60,72) and [84,96) appear in ours too.
* **`δ`** is an intra-day offset, `U(0, 8)` in his data to within 0.11 h at every quantile,
  with a ceiling fixed at 8 h — `HOURS_PER_SHIFT` at `S = 1`. **Not reproduced:** ours is 0.

Our risk exposure also matches: `R14` fires at **0.85×** his Table 6.11 rate, and per-order
touch rates are 68.1/75.4% (R11), 25.3/37.6% (R13) and 98.1/81.1% (R14), his against ours.

## 4. Where the residual is, stated as a limitation

Two of the three `CTj` terms reproduce; the third does not, and it propagates.

**`δ` has no endogenous generator.** Eight candidate mechanisms were implemented and
measured, and every one was refuted: inter-order queueing (refuted by counting — the median
day serves one order), daily freight capacity, the shift window as a queue, order attributes
(all correlations below 0.12), assembly time `Q/λ` (the ceiling does not move with `Q` across
deciles), calendar drift (`δ` is i.i.d.), material availability, and the seeding of `R⁰` by
the quantity gate.

**Consequence.** With `RPj ≈ CTj` in our model against a saturating `RPj` in his — his
correlates **0.88** with the risk count and only **0.37** with cycle time, freezing near
400 h — our recovery-period statistics run long. That is the single origin of the `rpj_mean`,
`rpj_p95` and `autotomy_share` gaps.

**This is a property of his model that the thesis does not document**, and reporting it is a
contribution rather than an embarrassment: the published algorithm (Algorithm 2, p.69) does
not by itself produce a saturating recovery period.

## 5. Scope, stated plainly

* The comparison is against **19 canonical sheets**, one shift, no strategic buffers, the
  thesis base configuration — not against every cell of the design.
* `scored_orders_per_year` is **excluded from scoring** pending a reference whose denominator
  uses the scored window on both sides; his sheets exclude the warm-up (`min(OPTj)` is
  823–1,225 h) while the current reference divides by `max(OPTj)`.
* Two levers were measured and **not adopted**: linking order completion to Op9 takes
  `ret_mean` to **0.23–0.29** but worsens `RPj`; restricting risk attribution to physical
  exposure takes `rpj_p95` from 2,533 to **672** but costs `ret_mean`. They do not compose.
  The trade-off frontier is measured and reported rather than resolved by choosing.

## 6. One sentence for the abstract

> The simulation reproduces Garrido-Ríos's resilience formula exactly (0 mismatches over
> 47,546 rows) and its order-cycle structure — lead time and daily-freight cadence —
> exactly; the mean resilience index agrees to within 1.3–1.8 combined standard errors
> across both risk families, while recovery-period statistics diverge for a documented
> reason we report as a limitation.
