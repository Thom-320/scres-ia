# Model validation — manuscript section (C&IE)

**Status:** `DRAFT_FOR_PI_REVIEW`. Every figure traces to a sealed artifact. Reference
`results/metric_audit/fidelity_reference_v4/result.json` (sha `32e23a79b43f76a7`), measured
runs `results/metric_audit/r14_seed_arms_v1/result.json`.

> **Revision history.** A first draft (2026-07-31) was withdrawn after self-review found
> three defects: `CF2` silently dropped from the reference, the autotomy gap misattributed,
> and the cadence claim resting on an unstated subpopulation. All three are fixed below and
> the reference has been rebuilt as v4. The corrected figures are **not materially
> different** — which is itself worth knowing.

---

## Why this section separates two claims

Our working note said the model was *"validated with <1% variance against the original
Excel."* That is true of **one** of two distinct validations and false of the other, and a
reviewer will separate them even if we do not.

* **Formula fidelity** — does our resilience metric compute what Garrido's workbook computes,
  on his own rows? **Yes, exactly.**
* **Behavioural fidelity** — does our discrete-event model *generate* order histories whose
  statistics match his? **Partly, and we can now say precisely where and why.**

Conflating them lets a single moment refute the strong claim.

## 1. Formula fidelity: exact

Re-computing Garrido's ReT expression over every formula-bearing row of the two canonical
workbooks gives **0 mismatches over 47,546 rows**, establishing that our implementation of

    ReT_j = IF(any risk, IF(APj > 0, APj/LT, 0.5/RPj), 1 − (ΣBt + ΣUt)/j)

is his and not an interpretation of it.

**Excluded source, disclosed:** `Rsult_1.xlsx` is not used. Its twelve configurations differ
from the raw workbooks by −1,949 to +735 rows and store `Re` as a pasted constant rather than
the live formula.

## 2. Behavioural fidelity: six moments over all twenty sheets

Reference built from **20 of 20 canonical sheets** — CF1–CF10 (R1r) and CF11–CF20 (R2r).

> **Note on CF2.** An earlier reference covered 19 sheets. `CF2`'s header sits on the second
> row, so a fixed `header=0` read returned only unnamed columns and the sheet was lost
> silently — 4,420 rows, about 20% of the R1r evidence. v4 detects the header row. CF2's
> statistics are typical of its family (`ret_mean` 0.0058 against a family mean of 0.006),
> so including it shifts the R1r reference by **under 2% on every moment**.

Each moment is computed **identically** on his rows and ours. Discrepancy is reported in
combined standard errors,

    d_k = |M_k − R̄_k| / sqrt( s_k²/n_sheets + se_k² )

where `s_k` is the between-configuration spread across his own sheets and `se_k` our
between-root standard error. `d_k = 1` means one combined standard error. This scale is used
because the moments have incommensurable units (a share, an hour count, a rate); expressing
each in units of *how much it varies between Garrido's own configurations* is the only
normaliser the data itself supplies. Twelve independent roots per cell.

| moment | R1r ours | `d_k` | R1r ref | R2r ours | `d_k` | R2r ref |
|---|---:|---:|---:|---:|---:|---:|
| **`ret_mean`** | 0.007 | **1.52** | 0.006 | 0.249 | **1.76** | 0.201 |
| **`scored_orders_per_year`** | 222.2 | **1.19** | 218.6 | 211.8 | **1.93** | 219.5 |
| `ret_above_one_share` | 0.000 | 4.47 | 0.001 | 0.000 | 3.99 | 0.001 |
| `rpj_mean` | 410.2 | 9.40 | 195.4 | 784.9 | 2.32 | 626.6 |
| `rpj_p95` | 2362.2 | 11.44 | 459.5 | 3479.4 | 1.49 | 3042.0 |
| `autotomy_share` | 0.000 | 12.40 | 0.004 | 0.000 | 4.56 | 0.001 |

**The endpoint the paper reports, `ret_mean`, sits at 1.5–1.8 combined standard errors**, and
the order-generation rate at 1.2–1.9. R2r reproduces on four of six moments. The gaps are
concentrated in R1r's recovery-period statistics and, in both families, in autotomy — each
with a distinct and identified cause (§4).

## 3. What reproduces exactly

The order cycle time decomposes as `CTj = 48 + k·24 + δ`, and this reconstruction returns his
observed `p25 = 75.00` and `p50 = 101.45` exactly.

* **`48 h`** = `LT` = Op10 (24 h) + Op11 (0 h) + Op12 (24 h), thesis §6.3 and §6.8.2.
  **Reproduced exactly.**
* **`k·24`**, the daily-freight cadence («at a daily freight rate, ROP = 24 hours»).
  `(CTj − 48) mod 24 = 0` for **100% of our delayed orders**, landing on 72, 96, 120, 144 …
  as his do, with his empty histogram bands at [60,72) and [84,96) present in ours.
  **Scope, and it must be stated:** only **36.5%** of our orders are delayed against
  **83.5%** of his, so we reproduce the grid over a smaller subpopulation than the one in
  which he exhibits it. The *structure* matches; the *incidence* does not.
* **`δ`**, an intra-day offset. In his data `U(0, 8)` to within 0.11 h at every quantile,
  with a ceiling fixed at 8 h — `HOURS_PER_SHIFT` at `S = 1`, verified independently of the
  shape (the p99 stays flat at ≈8.0 while `Q/λ` rises from 7.51 to 7.98 across `Q` deciles).
  **Not reproduced:** ours is identically 0.

Risk exposure matches: `R14` fires at **0.85×** his Table 6.11 rate, and per-order touch rates
are 68.1/75.4% (R11), 25.3/37.6% (R13) and 98.1/81.1% (R14), his against ours.

## 4. The two residuals, with distinct causes

**(a) Recovery-period statistics (`rpj_mean`, `rpj_p95`, R1r).** `δ` has no endogenous
generator. Eight candidate mechanisms were implemented and measured, and all eight refuted:
inter-order queueing (refuted by counting — the median day serves one order), daily freight
capacity, the shift window as a queue, order attributes (all correlations below 0.12),
assembly time `Q/λ`, calendar drift, material availability, and the seeding of `R⁰` by the
quantity gate. With `RPj ≈ CTj` in our model against a **saturating** `RPj` in his — his
correlates **0.88** with the risk count and only **0.37** with cycle time, freezing near
400 h above `CTj ≈ 500` — our recovery-period statistics run long.

**This is a property of his model that the thesis does not document.** Algorithm 2 (p.69) as
published does not by itself produce a saturating recovery period, and reporting that is a
contribution rather than an embarrassment.

**(b) Autotomy (`autotomy_share`, both families) — a different and simpler cause.** The
autotomy branch requires `CTj ≤ LTj`. Our fulfilment constant is 54 h against `LT = 48`, so
`CTj ≥ 54` for every order and **0 of 416 scored orders** can satisfy the condition: the
branch is **structurally unreachable**, independent of `δ`. Garrido's own minimum `CTj` is
48.0074, i.e. also above `LT`, and his 96 autotomy rows sit in a band `CTj − LT ∈ [0.0074,
0.048]` — so the gap is our floor being 6 h too high, not a difference in the branch rule.

**(c) `ret_above_one_share` (3.99–4.47).** Ours is identically 0 against his ≈0.001. His ReT
is unbounded above (Fig. 6.8a runs to ≈120 despite §5.6.3 declaring a [0,1] scale); ours does
not reach that tail at the base configuration. Small in absolute terms, and disclosed.

## 5. Scope

* Comparison is against the thesis base configuration — one shift, no strategic buffers —
  not against every cell of the design.
* Two levers were measured and **not adopted**: linking order completion to Op9 takes
  `ret_mean` to **0.23–0.29** but worsens `RPj`; restricting risk attribution to physical
  exposure takes `rpj_p95` from 2,533 to **672** but costs `ret_mean`. **They do not
  compose.** The trade-off frontier is reported rather than resolved by choosing.

## 6. One sentence for the abstract

> The simulation reproduces Garrido-Ríos's resilience formula exactly (0 mismatches over
> 47,546 rows) and the structure of its order cycle — lead time and daily-freight cadence —
> exactly; mean resilience agrees to within 1.5–1.8 combined standard errors across both risk
> families and order generation to within 1.2–1.9, while recovery-period statistics diverge
> for a documented reason we report as a limitation.
