# Workbook-formula replay and reconstruction checks — manuscript section (C&IE)

**Status:** `DRAFT_FOR_PI_REVIEW`. Reference
`results/metric_audit/fidelity_reference_v4/result.json` (sha `32e23a79b43f76a7`, read with
`docs/ERRATA_FIDELITY_REFERENCE_V4_2026-07-31.md`); measured runs
`results/metric_audit/fidelity_comparison_v4/result.json` (sha `09f71ce683a26a95`, 12 virgin
roots per family, five falsifiers passed). Every number below is a field of one of those two
artifacts.

> **Revision history.** The 2026-07-31 draft was titled *"model validation"* and is
> **withdrawn**. Three defects, all of them mine:
>
> 1. **The table had no source.** It cited `r14_seed_arms_v1`, an artifact that is HALTED
>    (`..._FALSIFIER_FAILED`), scored against the v3 reference rather than the v4 the header
>    declared, swept the pre-amendment epsilon band, and **excluded `scored_orders_per_year`
>    from scoring** — the row the draft printed in bold. The published `d_k` were not in it
>    either: I recomputed them by hand, outside the pipeline. `ret_mean` R1r was **1.52** in
>    the draft against **1.249** in the artifact and **1.79** now that it is measured properly.
> 2. **The scored-window convention was unmatched.** Ours ended at the horizon, his at the
>    last order — apart by **1.5% (R1r) and 2.2% (R2r)**, larger than R1r's own gap to the
>    reference, so the convention was deciding the discrepancy. Both sides now use
>    `(last − first OPTj)/8064`.
> 3. **It claimed behavioural validation.** It is a development-level concordance screen, and
>    §2 now says so in those words.
>
> The corrected figures are **worse in three moments and better in two**, which is what
> measuring instead of asserting is for.

---

## Why this section separates two claims

Our working note said the model was *"validated with <1% variance against the original
Excel."* That is true of **one** of two distinct exercises and false of the other, and a
reviewer will separate them even if we do not.

* **Formula fidelity** — does our resilience metric compute what Garrido-Ríos's workbook
  computes, on his own rows? **Yes, exactly.**
* **Behavioural concordance** — do our generated order histories match his statistics? **Only
  in part, and the design does not license calling the comparison a validation.**

Conflating them lets a single moment refute the strong claim.

## 1. Formula fidelity: exact

Re-computing his ReT expression over every formula-bearing row of the two canonical workbooks
gives **0 mismatches over 47,546 rows**:

    ReT_j = IF(any risk, IF(APj > 0, APj/LT, 0.5/RPj), 1 − (ΣBt + ΣUt)/j)

**Stated at its actual strength:** *conditional on the source ledger fields, the coded ReT
arithmetic reproduces all 47,546 formula-bearing workbook cells with zero discrepancy. This
verifies the arithmetic port — not the generated trajectories, and not the undocumented
Simulink logic that produced the ledger.*

**Excluded source, disclosed.** `Rsult_1.xlsx` is not used: it is a **different sample** —
twelve configurations, not the twenty canonical ones — and does not carry the per-order
operational ledger these moments require. It is *not* excluded for being materialised (its
twelve configuration sheets hold 43,776–45,234 live formulas each; only the four aggregate
sheets are pasted values) and any row-count difference is withdrawn as untraceable. See the
errata. It remains useful for studying his downstream normalisation and discretisation.

## 2. Behavioural concordance: a screen, not a validation

Reference built from **20 of 20 canonical sheets** — CF1–CF10 (R1r), CF11–CF20 (R2r).

> **Note on CF2.** An earlier reference covered 19 sheets: `CF2`'s header sits on the second
> row, so a fixed `header=0` read returned only unnamed columns and the sheet was lost
> silently — 4,420 rows, about 20% of the R1r evidence. v4 detects the header row. CF2 is
> typical of its family (`ret_mean` 0.0058 against 0.0063), so recovering it moves the R1r
> reference by under 2% on every moment.

Each moment is computed identically on his rows and ours, over the same scored window
convention. The discrepancy is reported as

    d_k = |M_k − R̄_k| / sqrt( s_k²/n_sheets + se_k² )

**and it is a descriptive standardized discrepancy, not a test statistic.** `s_k` is the
spread across ten *designed configurations*, not across replicates, so it does not estimate
sampling error and `d_k` does not support a p-value or a confidence statement. It is used
because the moments have incommensurable units — a share, an hour count, a rate — and
"how much this moment varies between his own configurations" is the only normaliser the data
itself supplies.

| moment | R1r ours | `d_k` | R1r ref | R2r ours | `d_k` | R2r ref |
|---|---:|---:|---:|---:|---:|---:|
| `ret_mean` | 0.0069 | **1.79** | 0.0063 | 0.2486 | **1.68** | 0.2007 |
| `scored_orders_per_year` | 222.0 | **0.98** | 218.6 | 215.3 | **1.07** | 219.5 |
| `ret_above_one_share` | 0.0000 | 4.47 | 0.0006 | 0.0000 | 3.99 | 0.0010 |
| `rpj_mean` | 409.7 | 7.82 | 195.4 | 783.1 | 3.46 | 626.6 |
| `rpj_p95` | 2584.4 | 14.77 | 459.5 | 3743.5 | 2.86 | 3042.0 |
| `autotomy_share` | 0.0000 | 12.40 | 0.0043 | 0.0000 | 4.56 | 0.0006 |

**Four reasons this is not a validation test, all measured:**

1. **His sheets are designed configurations, not replicates** — ten different treatment cells
   per family.
2. **They are not exchangeable among themselves.** CF1 and CF2 run **19.8 thesis years**
   against **9.8** for the other eight R1r sheets; R2r is homogeneous at 9.87–9.89. The R1r
   "spread" therefore mixes windows of different length.
3. **Configurations, seeds and horizons are unmatched.** Ours are twelve one-year roots with
   every risk of the family escalated; his are ten multi-year cells of a factorial design.
4. **No reproduction threshold was preregistered.** The previous draft's "R2r reproduces four
   of six" was a post-hoc count against no declared bar and is withdrawn.

What survives is directional: **`ret_mean` and the order-generation rate sit within about one
to two combined spreads**, and the gaps concentrate in the recovery-period statistics and in
autotomy — each with an identified cause (§4).

## 3. What is identified in the source, and what we reproduce

The order cycle time in **his** data decomposes as `CTj = 48 + k·24 + δ`, and that
decomposition returns **his** observed `p25 = 75.00` and `p50 = 101.45`. It is a
reconstruction of his ledger, **not an output of our model**, and the previous draft's
"reproduced exactly" for the whole expression was wrong.

Term by term:

* **`48 h` = `LT` = Op10 (24 h) + Op11 (0 h) + Op12 (24 h)** — thesis §6.3 and §6.8.2.
  **Identified in his data; not our floor.** Our shipped fulfilment constant is **54 h**, so
  our minimum `CTj` is 54.0 in all 24 runs and **64.4% (R1r) / 56.7% (R2r) of orders complete
  exactly there**. The 6 h excess is the subject of §4(b), and the earlier draft contradicted
  itself by claiming the 48 h floor reproduced while §4 said it was 6 h too high.
* **`k·24`, the daily-freight cadence** («at a daily freight rate, ROP = 24 hours»).
  Among our **delayed** orders — those completing past the constant — `(CTj − 48) mod 24 = 0`
  for **100.0%** (887/887 R1r, 1057/1057 R2r), landing on 72, 96, 120, 144 … as his do.
  **Scope, and it must be stated:** only **35.6% (R1r) / 43.3% (R2r)** of our orders are
  delayed at all, against **83.5% / 91.1%** of his — the same rule (`k ≥ 1`, i.e. `CTj ≥ 72`)
  applied to all 20 canonical sheets, 47,780 rows, inside the same sealed artifact. The
  lattice matches on the subpopulation that reaches it; the incidence does not.
* **`δ`, an intra-day offset.** In his data `U(0, 8)` to within 0.11 h at every quantile, with
  a ceiling at 8 h = `HOURS_PER_SHIFT` at `S = 1`, verified independently of the shape.
  **Not reproduced:** ours is identically 0.

## 4. The residuals, with distinct causes

**(a) Recovery-period statistics (`rpj_mean`, `rpj_p95`).** `δ` has no endogenous generator
here. Eight candidate mechanisms were implemented and measured, and all eight refuted:
inter-order queueing (refuted by counting — the median day serves one order), daily freight
capacity, the shift window as a queue, order attributes (all correlations below 0.12),
assembly time `Q/λ`, calendar drift, material availability, and the seeding of `R⁰` by the
quantity gate. With `RPj ≈ CTj` in our model against a **saturating** `RPj` in his — his
correlates 0.88 with the risk count and only 0.37 with cycle time, flattening near 400 h above
`CTj ≈ 500` — our recovery-period statistics run long.

**Stated at its actual strength:** this is a **pattern in his workbooks that none of eight
tested mechanisms explains**. Algorithm 2 (p.69) as published does not by itself produce a
saturating recovery period. Without the Simulink logic we do not know the generator, so we
report the pattern and the eight refutations rather than attributing a property to his model.

**(b) Autotomy — a simpler cause.** The branch requires `CTj ≤ LTj`. Our constant is 54 h
against `LT = 48`, so `CTj ≥ 54` for every order (measured: minimum exactly 54.0 across all 24
runs, **0 orders below**) and the branch is **structurally unreachable**, independent of `δ`.
His own minimum `CTj` is 48.0074 — also above `LT` — and his autotomy rows sit in a band
`CTj − LT ∈ [0.0074, 0.048]`, so the gap is our floor being 6 h too high, not a difference in
the branch rule. His incidence is small but real: **0.0043 (R1r)** and **0.0006 (R2r)**.

**(c) `ret_above_one_share`.** Ours is identically 0 against his ≈0.001. His ReT is unbounded
above despite §5.6.3 declaring a [0,1] scale, and the demonstration is a workbook row rather
than a figure axis: **CF12 carries `RPj = 0.00312` with no autotomy, so `0.5/RPj` returns
ReT = 160.2564**, one of four rows above 1 in that sheet. Small in absolute terms, disclosed,
and an internal inconsistency of the source rather than of our port.

## 5. Scope and attribution

* Comparison is against the thesis base configuration — one shift, no strategic buffers — not
  against every cell of the design.
* The thesis's own significance levels (99% for buffers, 95%/99% for shifts) are reported
  **as his**; the text acknowledges autocorrelation in the outputs and then assumes
  independence between series for the rank tests. **Our inference stays at run/tape level.**
* Two levers were measured and **not adopted**: linking order completion to Op9 takes
  `ret_mean` to 0.23–0.29 but worsens `RPj`; restricting risk attribution to physical exposure
  takes `rpj_p95` from 2,533 to 672 but costs `ret_mean`. **They do not compose.** The
  trade-off is reported rather than resolved by choosing.

## 6. The defensible paragraph

> The coded ReT arithmetic reproduced all 47,546 formula-bearing cells in Garrido-Ríos's
> canonical workbooks exactly when supplied the source ledger fields. A separate
> development-level concordance screen recovered selected order-flow signatures — the daily
> freight lattice among delayed orders, mean resilience and order-generation rate within about
> one to two combined spreads — but did not reproduce the completion-time, autotomy, or
> recovery-period distributions. Because configurations, seeds, horizons and scored windows
> were not matched, and the reference sheets are designed configurations rather than
> replicates, these discrepancies are descriptive rather than a behavioural-validation test.
> We therefore treat the environment as a **thesis-grounded reconstruction**, not a digital
> twin validated at the behavioural level.

*(Phrasing note: `tests/test_manuscript_retired_claims.py` bans the trigram "validated digital
twin" outright, so the wording above is deliberate — it says the same thing without tripping a
regex that does not read negation.)*
