# Result — the RPj clamp: preregistered, prediction correct, arm W not adopted

**Status:** `PREREGISTERED_NEGATIVE_ARM_W_NOT_ADOPTED`. Executes
`docs/PREREGISTRO_CLAMP_RPJ_2026-07-30.md`. Artifact
`results/metric_audit/rpj_onset_admission_v1/result.json`. Roots 2,400,001–12.

## 1. All three falsifiers passed

| falsador | resultado |
|---|---|
| arm C reproduces the frozen block on roots 2,300,001–12 | **exacto**: `rpj_p95 = 2405.5`, `ret_mean = 0.007` |
| `RPj ≤ CTj` everywhere, both arms | 0 violaciones |
| under W every `RPj > 0` order has an in-window onset | 0 violaciones, 1.917 órdenes revisadas |

The instrument is sound, so what follows is a result.

## 2. The result

| momento | R1r C | `d_k` | R1r W | `d_k` | R2r C | `d_k` | R2r W | `d_k` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `ret_mean` | 0.007 | **1.7** | 0.011 | **11.0** | 0.270 | 2.6 | 0.273 | 2.7 |
| `rpj_mean` | 401.2 | 9.5 | 383.6 | 8.7 | 775.9 | **2.9** | 966.0 | **5.3** |
| `rpj_p95` | 2440.6 | **12.0** | 2421.4 | **11.9** | 3575.4 | **1.8** | 4118.2 | **4.1** |
| `scored_orders_per_year` | 213.3 | 0.6 | 213.3 | 0.6 | 208.3 | 2.3 | 208.3 | 2.3 |
| suma `d_k` | | **38.9** | | **47.4** | | **18.1** | | **23.0** |

**`adopt_W = False`**, by the contract's own rule.

## 3. The prediction was right, on both halves

§3 of the preregistration declared, in `d_k`:

1. *W improves `d_k(rpj_p95)` in R1r* — **correct**, by 0.078. That is 2,440.6 → 2,421.4
   raw, **0.8%**.
2. *W probably does not close it* — **correct**, and by a wider margin than I expected.
3. *The declared risk is adverse: cutting `RPj` raises `ret_mean`, which is already above
   the reference* — **correct, and it is the decisive term.** `ret_mean` in R1r degrades
   **1.7 → 11.0**, a 9.3 `d_k` regression on the endpoint the manuscript reports.

Declaring the prediction on the same scale as the rule worked. Nothing here required
interpretation after the fact.

## 4. What is refuted, and it is my own mechanism

**The clamp is not the cause of the R1r tail.** Removing it buys **0.8%** of a 5.3× gap.

That is the fourth mechanism I proposed and measured today, and the fourth refuted:
per-event duration, the serial multiplier, recurrence and overlap, and now the onset clamp.

## 5. Why R2r got worse, which is informative

R2r's `rpj_mean` rises 775.9 → 966.0 and `rpj_p95` 3,575 → 4,118. That is not noise and it
has a clean cause: under W, an order whose only onsets precede `OPTj` fails Algorithm 2's
condition and leaves the `RPj > 0` population entirely. Those are disproportionately the
**short**-`RPj` orders, so removing them **raises** the mean and the p95 over what remains.

The literal reading therefore trades a small tail improvement in R1r for a population
truncation that hurts R2r. Both effects are real; neither is a fit.

## 6. Where this leaves the diagnosis

The honest state, all measured:

* **our physics matches his** — `CTj` p95 2,450 against his 2,239, within 9%;
* **his `RPj` saturates** near 400 for `CTj ≥ 1,000` while ours tracks `CTj` to 1.00;
* **no attribution rule we have tried reproduces that saturation.** `disruption`,
  `elapsed`-clamped and `elapsed`-within-window all fail, in different directions.

The saturation is the whole remaining object, and it was explicitly declared out of scope
in §8 of the preregistration because I could not name a mechanism for it. I still cannot,
and I am not going to propose a fifth one today without first finding text or data that
constrains it.

## 7. What changed in the code

`rpj_onset_admission` is added with **`"clamped"` as the default**. Arm C is bitwise
identical to the shipped behaviour — falsifier 1 proves it against a frozen artifact — and
no result moves. The `"within_window"` option stays, because Algorithm 2's condition is
real text and is now measurable rather than arguable; the measurement says adopting it
costs more than it buys.

**Nothing relabelled. No constant swept.**

## 8. Standing state

`ret_mean` under the shipped default is **1.7 SD (R1r) / 2.6 (R2r)** and unaffected. The
open gaps, in order: `rpj_p95` R1r (12.0), `autotomy_share` (11.2 / 4.6, cause **already
located** — the 54 h fulfilment delay exceeds `LT = 48`, so no order can ever be on time),
`rpj_mean` R1r (9.5), `ret_above_one_share` (3.9 / 4.0).

Of those, **autotomy is the only one with a known cause and no proposed fix**, which makes
it the cheapest next target rather than the tail.
