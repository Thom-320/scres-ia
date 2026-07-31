# Result — the R12/R13 reading test: preregistered, and negative

**Status:** `PREREGISTERED_NEGATIVE_ARM_P_NOT_ADOPTED`. Executes
`docs/PREREGISTRO_DURACION_R12_R13_2026-07-30.md`. Artifact
`results/metric_audit/procurement_delay_reading_v1/result.json`. Roots 2,300,001–12.

## 0. Correction carried first: my "249.8 SD" was wrong by 17×

I reported `rpj_p95` in R1r at **249.8 SD**. The correct figure is **14.6**.

The `d_k` this project defines (`supply_chain/fidelity_moments.py:24`) is

    d_k = |M_k − R̄_k| / sqrt( s_k²/n_ref + se_k² )

and my ad-hoc script **dropped the `se_k²` term**, dividing by the reference standard
error alone. That term is not small here: our between-seed SD for `rpj_p95` is 460.9 h,
so `se_k = 133.1` against a reference term of 5.67. Including it, 1,949 / 133.2 = 14.6.

Same class of error as the retraction earlier today: a second implementation of a
definition the module already owns. Every `d_k` I quoted from that script is affected;
the RPj ones most (`rpj_mean` 19.3 → 11.0), the others barely (`autotomy_share` is
unchanged at 11.2 because our SE there is exactly zero, `ret_mean` 1.6 → 1.7).

**The gap is still real and still large** — 2,405 h against 456 h, a 5.3× ratio, in a
family whose control (R2r) sits at 2.3. The headline magnitude was overstated; the
finding was not.

## 1. Both falsifiers passed

* **R2r bit-identical across arms** on all six moments. R2r contains neither R12 nor R13,
  so this confirms the change touched only what it should.
* **Under P every R12 event lasts exactly 168.0 h and every R13 event exactly 24.0 h.**

The instrument is sound, so the negative below is a result and not a malfunction.

## 2. The result

R1r, 12 roots:

| momento | S serial | `d_k` | P paralelo | `d_k` | referencia |
|---|---:|---:|---:|---:|---:|
| `ret_mean` | 0.007 | **1.7** | 0.007 | **1.8** | 0.006 |
| `scored_orders_per_year` | 212.4 | 0.7 | 217.6 | 0.8 | 215.1 |
| `ret_above_one_share` | 0.000 | 3.9 | 0.000 | 3.9 | 0.001 |
| `autotomy_share` | 0.000 | 11.2 | 0.000 | 11.2 | 0.004 |
| `rpj_mean` | 395.0 | 11.0 | 380.5 | 11.8 | 193.7 |
| `rpj_p95` | 2405.5 | **14.6** | 2255.5 | **17.1** | 456.5 |

R2r, both arms identical, sum of `d_k` = 18.2.

**`adopt_P = False`**, by the contract's own rule.

## 3. What actually happened, and it is worth stating precisely

In **raw hours P moves in the predicted direction**: `rpj_p95` falls 2,405.5 → 2,255.5,
a **6.2%** reduction, and `rpj_mean` falls 3.7%.

But `d_k` **worsens**, 14.6 → 17.1. That is not a contradiction. P also cuts the
between-seed SD (460.9 → 363.0), so the denominator shrinks faster than the numerator.
**P moves the estimate slightly toward Garrido while making the residual gap more
certain.** The contract's acceptance is on `d_k`, so it says do not adopt.

My declared prediction was that P would reduce `rpj_p95` and could not worsen it. On the
raw moment the first half held; on the `d_k` scale the acceptance is written in, the
second half did not. **I declared the prediction on the moment and the rule on `d_k`, and
those are not the same scale.** That is a defect in my preregistration, not in the result,
and the honest reading is the conservative one: the contract's rule governs.

## 4. What this refutes

**The serial multiplier is not the cause of the R1r tail.** It is real —
`supply_chain.py` multiplied a per-contract week by the count of delayed contracts, which
the thesis's own independence assumption (Table 6.6b(2)) does not support — but correcting
it closes **6.2% of a 5.3× gap**.

So the tail is not per-event length. It is **event recurrence and overlap plus queue
accumulation**: R12 re-draws every `op1_rop` and R13 every week, events stack, and Op1/Op2
stay down across draws. That is the next hypothesis, and it is not this one.

## 5. What changes in the code, and what does not

* `procurement_delay_accumulation` is added with **`"serial"` as the default** — the
  shipped behaviour is unchanged and no frozen figure moves.
* The `"parallel"` option stays, because the *reading* argument is independent of this
  test: the thesis sentence is singular and the same table cell declares the twelve
  processes independent. It is now measurable rather than arguable.
* **Nothing is relabelled.** No result is re-run, no constant is fitted, and 168 / 24 were
  fixed in both arms.

## 6. Standing state of R1r fidelity

`ret_mean`, the endpoint the manuscript reports, is at **1.7 SD in R1r and 2.1 in R2r**,
properly computed. It is unaffected by any of this and does not block the paper.

Open, in order of size: `rpj_p95` (14.6), `autotomy_share` (11.2, and 0.000 in *both*
families), `rpj_mean` (11.0). The autotomy and RPj gaps remain plausibly one defect, since
`RPj = CTj > LTj` always makes the `CTj <= LTj` autotomy branch unreachable.
