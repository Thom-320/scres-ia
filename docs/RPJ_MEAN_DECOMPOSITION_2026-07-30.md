# `rpj_mean` decomposed — one term is a definition mismatch, the rest is the tail

**Status:** `RETRACTED_IN_PART` by `docs/RETRACTACION_POBLACION_PUNTUADA_2026-07-30.md`.
> The scored-population claim below is **false**: the canonical ledger already
> filters to served orders (`ret_thesis.py:477`), and `scored_orders_per_year` was
> already at 0.7 SD. The 274.7 figure came from an ad-hoc script, not the pipeline.
> What survives: the RPj-mode correction, and that `rpj_mean`'s residual is the
> `CTj` tail.

**Original status:** `DEVELOPMENT_DIAGNOSIS`. Reference `fidelity_reference_v3` (horizon measured
per sheet). Roots 2,200,001–3.

## The decomposition

| term | worth |
|---|---:|
| `disruption` → `elapsed` (the thesis's own formula) | ~3.5 SD |
| scored-population mismatch | see below |
| residual `CTj` tail | **19.2 SD (R1r)** / 4.4 SD (R2r) |

## The definition mismatch, and it is ours

**Garrido's sheets contain zero orders without an `OATj`.** Checked CF1, CF3 and CF11:
4,241 / 2,151 / 2,165 rows, **0.0% missing** in every one. His scored population is, by
construction, the *served* orders.

Ours includes **22.5% that are never served**. And they are not an end-of-horizon
truncation artifact — their placement distribution is indistinguishable from the served
ones (p50 at 0.55 of the horizon against 0.56, and only 22.7% in the last 20%, which is
chance).

So we have been comparing a population that includes permanently unserved orders against
one that excludes them by definition. Aligning it:

| family | moment | before | after | reference |
|---|---|---:|---:|---:|
| R1r | `scored_orders_per_year` | 274.7 (**19.9 SD**) | **213.0 (0.7 SD)** | 215.1 |
| R2r | `scored_orders_per_year` | 278.7 (**22.5 SD**) | **204.7 (4.6 SD)** | 217.3 |

**The population moment collapses from 19.9 SD to 0.7 SD in R1r**, and from 22.5 to 4.6 in
R2r. That is not a model change and not a fit — it is scoring the same population he
scores.

## And it does not touch `rpj_mean`

`rpj_mean` is the mean over *positive* `RPj`, and unserved orders carry `RPj = 0`, so they
were already excluded. The value is 347.8 before and after.

So the earlier claim that the missing zeros were the dominant term in `rpj_mean` is
**wrong** — that counterfactual assumed the fast orders lacked `RPj`, and they do not:
100% of served orders in both the short and long groups have `RPj > 0`. The zeros are the
unserved, and they never entered the mean.

**`rpj_mean`'s 19.2 SD in R1r is the `CTj` tail, undiluted.** Our served orders have
`RPj = CTj` exactly, and our tail runs 45% longer than his.

## What is actionable now, and what is not

**Actionable, and independent of everything else:**

1. **Score only served orders**, matching his definition. Worth 19.2 SD on the population
   moment in R1r and 17.9 in R2r, changes no physics, and is verifiable against his sheets
   rather than chosen by us.
2. **Switch `ret_recovery_period_mode` to `elapsed`**, the thesis's Algorithm 2. Worth
   ~3.5 SD and better on all four `RPj` measurements.

Both are definition corrections, not calibrations. Neither needs the fitting discipline
because neither has a free parameter.

**Not actionable by any change tried so far:** the remaining 19.2 SD in R1r's `rpj_mean`.
It is the `CTj` tail, the matching sweep failed to move it, and the order-volume hypothesis
was refuted. R2r's `rpj_mean` is already at 4.4 SD and needs nothing.

## Correction carried

I said the residual was "the shape of the `CTj` distribution" and then that the dominant
term was "the missing zeros". The first was right and the second was wrong — the zeros
never entered the mean. The decomposition above replaces both.
