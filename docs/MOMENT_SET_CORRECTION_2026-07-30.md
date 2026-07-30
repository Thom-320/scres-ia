# The population moment was a horizon artifact — and it was driving R2r's ranking

**Status:** `DEVELOPMENT_CORRECTION`. `scored_rows` → `scored_orders_per_year`.
Reference regenerated as `results/metric_audit/fidelity_reference_v2/` (sha
`5013dcc811b5d28c`); sweep re-run as `fidelity_sweep_v3_rate`.

## The defect

`scored_rows` compared a **20-year** reference against **52-week** runs: 2,381 against
~215, a factor of 11 that is almost entirely calendar. I put it in the moment set without
thinking, and it contaminated every aggregate distance computed since.

Normalised to orders per year the residual is **1.8–2.0×**, and that part *is* fidelity —
we score roughly twice as many orders per year as he does.

## A prediction of mine that was wrong

I wrote that this "does not change rankings — it is an additive constant per cell within
each family". Measured:

| family | mean `d_k` before → after | cells in the same rank position |
|---|---|---:|
| R1r | 23.75 → 16.11 | **48/48** |
| R2r | 19.53 → 11.51 | **0/48** |

**In R1r the ranking held exactly, as I said. In R2r it changed completely.** So the
artifact was not merely inflating the scale there — it was *driving the order*. Every R2r
ranking produced before this correction was ordered by a calendar mismatch.

The reason is visible in the reference spreads: R2r's population moment has CV 0.040 — his
configurations are extremely consistent about how many orders get scored — so a fixed
offset lands as a huge `d_k` that swamps the other five. R1r's CV is 0.295, seven times
looser, so the same offset does not dominate.

## What the corrected set now says we should work on

`d_k` per moment, in combined standard errors, at a representative cell:

| moment | R1r | R2r |
|---|---:|---:|
| `rpj_mean` | **30.1** | 11.1 |
| `scored_orders_per_year` | 8.2 | **26.0** |
| `rpj_p95` | 8.8 | 10.2 |
| `autotomy_share` | 11.2 | 4.6 |
| `ret_mean` | 8.1 | 4.3 |
| `ret_above_one_share` | 3.9 | 4.0 |

**`rpj_mean` is the worst moment in R1r at 30 SD**, which confirms the priority. But
**`scored_orders_per_year` is the worst in R2r at 26 SD**, and I had dismissed it as an
artifact when part of it is real: we score about twice as many orders per year as he does,
against a reference whose between-configuration spread is 4%.

That is a new item, and it is not small. It says our order population differs from his in
a way his own configurations barely vary in — either our warm-up excludes far fewer orders,
or fewer of our orders are lost or left unresolved before scoring.

## Carried forward

- `RPj` remains the top priority in R1r and is second in R2r.
- The scored-order rate joins it, ahead of autotomy in both families.
- `autotomy_share` sits at 11.2 SD in R1r and 4.6 in R2r — real, but now clearly behind
  two moments with more leverage, which supports leaving it parked.
- Every aggregate distance reported before this correction should be re-read at the new
  scale, and **any R2r ordering from before it should be discarded outright.**
