# Q-R1 — deployable residual against the frozen c256 comparator

**Date:** 2026-07-22
**Branch:** `codex/q-r1-comparator-reconciliation`
**Comparator:** `qr1_v2_scenario_h4_c256_wf0.00_unone_expected_tol0.0000_legacy` (frozen,
`contracts/q_r1_comparator_v2_frozen_c256_v1.json`, entitlement re-derived from raw rows in
`results/q_r1/comparator_v2_c256_c1024_v1/freeze_audit_recomputed.json`)
**Roots:** burned 7570801–7570824, 48 campaigns (24 roots × 2 persistence strata)
**Claim status:** `BURNED_DEVELOPMENT_NO_CLAIM`

---

## Verdict

**`STOP_NO_CONVERTIBLE_RESIDUAL_OVER_FROZEN_C256` — M1/M2/M4 do not open.**

The gate in the approved route is: *GO to Gate 3 only if LCB95(residual over the frozen
comparator) ≥ 0.015.* The residual measured here is the **clairvoyant** one — the exact
maximum over the 4⁸ = 65,536 calendar space minus what the frozen comparator achieved — so
it upper-bounds **any** policy in that action space, learned or not.

| stratum | ceiling − frozen (mean) | LCB95 | gate 0.015 | SESOI 0.01 |
|---|---|---|---|---|
| κ = 0.90 | +0.009204 | **−6.0e-17 (exactly 0)** | fail | fail |
| κ = 0.75 | +0.033304 | +0.010761 | fail | pass |
| pooled | +0.021254 | +0.008227 | fail | fail |

No stratum clears 0.015, and κ=0.90 has no residual at all. A deployable learner would
capture strictly less than these numbers, so the ceiling failing is decisive.

## Why: the frozen comparator is already at the exact optimum

| stratum | campaigns where the frozen comparator picks the **exact global optimum** of 65,536 | median gap | campaigns carrying the entire gap |
|---|---|---|---|
| κ = 0.90 | **23 / 24** | 0.000000 | **1** |
| κ = 0.75 | **19 / 24** | 0.000000 | 5 |

Median calendars strictly better than the one chosen: **0** in both strata. The mean gap is
not a broad convertible headroom — it is a **tail event**. At κ=0.90 the entire +0.0092 comes
from one campaign (+0.2209); at κ=0.75, from five. A learner cannot systematically convert a
residual that is zero in the median campaign and concentrated in rare campaigns where the
belief was simply wrong.

This is the strongest form the "no neural premium" result has taken in this project: not
"we failed to find a gain", but "**the structured controller is provably at the answer key
in 23 of 24 campaigns**".

## What the same measurement establishes positively

All quantities clustered by history root, 10,000 draws, seed 20260722, leave-one-root-out
selection for the open-loop arm.

| bound | κ = 0.90 | κ = 0.75 | pooled |
|---|---|---|---|
| **frozen − open-loop** (value of structured feedback) | **+0.072614** (LCB +0.036263) | +0.050339 (LCB +0.013783) | +0.061477 (LCB +0.036471) |
| **frozen − reset** (value of retained knowledge) | +0.071816 (LCB +0.057223) | +0.067593 (LCB +0.041345) | +0.069704 (LCB +0.054576) |
| ceiling − open-loop (total headroom, not deployable) | +0.081818 | +0.083643 | +0.082731 |
| open-loop − reset | −0.000798 (CI straddles 0) | +0.017254 (CI straddles 0) | +0.008228 (CI straddles 0) |

The frozen structured controller captures **89%** of the total clairvoyant headroom over
open loop at κ=0.90 (0.0726 / 0.0818) and **60%** at κ=0.75. Both feedback bounds are
deployable-vs-deployable: the open-loop calendar is chosen without ever seeing the root it
is scored on.

The leave-one-root-out selection returned **one and the same calendar in all 24 folds**,
`[0,0,3,3,3,3,3,3]`, in both strata — the open-loop benchmark is stable, not a split
artifact.

## Method — the two contaminations that were removed

The superseded `scripts/run_q_r1_d3_residual_bound.py` computed
`best = max(candidates, key=early_ret_2w)` over a pool containing placebo and oracle arms:
a per-episode selection on the realized outcome. The external audits measured the damage —
placebos won 421/528 selected episodes.

The replacement, `scripts/run_q_r1_deployable_residual_vs_frozen_c256.py`, enforces both
fixes structurally:

1. **Allowlist.** Deployable arms are exactly `frozen_c256_retained`,
   `frozen_c256_reset`, `open_loop_fixed_calendar`. Placebos, oracle arms and per-episode
   realized selection are named in `excluded_from_every_deployable_bound` and cannot enter
   any deployable number. Clairvoyant quantities are emitted under separate keys listed in
   `clairvoyant_keys_not_deployable`.
2. **No selection on the evaluated outcome.** The open-loop calendar scored on root *r* is
   the one maximising mean objective over every root **except** *r*.

**Integrity check:** replaying each frozen comparator's own calendar through the exact
frontier reproduces the value the Pareto recorded for it in **48/48** campaigns at 1e-9.
Two independent code paths — the sequential per-campaign evaluator and the vectorised
65,536-calendar frontier — agree. The script aborts if any campaign disagrees.

## Scope — what this does NOT close

- **It does not bound a richer action space.** The ceiling is the maximum over the 4⁸
  *weekly* action space. The per-batch C6 variant (2²⁴ = 16.7M, `notebooks/
  scresia_david_perbatch_C6_FINAL.ipynb`) is a strictly larger space and this result says
  nothing about it. That door stays open, and it is now the *only* action-space door left.
- **Burned roots.** 7570801–7570824 have been reused across the c64 Pareto, the convergence
  ladder and the C1 diagnostic. This is development evidence for a gate decision, not a
  claim, and no fresh seed was opened.
- **It does not revisit the sealed confirmatory STOP.** Different roots, comparator and arms.
- **It is a mean-objective result.** The κ=0.75 service breach recorded in
  `clustered_inference.json` (worst-product LCB −0.05377) is untouched by this document.

## Consequence for the route

Steps 5–6 of the binding sequence (M1 terminal value, M2 belief correction) required a
deployable residual against the frozen comparator. **There is none.** The route therefore
advances to step 7 — learner-blind risk sensitivity — or terminates, and **no PPO, GRU or
confidence-gated learner is authorized on this action space.** The corrected north star is
satisfied in its structured form: retained decision knowledge causally improves cold-start
resilience (+0.0718, LCB +0.0572) and structured feedback beats the best open-loop calendar
(+0.0726, LCB +0.0363) — with no learner anywhere in the result.
