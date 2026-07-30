# Retraction — the "scored-population mismatch" was my measurement, not the model

**Status:** `RETRACTION`. Retracts the central claim of
`docs/RPJ_MEAN_DECOMPOSITION_2026-07-30.md` (commit 164f160) and Change A of
`docs/PREREGISTRO_POBLACION_PUNTUADA_2026-07-30.md` (commit 3a92745).

## What I claimed

That our ReT scored population included 22.5% never-served orders while Garrido's
included none, and that aligning it would collapse `scored_orders_per_year` from
19.9 SD to 0.7 SD in R1r.

## What is actually true

**The canonical ledger already filters to served orders.**
`ret_thesis.py:477` builds `visible_orders` with `getattr(order, "OATj", None) is
not None`. The metric has been aligned with Garrido's population all along.

The 274.7 figure came from my own ad-hoc script, which divided the *full* scored
count by the horizon. The pipeline never computed that. Measured through the real
path, `scored_orders_per_year` was **already 213.0 (0.7 SD)** in R1r and 204.7
(4.6 SD) in R2r.

I also suspected the moment helper mixed populations — `apj` over 277 orders
against a denominator of 217. It does read from the wider list, but every
unserved order carries `APj = RPj = 0`, so the moments are **bit-identical** under
both populations. Measured, not reasoned: all six moments match to the printed
precision.

**Change A is withdrawn.** It was a no-op that added a dead configuration knob, so
the code is reverted rather than kept "for safety".

## What survives

**Change B stands and is applied.** `RET_RECOVERY_PERIOD_MODE` migrates
`"disruption"` → `"elapsed"`, per Algorithm 2 (thesis p.69). It is a definition
correction with no free parameter, and `"disruption"` stays selectable so frozen
runs reproduce.

## The real state of fidelity, measured under `elapsed`

Roots 2,200,001–3, against `fidelity_reference_v3`:

| momento | R1r | `d_k` | R2r | `d_k` |
|---|---:|---:|---:|---:|
| `scored_orders_per_year` | 213.0 | **0.7** | 204.7 | 4.6 |
| `ret_mean` | 0.007 | **1.6** | 0.239 | **1.5** |
| `ret_above_one_share` | 0.000 | 3.9 | 0.000 | 4.0 |
| `autotomy_share` | 0.000 | 11.2 | 0.000 | 4.6 |
| `rpj_mean` | 348.9 | 19.3 | 737.1 | 4.2 |
| `rpj_p95` | 1869.6 | **249.8** | 3162.7 | **0.7** |

**The broken moment is not `rpj_mean`. It is `rpj_p95` in R1r, at 249.8 SD** —
1,869.6 h against a 456.5 h reference, a 4.1× tail. R2r's p95 is at 0.7 SD, so
whatever produces it is specific to the R1r risk family, not to the metric.

Two moments are already good in both families: `ret_mean` (1.6 / 1.5 SD) and, in
R1r, the population. `autotomy_share` is 0.000 in both families against a nonzero
reference — autotomy never fires at all, which is a separate and older defect.

## The lesson, which is the third time today

Six constants have now been fitted or assumed against a single observable, each
breaking another: `delay = 54`, the RPj mode, daily-freight congestion,
`op11_handling_hours`, `scored_rows`, `REFERENCE_HORIZON_YEARS`. This one is worse
in kind — I measured a discrepancy with a script that did not reproduce the
pipeline, then wrote a preregistration against it. **The falsifier belongs before
the diagnosis, not after it.** Here it would have been one line: check whether the
canonical ledger already filters.
