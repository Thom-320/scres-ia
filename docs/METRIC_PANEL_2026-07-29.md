# The five-column metric panel

**Status:** `DEVELOPMENT_SCREEN_NO_CLAIM`. Four tapes per cell decides nothing. Runner
`scripts/build_metric_panel.py`, result `results/metric_panel/panel_v1.json`.

## Why a panel and not a fourth metric

Three screens converged on the same conclusion by three different mechanisms:

| metric | how it fails to reward service |
|---|---|
| `ret_excel` | policy-dependent censoring; the omitted-order fraction ranges 3.9%–18.6% across postures, so each policy is scored on a different population |
| `ret_thesis` | every case collapses into the `recovery` bucket under risk |
| `R_cobb_douglas` | prices no lost order — an order never served leaves the backorder queue and stops costing anything |

**A resilience index is not a service guarantee.** So service is carried here as declared
constraints a policy passes or fails, never as a term inside an objective it can trade
away, and resource use is reported on a Pareto front rather than scalarised.

The panel also spans what nothing else spans. The Cobb-Douglas static screen varied shifts
but not heterogeneous buffer postures; the v2 comparator instrument varies all 216 postures
but pins `shifts=1`. Garrido's expanded contract is buffers **and** shifts. 5 postures × 3
shifts + DDMRP at each shift = 18 cells per family.

## The finding that survives everything: R1r

| cell | ret_excel | full ledger | **R_CD** | cvar10 | fill | lost | unresolved | κ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 168/0/168 \| **S1** | 0.005568 | 0.005547 | **0.5969** | 0.003442 | 0.99634 | 0 | 0 | **388,544** |
| 168/168/168 \| S1 | 0.005486 | 0.005468 | 0.5967 | 0.002965 | 0.99658 | 0 | 0 | 387,496 |
| ddmrp \| S1 | 0.005469 | 0.005451 | 0.5889 | 0.002867 | **0.99669** | 0 | 0 | 424,248 |
| 168/0/168 \| **S3** | **0.005594** | **0.005573** | 0.5400 | **0.003552** | 0.99634 | 0 | 0 | **834,377** |

**`R_CD` picks S1. `ret_excel`, `ret_excel_full_ledger` and `ret_excel_cvar10` all pick S3.**

And this disagreement is **completely insensitive to the service floor** — identical at
every threshold from 20,000 to 200,000 (`agreement_is_floor_robust: true`). It is a
property of the metrics, not of where a constraint was drawn.

The mechanism is as clean as it gets: at posture `168/0/168`, **S1 and S3 have identical
fill (0.99634), identical zero lost orders, identical zero unresolved backorder.** Three
shifts buy exactly zero service and cost **2.15×** (834,377 against 388,544). All three
ReT variants prefer S3 anyway. They are not rewarding extra service — they are rewarding
something that is neither service nor free.

That is the concrete, robust case for the third column.

## The finding I nearly got wrong: R2r

On first run, all four metrics agreed on a single winner among service-passing cells, and
that looked like the headline: *impose service as a constraint and the metric war
disappears.* **It is false.** The sweep:

| floor on unresolved backorder | cells passing | distinct winners |
|---|---:|---|
| 20,000 | **0** | — |
| 40,000 | **0** | — |
| 49,000 | **0** | — |
| 50,000 | 2 | **1 — all agree** |
| 55,000 | 15 | 3 |
| 60,000 | 15 | 3 |
| 200,000 | 15 | 3 |

The agreement existed only because the declared floor landed in a 5,000-wide window
admitting exactly two cells. `agreement_is_floor_robust: false`. The sweep is persisted in
the artifact so no reader repeats the error of reading a floor as a threshold.

**What R2r actually shows is not a ranking.** Below 49,000 units of unresolved backorder,
**zero of eighteen cells qualify.** Under this risk family at full escalation no posture in
the set delivers acceptable service, and reporting a winner would mean reporting the best
of a set that should not be deployed.

## Pareto front (resource ↓, fill ↑, lost ↓)

- **R1r:** `168/168/168|S1`, `ddmrp|S1`, `ddmrp|S2`, `ddmrp|S3`
- **R2r:** `168/0/168|S1`, `168/168/168|S1`

DDMRP is non-dominated at all three shift levels in R1r — it buys the highest fill in the
panel, at higher cost. It loses on every scalar metric and is still on the front, which is
the case for reporting the front.

## What this does not establish

Development tapes, four per cell, one risk corner per family, no paired intervals. The
comparison set is five buffer postures, not the full 216 — the v2 comparator instrument
owns that, and its MPC arm is deliberately excluded here because that run belongs to
another session. `κ̇` remains set-relative, so no number here is comparable to any table
with a different set. Unit costs are `c = 1`, Garrido's own §3.1 assumption (6), not costs
calibrated for this DES.
