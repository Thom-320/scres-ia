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

---

# Addendum — `ret_excel` is step-cadence dependent, and the v2 fold

## The cadence defect (found while building the fold)

Replaying a v2 MPC arm daily instead of at v2's 4-week epoch failed the replay gate on
24 of 24 arms by ~29%. Diagnosis, on one identical trajectory:

| step cadence | ret_excel | full ledger | fill | delivered |
|---|---:|---:|---:|---:|
| one step (8,736 h) — *buffer gate* | 0.004369 | 0.004353 | 0.99650 | 689,182 |
| 672 h — *v2 comparator* | 0.004369 | 0.004353 | 0.99650 | 689,182 |
| 168 h | 0.004401 | 0.004386 | 0.99650 | 689,182 |
| 24 h — *panel, C-D screens* | 0.005623 | 0.005603 | 0.99650 | 689,182 |
| 1 h | 0.005981 | 0.005960 | 0.99650 | 689,182 |

**The physics is invariant** — identical fill, identical delivered rations, identical
risk events, and `OPTj`, `OATj`, `APj` identical in all 311 scored orders. **`RPj`
differs in 175 of 311**, and `RPj` enters ReT through the `0.5/RPj` recovery branch.
Suspected mechanism: `_cumulative_down_hours` is advanced inside `step()`
(supply_chain.py:1852-1866), closing open downtime intervals at step boundaries.

Consequences, all now enforced in code:

- `ret_excel` numbers are comparable **only** across artifacts at the same `step_hours`.
  The panel now records `step_hours` and a `cadence_warning`.
- The panel was regenerated at 672 h to match v2.
- **Winners were verified stable** across 24 h and 672 h — all eight, both families —
  but the full rank order was not: only 2 of 18 positions held in R1r.

## The v2 fold, paired within tape

The v2 arms ran seeds 1,430,001+/1,530,001+; the panel ran 1,620,001-4. Merging those
directly would compare arms across different exogenous streams. Instead every reference
posture is **re-evaluated on each shard's own materialised tape**, so all comparisons are
paired. v2 pins `shifts=1`, so only S1 cells can be paired.

Partial run: 4 tapes R1r, 8 tapes R2r, of 12 each.

### Paired deltas, MPC minus reference (LCB95, bootstrap 10,000)

| family | reference | ret_excel | full ledger | cvar10 |
|---|---|---|---|---|
| R1r | 168/0/168 | **+0.000074** (LCB +0.000008, 4/4) | **+0.000074** (+0.000008, 4/4) | +0.000025 (−0.000015, 3/4) |
| R1r | 672/0/1344 | **+0.000116** (+0.000050, 4/4) | **+0.000115** (+0.000049, 4/4) | **+0.000322** (+0.000242, 4/4) |
| R2r | 168/0/168 | **+0.008194** (+0.004010, 6/8) | +0.000530 (−0.003278, 4/8) | −0.000004 (−0.000040, 5/8) |
| R2r | 672/0/1344 | **+0.015984** (+0.013369, 8/8) | **−0.008033** (−0.011608, **0/8**) | −0.000017 (−0.000055, 5/8) |

**In R1r the corrected MPC beats every static posture, and the advantage generalises**
across ret_excel, the uncensored ledger and (against the gate incumbent) the tail. It
also costs less: κ 450,898 against the incumbent's 469,333.

**In R2r it does not generalise, and the reason matters.** The MPC *optimises*
`ret_excel`, so winning that column is close to tautological. On the uncensored ledger it
**loses to 672/0/1344 on 0 of 8 tapes** (−0.008, LCB −0.0116) and to DDMRP by −0.0127.
The tail is flat to slightly negative. The advantage is specific to the metric being
optimised and reverses on the metric that is not.

This is precisely what the panel exists to catch, and it is the reason a single-metric
adjudication of the MPC arm would have been wrong in one of the two families.

Under `R_cobb_douglas` the MPC ranks **second in both families**, behind only
`greedy_pi_best_found_v2` — the clairvoyant oracle, which is a ceiling and not a
deployable policy.

## Corrections adopted from review

1. "S3 buys zero service" → **"S3 does not improve the aggregate service-quantity
   endpoints."** `tau` does change, and `fill_rate_on_time` is identically 0 in every
   cell — itself a defect worth a separate look, since a metric that is constant across
   18 cells is measuring nothing.
2. 49,000 is an **exploratory cap**, not a validated operational threshold. The defensible
   statement is "no cell met the exploratory backorder cap of 49,000", not "nothing is
   deployable".
3. **`agreement_is_floor_robust` was a real bug** — it compared the *count* of distinct
   winners, not the winners. Both families happened to give the right answer. Now compares
   the exact winner vector.
4. "DDMRP is exactly static in R1r" holds **only at S1**; S2 and S3 show three postures
   and two changes.
5. The Pareto front is specifically **(κ↓, fill↑, lost↓)** and is renamed
   `pareto_front_kappa_fill_lost_only`, with excluded axes listed.
6. Per-tape rows are now persisted in both the panel and the fold; the intervals above
   are computed from them.
7. Five postures, not 216 — the buffers×shifts crossing is exploratory and establishes no
   optimum of the expanded contract.
