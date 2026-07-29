# ReT endpoint audit and cadence correction

**Status:** `DEVELOPMENT_CORRECTIVE_AUDIT`. The RPj cadence carrier is fixed and
covered by passing invariance tests. The fulfilment-delay/autotomy issue remains an
open fidelity and measurement question, encoded as two `xfail(strict=True)` tests.
Historical artifacts retain their original code hash and metric semantics.

## 1. `ret_excel` cadence defect — resolved prospectively

Found when a daily replay of v2's MPC arms failed the fold's replay gate on 24 of 24
arms by ~29%. On **one identical trajectory**:

| step cadence | ret_excel | full ledger | fill | delivered |
|---|---:|---:|---:|---:|
| one step (8,736 h) — buffer gate, 90-config reproduction | 0.004369 | 0.004353 | 0.99650 | 689,182 |
| 672 h — v2 comparator | 0.004369 | 0.004353 | 0.99650 | 689,182 |
| 168 h | 0.004401 | 0.004386 | 0.99650 | 689,182 |
| 24 h — metric panel, Cobb-Douglas screens | 0.005623 | 0.005603 | 0.99650 | 689,182 |
| 1 h | 0.005981 | 0.005960 | 0.99650 | 689,182 |

37% spread, monotone in cadence. Physics invariant: identical fill, delivered rations
and risk events, and `OPTj`/`OATj`/`APj` identical in all 311 scored orders. **`RPj`
differs in 175 of 311.**

**Mechanism, corrected.** An earlier draft blamed `_cumulative_down_hours`; that is a
diagnostic accumulator and is not in the path. The carrier is `_op_down_since`:

1. `step()` advances the accumulator and then **resets `_op_down_since[op] = env.now`**
   at every step boundary (`supply_chain.py:1856`, `:1865`).
2. The legacy ongoing-disruption attribution measures a still-open disruption as
   `overlap_start = max(down_since, OPTj)` using that reset value
   (`supply_chain.py:5743`).
3. `RPj` receives those hours (`supply_chain.py:5811`) and ReT uses `0.5/RPj`.

More `step()` calls → more recent `down_since` → smaller overlap → smaller `RPj` →
**larger ReT**. That predicts the observed monotonicity exactly.

### What this invalidates, and what it does not

- **Invalidates:** comparing `ret_excel` across artifacts produced at different
  cadences.
- **Does not invalidate:** paired comparisons where every arm shares one cadence. The
  H2/H3 confirmation, Program Q (fixed weekly cadence) and the v2 comparator each hold
  internally.
- **Winners were verified stable** across 24 h and 672 h — all eight, both families —
  but only 2 of 18 rank positions held in R1r. Never quote a full ordering across
  cadences.

**Repair completed.** Completed disruptions are read from their immutable
`RiskEvent(start_time,end_time)` rows. A still-open disruption is read from
`_op_down_start`, which records its onset once and is never advanced by `step()`;
`_op_down_since` remains only the cumulative-hours cursor.

Corrective artifact:
`results/metric_audit/ret_cadence_corrective_v2/result.json`.

| step cadence | corrected ret_excel | orders with different RPj |
|---|---:|---:|
| one step | 0.004424198300 | 0 |
| 672 h | 0.004424198300 | 0 |
| 168 h | 0.004424198300 | 0 |
| 24 h | 0.004424198300 | 0 |
| 1 h | 0.004424198300 | 0 |

The corrected spread is exactly `1.0`. Physical endpoints, RPj order by order,
`ret_excel`, and `ret_excel_full_ledger` are invariant in the audit. This correction
does not relabel v2: its historical `ret_excel` remains the output of commit
`1dc40c1`; corrected replays live in a new artifact.

## 2. The autotomy branch of ReT is unreachable

Investigating why `fill_rate_on_time` is identically 0 across all 18 panel cells. It is
not a metric bug — `LTj` is correctly 48 for all 307 served orders, and the predicate
`CTj <= LTj` is correct.

**The minimum cycle time is 54 h against a 48 h lead-time promise, and no lever moves
it:**

| configuration | n | CTj min | CTj p50 | CTj ≤ 48 | autotomy % | fill_rate_on_time |
|---|---:|---:|---:|---:|---:|---:|
| risks off | 309 | 54.0 | 54.0 | **0** | 0.00 | 0.0000 |
| risks on, R1r escalated | 308 | 54.0 | 54.0 | **0** | 0.00 | 0.0000 |
| risks off, 3 shifts | 310 | 54.0 | 54.0 | **0** | 0.00 | 0.0000 |
| risks off, maximum buffer | 309 | 54.0 | 54.0 | **0** | 0.00 | 0.0000 |

Consequences:

- `fill_rate_on_time` is **saturated at zero** and does not discriminate between policies
  in this panel. The metric itself is computed correctly. `CTj − LTj` is **exactly +6.0 h
  at min, p05, p50 and p95** (301 of 310 orders sit at CTj = 54.0), so this is a uniform
  6-hour overshoot, not a dispersion of late orders.
- **`excel_case_pct_autotomy` is 0.00 in every configuration.** Autotomy — a disruption
  occurs and the order still arrives on time — is the thesis's central absorption
  mechanism, and our DES cannot express it. The `APj/LT` branch of ReT is dead code;
  every scored order takes the `0.5/RPj` recovery branch.
- This **explains mechanically** the previously recorded "100% of cases collapse to
  recovery under risk". That collapse is not risk-induced. It is structural: with risks
  off the orders sit in the no-disruption case, and the moment any risk touches an
  order it must land in recovery, because on-time delivery is impossible.

### Fidelity gap

Garrido's own output (`Rsult_1.xlsx`, Cf1) reports **`Media APj` = 0.4486** — positive,
so autotomy cases occur in his model. Ours produce zero, always.

Caveat on the source: `Rsult_1.xlsx` is **not** the thesis's final data — its 12
configurations differ from the thesis row counts by −1,949 to +735. It is still
Garrido's own model output, which is what the question needs: his system reaches the
branch, ours cannot.

## 3. What sets the 54 h floor — answered

`supply_chain/config.py:119`:

```python
GARRIDO_FULFILLMENT_DELAY_HOURS = 54.0  # Calibrated minimum CTj: no instant orders; just beyond LT=48.
```

It is a hardcoded calibration constant — `demand_on_hand_fulfillment_delay`, the delay
applied when demand is met from **on-hand stock** — not emergent physics. Its own comment
says it was placed *"just beyond LT=48"*. Being just beyond is what makes `CTj <= LTj`
unsatisfiable.

**ReT is effectively binary in this constant.** Custodied sweep
(`results/metric_audit/ret_defects_v1/result.json`):

| delay | ret_excel | autotomy % | CTj ≤ 48 | APj > 0 |
|---:|---:|---:|---:|---:|
| **54 (shipped)** | **0.004424** | 0.00 | 0 | 0 |
| 48 | 0.980513 | 98.05 | 301 | 301 |
| 47 | 0.980576 | 98.05 | 302 | 302 |
| 36 / 24 / 6 | 0.980576 | 98.05 | 302 | 302 |

**A six-hour change moves ReT by 221.6×**, and everything below the promise saturates to
the same value. The metric sits on a cliff and the shipped constant is six hours on the
far side of it.

### Is 54 faithful?

**Not established.** Delay 54 reproduces the aggregate order of magnitude of the
raw-Excel ReT, but that is endpoint calibration, not independent behavioral
validation. The workbook formula gates on `APj > 0` and does not require
`CTj <= LTj`; the DES currently creates APj only under that timing predicate. Matching
the aggregate level therefore does not resolve the micro-semantic divergence.

The defensible reading is therefore not "the constant is wrong" but:

1. ReT, as specified, is **discontinuous** in the fulfilment delay at exactly `LTj`, and
   our operating point is six hours from that discontinuity.
2. In that regime the autotomy branch is dead, so **essentially all of ReT's signal flows
   through `RPj`** — precisely the quantity defect 1 makes step-cadence dependent.
3. Garrido's own Cf1 reports `Media APj` = 0.4486 > 0, so his data is not *perfectly*
   autotomy-free. Our zero is stricter than his.

That composition is the real result: the endpoint every null in this project was measured
against is pinned to one branch by a hand-set constant, and that branch's only free
quantity is cadence-dependent.

**Do not change 54 retrospectively in existing experiments.** A successor contract
must select delay/APj semantics from fidelity evidence, not from which value preserves
more headroom. The full 216-posture delay diagnostic is separately labelled as
perfect-hindsight selection of one fixed posture per tape; even a zero there cannot
exclude state-contingent adaptation between epochs.

---

## 3. `RPj` attributes time, but delays can be caused by quantity risks

Found auditing tape R2r/1530011, where the MPC's per-tape `ret_excel` delta was −0.2634
against a family mean of −0.0111. Independently reproduced; custody in
`results/metric_audit/r2r_1530011_ret_tail_v1/`.

The gap is one order. Static incumbent `(336,0,168)`, scored population 262 orders:

```
order j=24    delivered 240 h after ordering = 5.0x the 48 h promise, 192 h LATE
              RPj = 0.006765 h = 24.4 seconds
              risk_indicators = {R23: 0.006765 (hours), R24: 2516.0 (rations)}
              ReT = 0.5 / RPj = 73.9082          <- the episode's HIGHEST value
```

**The most-delayed order in the episode receives the best possible score, by a factor of
74, on a metric defined on [0,1].**

The mechanism is not simply that `0.5/RPj` is unbounded as `RPj → 0`. It is that **`RPj`
is a time attribution while the delay was caused by a quantity risk.** `R24` is a
quantity risk — `_ret_quantity_risk_units = {"R14": 0.0, "R24": 0.0}` — so its 2,516
missing rations contribute **zero hours**. `RPj` drew only from `R23`'s 24-second overlap.
The more a delay is driven by shortfall rather than downtime, the *higher* it scores.
Monotonicity is inverted on exactly the worst-served orders.

This is orthogonal to defect 1 and survives its repair: the audit above was run with the
immutable-onset correction in place.

### Scope, measured over all 24 v2 tapes at each family's true incumbent

| family | scored orders | ReT > 1 | max ReT | mean inflation vs clipped |
|---|---:|---:|---:|---:|
| R1r `(0,0,336)` | 3,279 | **0** (0.00%) | 0.07 | 1.00x |
| R2r `(336,0,168)` | 3,108 | **7** (0.23%) | 73.91 | 1.06x |

**R1r is clean.** Its results are uncontaminated by this defect. **R2r is not**, and R2r
is precisely the family whose escalated set contains `R24` — the quantity risk. Seven
orders in 3,108 inflate the family mean by 6% and, on one tape, move a paired delta by
0.275.

### Consequence for the tape-1530011 verdict

| contrast | value |
|---|---:|
| raw `ret_excel`, MPC − static | **−0.263374** |
| leave the single maximum out of each arm | **+0.011164** |
| clip both arms to the metric's own [0,1] range | **+0.014757** |
| difference of medians | **+0.100000** |
| flow fill, MPC − static | **+0.003484** |

The frozen primary verdict is not changed here — the preregistered endpoint is
`ret_excel` as specified. But it can no longer be described as a physical failure of the
MPC: on that tape the MPC delivers **better** fill, zero lost orders, and a higher median
per-order ReT. Every robust statistic points the other way from the mean.

Two defects compound on this tape, and the second is the one already documented: the
warm-up is endogenous and **differs by arm** (631 h static against 943 h MPC), so the
scored populations differ too (262 against 253). Policy-dependent censoring sits on top
of the unbounded tail.

**Repair options, none taken yet:** clip to [0,1] as the metric's own definition implies;
or floor `RPj` at a physical quantum; or attribute quantity-risk delay to `RPj` in time
units. Each changes historical numbers, so none should be applied without preregistration.
