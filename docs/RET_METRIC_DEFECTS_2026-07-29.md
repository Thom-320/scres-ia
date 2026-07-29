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

**CORRECTED 2026-07-29.** An earlier draft called `j=24` "the most-delayed order in the
episode" receiving "the best possible score". Both are false: `j=62` on the same tape
reaches `CTj = 5,856 h`, and the formula is unbounded so there is no best possible score.
The verified statement is stronger than the false one:

| | most-delayed order | highest-scoring order |
|---|---|---|
| **R1r** (3,279 orders) | j=37, `CTj` 1,344 h → ReT **0.000462** | j=35, `CTj` **54 h** → ReT 0.0712 |
| **R2r** (3,108 orders) | j=34, `CTj` 7,152 h → ReT **0.000295** | j=24, `CTj` 240 h → ReT **73.9082** |

In both families they are **different orders**, and the highest-scoring one is among the
*least* delayed — in R1r it is at `CTj = 54 h`, the physical minimum. On tape 1530011 an
order 24x less late than the worst scores 148,000x higher. **ReT is not monotone in
lateness, in either direction.**

The mechanism is a **dimensional incompatibility**, confirmed: `R24` is a demand shock
measured in rations — `_ret_quantity_risk_units = {"R14": 0.0, "R24": 0.0}` — while `RPj`
accumulates hours, and `ret_thesis.py:125` takes `0.5/RPj` with no clipping. An order
carrying an R24 quantity indicator can therefore enter the recovery branch with a nearly
null *temporal* `RPj`.

**Not demonstrated, and corrected from an earlier draft:** that R24 *caused* the 192 h
delay. The `2516` is the recorded surge magnitude, an indicator, not a causal path — this
contract uses retrospective event attribution. The defensible statement is: *the order
combines an R24 quantity indicator with only 0.006765 h attributable to temporal risks,
and that incompatibility lets an order 192 h late receive ReT = 73.91. R24's exact causal
contribution to the delay is not identified.*

This is orthogonal to defect 1 and survives its repair: the audit above was run with the
immutable-onset correction in place.

### Scope, measured over all 24 v2 tapes at each family's true incumbent

| family | scored orders | ReT > 1 | max ReT | mean inflation vs clipped |
|---|---:|---:|---:|---:|
| R1r `(0,0,336)` | 3,279 | **0** (0.00%) | 0.07 | 1.00x |
| R2r `(336,0,168)` | 3,108 | **7** (0.23%) | 73.91 | 1.06x |

**R1r is clean of this tail** — and only of this tail, at the static incumbent, on these
twelve tapes. It absolves no other arm, posture, censoring effect, or ReT defect.
**R2r is not**, and R2r
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

**Repair options, none taken yet.** Preregistration drafted in
`docs/RET_REPAIR_PREREGISTRATION_2026-07-29.md`; measurements of all three below.

Custody for the family sweep and the per-arm warm-up:
`results/metric_audit/ret_tail_family_sweep_v1/`, runner
`scripts/audit_ret_tail_family_sweep.py`. Warm-up on tape 1530011 is **631 h** for the
static incumbent against **943 h** for the MPC, so the scored populations differ and
policy-dependent censoring compounds on top of the tail.

### What each repair produces

Measured, not argued: `scripts/audit_ret_repair_variants.py`, artifact
`results/metric_audit/ret_repair_variants_v1/result.json`. All 24 v2 tapes, MPC against
each family's true 216-posture incumbent, paired within tape, bootstrap 10,000.
**No repair is applied to the canonical metric.** Variants are produced by mutating the
inputs and re-calling the *official* ledger; the untouched variant must reproduce
`compute_episode_metrics`' `ret_excel` to 1e-9 or the run aborts. It did, in all 48 runs.

**R1r — every repair is a no-op.**

| variant | Δ (MPC − static) | CI95 | verdict |
|---|---:|---|---|
| all seven | **+0.000000** | [−0.00001, +0.00002] | `NOT_SEPARATED` |

Zero orders out of range, so there is nothing for a repair to change. R1r's verdict is
robust to all three repairs and to a 48x floor sweep.

**R2r — every repair flips the verdict.**

| variant | Δ (MPC − static) | CI95 | tapes | orders > 1 | max | verdict |
|---|---:|---|---:|---:|---:|---|
| **canonical** | **−0.011129** | [−0.05866, +0.01520] | 11/12 | 7 | 73.91 | `NOT_SEPARATED` |
| clip to [0,1] | **+0.011951** | [+0.00781, +0.01655] | 12/12 | 0 | 1.00 | **`MPC_AHEAD`** |
| quantity → time | **+0.013141** | [+0.00860, +0.01827] | 11/12 | 5 | 2.01 | **`MPC_AHEAD`** |
| quantity → time, then clip | **+0.013095** | [+0.00867, +0.01810] | 11/12 | 0 | 1.00 | **`MPC_AHEAD`** |
| `RPj` floor 0.5 h | +0.011951 | [+0.00774, +0.01660] | 12/12 | 0 | 1.00 | **`MPC_AHEAD`** |
| `RPj` floor 1 h | +0.012067 | [+0.00782, +0.01662] | 12/12 | 0 | 0.97 | **`MPC_AHEAD`** |
| `RPj` floor 6 h | +0.012847 | [+0.00831, +0.01768] | 11/12 | 0 | 0.97 | **`MPC_AHEAD`** |
| `RPj` floor 24 h | +0.012961 | [+0.00845, +0.01769] | 11/12 | 0 | 0.97 | **`MPC_AHEAD`** |

Three conceptually independent repairs, plus a **48x floor sweep**, all agree: Δ between
+0.0120 and +0.0131, CI entirely above zero. The floor sweep is the check the
Cobb-Douglas floors originally failed, and here it passes — the answer does not move with
the decision.

**So R2r's `NOT_SEPARATED` is produced by 7 orders in ~3,100.** Removing the
out-of-range tail by any principled route reverses it.

Two further findings:

- **`clip_0_1` and `rpj_floor_0.5` are numerically identical** (0.453447 / 0.441495).
  They should be — flooring `RPj` at 0.5 makes `0.5/RPj <= 1` by construction. Two
  independently written repairs coinciding exactly where they must is a check on both.
- **`quantity_time` alone is not sufficient.** It still leaves 5 orders above 1.0
  (max 2.01), because it only touches orders a quantity risk reached; an order with a
  tiny `RPj` from a pure timing risk stays unbounded. The prospectively registered
  sensitivity therefore applies that proxy and then clips. It remains a disclosed
  proxy, not an exact causal attribution model.

### What may and may not be claimed

**May not:** that the MPC beats the static incumbent in R2r. The preregistered endpoint
is `ret_excel` as specified, and under it the verdict is `NOT_SEPARATED`. Switching
metric after seeing the result is the Program G error.

**May:** that the R2r verdict is **not robust to a documented defect of the endpoint**,
and that every principled repair reverses it. That is grounds for preregistering a
repair — with the sign of the outcome already known and declared, which is exactly why
the preregistration has to be written before the rerun and not after.

**Unchanged either way:** R1r, where MPC and the true incumbent are not separated under
any variant.
