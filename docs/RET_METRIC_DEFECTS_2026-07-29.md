# Two defects in the ReT endpoint

**Status:** `DEVELOPMENT_FINDING`. Both encoded as `xfail(strict=True)` in
`tests/test_ret_metric_invariants.py`, so they fail loudly the day either is fixed.
Neither is a defect of any controller or screen; both are properties of the endpoint
and of the DES, and both affect how existing artifacts may be read.

## 1. `ret_excel` depends on how often `step()` is called

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

**Repair:** derive `RPj` from immutable risk intervals independent of `step()`, then
make `test_ret_excel_is_step_cadence_invariant` pass.

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

### Is 54 faithful, though?

Probably yes, and that is the uncomfortable part. Our R1r ReT (0.0038-0.0056) sits in the
same regime as the thesis's own R1r (0.0052-0.0087). At delay = 48 we would produce
~0.98, which matches nothing in his data. So the recovery-dominated regime **is** what
Garrido's numbers look like, and the 54 h floor is what reproduces them.

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

**Do not "fix" the 54 to 48.** It would break agreement with the thesis. The open question
is whether the ReT *specification* should be used as a primary endpoint at all when it is
discontinuous six hours from the operating point — which is an argument for the panel, and
an independent argument for the Cobb-Douglas index, which has no such branch.
