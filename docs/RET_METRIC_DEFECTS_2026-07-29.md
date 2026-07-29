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

- `fill_rate_on_time` is **structurally zero**. It cannot discriminate between any two
  policies and should not be reported as a service constraint until the floor is
  understood.
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

**This is a fidelity gap in the thesis's central resilience mechanism**, and it is more
consequential than the censoring already documented. Open question for the repair: what
sets the 54 h floor, and is it a modelling artifact or a faithful property of the case.
