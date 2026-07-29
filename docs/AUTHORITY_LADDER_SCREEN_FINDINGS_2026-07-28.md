# Authority-ladder cheap screens — findings, 2026-07-28

**Status:** `DEVELOPMENT_SCREEN_NO_CLAIM`. Development tapes only. No confirmation universe
was opened, no network was trained, no gate was declared passed.

Runners: `scripts/screen_procurement_authority_a1.py`, `scripts/screen_buffer_authority_gate.py`.
Artifacts: `results/authority_ladder/a1_procurement/screen_result.json`,
`results/authority_ladder/buffer_gate/screen_result.json`.

Protocol common to everything below: metric `ret_excel`, 52-week horizon, `shifts=1`,
`risks_enabled=True`, `risk_level="current"`, `strict_exogenous_crn=True`, thesis-faithful
year basis / warm-up / R14 mode, 12 tapes from seed 1,210,001 (procurement) and 1,220,001
(buffers).

## Why this exists

Garrido asked on 2026-07-28 to widen the decision surface rather than lengthen the episode,
and authorised buffers at unmodelled nodes upstream and downstream by name. He had already
prescribed the cheap precondition on 2026-07-03: freeze dispatch, vary only buffer targets
under common random numbers, and check for headroom **before** building or training
anything. These screens are that precondition, run on the families that need no new physics.

`H_PI` below is a **diagnostic ceiling**, not learner opportunity. The clairvoyant selects
per tape using outcomes no deployable policy observes, so `H_PI > 0` never authorises
training. A near-zero `H_PI` is nonetheless decisive downward: no policy of any kind can
exceed it.

## 1. Procurement (A1) is structurally void — and we can say why

`op1_rop`, `op2_rop`, `op2_q` already live in `MFSCSimulation.params` (lines 623/625/627)
and are re-read by `_op1_contracting` / `_op2_supplier_delivery` on every cycle. No Gym
contract has ever written them. A 125-point grid (0.5–2.0x base on each key) x 12 tapes:

| `procurement_contract_mode` | static bar | clairvoyant ceiling | `H_PI` | LCB95 |
|---|---:|---:|---:|---:|
| `legacy_independent` | 0.005157 | 0.005255 | 9.80e-05 | 5.64e-05 |
| `causal_coupled`     | 0.004970 | 0.005067 | 9.64e-05 | 4.85e-05 |

Two liveness observations make the null mechanistic rather than merely small:

- **`op2_q` moves 4.56e+06 units of external raw material for exactly 0.0 ReT change**, in
  both contract modes.
- **`op1_rop` is bit-identical at 0.5x / 1x / 2x when risks are off** (ReT, fill, raw
  material all unchanged). With risks on it does move ReT — but by *the same amount to six
  digits in both contract modes*, while raw material differs by ~450k units. The contract
  coupling demonstrably works and moves material; ReT ignores the material and responds only
  to a risk-phase side channel (the Op1 downtime loop shifting event timing). That is not
  procurement authority.

### The mechanism: the thesis operating point carries ~5x raw-material slack

Sweeping `op2_q` down from nominal, everything else fixed, seed 1,210,001:

| `op2_q` | ReT (risks off) | RM left at WDC | RM consumed | ReT (risks on) |
|---:|---:|---:|---:|---:|
| 1.00x | 0.718865 | 4,098,480 | 8,615,040 | 0.004689 |
| 0.50x | 0.718865 | 1,818,480 | 8,615,040 | 0.004689 |
| 0.30x | 0.718865 | 906,480 | 8,615,040 | 0.004689 |
| 0.25x | 0.718865 | 678,480 | 8,615,040 | 0.004689 |
| 0.20x | 0.718865 | 450,480 | 8,615,040 | 0.004689 |
| 0.15x | 0.654207 | 127,392 | 7,739,376 | 0.004783 |
| 0.10x | 0.465972 | **0** | 5,262,144 | 0.004345 |
| 0.05x | 0.243067 | **0** | 2,736,000 | 0.003558 |
| 0.02x | 0.104550 | **0** | 1,094,400 | 0.003664 |

ReT is **exactly flat** from 1.00x down to 0.20x, and consumption is pinned at 8,615,040 —
the quantity downstream capacity can absorb. Raw material only becomes binding below ~0.20x,
and by 0.10x the buffer is empty and the system is starving, not being optimised.

**Reportable statement.** Upstream quantity and cadence have no resilience authority anywhere
within a factor of five of the thesis operating point, because raw-material supply exceeds
the binding requirement by roughly 5x. The constraint is downstream, consistent with the
long-standing F11 finding. Adding upstream supplier nodes cannot change this without first
removing the slack — which would be changing the case, not widening the decision surface.

## 2. Buffer gate: level matters, contingency does not

Garrido's prescribed gate. Dispatch never written; 216 postures over `op3_rm`, `op5_rm`,
`op9_rations` (zero plus the five thesis ladder levels of Table 6.16) x 12 tapes.

| quantity | value |
|---|---:|
| static bar (best fixed posture) | 0.00609476 |
| clairvoyant ceiling | 0.00621055 |
| `H_PI` | 1.158e-04 |
| `H_PI` LCB95 | 7.78e-05 |
| **ReT span across the grid** | **1.204e-03** |
| best posture | `op3_rm` 61,440 · `op5_rm` **0** · `op9_rations` 126,000 |
| `best_is_all_max` | **False** |

Two things follow.

**Static sizing carries ~10x more value than adapting it.** The spread between the best and
worst fixed buffer posture is 1.20e-03, an order of magnitude larger than the 1.16e-04 a
clairvoyant gains by retuning per tape. This is the DDMRP question — where and how much —
answered in favour of getting the level right once, not in favour of contingency.

**Monotonicity does not hold.** The optimum is interior and `op5_rm` wants to be **zero**.
This refutes the intuition (which I had written into the plan) that under a free buffer
mechanism more buffer always wins. It does not, even here.

### Standing caveat on the mechanism

`_deliver_buffer_top_up` (`supply_chain.py:1137-1160`) satisfies a shortfall with an
unmatched `container.put(shortfall)`. There is no upstream `.get()`, containers are
effectively unbounded, and the injected quantity carries no price. **Mass still balances** —
`flow_ledger()` (`supply_chain.py:1956-1970`) counts `total_strategic_raw_injected` and
`total_strategic_rations_injected` as source terms and the residual is exactly zero, with
tests asserting it. So this is *accounted exogenous replenishment*, not an arithmetic leak;
calling it "broken conservation" would be wrong and reviewer-bait. The defect is that the
source is exogenous, uncapacitated and unpriced, which biases the screen toward large
buffers. A null measured under so generous a mechanism is therefore conservative.

## 3. The metric itself loses ordinal discrimination under risk

This is the most consequential finding of the session, and it reframes every headroom null
in the project.

Shift posture set at construction, no action written, 8 tapes from seed 1,220,001:

| risks | shift | `ret_excel` | `ret_thesis` | `ret_excel_full_ledger` | `ret_continuous` | fill | delivered |
|---|---|---:|---:|---:|---:|---:|---:|
| off | 1 | 0.721573 | 0 | 0.406049 | 1.000000 | 0.89361 | 619,297 |
| off | 2 | 0.812277 | 0 | 0.451835 | 1.000000 | 0.99642 | 693,043 |
| off | 3 | 0.812277 | 0 | 0.451835 | 1.000000 | 0.99642 | 693,043 |
| **on** | 1 | **0.005148** | 0.004193 | 0.004193 | 0.203594 | 0.81566 | 587,952 |
| **on** | 2 | **0.005113** | 0.004809 | 0.004809 | 0.206143 | 0.93980 | 681,035 |
| **on** | 3 | **0.005135** | 0.004935 | 0.004935 | 0.208960 | 0.96122 | 696,398 |

With risks off, `ret_excel` orders the postures correctly. With risks **on** it goes
non-monotone and nearly flat — it ranks shift 1 **first**, although shift 3 delivers 18% more
rations, drops lost orders to zero and gains 14.6 fill points. `ret_thesis`,
`ret_excel_full_ledger` and `ret_continuous` all stay correctly monotone.

### Mechanism: policy-dependent censoring plus case collapse

| risks | shift | visible_n | omitted_n | % omitted | `excel_case_pct_recovery` |
|---|---|---:|---:|---:|---:|
| off | 1 | 248.5 | 28.5 | 10.29% | 0 |
| off | 2/3 | 277.0 | 1.0 | 0.36% | 0 |
| on | 1 | 225.4 | 51.5 | **18.60%** | **100** |
| on | 2 | 261.2 | 16.8 | **6.03%** | **100** |
| on | 3 | 267.1 | 10.9 | **3.91%** | **100** |

Two failures at once. Every case collapses into the `recovery` bucket, so the case
decomposition carries no information. And the censored fraction is a **function of the policy
being evaluated** — the worst posture hides 18.6% of its orders, the best hides 3.9%. Each
policy is scored on a different population, and the censoring systematically flatters the
worse policy. The same effect appears in the buffer family: the omitted-order fraction spans
**18.44 percentage points** across the 216-posture grid.

**Scope limit, and it matters.** Our implementation is faithful — 0/47,546 formula mismatches
against the three real Excel workbooks. This is a property of the metric **as specified**, not
a coding defect. It is also precisely the ground on which Garrido already authorised the
Cobb-Douglas formulation from his 2024 paper as a fallback "for when normal resilience is not
enough". That fallback is not a convenience; this is the evidence that it is required.

### Does the censoring manufacture the nulls? No — and that strengthens them

Both screens recomputed on both metrics over the identical runs:

| screen | metric | static bar | `H_PI` | LCB95 | grid span |
|---|---|---:|---:|---:|---:|
| buffers (216 x 12) | `ret_excel` | 0.00609476 | 1.158e-04 | 7.78e-05 | 1.204e-03 |
| buffers | `ret_excel_full_ledger` | 0.00607120 | 1.085e-04 | 7.07e-05 | **2.026e-03** |
| procurement `legacy` (125 x 12) | `ret_excel` | 0.00515673 | 9.801e-05 | 5.64e-05 | 5.823e-04 |
| procurement `legacy` | `ret_excel_full_ledger` | 0.00427002 | 7.780e-05 | 3.23e-05 | **1.021e-03** |
| procurement `causal` | `ret_excel` | 0.00497022 | 9.639e-05 | 4.85e-05 | 3.958e-04 |
| procurement `causal` | `ret_excel_full_ledger` | 0.00392286 | 7.272e-05 | 3.52e-05 | **6.739e-04** |

The buffer screen returns the same optimal posture under both metrics. **Every headroom null
survives uncensoring, and on the uncensored metric procurement headroom is smaller still.**
So the nulls are not artefacts of the censoring — the censoring, if anything, was overstating
them.

What the censoring *does* destroy is dynamic range: the uncensored metric spans 68% more in
the buffer grid and 70–75% more in the procurement grids. That is signal the censored metric
throws away, which is why it cannot order the shift postures.

Both metrics are reported from now on and neither is chosen after seeing the result —
selecting the friendlier metric post hoc is exactly the Program G error.

## 4. Instrument defects found and fixed

- **`step()` now fails closed on unknown action keys** (`supply_chain.py:1765+`,
  `_PSEUDO_ACTION_KEYS` at 1114). Previously any key absent from `self.params` was dropped
  in silence.
- **`scripts/run_program_i_sensitivity.py:73` sends `op8_rop`**, which is not a `params`
  key — it is only a `CAPACITY_BY_SHIFTS` config field that `supply_chain.py` never reads.
  The `op7_release_period` factor was therefore a guaranteed no-op, and
  `results/program_i/morris/verdict.json` reports `mu = mu_star = sigma = 0.0` with
  `sign_stability = 1.0` for it. **That zero is by construction, not by physics.**
  It must be invalidated, not reinterpreted: mark it `GSA_NOT_TESTED_DUE_TO_UNKNOWN_KEY`.
  Note this does not by itself retract the catalogue's `transition_dead_configuration_field`
  classification, which also rests on static code evidence — but the two lines of evidence
  must stop being conflated, and the Program I index cannot count as physical evidence.
- **`dev_notes/CLAUDE.md`** carried `Track B — PPO wins / VALIDATED` with no superseded
  banner, and described the 8D contract as 7D. Corrected against C30 and against
  `env_experimental_shifts.py:2795-2818`.

## 5. Ground truth from Garrido's own workbooks

`Rsult_1.xlsx` (sheets `Cf1`–`Cf12`, `APj`, `RPj`, `DPj`, `Re`), plus `Raw_data1+Re.xlsx`
(`CF1`–`CF10`) and `Raw_data2+Re.xlsx` (`CF11`–`CF20`). The `Re` sheet holds 2,520 per-order
resilience values for each of the twelve configurations.

| configuration | n | mean `Re` | sd | max |
|---|---:|---:|---:|---:|
| Cf1 | 2520 | 0.01480558 | 9.468e-02 | 1.000155 |
| Cf2 | 2520 | 0.01228824 | 8.154e-02 | 1.000155 |
| Cf3 | 2520 | 0.01459251 | 9.139e-02 | 1.000155 |
| Cf4 | 2520 | 0.01200922 | 7.721e-02 | 1.000155 |
| Cf5 | 2520 | 0.01265754 | 8.016e-02 | 1.000155 |
| Cf6 | 2520 | 0.01169934 | 7.817e-02 | 1.000155 |
| Cf7 | 2520 | 0.01309959 | 8.477e-02 | 1.000155 |
| Cf8 | 2520 | 0.01200702 | 7.721e-02 | **1.000025** |
| Cf9 | 2520 | 0.01086218 | 7.106e-02 | 1.000155 |
| Cf10 | 2520 | 0.01149886 | 7.408e-02 | 1.000155 |
| Cf11 | 2520 | 0.01201819 | 7.657e-02 | 1.000155 |
| Cf12 | 2520 | 0.01186191 | 7.712e-02 | 1.000155 |

Three properties, all read directly off the author's own results.

**The metric is heavy-tailed and its mean is a rare-event frequency.** Cf1 has 1,684 distinct
values: 2.10% are exactly zero, roughly 97% sit around 0.001–0.01, and **0.99% sit near 1.0**.
The mean of 0.0148 is produced almost entirely by that ~1% tail, which is why the standard
deviation (9.5e-02) is over six times the mean.

**His twelve configurations are barely separable.** The spread between configuration means is
**3.94e-03**, against a within-configuration sd of ~8e-02. Treating the near-1 tail as the
event being estimated gives a binomial SE of 1.97e-03, so the full spread across all twelve
configurations is **2.00 SE**. The design does not cleanly rank its own alternatives.

**`Re` exceeds its nominal upper bound.** Eleven of twelve configurations top out at
`1.000155` and one at `1.000025`, where the index should be capped at 1. The excess is small
(1.55e-4) but it is a real formula artefact, and it happens to be the same order of magnitude
as every headroom estimate in this project.

### RETRACTION — the subsection below used the wrong workbook

Verified against the thesis PDF (Warwick, 2017) after the fact. Tables 6.26 and 6.27 publish
the KS degrees of freedom per configuration, which is the exact instance count, so provenance
is checkable row-for-row:

| workbook | verdict |
|---|---|
| `Raw_data1+Re.xlsx` (CF1–CF10), `Raw_data2+Re.xlsx` (CF11–CF20) | **IS the thesis data** — 18 of 20 match the published df exactly |
| `Rsult_1.xlsx` (Cf1–Cf12) | **is NOT** — all 12 mismatch by −1,949 to +735 rows; stores `Re` as a pasted constant rather than the live formula |

**The "ranking inverts" analysis immediately below was computed on `Rsult_1.xlsx` and therefore
says nothing about the published thesis.** It is struck through and kept only as a record. The
verified analysis of the genuine data follows it.

~~### The ranking of the twelve configurations rests on ~15 out-of-bound orders~~

Decomposing each configuration's mean into the tail (`Re >= 0.5`) and the rest:

| cfg | n | mean | n with `Re > 1` | n in tail | **tail share of the mean** | mean without tail |
|---|---:|---:|---:|---:|---:|---:|
| Cf1 | 2464 | 0.014330 | 17 | 23 | **60.92%** | 0.005653 |
| Cf2 | 2467 | 0.011741 | 10 | 18 | 55.26% | 0.005292 |
| Cf3 | 2484 | 0.011181 | 7 | 15 | 46.92% | 0.005971 |
| Cf4 | 2455 | 0.011512 | 9 | 15 | 47.82% | 0.006044 |
| Cf5 | 2504 | 0.011939 | 9 | 17 | 50.20% | 0.005987 |
| Cf6 | 2429 | 0.011314 | 8 | 16 | 52.29% | 0.005434 |
| Cf7 | 2429 | 0.012355 | 11 | 17 | 51.84% | 0.005992 |
| Cf8 | 2461 | 0.009044 | 3 | 9 | 33.71% | 0.006017 |
| Cf9 | 2491 | 0.009839 | 7 | 11 | 40.81% | 0.005850 |
| Cf10 | 2502 | 0.009983 | 7 | 11 | 40.06% | 0.006010 |
| Cf11 | 2510 | 0.010074 | 6 | 12 | 41.57% | 0.005914 |
| Cf12 | 2442 | 0.011422 | 9 | 15 | 48.51% | 0.005918 |

Between **34% and 61% of every configuration's mean comes from 9–23 orders out of ~2,460**.
Cf1's tail holds only eight distinct values, and the two most frequent are `1.000025` (9
orders) and `1.000155` (8 orders) — both **above the index's nominal maximum of 1**. Seventeen
of Cf1's twenty-three tail orders are out of bound.

**Removing the out-of-bound tail inverts the ranking.** On full means the order is
Cf1 (0.014330) best and Cf8 (0.009044) worst. On means excluding the tail the spread collapses
from 5.29e-03 to 7.5e-04 and the order becomes Cf4 (0.006044) best, Cf2 (0.005292) worst —
**Cf1 falls from 1st to 10th and Cf8 rises from 12th to 2nd**.

So the configuration ranking in the source workbook is not driven by systematic differences in
resilience across ~2,460 orders. It is driven by how many of roughly fifteen anomalous orders
each configuration happens to contain, and those orders sit outside the metric's own range.
This should be raised with Garrido directly: it is his workbook, and it bears on which
configuration the thesis reports as best.

### Verified analysis of the genuine thesis data

**The delivered workbooks are not the complete final runs.** The thesis runs **Cf1…Cf90**
(Table 6.25, Eq. 6.4). The files hold only Cf1–Cf20, which are DS1 (R1r) and DS2 (R2r) — the
samples for hypotheses H1a and H1b. **DS3 (Cf21–30, black-swan) and Cf31–Cf90 are absent, and
Cf31–Cf90 are exactly the configurations that test on-hand inventory buffers (I_tS) and
short-term manufacturing capacity (S), i.e. hypotheses H2 and H3.** The decision variables this
project is trying to extend are the ones missing from the delivered data. Two further
anomalies inside the genuine files: **CF5** has 4,241 rows and 21 columns where the thesis
publishes 2,279 and every sibling has 22 — and 4,241 is exactly CF1's count, so CF5 appears to
have been overwritten with a copy of CF1; **CF14** has 2,173 against a published 2,186.

**The ReT formula, recovered from the live cells** (`Raw_data1+Re.xlsx`, column U):

    ReT = IF(AVERAGE(risk columns) > 0,
             IF(APj > 0, APj / LT, 0.5 / RPj),
             1 − ((ΣBt + ΣUt) / j))

with `LT` fixed at 48. **The `APj / LT` branch has no normaliser and is unbounded above.**
Measured maxima in the genuine data:

| cfg | n | mean | orders ≥ 0.5 | tail share of mean | mean excl. tail | **max** |
|---|---:|---:|---:|---:|---:|---:|
| CF1 | 4241 | 0.006090 | 4 | 14.79% | 0.005194 | 1.000155 |
| CF5 | 2279 | 0.008693 | 7 | 29.64% | 0.006135 | 1.000870 |
| CF9 | 2061 | 0.005164 | 0 | 0.00% | 0.005164 | 0.478530 |
| CF11 | 2165 | 0.179780 | 457 | 97.00% | 0.006839 | 3.726893 |
| **CF12** | 2186 | 0.395142 | 824 | 99.08% | 0.005825 | **160.256410** |
| CF14 | 2173 | 0.172647 | 434 | 97.35% | 0.005720 | 8.434548 |
| CF16 | 2218 | 0.234931 | 587 | 98.04% | 0.006248 | 6.163708 |
| CF18 | 2277 | 0.260443 | 669 | 98.49% | 0.005579 | 1.381521 |

Three things follow, all from the author's own final data.

**ReT exceeds its own defined range, by a lot.** Thesis Eq. (7.1) categorises ReT as Low
`[0, 0.3]`, Medium `(0.3, 0.5]`, High `(0.5, 1.0]`. A value of **160.26** has no category. This
is not the 1.55e-4 rounding excess seen in the other workbook; it is two orders of magnitude
outside the scale, and it arises structurally from `APj / LT` being an unnormalised ratio.

**The two data samples are not on the same scale.** DS1 (CF1–10, operational risks) has 0–7
orders in the tail and means of 0.0052–0.0087. DS2 (CF11–20, natural disasters and attacks)
has **278–824 orders in the tail — 13% to 38% of all rows — producing 95–99% of the mean**,
with means of 0.116–0.395. That is a factor of 30–65 between samples.

**Excluding the tail, every configuration in both samples collapses to ~0.0048–0.0068.** So
essentially all between-configuration variation in ReT lives in the unbounded branch.

Whether this is a defect or the intended tail-autotomy semantics (Figure 5.2, the lizard-tail
metaphor — a longer autotomy period *should* read as more resilient) is a question for Garrido,
not for us to settle. What is not in doubt is that the quantity is unbounded while the analysis
that consumes it assumes a `[0, 1]` scale.

**Why this matters for our nulls.** Our screens return `H_PI ~ 1e-4` on a metric whose author's
own twelve-configuration spread is 3.94e-03 and whose noise is 8e-02. The nulls are therefore
consistent with the source data rather than symptomatic of our reconstruction: this metric has
very little discriminating power over policy in this system, in Garrido's hands as well as
ours. That is a finding about the instrument, and it is independent evidence for the
Cobb–Douglas route he authorised — see
`docs/COBB_DOUGLAS_RESILIENCE_PORT_SPEC_2026-07-28.md`.

## 6. What these screens do not establish

- Nothing about downstream finished-goods buffers at nodes that do not yet exist. Those
  need new physics and are untouched here.
- Nothing about `H_obs`. Every number above is a privileged-information ceiling.
- Nothing about Program Q, whose contract, products, horizon and risk regime are different.
  These screens run the thesis-grounded single-product physics with Garrido-native risks on.
- No claim of exhaustiveness for the buffer family: the grid is the thesis ladder plus zero,
  not a continuous search, so the bar is a **best-found incumbent**, not a certified optimum.
