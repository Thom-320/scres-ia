# Garrido's Cobb-Douglas resilience index, ported and measured

**Status:** `DEVELOPMENT_SCREEN_NO_CLAIM`. No confirmation universe opened, no learner
trained. Module `supply_chain/cobb_douglas_resilience.py`, runner
`scripts/calibrate_cobb_douglas.py`, contract
`contracts/cobb_douglas_calibration_v1.json`, result `results/cobb_douglas/score_v1.json`.

Source: Garrido, Pongutá & García-Reyes (2024), IJPR, DOI 10.1080/00207543.2024.2425771,
§3.4 Eq. (2)–(6), output variables at Algorithm 2 lines 33–38. Read in full for this port;
the spec in `COBB_DOUGLAS_RESILIENCE_PORT_SPEC_2026-07-28.md` was written from §3.4 alone
and is corrected in three places below.

## 1. What was built

    R = 1 / (1 + exp(-(a·ln ζ - b·ln ε + c·ln φ - d·ln τ - n·ln κ̇)))        Eq. (6)

with the five variables sampled daily off the physical ledger — no case classification,
no order-visibility filter. Exponents are **re-derived** with Garrido's own rule
`0.20/ln(x_max)` from our maxima over a 108-episode development sweep, never copied.

| | Garrido's | ours | our x_max |
|---|---:|---:|---:|
| ζ inventory | 0.024 | 0.014200 | 1,308,556 |
| ε backorders | 0.026 | 0.016990 | 129,521 |
| φ spare capacity | 0.04 | 0.025582 | 2,485 |
| τ time-to-fulfil | 0.06 | 1.075378 | 1.204 |
| κ̇ cost | 0.1771 | 0.354673 | 1.758 |

Costs use `c = 1` for all seven coefficients. That is not a placeholder: it is Garrido's
own §3.1 assumption (6), and his §5 varies the sensitive ones over [1,2] and finds the
ranking unchanged. **This retires open item 4 of the port spec** — his sign-off was
already in the published baseline.

## 2. Three corrections to the port spec

**The published exponents cannot be inverted.** `exp(0.20/0.024) = 4,160` against his
stated ζ_max ≈ 3,612 — a 15% error from a 1.7% rounding, because the exponent enters
through `exp(0.20/a)`. An independent reason not to copy the five numbers, separate from
the scale argument. Test: `test_published_exponents_are_too_rounded_to_invert`.

**ζ enters positively — the index does not punish hoarding directly.** More inventory
*raises* R. The only thing that penalises the DDMRP-style 26× overstocking is the holding
cost inside κ̇, whose exponent is ~25× ζ's in our fit. Without a costed ledger this index
is *worse* than ReT for that question, not better. The cost term is load-bearing.

**The floor choice can dominate the index.** Declaring τ's floor at 1e-4 (first run) gave
its term a magnitude of 9.9 against a budget of 0.20 — the least informative of the five
variables would have driven R outright. Fixed by flooring all one-sided variables at 1.0,
where `ln(1) = 0` and a variable at its floor contributes exactly nothing.
`assert_terms_bounded` now enforces that every term stays within ±1/5 across the
calibration range, which is what the exponent rule exists to guarantee.

## 3. Two variables do not survive the transfer

**τ is dead.** It is exactly 0 in 88 of 108 calibration episodes, and non-zero
essentially only at the zero-buffer posture. Net requirements `max(GR_t − I_{t−1} +
B_{t−1}, 0)` never go positive when the operating point carries months of stock —
the same ~5× material slack the A1 procurement screen measured. Its term is `−0.0` for
every scored policy. **The five-variable index is a four-variable index here.**

**τ and κ̇ are ill-conditioned.** Relative sensitivity `1/ln(x_max)` is 5.4× for τ and
1.8× for κ̇, against 0.07× for ζ. A single unusual calibration episode moves those two
exponents by more than the error in the maximum. Recorded in the contract's
`conditioning` block, not silently accepted.

## 4. Results

Nine postures × 4 tapes × 2 risk families, comparison set declared before evaluation.
All metrics reported side by side, never selected.

### R1r

| policy | **R_CD** | ret_excel | full ledger | fill | delivered | lost | κ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **I168_S1** | **0.6061** | 0.005615 | 0.005596 | 0.99676 | 767,790 | 0 | 412,756 |
| I168_S2 | 0.5795 | 0.005615 | 0.005597 | 0.99677 | 770,323 | 0 | 603,108 |
| I1344_S1 | 0.5689 | 0.005615 | 0.005596 | 0.99676 | 767,790 | 0 | 647,432 |
| I168_S3 | 0.5508 | 0.005610 | 0.005592 | 0.99678 | 772,810 | 0 | 880,928 |
| I1344_S2 | 0.5482 | 0.005611 | 0.005593 | 0.99677 | 770,323 | 0 | 871,429 |
| I1344_S3 | 0.5235 | 0.005608 | 0.005590 | 0.99678 | 772,810 | 0 | 1,203,678 |
| I0_S1 | 0.5092 | 0.004547 | 0.003463 | 0.75441 | 518,656 | 11 | 464,903 |
| I0_S2 | 0.5002 | 0.004725 | 0.004699 | 0.99446 | 686,042 | 0 | 578,549 |
| I0_S3 | 0.4709 | 0.004719 | 0.004702 | 0.99639 | 689,216 | 0 | 841,738 |

### R2r

| policy | **R_CD** | ret_excel | full ledger | fill | delivered | lost | κ |
|---|---:|---:|---:|---:|---:|---:|---:|
| **I168_S1** | **0.5946** | 0.494190 | 0.289960 | 0.95228 | 797,155 | 0 | 436,345 |
| I0_S1 | 0.5828 | 0.308931 | 0.177177 | 0.76027 | 570,994 | 16 | 480,378 |
| I168_S2 | 0.5616 | 0.493411 | 0.291125 | 0.95246 | 800,273 | 0 | 691,930 |
| I1344_S1 | 0.5589 | 0.494190 | 0.289960 | 0.95228 | 797,155 | 0 | 671,541 |
| I0_S2 | 0.5554 | 0.359736 | 0.228428 | 0.85636 | 645,765 | 1 | 714,457 |
| I168_S3 | 0.5342 | 0.493411 | 0.291125 | 0.95246 | 800,273 | 0 | 992,373 |
| I1344_S2 | 0.5330 | 0.493411 | 0.291125 | 0.95246 | 800,273 | 0 | 962,499 |
| I0_S3 | 0.5289 | 0.359736 | 0.228428 | 0.85636 | 645,765 | 1 | 1,008,743 |
| I1344_S3 | 0.5086 | 0.493411 | 0.291125 | 0.95246 | 800,273 | 0 | 1,322,873 |

### It does what it was brought in to do

**Shift 3 stops being free.** Under `ret_excel` in R1r, I168_S1 and I168_S3 are tied to
5e-6 while S3 delivers 5,020 more rations — the metric prices no capacity, so nothing
restrains always choosing three shifts. Under R_CD the gap is 0.055 and S3 loses,
because it costs 2.13× more for 0.65% more delivery. The mechanism is Garrido's own
`c_u·U_t` charge on spare capacity, not a term of our invention.

Note carefully: **R_CD agrees with ReT's ranking here rather than reversing it.** What
changes is the epistemic status. ReT's preference for one shift is an artefact of
policy-dependent censoring; R_CD's is a defensible judgment that 0.65% more service is
not worth 2.13× the resource. Same order, different reason — and only one of them
survives being asked why.

Buffer/shift ordering is otherwise clean and monotone: within every buffer level,
S1 ≻ S2 ≻ S3, and I0 collapses to the bottom in R1r.

## 5. The index has its own service blindness

**In R2r, `I0_S1` ranks second** — ahead of `I168_S2`, `I1344_S1` and six others —
while filling **76.0% against 95.2%** and losing **16 orders against zero**. It wins on
being cheap.

This is not a calibration accident. Running Garrido's own §5 procedure over
`c_b, c_i ∈ {1,2}`:

| c_b | c_i | winner | I0_S1 rank |
|---|---|---|---|
| 1 | 1 | I168_S1 (0.5946) | **2 of 9** |
| 1 | 2 | I168_S1 (0.5948) | **2 of 9** |
| 2 | 1 | I168_S1 (0.5943) | **2 of 9** |
| 2 | 2 | I168_S1 (0.5946) | **2 of 9** |

The top of the ranking is robust — matching his own finding that S12's superiority
survives c ∈ [1,2]. But **I0_S1's second place is equally robust**, and doubling the
backorder cost does not dislodge it. The reason is structural: κ charges backorders
`B_t`, and a *lost* order leaves the backorder queue entirely. Orders that are never
served stop costing anything.

**So all three metrics fail to reward service, by three different mechanisms:**
`ret_excel` through policy-dependent censoring; `ret_thesis` through case collapse under
risk; `R_CD` through under-pricing stockouts and not pricing lost orders at all.

That is the real finding, and it is more useful than a fourth metric would have been.
The problem is not which resilience index we pick — **it is that a resilience index is
not a service guarantee.** Service has to be carried as a separate constraint, which is
exactly the role of Program Q's worst-product fill guardrail, and exactly the guardrail
whose failure closed Program O. This port is the third independent confirmation.

## 6. Standing rules, unchanged

- **Triangulate, never select.** `ret_excel`, `ret_excel_full_ledger` and `R_cobb_douglas`
  are reported for every policy, always. A result appearing only under one is
  metric-dependent, not a win. Program G is the precedent.
- **κ̇ is set-relative** (Eq. 5 normalises by the whole comparison set's cost), so R is
  not comparable across sets and never across papers. Every table states its set.
- Exponents, maxima, floors, costs and the comparison set are frozen in the contract
  before scoring, with the calibration sweep's hash.

## 7. What this does not establish

Development tapes only, one risk corner per family, four tapes per cell, no paired
intervals. Nothing about `H_obs`, nothing about a learner. The comparison set is nine
static postures — the index has not yet been applied to a dynamic controller, which is
where the DDMRP and MPC questions live.

## 8. One thing the 2024 paper gives us

§6.2, his own limitations: future work should include "other recurrent risky events …
such as power failure, machines breakdowns, absenteeism, material shortages, or
reprocessing orders"; should "concentrate on measuring the impact of other tactical
manufacturing processes – e.g. the purchasing and material requirements planning";
and "would imply the combined use of discrete simulation techniques and robust
learning-based algorithms".

That is the MFSC DES risk set (R11–R24), DDMRP, and the RL work, named as future
research by the index's own author. Worth citing directly in the framing.
