# Fidelity check vs Garrido (2026-06-28) — RESOLVED: GATE PASSED, our DES is faithful

**FINAL VERDICT (paired H2/H3 gate, 300 episodes, thesis frequencies):** our DES REPRODUCES Garrido's H2
(inventory raises ReT: R1 10/10, R2 10/10, R3 9/10) and H3 (shifts raise ReT: R1 10/10, R2 10/10, R3 7/10 —
weak for R3, matching his thesis) + baseline ReT (R1 0.0058≈0.006, R2 matched-freq 0.19≈0.18). The DES is
FAITHFUL. Corrections: "shifts harmful" was wrong (shifts HELP at I0; redundant at high buffer); buffer
saturation at I168 is faithful-consistent (his tests are lever-vs-none). The "cheap sweet spot / no
decision frontier with buffer×shift" is a LEGITIMATE property. Full data:
`outputs/benchmarks/garrido_static_fidelity_stress/paired_h2_h3_full_cf1_30_thesis_1rep_2026_06_28/`
(`FIDELITY_GATE_ANALYSIS.md`, `policy_family_summary.csv`).

---
(historical interim notes below — superseded by the PASS verdict above)


**STATUS: initial "critical fidelity gap / contradiction" was OVERSTATED — corrected below after reading
Garrido's actual hypothesis TESTS (Ch.7), not just the §8.2 summary.** The user's skepticism ("how can a
cheap low-buffer policy beat everything, and Garrido not notice?") was a good catch, but the resolution is:
our DES is approximately faithful to Garrido's RIGOROUS findings; f0.10_S1 winning answers an efficiency
question he never asked; one residual (exact saturation point) is genuinely untestable from the data we have.

## Correction: what Garrido actually tested (Ch.7, Tables 7.4–7.9)
His tests are "lever PRESENT vs ABSENT" with treatment sets randomly MIXING levels — NOT monotonic sweeps:
- **H2 (inventory):** DS1 (I0) vs DS4 (buffer present, random mix of I168..I1344) → buffer > no buffer (strong).
- **H3 (shifts):** DS1 (S1) vs DS7 (S2/S3 present) → shifts > S1, but WEAK/non-universal (1 of 10 non-sig in
  H3a, 2 of 10 in H3c; needed a binomial test to claim 95% support).
- **He never rigorously isolated `I1344 > I168` or `S3 > S2`.** "The more, the better" (§8.2) is summary
  framing, not what the tests establish. His 3 Excel files (`Rsult_1`, `Raw_data1/2+Re`) are the BASELINE
  configs Cf1–20 — they do NOT contain the buffered/shifted configs (Cf31–90), so his per-level buffer/shift
  response is NOT checkable from the data we have.

## How our DES compares (revised verdict: ~faithful)
- **Buffer:** ours shows I0 < I168 (buffer helps) → AGREES with Garrido's rigorous result. Saturation beyond
  I168 is neither confirmed nor refuted by his isolated tests.
- **Shifts:** ours ≈ neutral (S3 vs S1 within ±1%); Garrido weakly positive + often non-significant → MINOR
  difference, not a contradiction; war-stress downstream cap explains the neutrality.
- So "f0.10_S1 dominates / no decision frontier" is PLAUSIBLY A REAL RESULT (buffer saturates early →
  constant buffer optimal → no dynamic frontier), NOT a clear artifact. Lane A's null is re-validated as
  plausibly real, contingent on the residual below.

## (kept for record) the raw discrepancy that triggered the check

## Garrido's validated finding (thesis §8.2, H2a–c & H3a–c, ≥95–99% confidence)
- **On-hand inventory buffers (I_LS at Op3/Op5/Op9) increase resilience MONOTONICALLY** as they rise through
  I168, I336, I672, I1344 (1→8 weeks), in the presence of R1/R2/R3. "The more, the better."
- **Short-term manufacturing capacity (shifts S) increases resilience** (S2, S3 > S1).
- Inventory preferred over capacity in all cases except contingent demand (R24).

## Our DES (fidelity check `scratchpad/fidelity_check.py`, S1, Re=ret_excel, 3 seeds)
| risk level | I0 | I168 | I336 | I672 | I1344 |
|---|---|---|---|---|---|
| current | 0.00456 | 0.00562 | 0.00562 | 0.00562 | 0.00562 |
| increased | 0.00215 | 0.00310 | 0.00310 | 0.00310 | 0.00310 |
| severe | 0.00033 | 0.00109 | 0.00109 | 0.00109 | 0.00109 |
- **Buffer benefit SATURATES at I168 (1 week).** I168 = I336 = I672 = I1344 exactly → 8 weeks adds ZERO
  resilience over 1 week. Garrido found benefit all the way to I1344.
- **Shifts S3 < S1 (HARMFUL)** at current & increased (S3 only ≥ S1 at severe). Garrido found shifts help.

## Conclusion: our "low-buffer wins / no decision frontier" was an ARTIFACT
Because buffer beyond I168 does nothing and shifts hurt in our DES, the cheapest sufficient config (~f0.10,
S1) looks optimal and the dense static frontier "dominates." **In a faithful model where buffer helps to
I1344 and shifts help, a low-buffer policy does NOT dominate** — exactly why Garrido never observed it. All
recent conclusions (dense-CRN falsification, A0 no-headroom, Lane-A "exhausted") are **suspect**: they were
measured in an environment that does not reproduce the thesis's core relationship.

## Likely cause (to diagnose)
The downstream LOC dispatch is a FIXED cap (op9–12 ship U(2400,2600)/day ≈ demand). Once buffer ≥ ~1 week,
the downstream can't dispatch faster → extra buffer sits idle → no further ReT. Garrido's risks are short
(hours–5 days) but FREQUENT over 20 years; in his model the cumulative backlog apparently keeps growing so
more buffer keeps helping. Candidate fidelity bugs that would cause early saturation in ours:
- backorder accumulation / recovery mechanism (the `BACKORDER_OVERFLOW_MODE`, Bt-cap, R14 period fixes) may
  clear/cap backlog too aggressively, so buffer beyond 1 week is never drawn down;
- the downstream dispatch cap may be too binding vs Garrido's (or his demand/backlog grows unbounded);
- horizon: we run h104 (2 yr); Garrido runs up to 20 yr — cumulative backlog over 20 yr may be where I1344
  pays off. **Test h520/h1040 (10–20 yr) buffer sweep.**
- the R1 *level* was validated (ReT scale ≈ 0.0044 vs Excel 0.0063) but the buffer-RESPONSE CURVE was never
  validated against Garrido's I_LS sweep — that is the untested gap the user found.

## Required next steps (before ANY win/frontier claim)
1. **Verify against Garrido's data:** map `Rsult_1.xlsx` Cf→(I_LS,S,risk) and confirm his ReT rises I168→I1344
   (his §8.2 says yes; get the numbers). Use `Raw_data1/2+Re.xlsx` order ledgers + the design tables 6.16/6.20.
2. **Diagnose our early saturation:** does delivered/backorder differ at all between I168 and I1344? Trace
   why extra buffer is never used (downstream cap vs backlog clearing).
3. **Test horizon:** buffer sweep at h520/h1040 (10–20 yr) — does I1344 beat I168 over Garrido's horizon?
4. **Fix the fidelity gap** so our DES reproduces monotonic buffer benefit, THEN re-ask the frontier/win
   question. The current "no frontier" result is not trustworthy until the DES matches the thesis.

This supersedes the "Lane A exhausted" framing: Lane A's null is an artifact of the fidelity gap, not a real
characterization. The paper cannot claim "no frontier" on a DES that contradicts the thesis it replicates.
