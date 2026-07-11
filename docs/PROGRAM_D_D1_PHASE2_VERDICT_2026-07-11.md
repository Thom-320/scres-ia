# Program D — Lever D1 (Op9 rationing rule), Phase 1+2 verdict (2026-07-11)

**Verdict:** `AUTHORITY_PRESENT` — proceed to Phase 3 (branching).
**Artifact:** `results/program_d/d1_authority_screen/verdict.json` (+ d1_rows.csv).
**PPO trained:** no. **Virgin tapes:** none. 30 calibration tapes × 104 weeks,
strict-CRN paired, garrido_proxy_v1 physics (sha 3d7aaa14…).

## What this is

The FIRST lever in the entire research program (after Tracks A / B-P / C, Program
L, and the reserve v2/v3 gates all measured constants-optimal) to clear the
Phase-2 authority gate: the choice of rationing rule for the standing Op9
backlog materially moves the Garrido resilience index.

## Results

Mean ret_excel by constant rule (30 tapes):

| Rule | ret_excel | mean Ut (lost) |
|---|---|---|
| **age_threshold** (best) | **0.5442** | 109.7 |
| fifo_contingent = fifo_flat | 0.5301 | 114.5 |
| spt_contingent = spt_flat (thesis) | 0.5056 | 109.2 |
| lpt_contingent (worst) | 0.4898 | 120.3 |

- Best − worst gap: **+0.0544** ret_excel, CI95 **[0.0525, 0.0563]**.
- Best − thesis SPT: **+0.0387** ret_excel (~7.6% rel.), CI95 **[0.0366, 0.0407]**.
- Best − worst unattended-order reduction: **10.6 fewer lost**, CI95 [10.4, 10.7].

(spt_contingent==spt_flat and fifo_contingent==fifo_flat because contingent
R24 orders are a small share; the contingent-first tier only binds when R24
fires. This is expected, not a bug.)

## What this DOES and does NOT establish

**DOES (managerial, static):** Garrido's thesis SPT rule is NOT the best fixed
rationing rule for this chain. An age-threshold rule — serve orders that have
waited past τ before applying SPT — scores ~7.6% higher ReT with a tight CI95.
This directly answers a lever the thesis fixed by assumption (§6.5.4) and never
varied. It is publishable managerial content on its own and gives Program D its
first authority pass.

**DOES NOT (dynamic — the actual open question):** Phase 2 compares CONSTANT
rules. It shows a *different constant* beats the thesis constant; it does NOT
yet show that *switching the rule by state* beats the best constant
(age_threshold). That is Phase 3 (counterfactual branching): does the optimal
rationing rule VARY with the observable queue/inventory state, and does a
state-contingent policy beat age_threshold by ≥ δ_oracle with CI95? Only a
Phase-3 pass would demonstrate that a dynamic decision variable matters here.

## Honesty flags (carry into any write-up)

1. **Primary/co-primary not co-directional at the top.** age_threshold wins on
   ret_excel but has marginally MORE lost orders than SPT (109.7 vs 109.2). Its
   ReT gain comes from serving order/aging structure, not from shedding fewer
   orders. The thesis SPT rule is already near-optimal on Ut. Report both.
2. **Threshold not tuned.** τ=336 h is the untuned default; a τ sweep is a
   Phase-2 sensitivity, not an optimization licence. The claim is "authority
   exists," not "age_threshold@336h is optimal."
3. This does not reopen Tracks A/B/C/L or Program L's boundary verdict — those
   levers remain constants-optimal. D1 is a *different* decision right.

## Next step (authorized by this verdict)

Phase 3 branching for D1: reuse the prefix-replay machinery
(run_l_branching_headroom) over the rationing-rule action set at sampled
decision states, with age_threshold as the constant comparator. Preregistered
δ_oracle = 5% service-loss gain, co-directional on ret_excel, CI95 > 0;
action-variability ≥15% for ≥2 rules. No PPO until the Phase-4 observable gate.
