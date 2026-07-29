# Track C oracle phase — TERMINAL VERDICT (2026-07-10, overnight session)

**Headline: NULL. Across five pre-registered calibrations spanning three
mechanism families, the best state-switching policy's advantage over the best
same-contract constant tops out at ~1.6% of baseline Excel ReT — an order of
magnitude below the pre-registered materiality bar (5%). Per the terminal
rule frozen in `docs/TRACK_C_PREREGISTRATION_2026-07-10.md` (A2), the Track C
oracle phase ENDS without training a single PPO seed. Constant full-contract
policies are structurally near-optimal in this DES class, even under
engineered non-stationarity with priced economics.**

## What was run (all eval-only; no learner; tape discipline per prereg)

Environment: track_bp_v1 11D contract + campaign regime process (CRN-safe
exogenous schedule, exact thinning) + route-aware replenishment +
surge_inertia + J_v3 = ret_excel − λ·(actual holding, dispatch excess, shift),
λ frozen from Cf_0 anchors before any optimization, per iteration.
Ladder per iteration: C0 anchors/λ → Sobol-96 constant screen → refinement
(frozen constant) → true-regime switch pairs (frozen switcher) → C1 verdict
(24 tapes, CRN-paired) → detector fit → C2 verdict.

| Iter | Family | C1 switcher−constant (J) | Consistency |
|---|---|---|---|
| 1 | 2-state campaign, R22-led, moderate | −5.9e-6 [−1.2e-5, +2.4e-8] | 6/24 |
| 2 | R22-led, drain-heavier (freq×6, impact×4) | +8.8e-6 [−8.4e-6, +1.8e-5] | 23/24 |
| 3 | + pre_campaign ramp (A1) | +2.26e-5 [+1.40e-5, +3.27e-5] | 23/24 |
| 4 | max interdiction amplitude (freq×8, impact×6) | +6.47e-5 [+5.4e-6, +1.36e-4] | 12/24 |
| 5 | R13/R24-led supply/demand stress (A2, FINAL) | +2.78e-5 [+2.43e-5, +3.14e-5] | **24/24** |

Thresholds (0.05·ReT_base) ranged 1.5e-4–2.1e-4. No iteration came within
2.4× of its bar. Iter4's larger mean came with collapsed consistency (noise,
not signal). Stop rules fired exactly as pre-declared.

## The two real findings (both CI-clean, worth keeping)

1. **Hysteresis beats clairvoyance.** In iters 3 and 5, the NON-privileged
   hazard detector (EWMA of ops-down + realized event starts) outperformed
   the TRUE-state instant switcher — iter5: detector +6.50e-5
   [+6.25e-5, +6.74e-5] vs oracle +2.78e-5, capture ratio 2.34, both ~24/24.
   Under commitment lags (168h replenishment lead), smoothed persistence
   (staying provisioned across phase boundaries) is worth more than exact
   knowledge of the regime switch. A genuinely interesting control-theoretic
   nugget: in this world, *knowing the state instantly is not the binding
   constraint; commitment latency is.*
2. **Lever authority exists; priced economics neutralize it.** Buffer/
   capacity anchors separate cleanly on raw ReT in every campaign world
   (Cf_0 0.0042 → I1344_S2 0.0048 → heavy 0.0056 in iter5), but at a holding
   price calibrated to 15% of baseline ReT per 2× stock, the J-ranking
   compresses to near-indifference — a quantified instance of Garrido §8.3's
   "the more…the better, but…" position, answering his §8.5.2/8.6.2 cost
   question: with the cost factor included, the optimum buffering strategy in
   this class is a *moderate constant*, and dynamic modulation of it is worth
   ≤2% of baseline.

## Why constants keep winning here (mechanism synthesis, five worlds later)

- **Route-block paradox (interdiction family):** campaign outages block
  exactly the in-phase buffer commitments that switching would exploit;
  only instant levers (dispatch/shift) retain in-phase authority (~2% ceiling
  at max amplitude, iter4).
- **Weekly-mean metric physics:** Excel ReT is dominated by steady-state
  branch mix; week-scale posture changes move it through slow inventory
  dynamics that a well-chosen constant already near-optimizes (supply/demand
  family, iter5: cleanest CI, same tiny magnitude).
- **Cost symmetry:** any λ large enough to punish an always-heavy constant
  also punishes the switcher's campaign posture; the differential is a
  second-order difference of near-break-even quantities.

This generalizes the same-contract reversal
(`docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`): the Track B
"adaptive win" was a comparator artifact in a stationary world, and even
ENGINEERED non-stationarity with observable ramps, priced costs, and
commitment physics does not open material state-contingent headroom in this
DES class. The result is now a positive statement about the world, not an
absence of trying: three mechanism families, five calibrations, pre-registered
gates, zero trained seeds.

## Paper disposition

- The C&IE pivot paper gains a strong boundary-extension section: "the
  comparator lesson survives non-stationarity" — with the oracle-gate
  protocol (design → C1/C2 before any training) as a reusable methodology
  contribution, and the hysteresis-beats-clairvoyance nugget as a
  discussion point.
- Any future attempt at an adaptive/preventive win in this benchmark class
  requires CHANGING THE CLASS: mass-conserving physics with finite sources,
  downstream emergency reserves behind threatened arcs, an explicit imperfect
  advance-alert channel, and/or sub-weekly decision epochs — each a new
  pre-registration (candidate knobs catalogued in the 2026-07-10 external
  review and `docs/TRACK_C_FROM_ZERO_REDESIGN_2026-07-10.md` §3.6).
  None of these may be motivated by *this* phase's verdict tapes.

## Artifacts

`outputs/experiments/track_c_gates_iter{1..5}_2026-07-10/` (each with
campaign_config.json, lambdas.json, all stage CSVs, c1/c2 verdicts);
runner `scripts/run_track_c_gates.py`; env machinery
`supply_chain/track_c_env.py` + campaign/route-aware sim extensions
(flags-off bitwise identity verified against the crossed-eval ledger;
`tests/test_track_c_env.py` 8/8). Ready-but-unused: Gate C3 trainer
(`scripts/run_track_c_gate_c3.py`, J_v3-aligned reward, scratch + BC-warm).
