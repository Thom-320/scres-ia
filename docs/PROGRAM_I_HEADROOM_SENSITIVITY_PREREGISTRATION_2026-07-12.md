# Program I — Headroom Sensitivity Map (preregistration, frozen 2026-07-12)

Status: **FROZEN. This is a sensitivity/characterization study, NOT a training lane.** A learner
is authorized only if a located region passes the frozen G/H gate as its own preregistered follow-up.
Contract: `contracts/program_i_headroom_sensitivity_v1.json`.

## Question
Across the broad structural parameter space of the stylized two-CSSU shared-convoy contract, WHERE
does adaptive headroom live and WHICH factors (and interactions) drive it? Run global sensitivity
analysis (GSA) with the HEADROOM as the output — not the raw ReT (output-sensitivity ≠ adaptive value,
the project's core lesson). Outputs: **H_PI = EVPI** (clairvoyant, `oracle − best static`) and the
**adaptive gap H_obs ≈ VSS** (`best observable policy − best static`), on `ret_order`; plus η=H_obs/H_PI.

## Method (three-stage GSA; estimators hand-rolled on scipy.stats.qmc — no SALib)
1. **Morris elementary effects** — μ\* (effect) and σ (interaction/non-linearity) per factor; cheap
   screen, `(k+1)·r` runs. σ is a PRIMARY output (headroom, if any, is a conjunction/interaction).
2. **Sobol indices** (Saltelli/Jansen) on survivors — first-order S_i, total S_Ti; the gap S_Ti−S_i
   is the interaction quantifier.
3. **GP surrogate + expected-improvement** to LOCATE argmax H_obs (the principled version of the
   user's "find the points," not brute Monte Carlo).

## Factor space (broad structural; frozen ranges)
`signal_q`∈[0.50,0.95] · `lead`∈[1,3]wk · `surge_mult`∈[1.1,3.0] (scarcity vs fixed convoy) ·
`persistence` short/long dwell · `commonality`∈[0,0.90] (A/B co-surge) · `r22_prob`∈[0,0.30]
(risk-magnitude CONTROL axis, expected inert). Metric axis (ret_order/ret_quantity/service-loss/
worst-fill/Cobb-Douglas) is a documented secondary study reusing `program_g.metrics_all`.

## Reuse (no reinvention)
`supply_chain/headroom_sensitivity.py` (`headroom_at(theta)`) reuses `program_g.materialize_tape`
(base tape byte-identical at commonality=0), `enumerate`/`ret_order_metrics`/`periodic_calendars`,
and `program_h_belief.CSSUFilter` (belief observable policy). `supply_chain/gsa.py` = Morris/Sobol/GP.
Runner `scripts/run_headroom_gsa.py`.

## Estimator validation (done before any GSA run — binding)
- **Sobol on Ishigami** reproduces analytic indices (S1≈[.31,.44,0], ST≈[.56,.44,.24], x1↔x3
  interaction gap ≈+0.24). PASSED.
- **Program-G anchor point-check**: `headroom_at` at commonality=0 returns H_PI≈+0.015, H_obs≈−0.02,
  STABLE across seed blocks. PASSED only after fixing a real bug: a calibration-frozen static flipped
  between ABAB and BABA (the contract has a mild A/B asymmetry), inflating H_PI ~4× and flipping H_obs
  positive. Fix: the static baseline is the STRONGEST full-contract periodic calendar evaluated
  in-sample on the eval tapes (A/B-symmetry-aware, stable, conservative headroom → no false positives).

## Decision rule (anti-p-hacking, binding)
- The WHOLE map is reported, nulls included; no factor range is chosen by learner performance.
- If GP-locate finds a region with **H_obs LCB95 ≥ 0.01 AND η ≥ 0.30** on ≥200 fresh tapes, that
  region becomes a NEW preregistered lane with the frozen G/H gate protocol (calibration → holdout →
  virgin), NOT a rescue and NOT a managerial claim from the GSA.
- If none qualifies, the map IS the result: a formal boundary characterization (which factors move
  H_PI, why none move H_obs) — a methodological contribution (GSA on value-of-adaptivity) for the
  manuscript.

## The two non-parametric levers (the honest "alternatives to create headroom")
The GSA probes parametric levers. The real bottleneck the project identified is non-parametric and the
GSA will confirm it: (1) change WHAT is OBSERVED (a signal predictive of the future before acting and
not redundant with current inventory), (2) change WHAT resilience MEANS (a metric valuing the adaptive
objective). Both must come from Garrido/operational reality, not from wanting RL to win.

## Provenance
Factor ranges + estimator SHAs recorded; `results/headroom_gsa/verdict_*.json` is the source of truth.
Program I does not block the manuscript; it runs in parallel and characterizes the boundary.
