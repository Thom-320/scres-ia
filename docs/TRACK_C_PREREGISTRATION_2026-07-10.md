# Track C pre-registration — Gates C0-C2 (frozen 2026-07-10, before any gate run)

Status: **frozen before reading any result from any stage below.** Runner:
`scripts/run_track_c_gates.py`. Design: `docs/TRACK_C_FROM_ZERO_REDESIGN_2026-07-10.md`.
Environment: `make_track_c_env` = track_bp_v1 11D contract + `campaign_v1`
regime (exact thinning; CRN-safe exogenous schedule) + route-aware
replenishment + surge_inertia + risk_level `current`, h104 × 168h,
`stochastic_pt` on, thesis year basis. Sim changes verified bitwise-inert
with flags off (static S2/2.00/1.50 on tapes 200001/200013 reproduces the
crossed-eval ledger exactly; `tests/test_track_c_env.py` 6/6).

## Objective and endpoint

Primary endpoint `J_v3 = ret_excel − λ_h·holding_actual − λ_d·dispatch_excess
− λ_s·shift_excess`, where holding_actual is the ACTUAL time-weighted mean
container stock (op3/op5/op9) normalized by I_1344 capacity (operating floor
included on purpose: policies pay for the stock they really hold; the floor
cancels in paired contrasts). ret_excel reported alongside, always.

## Lambda freezing rule (Gate C0, BEFORE any optimization)

From the Garrido Cf_0 anchor (S=1, base multipliers, no strategic stock) on
tapes 600001-600012:
- λ_h = 0.15 · ReT_base / holding_base  (holding 2× the Cf_0 stock costs 15%
  of Cf_0 ReT),
- λ_d = 0.05 · ReT_base per unit of (m10−1)⁺+(m12−1)⁺,
- λ_s = 0.05 · ReT_base per unit of (S−1).
Frozen at `lambdas.json` on first baseline run of a calibration; never edited
within a calibration iteration.

## Tape discipline (universe 600001+; ranges never mixed)

| Use | Tapes |
|---|---|
| C0 anchors / λ | 600001-600012 |
| Sobol screen (96+anchors, J_v3) | 600001-600008 |
| Refinement (top-6 ± 0.15) → frozen CONSTANT | 600001-600016 |
| Switch-pair fit (top-6 × top-6 + lean/heavy) → frozen SWITCHER | 600017-600024 |
| Detector fit (θ×half-life grid on frozen pair) | 600017-600024 |
| C1/C2 verdict battery | 600031-600054 |
| C3 (PPO) confirmatory battery — untouched until C3 | 610001+ |

## Gate C1 — switching-oracle headroom (decides whether the env is worth training in)

Contrast: frozen TRUE-regime switcher − frozen constant, CRN-paired on
600031-600054, bootstrap CI95 (10k).
**PROMOTE iff mean ≥ 0.05·ReT_base AND CI95 wholly > 0.**
(The design doc's provisional "+0.002" is superseded by this scale-aware form
— set before any campaign-world result was read; ReT scale under campaign_v1
was unknown at freeze time.)

## Gate C2 — non-privileged detectability

Detector: EWMA (half-life ∈ {2,3,4} wk) of an operator-observable signal =
(# ops currently down) + (# recorded R21/R22/R23/R24 event starts in the last
week); campaign iff EWMA ≥ θ ∈ {0.5,0.75,1.0,1.5,2.0}. Fit on 600017-600024,
frozen, then verdict on 600031-600054.
**PASS iff detector-switcher − constant ≥ 50% of the C1 gap AND CI95 > 0.**

## Environment-iteration policy (the only tuning allowed, all pre-training)

If C1 fails, the environment calibration may be revised and the WHOLE ladder
rerun in a NEW output directory, logging each iteration in the table below.
Permitted knobs: campaign frequency/impact multipliers, dwell means, λ scale
factors (0.10/0.15/0.20 for the holding share), replenishment lead (168/336h).
NOT permitted: touching 600031-600054 selection logic, the 610001+ battery,
or any learner. If no calibration within the knob ranges passes C1, Track C
STOPS and the null becomes a boundary-extension section of the C&IE paper.

| Iter | Output dir | Knobs changed | C1 result |
|---|---|---|---|
| 1 | outputs/experiments/track_c_gates_iter1_2026-07-10 | none (campaign_v1 as designed) | (pending) |

## Falsifiers / sanity (Gate C0)

- campaign_frac (fraction of episode steps in campaign) in [0.15, 0.45];
- anchor ordering sanity: I1344_S2 ≠ cf0 on ret_excel (buffers must matter);
- route-aware mechanics unit-tested (top-ups blocked while route down);
- exact-zero detection: if switcher and constant are bit-identical on the
  verdict battery, the gate is void (investigate before interpreting).

## What C1/C2 passing does and does not mean

Passing C1+C2 means: a detectable, materially-valuable state-contingent
policy EXISTS in this world — training is justified (Gate C3, separate
pre-registration: 5×200k PPO, scratch + BC-warm arms, same-contract Sobol
constant re-optimized under the final calibration, virgin battery 610001+).
It does NOT itself establish an RL result, and the C&IE pivot paper does not
depend on any of this.
