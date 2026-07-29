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
| 1 | outputs/experiments/track_c_gates_iter1_2026-07-10 | none (campaign_v1 as designed) | **FAIL**: switcher−constant −5.9e-6 CI [−1.2e-5, +2.4e-8], 6/24 tapes; threshold 1.96e-4. Diagnosis: phase asymmetry ≈1.2e-5 (swapping calm/campaign roles is inert) — campaign hits do not drain stock (R22 recovery 24h→36h trivial; R21 present in only ~20% of campaigns); pair-candidate set also near-degenerate (similar Sobol leaders). C0 sanity all passed (campaign_frac 0.373; anchors separate: cf0 0.00391 → I1344_S2 0.00469 → heavy 0.00534 ret). Incident note: the runner was edited mid-run (worker tuple mismatch crashed the verdict stage; stages baseline/screen/refine/pairs completed under identical env defaults and were kept; verdict/c2fit/c2verdict rerun cleanly). Script change before iter2 (pre-training, no verdict tapes consulted for design): pair grid now crosses leaders × {leaders, heavy, I1344S2, campaign-boosted(+0.35 buffers, S3, +0.3 dispatch)} and calm-side {leaders, lean, calm-leaned}. |
| 2 | outputs/experiments/track_c_gates_iter2_2026-07-10 | freq {R21:6,R22:6,R23:4,R24:3}; impact {R21:4,R22:4,R23:2,R24:2}; dwell campaign 6wk (calm 8wk); λ share unchanged 0.15; lead 168h | **FAIL but direction flipped**: switcher−constant +8.8e-6 CI [−8.4e-6, +1.8e-5], **23/24 tapes positive** (real, tiny); threshold 1.52e-4. C2 detector +3.6e-6, 23/24, capture 0.41. Physics diagnosis: with route-aware + 168h lead, buffer raises AT campaign onset arrive mid-outage by construction — the instantaneous-state oracle cannot pre-position; only the instant levers (shift/dispatch) generate switching value, hence tiny gap. Anticipation requires an observable ramp BEFORE the heavy phase. |

**AMENDMENT A1 (2026-07-10, pre-training, before iter3):** the design doc's
`pre_campaign` ramp state (v2 option) is activated as a structural knob: cycle
calm → pre_campaign → campaign with per-state multiplier tables (implemented
as the `cycle` config format; legacy 2-state format unchanged; flags-off
bitwise identity re-verified). Rationale: measured — in-campaign buffer
commitment is physically blocked (route-aware + lead), so a detectable low-
damage precursor phase is the only channel through which ANY non-privileged
policy could prepare; thesis-plausible (intel/harassment upticks precede
offensives). The oracle/switcher definition becomes "campaign config when
state ≠ calm". Verdict tapes were consulted only as the pre-registered
iteration trigger; no design choice used per-tape verdict data.

| 3 | outputs/experiments/track_c_gates_iter3_2026-07-10 | cycle: calm 8wk (native) → pre_campaign 3wk (freq {R22:2,R23:1.5,R24:1.5}, impact 1.0) → campaign 6wk (freq {R21:6,R22:6,R23:4,R24:3}, impact {R21:4,R22:4,R23:2,R24:2}); λ share 0.15; lead 168h | **FAIL on magnitude, first clean CI**: switcher−constant +2.26e-5 CI95 [+1.40e-5, +3.27e-5], 23/24 tapes; threshold 1.72e-4. NOTE: the non-privileged detector (+3.72e-5 CI [+1.90e-5, +5.69e-5], capture 1.65) BEATS the true-state oracle — EWMA hysteresis pre-positions through ramps better than instant switching; the anticipation channel works mechanically. Diagnosis: winning pair differs only in soft buffer levels (calm 0.02/0.42/0.16 → camp 0.22/0.47/0.36, both S1); boosted variants lose — campaign buffer VALUE still ≈ its λ cost, so the phase differential is second-order. corr(ΔJ, campaign_frac)≈0. Script fix: c2verdict now requires c1.passed (the iter3 c2_verdict.json 'PROMOTE' string predates this fix and is void — C1 failed). |
| 4 | outputs/experiments/track_c_gates_iter4_2026-07-10 | freq {R21:8,R22:8,R23:6,R24:4}; impact {R21:4,R22:6,R23:3,R24:2}; pre_campaign 4wk freq {R22:3,R23:2,R24:2}; calm 10wk / campaign 7wk; λ share 0.15; lead 168h. Rationale: push into the measured headroom-on region (R21 f8; R22 6-day outages) to raise the CAMPAIGN value of buffers, which — not cost pressure — binds the differential. **Pre-declared stop rule: if iter4 mean < 1e-4, the trajectory (−6e-6, +9e-6, +23e-6, ...) indicates a mechanism ceiling ≈3% of base — STOP and write the null verdict.** | **FAIL — stop rule FIRES for the R22-led family**: mean +6.47e-5 (< 1e-4), CI [+5.4e-6, +1.36e-4], only 12/24 tapes (consistency collapsed vs iter3's 23/24 — max amplitude buys variance, not signal); threshold 1.54e-4. C2 capture 0.48, CI crosses zero. The transport-interdiction (R22-led) campaign family is CLOSED: its measured ceiling is ~2% of base. Root cause (established iter3-4): the route-block paradox — campaign outages block exactly the in-phase buffer commitments that switching would exploit. |

**AMENDMENT A2 (2026-07-10, pre-training; iter5 rationale documented BEFORE
reading iter4 — see the session record):** one FINAL calibration family is
declared: the supply/demand-stress campaign (R13 supplier delays + R24
contingent-demand surges lead; routes remain mostly OPEN so in-phase buffer
replenishment has authority — this breaks the route-block paradox; R13/R24
are precisely the risks Garrido's Ch7 finds buffer/capacity-mediated).
Technical fix included: binomial risks (R12/R13) take the state-CURRENT
multiplier at their weekly re-sample (exact; thinning only applies to
uniform-window risks); R11 got the acceptance gate for completeness;
flags-off bitwise identity re-verified; tests 8/8.
**TERMINAL RULE: iter5 is the last calibration. If C1 fails at any margin,
the Track C oracle phase ENDS and the null verdict is written (constants are
structurally near-optimal in this DES class across three mechanism families:
stationary, interdiction-campaign, and supply/demand-stress campaign).**

| 5 | outputs/experiments/track_c_gates_iter5_2026-07-10 | cycle: calm 10wk (native) → pre_campaign 3wk (freq {R13:2,R24:1.5}) → campaign 8wk (freq {R13:6,R24:4,R22:2,R23:2}, impact {R24:2,R22:2,R23:2}); λ share 0.15; lead 168h | **FAIL — TERMINAL**: +2.78e-5 CI [+2.43e-5, +3.14e-5], **24/24 tapes** (cleanest signal of the night) vs threshold 2.09e-4. Detector +6.50e-5 [+6.25e-5, +6.74e-5], capture 2.34 — again beats the true-state oracle, still 3.2× below bar. **Oracle phase CLOSED per the terminal rule. Null verdict: `docs/TRACK_C_ORACLE_PHASE_VERDICT_2026-07-10.md`. No PPO seed was ever trained.** |

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
