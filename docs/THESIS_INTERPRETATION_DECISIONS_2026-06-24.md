# Thesis Interpretation Decisions — 2026-06-24

**Context.** Garrido retained no original Simulink/MATLAB files or event traces from
the thesis. Exact reproduction is therefore impossible and is **not claimed**. Where
the thesis text is ambiguous, *we own the interpretation*. The governing principle is:

> Choose the interpretation that best reproduces the **published** thesis numbers
> (Tables 6.10–6.12, 6.16, 6.20). Where two interpretations match equally, choose the
> simpler. Document every choice here. Keep the rejected interpretation available as a
> sensitivity / negative-control lane, never silently deleted.

This makes the DES **thesis-grounded**, not "thesis-faithful/exact". The manuscript must
use "thesis-grounded reconstruction" wording.

Status legend: `MATCHES-PUBLISHED` (reproduces a printed number), `CHOSEN-AMBIGUOUS`
(genuine ambiguity, decided by number-match or parsimony), `CALIBRATED` (a parameter we
set ourselves, justified by fit, no thesis source).

| # | Item | Decision | Status | Justification / number |
| --- | --- | --- | --- | --- |
| D1 | Uniform-risk scheduling (R11,R21,R22,R23,R24,R3) | `thesis_window`: one event per non-overlapping `b`-hour window, offset `U(1,b)`; occurrence clock independent of recovery time | MATCHES-PUBLISHED | Reproduces Table 6.11: 48=8064/168 (R11), 0.5=8064/16128 (R21), 2=8064/4032, 1=8064/8064 (R3). Renewal `U(1,b)`-from-last-event gives ~`b/2` interval → ~2× frequency. Keep `legacy_renewal` as negative-control. |
| D2 | R11 topology (Op5 vs Op6) | Single breakdown clock at 48/yr; each incident selects **one** affected station | CHOSEN-AMBIGUOUS | 48/yr is the *total*; a single clock reproduces it. Current code that downs Op5+Op6 simultaneously double-counts station-incidents. Sensitivity: per-station clocks. |
| D3 | R13 delivery delays | 48 weekly draws/yr of `Binomial(12, 0.1)` → ≈58/yr | MATCHES-PUBLISHED | 48×12×0.1 = 57.6 ≈ 58 reproduces Table 6.11. The "monthly Op2 delivery" reading gives 12×12×0.1 = 14.4 and does **not** match. Documented inconsistency; we follow the published frequency. |
| D4 | Binomial risk units (R12/R13/R14) | Count *affected units* (delayed contracts / delayed deliveries / defective products), distinct from *incident count*; `RiskEvent` carries `magnitude` + `unit` | CHOSEN-AMBIGUOUS | Table 6.11 mixes units. Needed so "5 affected ops" ≠ "5 disasters". |
| D5 | Raw-material quantities (Op2 190k, Op3 15.5k, Table 6.16 buffer) | Per individual raw material `rm1..rm12`; aggregate representation must preserve 12 component-units per ration-kit | CHOSEN-AMBIGUOUS | Thesis text: Op2 handles 190,000 of *each* rm; Op3 sends 15,500 of *each*; same `I_{t,S}` level applied at Op3/Op5/Op9. |
| D6 | Buffer replenishment semantics | Order-up-to (top-up to target) every `t` hours; strategic buffer modeled within operating inventory unless evidence says otherwise | CHOSEN-AMBIGUOUS | "Add full quantity every t" would grow inventory unboundedly; order-up-to is the only non-divergent reading. |
| D7 | Raw-material order-up-to multiplier `2.0` | Calibration parameter we set; **not** thesis-sourced | CALIBRATED | Justified only by fit to throughput/ReT targets. MUST be reported as a calibrated knob with a one-line fit criterion, and sensitivity-tested at {1.0, 1.5, 2.0}. Not presented as a thesis value. |
| D8 | R14 defect mode | `thesis_strict_op6` | CHOSEN-AMBIGUOUS | Per current frozen contract; magnitude = defective units via D4. |

**Hard rule.** Items marked `CALIBRATED` (currently only D7) may not be described as
thesis reproduction anywhere in the manuscript. Items marked `CHOSEN-AMBIGUOUS` must be
named as our interpretation in the methods section with the rejected alternative cited.

## Verification status (checked 2026-06-24, full 20-year run, ~2 s)

D1–D4 are **implemented and locked by tests**; G1 is GREEN, not analytical.

- D1/D2/D3 empirical per-year counts under `risk_occurrence_mode=thesis_window`
  (`seed=7`, 161,280 h): R11 48.0, R21 0.5, R22 2.0, R23 1.0, R24 12.0, R3 0.05 —
  exact vs Table 6.11. Negative control `legacy_renewal` ≈ doubles them (R11 92.9,
  R22 3.7, R24 24.4) and collapses R13, confirming the two modes are genuinely
  different processes.
- D3/D4 unit-vs-incident split confirmed: R13 = 34.5 **incidents**/yr but 60.6
  **delayed-deliveries**/yr ≈ Table 6.11's 58 (expected 57.6). The published
  number is a *unit* count, not an incident count — exactly why `magnitude`/`unit`
  exist on `RiskEvent`.
- Tests pinning this: `tests/test_thesis_faithful_lane.py::`
  `test_thesis_risk_frequency_reporter_passes_thesis_window_gate`,
  `test_thesis_risk_frequency_reporter_preserves_legacy_gap`,
  `test_r11_incidents_affect_one_workstation_not_both` (3 passed).
- Reporter: `scripts/report_thesis_risk_frequency.py` (9-row PASS/FAIL gate).

**Implication for next work.** The DES fidelity gate is closed. The open frontier is
the contract's frozen learning regime (persistence `rho` + operational-tempo
stochastic demand), which has no implementation yet; the retained/reset harness
(`scripts/evaluate_retained_reset_learning.py`) already exists and is ready to
consume it.
