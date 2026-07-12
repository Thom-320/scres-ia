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

## Addendum — 2026-07-02 (Garrido/David meeting)

| # | Item | Decision | Status | Justification / evidence |
| --- | --- | --- | --- | --- |
| D9 | Definition of "CDC" (Centro de Distribución) | `CDC = Op3` (`OPERATIONS[3]`, "Warehouse & Distribution Centre"). For Garrido/David meeting language, "del CDC hacia abajo" means **CDC-inclusive**: Op3 may be a decision node, while upstream Op1/Op2 (contracting/suppliers) are excluded. For a stricter reviewer-facing diagnostic, use the explicit label **`post_cdc_only`** to mean Op3-excluded: `op3_q`/`op3_rop` frozen at Garrido baseline and only downstream authority retained. | CHOSEN WITH DIAGNOSTIC | Garrido states the inclusive boundary twice in the 2026-07-02 transcript: *"nodos posibles del centro de distribución hacia abajo"* and *"del CDC hacia abajo, que tú puedas poner literalmente buffers en cada uno de esos."* The repo now keeps both meanings separate: Track A (`op3,op5,op9,shift`) and Track B (8D, dims 0/2 = `op3_q`/`op3_rop`) comply with the CDC-inclusive meeting convention; `scripts/run_track_b_ablation.py::PostCdcOnlyWrapper` is the Op3-excluded diagnostic arm. Do not use "post-CDC" in the manuscript without specifying whether it is CDC-inclusive or Op3-excluded. |
| D10 | Buffer replenishment conservation (does a buffer top-up create material from nothing?) | `_top_up_inventory_buffer` (used by Track A's `per_op_buffer` contract for Op3/Op5/Op9) is an **unconditioned exogenous top-up** (`container.put(shortfall)`) — it does not check upstream availability. This is a known, disclosed simplification, not a silent bug. Track B's downstream dims (`op9_q`, `op10_q`, `op12_q`) are **not** affected: their dispatch mechanics (`_op9_sb_dispatch`, `_op10_transport_to_cssu`, `_op12_transport_to_theatre`) use `dispatch_qty = min(target, available)` and never exceed on-hand inventory. | CHOSEN-AMBIGUOUS / partially open | Garrido, reading the same code live in the meeting: *"la simulación es el buffer no se puede simplemente abastecer de la nada... él debe, a su vez, implementar, incrementar el shift, o lo que sea que haya que hacer en aras de abastecer ese punto."* His concrete ask: add a per-post-CDC-node decision variable for "replenish or not," tied to upstream shift/production — not yet implemented. Partial mitigation in flight: `outputs/experiments/track_a_repair_leadtime_robustness_2026-07-02/` (holding_cost + lead_time friction on the existing top-up, 3 arms × 5 seeds) tests whether Track A's negative headline result survives removing the "free instant" property; this is a defensive robustness check, **not** a full implementation of Garrido's explicit-decision-variable ask. Track B's positive headline result already satisfies the no-free-lunch requirement by construction (verified 2026-07-02) and needs no further mitigation on this axis. |
| D11 | `op5_q_multiplier_signal` is inert in every Track B / native-Track-A-6D run to date | The action's only consumer is `supply_chain.py` line ~978: `elif k == "op5_q" and "op5_rm" in self.inventory_buffer_targets:`. `self.inventory_buffer_targets` is populated only from the `initial_buffers` constructor kwarg, which defaults to `{}` and is never overridden by any training script (`run_track_b_smoke.py`, `run_track_a_training_repair.py`, etc. — grepped, zero hits). So `op5_q` is silently a no-op dimension: the agent emits a value, the DES ignores it. This does **not** invalidate Track B's positive result (Op10/Op12 are the load-bearing new dims); it does mean the Track A "6D native contract" (`op3_q, op9_q, op3_rop, op9_rop, op5_q, shift`) is functionally **5D** (`op3_q, op9_q, op3_rop, op9_rop, shift`) under the standard env construction. | CONFIRMED (code trace, not ambiguous) | Re-enabling `op5_q` would require passing non-empty `initial_buffers={"op5_rm": ...}` at construction, which *also* starts `_inventory_buffer_replenishment()` for whichever keys are present — reintroducing the exact unconditioned top-up mechanism D10 flags as non-conservation-respecting. The "Track A v2 conservation-respecting" confirmatory design (2026-07-03) therefore uses the 5 functionally-effective dims only and drops `op5_q` rather than resurrecting the flawed mechanism to make it do something. |
