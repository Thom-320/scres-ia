# Program G terminal metric audit preregistration — 2026-07-12

Status: **FROZEN BEFORE OPENING 1040001+ OR 1050001+.**

This is a corrective, terminal audit of the stylized Program G order adapter. It is not a
rescue of G5, not a port to the full Op1–Op13 DES, and not a new confirmatory claim about the
MFSC. The previously opened 1020001+/1030001+ results exposed two implementation defects:
the six-day operating week was compressed to 144 hours and two runners scored ReT before
updating the current order's cumulative backlog/unattended counters.

## Frozen corrections

1. Calendar time is `week * 168 + operating_day * 24`; demand is still generated six days/week.
2. Orders pending at the episode horizon are marked unattended/lost at that horizon.
3. Order ReT is computed only through `compute_order_level_ret_excel_formula` with
   `j_source=row_index`; no runner may reimplement its cumulative ledger.
4. `ReT_quantity` is the quantity-weighted mean of the same canonical per-order scores.
5. The quantity endpoint formerly called `service_loss` is named
   `unfulfilled_rations_at_horizon`; it is not described as service-loss AUC.
6. All metrics for a policy/tape are derived from one disclosed stylized order trajectory.

## Frozen policies and data

- Action contract: weekly priority `{A,B,HOLD}`, S1 fixed, Program G TRS physics unchanged.
- Full-contract static bar: all periodic calendars of periods 1–4, selected once on calibration.
- Observable policies: cover, MPC, depth-3 service tree, depth-3 ReT tree.
- No PPO training or tuning is permitted.
- Calibration: 200 tapes beginning at `1040001`.
- Locked terminal test: 400 tapes beginning at `1050001`.
- Region: the 12 previously frozen `surge=1.50` cells; no cell selection by outcome.

## Primary terminal rule

An observable policy passes only if, against the frozen best full-contract static calendar on
the locked test, all are true:

1. canonical `ReT_order` paired CI95 lower bound is positive;
2. `ReT_quantity` paired CI95 lower bound is non-negative;
3. attended-order delta paired CI95 lower bound is non-negative;
4. worst-CSSU fill delta paired CI95 lower bound is non-negative;
5. unfulfilled-rations-at-horizon paired CI95 upper bound is non-positive;
6. resource rights are identical by construction.

If no observable policy passes every condition, the terminal verdict is
`STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT`. A Cobb-Douglas-inspired
secondary score cannot change this verdict. If a policy passes, the allowed result is only
`PASS_PROGRAM_G_STYLIZED_ROBUST_OBSERVABLE_VALUE`; full-DES validation remains separate.

## Secondary construct analysis

The fixed Garrido-2024 exponents are reported only as a Cobb-Douglas-inspired sensitivity.
They were not calibrated for this MFSC. `phi` and `kappa` are constant in Program G v1.2.
Metric disagreement is reported as preference sensitivity, never as a replacement primary.
