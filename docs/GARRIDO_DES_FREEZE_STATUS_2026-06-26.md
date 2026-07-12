# Garrido DES Freeze Status - 2026-06-26

## Purpose

This is the pre-PPO freeze gate for the thesis-faithful Garrido lane.  The goal
is to ensure that training starts from the closest current DES environment to
Garrido-Rios (2017), while keeping the claim boundary explicit.

## Defaults Frozen for Training

The base DES and the shift-control Gym environment now default to the
thesis-faithful lane:

- `warmup_trigger = op9_arrival`
- `downstream_q_source = figure_6_2`
- `r14_defect_mode = thesis_strict_op6`
- `risk_occurrence_mode = thesis_window`
- `raw_material_flow_mode = kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier = 2.0`
- `demand_on_hand_fulfillment_delay = 54.0`
- `ret_recovery_period_mode = disruption`
- `backorder_overflow_mode = largest`
- `BACKORDER_QUEUE_CAP = 60`

Legacy alternatives remain available as explicit sensitivity modes, but they
are no longer the default path for training.

## Workbook / Excel Gate

Source workbooks:

- `Raw_data1+Re.xlsx`: primary order-level target for CF1-CF10.
- `Raw_data2+Re.xlsx`: primary order-level target for CF11-CF20.
- `Rsult_1.xlsx`: secondary aggregate/distribution workbook.

Audit artifact:

- `outputs/audits/freeze_check_garrido_workbooks_2026-06-26/audit_summary.json`

Result:

- Raw rows audited: `47,546`
- Recomputed Excel formula mismatches: `0`
- CF2 dynamic header detected: `true`
- Formula: `IF(AVERAGE(risk_cols)>0, IF(APj>0, APj/LT, 0.5*(1/RPj)), 1-((sumBt+sumUt)/j))`

## Forensic Replication Gate

Artifact:

- `outputs/audits/freeze_check_replicate_garrido_2026-06-26/replication_audit.json`

Configuration:

- `demand_source = excel_order_tape`
- `risk_occurrence_mode = thesis_window`
- `risk_attribution_source = excel_risk_tape`
- `seed_stream_mode = split`

Result:

- `replication_status = passed_gate`
- Global MAE: `0.0023058419`
- Max branch-share gap: `0.0 pp`
- R1 target ReT: `0.0062823`; simulated replay ReT: `0.0063017`
- R2 target ReT: `0.2007417`; simulated replay ReT: `0.2051462`

This is the numeric Garrido Excel replication claim.

## Thesis Risk-Frequency Gate

Artifact:

- `outputs/audits/freeze_check_risk_frequency_2026-06-26/thesis_risk_frequency.json`

Result:

- Overall status: `PASS`
- R11, R12, R13, R14, R21, R22, R23, R24, R3: `MATCH`

The risk process frequencies match the thesis tables under `thesis_window`.

## Endogenous DES Gate

Artifacts:

- `outputs/audits/freeze_check_risk_calibration_2026-06-26/audit_report.md`
- `outputs/audits/freeze_check_tail_fidelity_2026-06-26/audit_report.md`
- `outputs/audits/garrido_des_family_match_after_bt_cap_2026-06-26/summary.json`

Current result:

- R1 endogenous ReT is in the thesis scale and close enough for training:
  Excel `0.006253`; DES `0.004461`.
- R2 endogenous ReT is not an exact trajectory replication:
  Excel `0.201916`; DES `0.482692`.
- Tail fidelity remains imperfect:
  - R1 CTj p99 ratio: `2.05`
  - R1 RPj p99 ratio: `4.61`
  - R2 CTj p99 ratio: `2.14`
  - R2 RPj p99 ratio: `0.16`

Interpretation:

- Event frequencies are calibrated.
- Formula replay is calibrated.
- The remaining endogenous gap is event-to-order attribution and backlog/tail
  dynamics, especially R2.  This is a limitation for claiming full endogenous
  trajectory replication, not a blocker for a fair intra-DES PPO vs static
  comparison.

## Test Gate

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

- `380 passed`
- `3 warnings`

## Claim Boundary

Passed:

- Excel formula replication.
- CF1-CF20 forensic replay replication.
- Thesis risk-frequency gate.
- Training defaults aligned to the thesis-faithful lane.
- Full test suite.

Not passed:

- Full endogenous DES trajectory equivalence to Garrido Excel for R2/tails.

Approved next step:

- PPO / dynamic-vs-static experiments may start from this frozen environment.
- Claims must say that numeric Excel replication uses the forensic tape lane;
  endogenous DES results are a calibrated thesis-faithful simulation with a
  documented R2/tail limitation.
