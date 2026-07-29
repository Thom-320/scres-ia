# Garrido Original Runs Gate

Date: 2026-06-26

## Decision

The original Excel files are now treated as the reference data contract for the
Garrido replication lane. The DES may still be used for experiments, but a
behavioral fidelity claim is not allowed until this gate is explicit about what
matches and what remains blocked.

Current gate status: **partial pass, tail blocker remains**.

## Workbook Roles

- `Raw_data1+Re.xlsx`: primary row-level target for `CF1-CF10`.
  These sheets use the R1 risk family columns:
  `R11_1`, `R11_2`, `R12`, `R13`, `R14`.
- `Raw_data2+Re.xlsx`: primary row-level target for `CF11-CF20`.
  These sheets use the R2 risk family columns:
  `R21_1`, `R21_2`, `R21_3`, `R21_4`, `R21_5`, `R22_1`,
  `R22_2`, `R22_3`, `R22_4`, `R23`, `R24`.
- `Rsult_1.xlsx`: secondary post-processing workbook. It is useful for
  aggregate/distribution checks of `APj`, `RPj`, `DPj`, and `Re`, but should
  not be forced to act as a row-level replay equivalent of the raw workbooks.

## Excel Formula Gate

Audit artifact:

- `outputs/audits/garrido_excel_schema_formula_gate_2026-06-26/`

Result:

- Raw rows audited: `47,546`.
- Formula mismatches: `0`.
- `CF2` is present.
- Headers are read dynamically, so `Raw_data2+Re.xlsx` uses its 11 risk columns
  instead of the 5-column `Raw_data1+Re.xlsx` layout.
- Raw Excel `ReT > 1` values are genuine formula outputs, not stale cache
  artifacts. Total count across `CF1-CF20`: `38`.

Important correction: the earlier diagnosis that `Raw_data2+Re.xlsx` had stale
cached `ReT` values was caused by applying the `Raw_data1` risk-column layout to
`Raw_data2`. With the correct dynamic headers, the cached Excel values
recompute exactly.

The exact raw Excel formula remains:

```text
IF(AVERAGE(risk_columns)>0,
   IF(APj>0, APj/LT, 0.5*(1/RPj)),
   1-((sumBt+sumUt)/j))
```

No clamp is applied in the raw Excel reproduction lane.

## Fulfillment Delay Gate

Code path fixed:

- Direct on-hand fulfillment and pending-backorder fulfillment now both honor
  `demand_on_hand_fulfillment_delay`.
- The Garrido-facing default is currently
  `GARRIDO_FULFILLMENT_DELAY_HOURS = 54.0`.

Interpretation:

- `delay=0` reproduces legacy DES instant completions, but not Garrido's
  original-run structure.
- `delay=48` removes instant completions but leaves many orders exactly at
  `LT=48`, which keeps the Excel fill-rate branch high.
- `delay=54` is the smallest tested value that crosses the `LT=48` cliff and
  reproduces the Garrido raw-Excel order of magnitude for `ReT`.

This is a **provisional reproduction default**, not a complete behavioral
calibration. The backlog tail still has to be fixed or bounded before claiming
trajectory-level fidelity.

## Static Panel With Current Default

Artifact:

- `outputs/experiments/static_baseline_panel_delay54_default_2026-06-26/`

Because `delay=54` makes all served orders later than `LT=48`, the order-level
`fill_rate` column is `0.0` by construction in this raw-Excel contract. That
metric is therefore not a useful service objective under `delay=54`; service
must be evaluated with lost orders, pending backlog, flow/service-loss metrics,
and resource use.

Selected static results:

| Regime | Efficient low-resource policy | Ret Excel | Lost orders | Backlog qty | Shift hours | Buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| current | `I168_S1` | 0.0092 | 0.0 | 0.0 | 161k | 46k |
| increased | `I168_S1` | 0.0073 | 88.8 | 133,681.0 | 161k | 46k |
| severe | `I168_S1` | 0.0051 | 1,137.0 | 151,876.0 | 161k | 46k |

`I1344_S3` slightly improves lost orders in `increased` and `severe`, but at a
large resource cost: 484k shift-hours and 372k buffer units.

## Threshold Heuristic With Current Default

Artifact:

- `outputs/benchmarks/garrido_dynamic_vs_static/threshold_heuristic_delay54_5seed_104w_2026-06-26/`

Result: the threshold heuristic is not the new winner. The buffered heuristic
beats expensive or no-buffer comparators, but it does not beat the efficient
frontier:

- `current`: `heuristic_threshold_buffer` is `-0.0302` CD below `S1_I168`.
- `increased`: `heuristic_threshold_buffer` is `-0.0204` CD below `S1_I168`.
- `severe`: `heuristic_threshold_buffer` is `-0.0103` CD below `S2_I168`.

Conclusion: a simple backlog-reactive rule is a useful comparator, but not yet
the paper win. A dynamic policy would need anticipation or a better
service/resource reward, not just threshold reaction.

## Tail Blocker

Artifact:

- `outputs/audits/garrido_tail_r3_separation_2026-06-26/`
- `outputs/audits/garrido_tail_fidelity_after_overflow_fix_2026-06-26/`

The p99 tail is not explained by R3 alone.

At `delay=54`, `risk_level=increased`, `S1/I0`, 3 paired seeds:

| Horizon | Scenario | p50 CT | p90 CT | p99 CT | Lost | R3 events |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 10yr | all risks | 54 | 1,192 | 20,304 | 672 | 1.0 |
| 10yr | no R3 | 54 | 1,000 | 14,544 | 649 | 0.0 |
| 10yr | R1 only | 54 | 272 | 19,992 | 288 | 0.0 |
| 10yr | R2 only | 54 | 1,024 | 19,176 | 322 | 0.0 |
| 10yr | R3 only | 54 | 54 | 54 | 10,568 | 0 | 1.0 |

Garrido `CF1` has p99 around `6,628 h`. Removing R3 reduces some tail mass but
does not close the gap. Therefore the dominant blocker is the DES
backorder/recovery flow under ordinary R1/R2 disruptions, not the black swan by
itself.

Follow-up family audit:

| Combo | Family | CTj p99 ratio vs Excel | RPj p99 ratio vs Excel |
| --- | --- | ---: | ---: |
| `disruption/largest` | R1 | 2.05 | 4.52 |
| `disruption/largest` | R2 | 2.14 | 0.16 |
| `disruption/oldest` | R1 | 1.60 | 3.49 |
| `disruption/oldest` | R2 | 1.28 | 0.11 |
| `elapsed/largest` | R1 | 2.05 | 18.37 |
| `elapsed/largest` | R2 | 2.14 | 4.69 |

Interpretation:

- `ret_recovery_period_mode="disruption"` is the workbook-aligned default for
  `RPj`; `elapsed` is the legacy sensitivity and over-inflates RPj.
- `backorder_overflow_mode="largest"` remains the thesis-faithful default
  because the pending list is SPT-sorted and the thesis removes the last order
  in that list. `oldest` is useful as an age-based sensitivity: it reduces CTj
  tail mass but does not fully reproduce Garrido, and it is not the primary
  thesis reading.
- The residual mismatch is risk generation/overlap/catch-up, not just queue
  eviction.

## Consensus With Claude

Consensus:

- Dynamic header extraction is mandatory.
- `Raw_data2+Re.xlsx` must not use the `Raw_data1` risk-column map.
- `delay=54` is the current best default for reproducing the raw Excel
  late-order/ReT structure.
- The next fidelity blocker is the extreme backlog/recovery tail.

Correction to the proposed R3 hypothesis:

- Separating R3 was the right test, but R3 is not the sole cause. The p99 tail
  remains too heavy without R3, so the next investigation should target
  backorder aging, discharge priority, production bottlenecks, and lost-order
  handling under R1/R2.

## Next Gate

Do not claim that the DES fully behaves like Garrido's original model yet.
The R11-R24 calibration audit has now been run:

- `outputs/audits/garrido_risk_calibration_2026-06-26/`

Result:

- Event frequencies are broadly calibrated. Across `CF1-CF20`, R11, R13,
  R14, R21, R22, R23, and R24 are near the thesis-window expected rates; R12 is
  slightly low in this 1-seed audit but not enough to explain the tail.
- Workbook-visible order attribution is not fully calibrated:
  - R1: `R11_1/R11_2` are close, `R14` is close, `R13` is high
    (`1.27x`), and `R12` is low (`0.38x`).
  - R2: `R21_1..R21_5` are close, but `R22`/`R23` are low
    (`~0.56-0.62x`) and `R24` is low (`0.45x`).
- Backlog catch-up remains the binding blocker. With calibrated event
  frequencies, S1 has near-continuous backlog under R1/R2. R2 stays nearly
  continuous even with S2/S3, suggesting the R2 tail is downstream/transport
  recovery rather than assembly capacity alone.

Updated next gate:

1. Inspect event-to-order attribution for `R12`, `R22`, `R23`, and `R24`
   because event frequencies are right but visible order shares are low.
2. Audit downstream/backorder discharge after R2 events (`R21-R24`), where
   adding shifts does not clear the queue.
3. Test whether Garrido's recovery transient includes an implicit catch-up
   rule, transport priority, or buffer rebuild that our DES does not model.
4. Only after the endogenous tail is bounded, freeze whether `54 h` stays as
   the final default or whether `48 h` remains a service-oriented sensitivity.

## R2 Downstream Attribution Audit

Artifact:

- `outputs/audits/garrido_r2_downstream_2026-06-26/`

Result:

- Current event-to-order attribution is low for the workbook-visible R2
  downstream columns:
  - `R22`: DES order-positive share is about `0.62x` the Excel share.
  - `R23`: DES order-positive share is about `0.64x` the Excel share.
  - `R24`: DES order-positive share is about `0.44x` the Excel share.
- Extending attribution from event overlap to "until the backlog clears" is
  **not** a valid fix. It overcorrects by orders of magnitude because the R2
  backlog often never clears during the observed horizon:
  - tail-window `R22` share is about `29x` the Excel share;
  - tail-window `R23` share is about `93x`;
  - tail-window `R24` share is about `144x`.
- The R2 backlog tail is persistent, not a small local transient. Across
  `CF11-CF20`, 3 seeds, terminal pending backorder quantity averages about
  `144,902` rations, with max backlog interval fixed at `80,562 h`.

Interpretation:

- The DES is probably under-attributing R22/R23/R24 to orders if it only marks
  direct event overlap.
- But the correct Garrido-like attribution cannot be "mark every order until
  the backlog finally clears"; our endogenous R2 backlog is too persistent for
  that rule.
- The next calibration target is a **finite downstream recovery/catch-up
  transient** after R2 events: how Garrido releases transport/theatre stock,
  rebuilds buffer, or prioritizes backorders so the downstream queue does not
  remain open for years.

Updated blocker:

The current blocker is no longer raw R2 event frequency. It is the mismatch
between (a) finite R2 event durations and visible Excel order shares, and (b)
the DES downstream discharge process, which leaves a chronic backlog tail. Do
not run RL on the endogenous R2 lane until this recovery transient is bounded
or explicitly documented as an intentional extension.

## R2 Recovery Transient Sensitivity

Artifacts:

- `outputs/audits/garrido_r2_recovery_transient_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_transient_buffers_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_window_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_release_sweep_2026-06-26/`
- `outputs/audits/garrido_r2_stock_conserving_sweep_2026-06-26/`

Audit-only variants tested:

- faster downstream checks: `op9/op10/op12 ROP = 12 h`;
- larger downstream dispatch lots: `2x Q` for `op9/op10/op12`;
- non-blocking pending-order service: serve any pending order that fits theatre
  stock instead of blocking on the SPT queue head;
- downstream pipeline priming: `31,500` rations placed at CSSU/theatre/pipeline
  buffers before the run.

Summary:

| Variant | CT p99 / Excel | RP p99 / Excel | Pending qty | Backlog max h |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.09 | 0.56 | 144,902 | 80,562 |
| downstream `2x Q` | 1.69 | 0.46 | 141,556 | 80,562 |
| downstream `ROP=12h + 2x Q` | 1.69 | 0.45 | 141,370 | 80,562 |
| fit-any backorder | 2.11 | 0.58 | 288,723 | 80,562 |
| CSSU buffer 31,500 | 1.54 | 0.39 | 147,526 | 80,428 |
| downstream pipeline buffer 31,500 | 0.88 | 0.26 | 142,951 | 79,567 |
| pipeline buffer + `ROP=12h + 2x Q` | 0.73 | 0.21 | 140,568 | 77,893 |
| R2 window 336 h, no release | 1.70 | 0.45 | 141,300 | 80,562 |
| R2 window 336 h, release 2,500 | 1.15 | 0.36 | 43,097 | 52,657 |
| R2 window 336 h, release 5,000 | 0.23 | 0.09 | 0 | 13,362 |
| R2 window 336 h, release 10,000 | 0.06 | 0.05 | 0 | 3,714 |
| R2 window 336 h, release 31,500 | 0.01 | 0.03 | 0 | 362 |
| R2 window 336 h, stock-conserving move 2,500 | 1.79 | 0.49 | 141,665 | 80,562 |
| R2 window 336 h, stock-conserving move 5,000 | 1.71 | 0.49 | 146,268 | 80,562 |
| R2 window 336 h, stock-conserving move 10,000 | 1.89 | 0.52 | 143,396 | 80,562 |

Interpretation:

- Faster/larger downstream dispatch improves `CTj p99`, so downstream movement
  is part of the missing mechanism.
- Pipeline priming can make served-order `CTj p99` look Garrido-like or even
  faster than Garrido, but it does **not** drain the chronic terminal backlog.
- Non-blocking service priority does not solve the blocker; by itself it worsens
  terminal pending quantity.
- A finite R2 recovery window without released stock is not enough. Adding
  released stock identifies the missing mechanism direction, but the magnitude
  is delicate:
  - `2,500` rations per R2 event gives the closest CTj tail (`1.15x` Excel), but
    still leaves `43k` terminal pending rations.
  - `5,000+` rations clears terminal backlog, but makes CTj/RPj far too small
    and creates excessive theatre stock.
- A stock-conserving recovery window that only moves existing downstream stock
  from CSSU/SB/SB-dispatch toward theatre does **not** pass the stop-rule:
  `CTj p99` remains `1.71-1.89x` Excel and the backlog interval remains open
  for `80,562 h`.

Conclusion:

The missing Garrido behavior is not a simple permanent increase in downstream
throughput, not a queue-priority-only issue, and not just initial downstream
stock. The next candidate is a **finite post-R2 recovery window**: after R22,
R23, or R24, the original model appears to bound how long the risk affects
later orders and/or applies a temporary catch-up/rebuild process that prevents
the backlog interval from staying open for the rest of the horizon.

Do not promote any release magnitude as a default yet. The release sweep shows
where the mechanism lives, but not a faithful calibrated rule. The next
implementation should model a bounded, stock-conserving downstream recovery
process rather than injecting unlimited theatre inventory.

Stop-rule result:

The one bounded stock-conserving attempt did **not** close the endogenous R2
tail to `<1.5x` Excel. For the next paper-facing experiment, the defensible
path is to keep the forensic `excel_risk_tape` lane as the verified Garrido
replication, document the endogenous R2 tail as an open fidelity limitation,
and avoid claiming full endogenous trajectory replication until a stronger
stock-conserving recovery model is specified.

## R1 R14 Attribution Fix

Artifact:

- `outputs/audits/garrido_r14_attribution_2026-06-26/`

Correction:

The earlier branch-composition suspicion was incomplete. The R1 mean gap was
not driven by no-risk fill-rate rows: in the endogenous DES those rows are
mostly lost or pending and contribute almost zero to the Excel formula. The real
inflator was R14's period contribution.

Before the fix, R14 quantity-risk attribution marked later orders correctly but
contributed only `1 h` to `APj/RPj`. That made `R14_only` orders score roughly
`0.5 / 1 = 0.5`, while Garrido's `R14_only` rows have `RPj` median around
`72 h` and mean ReT around `0.008`.

The DES now keeps the R14 visible indicator as `1.0`, but uses
`GARRIDO_R14_RET_PERIOD_HOURS = 72.0` for the AP/RP/DP period contribution.

Verification over `CF1-CF10`, seeds `1,2,3`:

| Metric | Excel R1 | DES after R14 fix |
| --- | ---: | ---: |
| mean ReT | 0.00625 | 0.00441 |
| risk-active share | 0.99965 | 0.98142 |

This closes the main R1 headline ReT gap to the right order of magnitude. The
remaining difference is now small compared with the previous `~0.089` DES mean
and should be treated as residual calibration, not as the dominant blocker.
