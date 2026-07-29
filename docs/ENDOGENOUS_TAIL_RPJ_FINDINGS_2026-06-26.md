# Endogenous Tail & RPj Divergence — Forensic Findings

Date: 2026-06-26
Scope: GARRIDO_ORIGINAL_RUNS_GATE next-gate items 1-3 (CTj/RPj/DPj quantiles by
family, oldest-backorder inspection, old-order handling). Replaces the prior
"backorder aging / discharge priority" hypothesis with a measured root cause.

## TL;DR

The forensic `excel_risk_tape` lane reproduces Garrido (ReT MAE 0.0038). The
**endogenous** `des_events` lane still diverges. Two mechanisms matter:

1. `RPj` must be bounded to disruption/recovery duration, not elapsed queue
   wait. The default is now `RET_RECOVERY_PERIOD_MODE="disruption"`.
2. The remaining `CTj/DPj` tail is backlog starvation under high R11/R13
   frequency: some orders wait for years even though individual risk events are
   short and bounded (R1 max 504 h, R2 max 822 h).

## 1. Excel targets (order-level, pooled)

| Family | Metric | p50 | p90 | p99 | max |
| --- | --- | ---: | ---: | ---: | ---: |
| R1 (CF1-10) | CTj | 101.5 | 1539.6 | 7,423.8 | 52,137 |
| R1 | RPj | 99.4 | 428.9 | 831.7 | 1,156 |
| R2 (CF11-20) | CTj | 116.2 | 1465.3 | 10,087 | 37,617 |
| R2 | RPj | 96.0 | 1410.0 | 4,544 | 7,116 |

- `CTj ≡ DPj` in the workbooks (DPj = CTj in the recovery branch).
- `RPj` is **bounded and much smaller than CTj** for long-tail orders.

## 2. DES endogenous (`des_events`, delay=54, 3 seeds) vs Excel

| Family | Metric | DES p99 | Excel p99 | ratio |
| --- | --- | ---: | ---: | ---: |
| R1 | CTj | 15,311 | 7,424 | 2.1x |
| R1 | RPj | 3,770 | 832 | 4.5x |
| R2 | CTj | 21,549 | 10,087 | 2.1x |
| R2 | RPj | 737 | 4,544 | 0.16x |

The legacy DES made `RPj == CTj` for many R1 orders because
`_set_order_ret_indicators` computed `RPj = OATj - max(earliest_risk_start,
OPTj)`, which collapses to elapsed queue wait when a risk is ongoing at order
placement. Garrido's RPj is the bounded disruption duration, so the default
was changed to `ret_recovery_period_mode="disruption"`. This fixes the gross
RPj inflation but exposes a second problem: R1 disruption overlap is now too
large, while R2 overlap is too small.

## 3. Root cause of the CTj tail: chronic starvation, not overflow

Inspection of the top-CTj served orders (`sandbox/inspect_tail_orders.py`):

- R1, j=501: CTj = 142,368 h over a window with **1,476 overlapping risk events**
  (848 R11 + 603 R13 + 25 R12), but the summed overlap is only 32,639 h. The
  remaining ~110,000 h is pure backlog wait with production chronically behind.
- R2, j=159: CTj = 138,672 h with 62 events (overlap 6,276 h).
- Single-event durations are fine: R1 max **504 h**, R2 max **822 h**.

Over 20 years the R1 lane generates ~7,397 risk events; a single order can
overlap ~1,476 of them. The cumulative downtime keeps the backlog deeper than
shift capacity can clear, so orders at the back wait essentially the full
horizon. Garrido's system recovers between events; ours does not catch up.

## 4. Hypotheses tested (measured)

- **Overflow direction.** The faithful default remains
  `BACKORDER_OVERFLOW_MODE="largest"` because the queue is SPT-sorted and thesis
  Sec. 6.5.4 says the last order in the pending list is removed. A corrected
  age-based sensitivity, `oldest`, drops the earliest `OPTj` and reduces CTj
  p99 (R1 15,311 -> 11,959; R2 21,549 -> 12,879), but it does not fully match
  Garrido and is not the primary thesis reading.
- **RPj formula.** `ret_recovery_period_mode="disruption"` reduces legacy RPj
  inflation (R1 p99 15,311 -> 3,770; R2 21,331 -> 737) and is now the default.
  It still does not match Excel per-family because the underlying risk
  generation/overlap differs.
- **Single black-swan / R3.** Already ruled out by the R3-separation tail audit.

## 5. What this means for the gate

- The forensic lane remains the validated replication path (ReT MAE 0.0038).
- The endogenous lane cannot match Garrido's CTj/RPj until the **risk
  frequency/recovery generation** is calibrated against thesis Tables 6.6b/6.7b
  and the workbook-visible risk columns. This is the real next blocker.
- The smoke test has been moved from the historical `legacy_renewal` negative
  control to the paper-facing `thesis_window` lane and now passes with the
  delay-54 default.

## 6. Smoke test correction

`tests/test_garrido_replication_harness.py::test_replicate_garrido_excel_smoke_with_order_tape`
now uses `risk_occurrence_mode=thesis_window`, matching the paper-facing
Garrido lane. `legacy_renewal` remains a historical negative control and should
not be the smoke for delay-54 reproduction.

## 7. Next step

The R11-R24 calibration audit was executed in
`outputs/audits/garrido_risk_calibration_2026-06-26/`.

Result:

- Event frequencies are not the main blocker: R11/R13/R14 and R21-R24 are near
  thesis-window expected rates across `CF1-CF20`; R12 is somewhat low in the
  1-seed audit but not tail-dominant.
- Workbook-visible order attribution still differs. R12, R22, R23, and R24 are
  under-attributed at the order level even when event counts are right.
- Backlog catch-up remains the dominant behavioral gap. R1 improves under
  S2/S3 in most seeds, but R2 remains nearly continuously backlogged even with
  S2/S3, pointing to downstream/transport recovery rather than assembly shifts
  alone.

Next: inspect event-to-order attribution and downstream backlog discharge after
R2 events before running RL.

Artifacts: `sandbox/excel_quantiles_by_family.py`,
`sandbox/des_quantiles_by_family.py`, `sandbox/inspect_tail_orders.py`,
`sandbox/results/*.json`, and the formal audit
`outputs/audits/garrido_tail_fidelity_after_overflow_fix_2026-06-26/`.

## 8. R11-R24 calibration results (2026-06-26, continued)

The three calibration hypotheses in section 7 were tested and **ruled out**:

1. **R11 frequency is correct.** Thesis Table 6.6b specifies one breakdown per
   168 h window, exponential repair β=2 h. The DES generates exactly that: 848
   R11 events over 142,368 h = 1 per 168 h. Risk frequency is NOT the cause.
2. **No throughput bug.** With risks disabled, S1 serves 100% of demand
   (719,702 rations/yr served = demanded), zero lost orders. Capacity exceeds
   demand; production is not the bottleneck.
3. **The tail is a high-utilization queueing effect.** S1 runs at ~97.5%
   utilization (capacity 2,564/day vs demand ~2,500/day). Even with NO risks,
   S1 gives CTj p99 = 3,561 h (max 34,488 h) from demand/production variability
   alone. Risks amplify it to p99 14,454 h. S2/S3 (~49%/33% utilization) are
   fine: p99 ~ 1,200 h. This is faithful to the thesis's deliberately tight S1.

**Excel RPj semantics resolved.** Per-order correlation on CF1 shows
`RPj ~ risk_event_count x 168 h` (R11_1 ratio 168.03, R11_2 ratio 168.44,
R14 ratio 24.63). Crucially, in the workbooks one order overlaps at most ~6 R11
events (R11_1 max = 6), whereas a long-waiting DES order overlaps 848. The
divergence is therefore **circular**: long backlog wait -> many overlapping
events -> inflated RPj/CTj. Bounding the wait bounds RPj automatically.

**Conclusion.** The remaining endogenous-vs-Excel gap is NOT a raw
risk-generation calibration problem (frequencies and throughput match the
thesis closely enough for this gate). It is a **backlog-clearing dynamics**
problem at high utilization: after a disruption the DES queue takes longer to
drain than Garrido's. The next investigation target is the post-disruption
recovery transient (shift coverage, warmup priming, transport priority, buffer
rebuild), not another broad risk-frequency sweep.

Current defaults are split by purpose:

- `ret_recovery_period_mode="disruption"` is now the workbook-aligned RPj
  default.
- `backorder_overflow_mode="largest"` remains the thesis-faithful pending-list
  default, because the queue is SPT-sorted and thesis Sec. 6.5.4 removes the
  last order in that list.
- `oldest` remains only an age-based sensitivity.

## 9. R2 downstream audit (2026-06-26, continued)

Artifact:

- `outputs/audits/garrido_r2_downstream_2026-06-26/`

The R2-specific audit inspected R22/R23/R24 event-to-order attribution and the
downstream discharge state after R2 events. It confirms a two-part blocker:

1. **Direct event-overlap attribution is too low.** Compared with the Excel
   visible order shares, DES order-positive shares are about `0.62x` for R22,
   `0.64x` for R23, and `0.44x` for R24.
2. **Attributing every delayed order until backlog clearance is much too high.**
   Because the R2 backlog often never clears in the observed horizon, a
   tail-window attribution explodes to about `29x` Excel for R22, `93x` for
   R23, and `144x` for R24.

The terminal downstream state makes the mechanism visible. Across `CF11-CF20`
and three seeds, terminal pending backorder quantity averages about `144,902`
rations, and the maximum positive-backlog interval is `80,562 h` in every run.

Therefore the right next experiment is not RL and not a blanket propagation of
R2 flags. The next calibration target is a finite Garrido-like downstream
recovery/catch-up transient: how long an R2 event should keep affecting later
orders, how theatre/CSSU stock is released, and whether backorders receive an
implicit priority or buffer rebuild after disruption.

## 10. R2 recovery transient sensitivities (2026-06-26, continued)

Artifacts:

- `outputs/audits/garrido_r2_recovery_transient_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_transient_buffers_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_window_2026-06-26/`
- `outputs/audits/garrido_r2_recovery_release_sweep_2026-06-26/`
- `outputs/audits/garrido_r2_stock_conserving_sweep_2026-06-26/`

Audit-only variants tested the candidate mechanisms directly:

| Variant | CTj p99 / Excel | RPj p99 / Excel | Terminal pending qty | Max backlog interval |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2.09 | 0.56 | 144,902 | 80,562 h |
| downstream `2x Q` | 1.69 | 0.46 | 141,556 | 80,562 h |
| downstream `ROP=12h + 2x Q` | 1.69 | 0.45 | 141,370 | 80,562 h |
| fit-any backorder priority | 2.11 | 0.58 | 288,723 | 80,562 h |
| CSSU buffer 31,500 | 1.54 | 0.39 | 147,526 | 80,428 h |
| downstream pipeline buffer 31,500 | 0.88 | 0.26 | 142,951 | 79,567 h |
| pipeline buffer + `ROP=12h + 2x Q` | 0.73 | 0.21 | 140,568 | 77,893 h |
| R2 window 336 h, no release | 1.70 | 0.45 | 141,300 | 80,562 h |
| R2 window 336 h, release 2,500 | 1.15 | 0.36 | 43,097 | 52,657 h |
| R2 window 336 h, release 5,000 | 0.23 | 0.09 | 0 | 13,362 h |
| R2 window 336 h, release 10,000 | 0.06 | 0.05 | 0 | 3,714 h |
| R2 window 336 h, release 31,500 | 0.01 | 0.03 | 0 | 362 h |
| R2 window 336 h, stock-conserving move 2,500 | 1.79 | 0.49 | 141,665 | 80,562 h |
| R2 window 336 h, stock-conserving move 5,000 | 1.71 | 0.49 | 146,268 | 80,562 h |
| R2 window 336 h, stock-conserving move 10,000 | 1.89 | 0.52 | 143,396 | 80,562 h |

Findings:

- Faster/larger downstream dispatch helps the served-order CTj tail, but does
  not close the backlog interval.
- Pipeline priming can match or beat Garrido's served-order CTj p99, but still
  leaves a large terminal backlog. It changes the sample of served orders more
  than it proves true catch-up.
- Non-blocking "serve any order that fits" is not the missing Garrido rule. It
  reduces some lost-order pressure but increases terminal pending quantity.
- A finite R2 catch-up window without inventory release is too weak. A release
  of `2,500` rations per R2 event gets the CTj tail closest to Excel (`1.15x`)
  but still leaves `43k` pending rations. Releases of `5,000+` clear backlog
  but overcorrect CTj/RPj and create excess theatre stock.
- A stock-conserving finite window that moves only existing downstream inventory
  does not clear the gate. CTj p99 remains above the `<1.5x` stop-rule and the
  backlog interval remains open through the horizon.

Therefore the next calibration experiment should implement a bounded,
stock-conserving **post-R2 recovery process** rather than a permanent capacity
change or an unlimited stock injection. The plausible missing mechanism is
finite downstream catch-up/rebuild after R22/R23/R24, but the current release
sweep is only diagnostic. This is the last fidelity gate before any RL claim on
the endogenous R2 lane.

Stop-rule outcome: the first stock-conserving implementation did not close the
tail. The project can now move forward with the forensic Garrido-replay lane as
the verified replication path, while documenting the endogenous R2 catch-up
tail as a known limitation rather than spending more cycles on unbounded
fidelity tuning.

## 11. R1 R14 period-attribution fix

Artifact:

- `outputs/audits/garrido_r14_attribution_2026-06-26/`

The R1 mean ReT mismatch was traced to R14's period contribution. The DES
already fed R14 events into the persistent quantity-risk queue, so the problem
was not dead code and not simply a missing risk gate. The problem was that each
R14 quantity-risk attribution contributed only `1 h` to the AP/RP/DP period
logic. That made `R14_only` orders score near `0.5`, because the Excel formula
uses `0.5/RPj`.

Raw Excel R1 shows `R14_only` rows with `RPj` median around `72 h`, so the DES
now keeps the visible `R14` indicator as `1.0` but applies
`GARRIDO_R14_RET_PERIOD_HOURS = 72.0` as the period contribution.

Verification over `CF1-CF10`, seeds `1,2,3`:

- Excel R1 mean ReT: `0.00625`.
- DES R1 mean ReT after fix: `0.00441`.
- DES R1 risk-active share after fix: `0.98142`.

This fixes the dominant R1 headline ReT mismatch. The endogenous R2 downstream
tail remains a separate limitation for CTj/RPj/CVaR, but it is no longer the
dominant explanation for the R1 mean-ReT gap.
