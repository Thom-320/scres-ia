# Garrido fidelity mechanism sprint verdict (2026-07-10)

## Executive verdict

`garrido_reference_v2` is **not promoted**. The corrected workbook-view gate
passes mass conservation but fails the five empirical gates: ReT, risk-active
share, warm-up, CT quantiles, and RP quantiles. The defensible statement remains:

> The environment reproduces Garrido's structural specification and has audited
> physical causality, but it does not yet reproduce the experimental
> distributions endogenously.

No RL training should use the label `garrido_reference_v2` until this gate is
closed or replaced by a preregistered distributional-validation design.

## Audit of the prior diagnosis

| Claim | Verdict | Evidence |
|---|---|---|
| Too few demand orders are generated | **False** | CF1: Excel has 4,241 visible rows but `max(j)=5,714`; DES places 5,724. CF11: 2,165 rows but `max(j)=2,844`; DES places 2,845. Missing rows are lost/pending/in-flight orders. |
| Workbook rows are attended orders only | **Supported** | Sparse `j`, dynamic `Bt`, and cumulative `Ut` jointly establish a workbook view rather than a placement ledger. |
| Zero workbook orders have `CT<=48 h` | **Supported** | CF1 and CF11 minima are 48.00744 h; neither contains an on-time row under the strict formula. |
| Op10/Op12 capacity-one convoy queues explain the CT body | **Falsified as a major mechanism** | CF1 is bitwise unchanged. In CF11, the Op12 resource-queue p95 is only 120 h and CT p95 is essentially unchanged. |
| Explicit serial WIP at Op5-Op7 will repair the CT body | **Falsified as implemented** | Across all ten odd CFs, mean CT-p95 log error worsens from 0.547 to 0.776 and mean absolute ReT error worsens from 0.0692 to 0.0881. |
| `elapsed` is the thesis RP definition | **Supported** | Thesis Algorithm 2 defines `RPj = OATj - first-R0cr`. |
| Current overlap attribution correctly identifies `first-R0cr` | **False** | It marks every event overlapping `[OPTj,OATj]`, including events unrelated to the order's physical delay. Risk-conditional RP tails then reach tens of thousands of hours. |
| A fixed 6 h Op9 departure phase is thesis-supported | **Unsupported** | Daily freight is supported; an absolute 06:00 phase is not specified. |
| A fixed 168 h R24 attribution window is proven | **Bounded screen only** | It improves aggregate R2 gaps, but it is inferred from marked-orders/event and is not tied to the lifetime of contingent demand or its induced backlog. |

## Corrected gate result

Artifact: `outputs/audits/garrido_reference_v2_gate_r3/gate.json`.

| Gate | Result |
|---|---:|
| Mass conservation | PASS |
| Mean absolute ReT gap <= 0.02 | FAIL (0.03547) |
| Mean absolute risk-share gap <= 0.05 | FAIL (0.06560) |
| Every validation warm-up within 10% | FAIL |
| Every validation CT p50/p95 ratio in [0.75,1.33] | FAIL |
| Every validation RP p50/p95 ratio in [0.75,1.33] | FAIL |

The R24 168 h proxy improves the first two empirical gaps relative to gate r2
(0.07262 and 0.13397), but it does not promote the environment.

## Mechanism ablations on calibration CFs only

Artifacts:

- `outputs/audits/garrido_mechanism_audit_cf1_cf11/`
- `outputs/audits/garrido_mechanism_audit_serial_calibration/`
- `outputs/audits/garrido_mechanism_audit_release_calibration/`
- `outputs/audits/garrido_mechanism_audit_release_clock_screen/`
- `outputs/audits/garrido_mechanism_audit_r12_initial_screen/`

### Where CT is generated

For the current clock/parallel model, the Op9 pre-release wait dominates CT:

| CF | CT p95 | Op9 release-wait p95 | Downstream resource-wait p95 |
|---|---:|---:|---:|
| CF1 | 1,085 h | 1,037 h | 0 h |
| CF11 | 2,732 h | 2,684 h | 0 h |

Therefore the first-order mechanism is stock/release dynamics at Op9, not an
unmodelled capacity queue on Op10/Op12.

### Serial assembly stop rule

The explicit balanced-line candidate increases attended orders but compresses
the CT distribution in the wrong direction. It is rejected as the next repair.
Keep it as an appendix ablation; do not add more station-level tuning.

### Release-clock finding

The legacy code measures Op2/Op3 ROP periods after completing processing and
transport. Thus Op2 drifts from a 672 h start cadence to 696 h and Op3 from
168 h to 216 h. A start-to-start candidate is directly grounded in the stated
ROP semantics. It moves CF11 warm-up from 943 h to 816 h against Excel 823.65 h,
visible orders from 1,903 to 2,130 against 2,165, and CT p50 from 118 h to
101 h against 103 h. This is the strongest physical correction found.

It is not yet promoted because it increases R1 supply and compresses CF1 CT
tails unless the first contracting-cycle R12 delay is represented correctly.

### R12 initialization and RNG confounding

The thesis samples R12 for each contracting process and adds 168 h per delayed
contract. The current generator waits 4,032 h before its first draw, so R12
cannot affect the initial contract or warm-up. An initial-cycle option was
implemented for audit.

However, all risk families consume one shared `risk_rng`. Adding the initial
R12 draw changes all subsequent R11/R13 draws even when R12 itself is zero.
The first screen was therefore confounded. A stable per-risk RNG option was
subsequently implemented and the three-arm comparison was repeated.

Artifact:
`outputs/audits/garrido_mechanism_audit_rng_release_r12_calibration/`.

| Arm | R1 warm-up MAPE | R2 warm-up MAPE | R1 visible MAPE | R2 visible MAPE | R1 CT-p95 log error | R2 CT-p95 log error |
|---|---:|---:|---:|---:|---:|---:|
| Legacy release, deferred R12 | 0.165 | 0.128 | 0.029 | 0.059 | 0.773 | 0.400 |
| Start-to-start, deferred R12 | 0.203 | **0.024** | 0.124 | **0.037** | 1.103 | **0.377** |
| Start-to-start, initial R12 | 0.173 | **0.024** | 0.140 | **0.037** | 1.285 | **0.377** |

The release correction is compelling for R2 but fails the joint promotion
rule because it worsens R1 and CT/RP behavior. Initial-cycle R12 does not rescue
the combined mechanism. **Stop:** do not run these arms on even CFs.

## Risk attribution diagnosis

Under CF1 clock/parallel/elapsed:

| Risk | DES share | Excel share | DES RP p95 | Excel RP p95 |
|---|---:|---:|---:|---:|
| R11 | 0.529 | 0.715 | 3,312 h | 461 h |
| R12 | 0.048 | 0.092 | 44,868 h | 940 h |
| R13 | 0.538 | 0.291 | 3,269 h | 515 h |
| R14 | 1.000 | 0.984 | 1,062 h | 450 h |

Under CF11:

| Risk | DES share | Excel share | DES RP p95 | Excel RP p95 |
|---|---:|---:|---:|---:|
| R21 | 0.116 | 0.199 | 25,734 h | 4,482 h |
| R22 | 0.108 | 0.291 | 26,277 h | 4,359 h |
| R23 | 0.267 | 0.343 | 14,607 h | 4,293 h |
| R24 | 0.549 | 0.750 | 5,377 h | 3,728 h |

`elapsed` is mathematically correct once the responsible risk is known. The
error comes from using temporal overlap as responsibility. A long-waiting order
collects unrelated later events; `first-R0cr` then becomes arbitrarily early.

## Frozen stop and promotion rules

### Stop now

- Do not implement further Op10/Op12 convoy capacity tuning.
- Do not tune the 6 h departure offset.
- Do not refine serial Op5-Op7 WIP parameters.
- Do not evaluate more mechanism variants on even CFs.
- Do not start RL.

### Completed in this sprint

1. Each risk family now has an opt-in independent, stable RNG stream.
2. The following arms were run on odd CFs only:
   - legacy release + deferred R12;
   - start-to-start release + deferred R12;
   - start-to-start release + initial-cycle R12.
3. The preregistered joint rule was not met; the mechanism was not promoted.

### Mandatory next experiment

1. Replace retrospective overlap with a causal exposure ledger. For each order,
   distinguish stock/release waiting, resource-queue waiting, and time actually
   blocked by an affected operation. `elapsed` may use only the first risk that
   contributes to one of those causal intervals.
2. Replace the fixed R24 window with a causal exposure state that closes when
   the contingent order and its induced backlog are resolved. Promote only if
   R24 share and RP quantiles improve without degrading R21-R23.
3. Promotion requires, jointly across odd CFs:
   - warm-up absolute percentage error decreases for both R1 and R2 families;
   - visible/lost-order error decreases;
   - CT p50 and p95 errors do not worsen in either family.
4. Only after freezing one candidate, execute the unchanged gate once on even
   CFs.

## Code added in this sprint

- `scripts/audit_garrido_mechanisms.py`: calibration/validation-safe mechanism
  audit with per-risk shares, conditional CT/RP/DP, event counts, release waits,
  transport waits, and warm-up milestones.
- `assembly_flow_mode`: `aggregate_line` versus `serial_wip` ablation.
- `periodic_release_mode`: legacy completion-relative versus start-to-start ROP.
- `operational_risk_initialization_mode`: deferred versus initial R12 cycle.
- `risk_rng_mode`: shared legacy stream versus stable per-risk streams.
- diagnostic per-order causal wait ledger; it does not enter ReT.

These options are opt-in. Frozen adaptive lanes retain their previous defaults.

## Addendum: causal-exposure attribution arm (Fable, post-sprint)

The mandated causal-exposure candidate was implemented
(`risk_attribution_source="causal_exposure"`: per-event endogenous exposure
ending when the order backlog returns to its pre-event level, computed lazily
from a queue-length history; R24 exposure from the surge until its induced
backlog is absorbed; AP keeps raw outage overlap) and evaluated against the
pre-registered rule on the ten odd CFs, three arms:

| Arm (odd-CF means) | R2 ret_gap | R2 risk_gap | R2 RP95 log-err | R1 (all metrics) |
|---|---:|---:|---:|---|
| raw overlap | 0.137 | 0.234 | 0.312 | identical across arms |
| 168 h R24 window | 0.054 | 0.064 | 0.200 | identical |
| causal exposure | 0.135 | 0.231 | 0.310 | identical |

**Verdict: the causal-exposure arm is FALSIFIED as implemented** — it adds
almost nothing over raw overlap. Mechanism: our backlog is too SHALLOW most
of the time (the very CT-body defect this sprint isolated), so the
queue-relief close rule fires immediately and exposures collapse to the raw
outage window. The single-CF read (CF11 RP95 ratio 1.16) was not
representative. Artifact: `outputs/audits/attribution_three_arm_odd_cfs.json`.

**Implication (ordering of repairs):** attribution surgery is DOWNSTREAM of
queue-depth physics. Until the Op9 stock/release dynamics reproduce Garrido's
standing mid-band backlog (the start-to-start release-clock candidate is the
strongest lead: CF11 warm-up 816 vs 823.65 h, visible orders 2,130 vs 2,165,
CT p50 101 vs 103 h — blocked only by its R1-side interaction with the first
R12 contracting cycle), no exposure mechanism has the substrate it needs. The
168 h window remains a disclosed calibration, not a mechanism, and even it
leaves odd-CF ret_gap 0.054 vs the 0.02 bar. Next sprint = resolve the
release-clock x initial-R12 interaction on the R1 side, then revisit
exposure.
