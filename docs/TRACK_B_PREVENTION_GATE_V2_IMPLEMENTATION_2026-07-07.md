# Track B Prevention Gate v2 implementation - 2026-07-07

## Verdict

Implemented. The old `R_full - R_reset(pre-risk)` splice gate remains
retracted as prevention evidence. Gate v2 is now available as an experimental
audit harness based on complete reruns under replayed/edited risk-event tapes.

This does **not** create a new prevention claim. It creates the machinery needed
to test whether selected Case C has any causal preventive headroom before
training another policy.

## What changed

### Risk-event tape/replay

Added `supply_chain/risk_event_tape.py`:

- serializes `sim.risk_events` to JSON/CSV;
- reloads tapes into normalized records;
- supports anchor removal and simple event insertion helpers;
- supports R24 clustering helper for future cluster-level gates.

Added opt-in `risk_event_tape` plumbing to Track B:

- `MFSCSimulation(..., risk_event_tape=...)`;
- `MFSCGymEnvShifts(..., risk_event_tape=...)`;
- default behavior is unchanged when `risk_event_tape=None`;
- when a tape is supplied, stochastic risk generators are replaced by a replay
  process and adaptive risk-regime risk generation is disabled.

Replay semantics currently cover the prevention-gate target risks:

- R22/R23 and other duration events with `affected_ops`: replayed as down/up
  intervals;
- R24: replayed as a point contingent-demand surge using the recorded
  `magnitude`;
- R14/other point events: retained in the event ledger conservatively for
  attribution/audit.

### Gate v2 auditor

Added `scripts/audit_track_b_prevention_gate_v2.py`.

Modes:

- `forced_prep_sweep`: complete reruns with calm vs forced-prep action during
  the pre-anchor window.
- `event_on_off`: complete reruns with the anchor present vs removed from the
  replayed tape.
- `oracle_warning`: perfect-warning/preaction assay implemented as prep allowed
  vs preaction blocked, without training.

Outputs:

- `gate_v2_event_rows.csv`;
- `gate_v2_summary.csv`;
- `gate_v2_placebos.csv`;
- `metadata.json`;
- optional serialized `tapes/*.json`.

The auditor filters out anchors whose pre-window is not fully inside the
post-warmup controllable period. Default: `control_start_hours=168`, `lead=4`
weeks.

## Outcome definition

Gate v2 uses local event outcomes rather than episode-level ReT:

- `event_ret_excel_mean`;
- `event_ret_excel_cvar05`;
- `event_CTj_p95`;
- `event_CTj_p99`;
- `event_RPj_p95`;
- `event_RPj_p99`;
- `event_service_loss_auc`;
- `event_backlog_clearance_time`;
- `event_fill_branch_rate`.

The local order universe is:

> orders open at the anchor time or created/completed in
> `[anchor_time, anchor_time + L]`, with `L=8` weeks by default.

## Promotion rule

`metadata.json` records `prevention_headroom_found=true` only if any summary row
passes all current gate conditions:

- `mean_delta_event_ret_excel > 0`;
- `positive_pair_rate > 0.60`;
- positive seed count passes the seed gate;
- seed-clustered CI95 lower bound is positive.

This is necessary but still not sufficient for a manuscript prevention claim:
placebos, reactive-null, and label-permutation controls must also be null.

## Smoke tests run

All smoke tests used selected Case C:

- enabled risks: R22/R23/R24;
- R24 frequency x3;
- R22/R23 impact x1.5;
- h104 contract, run at `max_steps=24` for smoke speed;
- static-medium base policy;
- no training.

### forced_prep_sweep

Command:

```bash
.venv/bin/python scripts/audit_track_b_prevention_gate_v2.py \
  --output-dir outputs/experiments/track_b_prevention_gate_v2_smoke2_2026-07-07 \
  --mode forced_prep_sweep \
  --seeds 1 --eval-episodes 1 --max-steps 24 \
  --max-anchors-per-episode 1 --placebo-anchors-per-episode 0 \
  --target-risks R24 --prep-actions high_dispatch --write-tapes
```

Result:

- anchor selected after warmup/pre-window filter: R24 at `982h`;
- `high_dispatch_minus_medium` local Excel ReT delta: `0.0`;
- `prevention_headroom_found=false`.

### event_on_off

Command:

```bash
.venv/bin/python scripts/audit_track_b_prevention_gate_v2.py \
  --output-dir outputs/experiments/track_b_prevention_gate_v2_event_on_off_smoke2_2026-07-07 \
  --mode event_on_off \
  --risk-event-tape outputs/experiments/track_b_prevention_gate_v2_smoke2_2026-07-07/tapes/seed1_episode1.json \
  --seeds 1 --eval-episodes 1 --max-steps 24 \
  --max-anchors-per-episode 1 --target-risks R24
```

Result:

- `event_on_minus_event_off` local Excel ReT delta:
  `-0.0000025932156669170037`;
- `prevention_headroom_found=false`.

### oracle_warning

Command:

```bash
.venv/bin/python scripts/audit_track_b_prevention_gate_v2.py \
  --output-dir outputs/experiments/track_b_prevention_gate_v2_oracle_warning_smoke_2026-07-07 \
  --mode oracle_warning \
  --risk-event-tape outputs/experiments/track_b_prevention_gate_v2_smoke2_2026-07-07/tapes/seed1_episode1.json \
  --seeds 1 --eval-episodes 1 --max-steps 24 \
  --max-anchors-per-episode 1 --placebo-anchors-per-episode 0 \
  --target-risks R24 --prep-actions combined_prep
```

Result:

- `combined_prep_minus_medium` local Excel ReT delta: `0.0`;
- `prevention_headroom_found=false`.

## Interpretation

The implementation works end-to-end at smoke scale. The smoke results are not a
final headroom verdict; they only prove the new causal-audit plumbing runs and
does not revive the old prevention claim by default.

## Case C forced-prep screen

Completed the first cheap headroom screen:

```bash
.venv/bin/python scripts/audit_track_b_prevention_gate_v2.py \
  --output-dir outputs/experiments/track_b_prevention_gate_v2_case_c_screen_2026-07-07 \
  --mode forced_prep_sweep \
  --seeds 1 2 3 --eval-episodes 8 --max-steps 104 \
  --max-anchors-per-episode 4 --placebo-anchors-per-episode 2 \
  --target-risks R22 R24 \
  --prep-actions high_dispatch combined_prep \
  --write-tapes
```

Result:

| Comparison | Event-local pairs | R24 anchors | R22 anchors | Mean delta event ReT | Positive rate | Seed CI95 | Gate |
|---|---:|---:|---:|---:|---:|---:|---|
| `high_dispatch_minus_medium` | 96 | 95 | 1 | 0.000000 | 0.000 | [0.000000, 0.000000] | fail |
| `combined_prep_minus_medium` | 96 | 95 | 1 | 0.000000 | 0.000 | [0.000000, 0.000000] | fail |

Placebos were also generated (`96` rows total) and did not create a positive
Excel-ReT signal.

This first screen says the simple forced-preparation postures tested here do
not create measurable local Excel-ReT headroom under selected Case C. That is
consistent with the current paper stance: the confirmed mechanism is adaptive
recovery/exposure control, not proven anticipatory prevention.

Do **not** train a preventive policy yet. If prevention remains strategically
important, the next step should be a richer action-value ceiling assay (more
lead-window action schedules, not a learned policy) or an explicit environment
variant with a real preparation lead time. If that ceiling is still near zero,
stop the preventive-policy search for the current environment and keep the
paper framed as adaptive recovery/action-space alignment.
