# DRA-2 software preflight verdict

Status: **PASS SOFTWARE PREFLIGHT — AWAITING GARRIDO FACE VALIDATION**.

This preflight used only deterministic fixtures and a disposable routine tape.
It opened no calibration, holdout, virgin or PPO data.

## G-A intertemporal commitment

PASS. From identical staged-inventory states, `DISPATCH_NOW` and `HOLD` have
different feasible action/resource states 24 hours later and retain different
inventory and departure histories after the 48-hour cycle. This was verified
at staged quantities 1,000, 2,500 and 5,000.

## Reported 54-hour return

Not reproduced. In the integrated disposable fixture:

```text
departure              968 h
nominal return         1016 h
actual return          1016 h
cycle                     48 h
route wait                 0 h
```

The earlier observation at 54 hours resulted from advancing or polling the DES
without processing events scheduled exactly at the target boundary. The
preflight uses `advance_including`, which processes every event with timestamp
less than or equal to the requested target. A regression test now requires
nominal and actual return times to match when the route remains available.

## G-B strong liveness

The definition is now frozen: dispatch feasibility alone is insufficient. Both
actions must be admissible and exact CRN branches must create different
physical/resource trajectories. The deterministic fixtures pass this
definition. The preregistered 20% population fraction remains uncomputed and
cannot be claimed before calibration is authorized.

## Resource equivalence

The primary comparator is frozen as the best calibration-static policy within
the candidate policy's joint resource envelope: no more departures and no more
vehicle-hours. Pareto reporting is secondary and non-dominance alone cannot
support an adaptive-superiority claim.

## Remaining blocker

Garrido must validate the 24-hour return leg, full-slot cost of a partial load,
and route-outage semantics before the 60 calibration tapes may be opened. See
`PROGRAM_D_DRA2_GARRIDO_FACE_VALIDATION_REQUEST_2026-07-11.md`.

Machine verdict:
`results/program_d/dra2_preflight/verdict.json`.
