# The dominance sweep with the transport axis

**Status:** `DEVELOPMENT_FIDELITY_SWEEP_NO_CONSTANT_CHANGED`. Runner
`scripts/run_fidelity_delay_sweep.py`, artifact
`results/metric_audit/fidelity_sweep_transport_v2/result.json`. 48 cells per family:
6 delays × 2 transport modes × 2 semantics × tolerances. Reference
`results/metric_audit/fidelity_reference_v1/`.

## 1. In R1r the transport mode is a strict no-op

Every one of the six moments is identical between `skip_wave` and `retry_when_ready`, to
the last digit, at every delay. Mean discrepancy 23.754 under both.

That is the correct behaviour and it is a real constraint on this route: R11–R14 hit
assembly workstations, and the thesis assigns R22 to Op10 and Op12. **A transport fix
cannot help a family where no risk touches transport.** Half the calibration problem is
untouchable from here.

## 2. In R2r it helps, modestly and in the right direction

| moment | skip_wave | retry_when_ready | reference |
|---|---:|---:|---:|
| `rpj_mean` (delay 48) | 305.53 | **316.39** | 626.60 |
| `rpj_p95` (delay 48) | 931.66 | **951.08** | 3042.02 |
| `scored_rows` | 206.25 | **209.25** | 2169.30 |
| **mean discrepancy** | 19.795 | **19.260** | — |

Both `RPj` moments move *toward* the reference, and the aggregate discrepancy improves.
The improvement is small — we sit at roughly half the reference `RPj` mean and a third of
its p95 — but it is in the right direction on the moments the change should touch, which
is the minimum a mechanism fix should show.

## 3. The reference sits between two grid points and neither is close

The autotomy moment, which is the whole reason for this exercise:

| delay | autotomy share (R1r) | reference |
|---|---:|---:|
| 48 | **0.652** | 0.00436 |
| 54 | **0.000** | 0.00436 |

At 54 the branch is dead. At 48 it fires for roughly two thirds of orders — **150× the
reference**. The truth is between two adjacent grid points and neither endpoint is within
two orders of magnitude of it.

**This number needs verification before it is used.** 0.652 is surprising: the
reclassification in `moments_under` only touches orders with `RPj > 0`, and at delay 48 the
on-hand orders should carry no risk indicators, so a share that large is not obviously
consistent with the earlier finding that on-time and risk-touched orders are disjoint. It
may be that at `delay = LT` the DES's own `CTj <= LTj` predicate fires for a population I
have not characterised. Flagged rather than interpreted.

## What this settles and what it does not

**Settles:** the transport gate was a real defect and fixing it helps R2r on the moments
it should, while leaving R1r untouched for a principled reason. The mechanism is not the
whole gap.

**Does not settle:** the autotomy share. No cell in the grid lands near 0.44%, and the two
adjacent delays straddle it by two orders of magnitude in opposite directions. Closing that
needs the change already identified — a committed order not waiting on the next material
wave — and this sweep is the baseline it will be measured against.

Both families remain `epsilon_stable: false`, so the non-dominated sets (24/48 in R1r,
22/48 in R2r) are reported as unstable rather than shown as a result.
