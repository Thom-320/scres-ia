# Program O fixed-clock physical preflight freeze

**Status:** `FROZEN_BEFORE_SEALED_VALIDATION — preflight only`

This freeze converts the favorable fixed-clock result from an accounting convention into a
falsifiable physical extension. It authorizes implementation tests and burned-fixture parity only.
It does **not** authorize opening `7420049–7420096`, claiming H_obs, training a learner, or starting
Paper 3.

## Why the physical preflight is mandatory

The parent Program O implementation physically dispatches a loaded order only when matching stock
and backlog exist. Its resource ledger nevertheless charges 112 daily slots and 5,376 downstream
vehicle-hours to every policy. The dual-resource diagnostic therefore establishes a development
signal conditional on fixed-clock accounting, but it does not by itself prove equal physical use.

The thesis supports the following native facts:

- Op9, Op10, and Op12 operate at a daily freight rate (ROP = 24 h; thesis Tables 6.20–6.22).
- Op10 and Op12 each have 24 h processing time.
- Distribution-vehicle availability and route planning are taken for granted (thesis §6.5.5).

The thesis does **not** state that empty missions run or that freight is paid at a flat price. The
dedicated shuttle is therefore explicitly a **researcher-introduced extension**, not a recovered
thesis fact.

## Frozen intervention

`downstream_freight_physics_mode = fixed_clock_physical_v1`

At each daily clock slot during the eight-week decision window plus clearance:

1. exactly one downstream mission is committed;
2. if a product-feasible order is ready, the mission carries that order;
3. otherwise it runs empty;
4. loaded and empty missions occupy the same capacity-one Op10 and Op12 SimPy resources for 24 h
   per leg;
5. each mission receives the same 2,600-ration capacity entitlement;
6. policies cannot purchase, cancel, accelerate, or add missions.

The resulting per-episode envelope is exactly 112 missions, 5,376 vehicle-hours, 5,376 crew-hours,
and 291,200 ration-units of payload capacity. Gross policy production remains 120,000 and setup
remains zero. Actual payload is reported as utilization/output; it is not used as the fixed-clock
resource gate. The pay-per-use interpretation remains a mandatory secondary boundary estimand.

## Fail-closed preflight

Before any sealed tape can be opened:

1. Default `loaded_only` behavior is unchanged.
2. Every scheduled mission is logged as loaded or empty; loaded + empty = 112.
3. Empty missions occupy both real convoy resources; they do not bypass Op10 or Op12.
4. Scheduled missions, vehicle-hours, crew-hours, payload capacity, production, and setup are
   identical across calendars and policies.
5. On burned fixtures, adding empty missions leaves the physical trajectory, canonical metrics,
   order completion, conservation, and action sequence identical to the parent result. Any queueing
   change is a preflight failure, not a tunable discrepancy.
6. Product and aggregate mass residuals remain at most `1e-8`.
7. The fully fungible null remains exact.
8. The same-time convention and action-trajectory tests pass.

Failure label: `STOP_PROGRAM_O_FIXED_CLOCK_PHYSICAL_PREFLIGHT`.

## Frozen prospective policy and gate

The sole primary controller is `belief_mpc__3`, selected for the shorter planning horizon and
parsimony among the two development-passing MPCs. `belief_mpc__4` is a sensitivity, not a replacement.
The three connected cells, nine information placebos, 65,536-calendar frontier, classical comparator
set, metrics, guardrails, bootstrap, resource envelope, and action certificate are frozen in
`contracts/program_o_fixed_clock_physical_hobs_validation_v1.json`.

Only an immutable preflight PASS plus a fresh execution freeze may authorize the one-time opening of
`7420049–7420096`. A prospective pass establishes **classical observable headroom under the disclosed
dedicated-shuttle extension**. It does not establish learned advantage or Paper 2 neural value.

## Claim limit and external calibration

Garrido Q13/Q14 remain useful for determining how far the extension represents the actual MFSC.
They are not dependencies for studying the disclosed extension. A pay-per-use answer forbids a
pay-per-use positive claim; a fixed-schedule answer upgrades external validity. Neither answer may
alter the already frozen validation contract after results are seen.

