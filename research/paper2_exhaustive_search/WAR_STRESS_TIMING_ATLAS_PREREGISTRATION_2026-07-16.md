# War-stress timing atlas v1 — preregistration

**Date:** 2026-07-16

**Status:** `FROZEN_DESIGN_BEFORE_IMPLEMENTATION_AND_SEED_ACCESS`

**Contract:** `contracts/war_stress_timing_atlas_v1.json`

## Why this is a new contract

The completed Garrido sensitivity established a bounded null: exact Cf1–Cf20
frequency profiles and individual impacts up to 2x did not create material
regime-tailoring value over 18 Track-A constants. It did not test the broader
question of whether timing interventions around concrete wartime events have
value inside a fixed high-stress regime.

The obsolete timing freeze required the best constant to reverse across risk
regimes before timing could run. That prerequisite is removed. A constant can
remain best on average while temporary, physically constrained intervention
around an event still has value.

This is not a retrospective rescue. The entire stress family, inference family,
region rule, comparators, nulls, guardrails and fresh seed blocks are frozen
before implementation and before any scientific tape is opened.

## Conditional wartime envelope

The primary atlas contains exactly 144 cells:

```text
3 operational masks
x 4 frequency levels phi = {1, 2, 4, 8}
x 3 impact levels psi = {1, 2, 4}
x 4 temporal coupling modes
= 144 cells
```

The masks are fixed by operational mechanism:

1. `LOC_SURGE`: R22 attacks on lines of communication plus R24 contingency
   demand.
2. `THEATER_CAPACITY_SURGE`: R21 environmental disruption and R23 attacks on
   forward units plus R24 demand.
3. `PRODUCTION_QUALITY_SURGE`: R11 assembly breakdown and R14 defects plus R24
   demand.

The coupling modes are independent native tapes, disruption 72 h before surge,
coincident onset, and disruption 72 h after surge. All tapes are generated
before policy evaluation and replayed identically across policies.

In a coupled mode, R24 is the cluster driver and every non-R24 risk in the mask
is scheduled at the frozen offset. Its native occurrence stream is replaced,
not duplicated. Impact draws remain risk-specific and policy-independent.

High values are interpreted as conditional wartime stress tests, not estimates
of empirical conflict probability. Every cell is reported. No additional mask,
amplitude or offset may be added after seed access.

The current DES caps pending R24 demand at 13,000 rations. Every run must ledger
generated, admitted and clipped surge quantity plus cap-hit counts. A cell with
positive clipped demand or any unaccounted risk saturation is reported as model
saturation and cannot enter the promoted component. High stress may expose a
model boundary; it may not win by silently discarding stress.

## R3 boundary

R3 is not mixed into the connected-region atlas. One scheduled 672 h R3 event
at the one-year midpoint is reported separately. A second report-only fixture
contains three 672 h disruptions at six-month intervals; it is named
`PERSISTENT_PRODUCTION_DISTRIBUTION_DISRUPTION_NOT_R3`. It cannot be described as
repeated black swans or used to select the primary region.

## Estimand

For each primary cell:

\[
H_{\mathrm{timing}}^{\mathrm{restricted,safe}}
=
R^{\mathrm{canonical}}_{\mathrm{best\ safe\ restricted\ event\ timing}}
-
R^{\mathrm{canonical}}_{\mathrm{strongest\ same-cell\ comparator}}.
\]

The privileged controller knows realized future onset and end times, but may
only intervene at offsets `{-168,-72,-24,0,+24,+72}` h and exit at
`{+24,+72,+168}` h after a risk cluster. Between boundaries it must HOLD. The
complete 18-posture Track-A frontier is available; the implementation freeze
must publish the exact finite template count before seed access.

The denominator includes all 18 constants, an exactly enumerated weekly
open-loop family, a weekly-boundary clairvoyant version, and a resource-matched
open-loop comparator. Comparator selection is repeated inside every bootstrap.

The estimand is a restricted perfect-information timing ceiling, not total H_PI
and not H_obs.

## Physical intervention

Review occurs every 24 h, but control is multiscale:

- `HOLD` preserves policy exactly and creates no top-up or exogenous RNG call;
- shifts ramp by at most one level per day and consume the existing finite
  surge-hour budget;
- buffers activate after an immutable 168 h lead, cannot be cancelled or
  overwritten, and accept at most one new commitment per week;
- no artificial churn price is introduced;
- intervention count and magnitude are reported.

The safe oracle excludes any tape-level candidate that worsens losses, omitted
orders, full-ledger or quantity ReT, backlog, age, worst-node/product service,
or scheduled/realized resources. A known shed-to-win fixture must be rejected.

## Metrics

The sole primary endpoint is `ret_excel_request_snapshot_v2` through the
canonical episode aggregator. Full-ledger ReT, quantity ReT, CVaR10, losses,
omissions, backlog, age, worst node/product and resources are mandatory
guardrails.

The temporal panel—service-loss AUC, maximum service drop, recovery to 95% for
seven days, backlog-age CVaR and worst node/product—is report-only. It cannot
select a cell or open the observable/learner stages.

## Region rule

A cell passes development only if safe canonical timing improvement is at least
0.01, the simultaneous one-sided LCB95 is positive, at least 8/12 tapes favor
timing, every guardrail passes, resources are non-superior, and actions contain
multiple intervention times and sequences.

Adjacency holds only within one mask and coupling mode. Neighboring cells differ
by one ordinal phi or psi step. Promotion requires at least four adjacent cells
spanning at least two phi and two psi levels. A single positive cell is reported
but cannot promote.

If several components pass, select the component with the lowest minimum
`log2(phi)+log2(psi)`, then the lowest maximum severity, then the frozen
mask/coupling order. This makes the selected result the least severe robust
wartime envelope, not the largest observed headroom.

Inference is simultaneous across all 144 cells and mandatory estimands.

## Seed sequence

- Development atlas: `7470001–7470012`, unopened; all 144 cells use CRN.
- Selected-component validation: `7470101–7470148`, sealed.
- Classical observable validation: `7470201–7470248`, reserved but unauthorized.
- Learner: no seed block allocated.

Validation opens only after a development component passes and a new immutable
execution freeze commits its cells and policies. Failure closes the family with
no cell deletion, amplitude increase, offset change, policy replacement or
metric change.

## Promotion sequence

```text
restricted privileged timing ceiling
    -> frozen observable EWMA/hysteresis controller
    -> fresh OOS classical H_obs validation
    -> separately frozen learner vs all classical/open-loop comparators
    -> Paper 2
    -> Paper 3
```

No learner, Paper 2 claim or Paper 3 protocol is authorized by this
preregistration.
