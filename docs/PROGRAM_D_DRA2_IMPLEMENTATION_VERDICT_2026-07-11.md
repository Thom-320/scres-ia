# Program D DRA-2 — implementation verdict

Status: **IMPLEMENTED; CALIBRATION BLOCKED PENDING GARRIDO FACE VALIDATION**.

The opt-in finite-convoy lane, static frontier runner, exact one-action
branching and exact seven-day sequence search are implemented. Historical
transport remains the default.

## Verified physical behavior

- one convoy, 5,000-ration capacity;
- partial loads consume the full convoy opportunity;
- no overlapping departure while the convoy is away;
- 24 h outbound plus 24 h return under an open route;
- Op8 downtime pauses outbound and return progress;
- HOLD preserves staging stock and vehicle location;
- continuous Op7 finished-ration staging in the DRA-2 lane;
- mass and single-convoy conservation;
- no future or latent observation fields;
- resource ledger for departures, vehicle hours, route waiting, idle time,
  load factor, ration-hours in transit and masked attempts.

## Smoke artifacts

Static frontier:
`results/program_d/dra2_static_frontier_smoke/verdict.json`.

- four disposable tapes, one per family;
- nine same-contract threshold/wait policies;
- CRN, mass and convoy conservation: PASS;
- mean live-epoch fraction: approximately 0.498;
- calibration tapes opened: false.

Branching:
`results/program_d/dra2_exact_branching_smoke/verdict.json`.

- 16 live states;
- 32 one-action branches;
- 512 seven-day sequence rollouts over four family-stratified states;
- 34 distinct realized feasible departure patterns in the initial smoke;
- prefix identity, CRN, mass and convoy conservation: PASS;
- no virgin tapes and no PPO.

Smoke outcomes are implementation diagnostics only. They do not promote the
family or estimate scientific headroom.

## Remaining external gate

Before opening the 60 calibration tapes, Garrido must validate or explicitly
authorize the modeled 24 h convoy return leg. If accepted, the runners expand
without changing physics or thresholds. If rejected, the contract must be
re-preregistered before any calibration result is observed.
