# Program D DRA-2 — implementation verdict

Status: **IMPLEMENTED; CALIBRATION REQUIRES CONTRACT-BOUND PI AUTHORIZATION**.

The opt-in finite-convoy lane, static frontier runner, exact one-action
branching and exact seven-day sequence search are implemented. Historical
transport remains the default.

## Verified physical behavior

- one researcher-imposed indivisible convoy slot, with capacity interpreted from
  the thesis-grounded 5,000-ration release batch;
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
- mean dispatch-feasible fraction: approximately 0.498;
- strong-live fraction: not measured in this static smoke;
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

## Remaining authorization gate

Before opening the 60 calibration tapes, the runner requires a structured record
bound to the exact contract hash. The PI has delegated modeling autonomy; therefore
the lane is disclosed as a stylized researcher-imposed extension rather than a
validated operational reconstruction.
