# Program D DRA-2 — finite-convoy batch-release preregistration

Status: **FROZEN BEFORE DRA-2 IMPLEMENTATION AND CALIBRATION RESULTS**.

DRA-2 is a new decision family. It does not modify or rescue the terminal DRA-1
result. The objective is to test whether a physically persistent convoy
commitment creates material, observable, state-contingent headroom before any
RL training.

## Physical contract

- Opt-in mode `op8_dispatch_mode="finite_convoy_v1"`; historical modes remain
  bitwise default.
- One convoy, capacity 5,000 rations.
- Op7 finished rations enter an explicit staging stock continuously rather than
  materializing only after a full 5,000-ration release batch.
- `DISPATCH_NOW` loads `min(staged_inventory, 5000)` and consumes the entire
  convoy opportunity, including on a partial load.
- Outbound travel is 24 h; return is 24 h. The return leg is a disclosed
  Garrido-informed extension requiring face validation before confirmatory use.
- Op8 outages pause physical outbound and return progress.
- `HOLD` forfeits today's opportunity but leaves the convoy available at Op7
  for the next 24 h epoch.
- No action creates inventory, vehicles, routes or future information.

## Actions and static comparators

Dynamic action set: `{HOLD, DISPATCH_NOW}` at 24 h epochs, with feasibility
masking.

The same-contract static frontier is the 3x3 factorial:

```text
inventory threshold in {1000, 2500, 5000}
maximum staging wait in {24h, 48h, 72h}
```

A static policy dispatches when the threshold is met OR the oldest staged
inventory reaches maximum wait. Thesis full-batch and dispatch-whenever-possible
are named diagnostics, not weaker headline comparators.

## Information contract

Allowed observations are contemporaneous inventory, backlog, production,
demand, convoy availability/ETA, staging age, route status and action history.
Future shocks, repair durations, demand, regimes and retrospective labels are
forbidden.

## Experimental sequence

1. Physical liveness and invariant tests.
2. Nine-policy same-contract static frontier on 60 calibration tapes.
3. Exact one-action branching at 240 eligible states, horizons 72 h and 28 d.
4. Exact seven-day binary sequence search on a frozen 60-state subset; only
   feasible masked sequences are interpreted.
5. Cross-fitted depth-3 tree and rollout heuristic on calibration data.
6. Observable-policy confirmation on 40 separate holdout tapes.
7. PPO may be defined only after all prior gates pass; its final claim uses 60
   additional virgin tapes.

R3 is excluded from calibration, policy fitting and PPO training. It is OOD
only.

## Outcomes and resources

Primary: `ret_excel_visible_v1`.

Required physical consistency: service-loss AUC. Secondary: fill, backlog AUC,
maximum backlog age, CT/RP, lost orders, tail service loss, departures,
vehicle-hours, route waiting and load factor.

## Promotion

Promote to observable policy only if all frozen thresholds in
`contracts/op7_op8_finite_convoy_v1.json` pass, including:

- at least 20% live epochs;
- each binary action optimal in at least 15% of states;
- clustered oracle ReT lower CI above zero and point delta at least 0.01;
- service-loss AUC reduction at least 5%;
- mass and convoy conservation;
- equal-resource or Pareto-valid comparison;
- no tail or lost-order contradiction.

No threshold or physical parameter may be changed after calibration results to
rescue the family. If the pre-RL gate fails, emit
`STOP_NO_DYNAMIC_CONVOY_HEADROOM` and train no PPO.
