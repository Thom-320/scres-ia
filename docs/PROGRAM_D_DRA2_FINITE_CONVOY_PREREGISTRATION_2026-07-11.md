# Program D DRA-2 — finite-convoy batch-release preregistration

Status: **FROZEN BEFORE DRA-2 IMPLEMENTATION AND CALIBRATION RESULTS**.

DRA-2 is a new decision family. It does not modify or rescue the terminal DRA-1
result. The objective is to test whether a physically persistent convoy
commitment creates material, observable, state-contingent headroom before any
RL training.

## Physical contract

- Opt-in mode `op8_dispatch_mode="finite_convoy_v1"`; historical modes remain
  bitwise default.
- One researcher-imposed indivisible convoy slot. Its 5,000-ration capacity is an
  interpretation anchored to the thesis release batch, not a recovered vehicle fact.
- Op7 finished rations enter an explicit staging stock continuously rather than
  materializing only after a full 5,000-ration release batch.
- `DISPATCH_NOW` loads `min(staged_inventory, 5000)` and consumes the entire
  convoy opportunity, including on a partial load.
- Outbound travel is thesis-supported at 24 h; return is a researcher-imposed 24 h
  parsimonious extension authorized under PI-delegated modeling autonomy.
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

### Frozen resource-equivalence estimand

The primary adaptive comparison is not an unconstrained comparison with the
globally best static policy. For each candidate dynamic policy, the comparator
is the best calibration-selected static policy inside the candidate's resource
envelope:

```text
static departures <= candidate departures
AND static vehicle-hours <= candidate vehicle-hours.
```

If no static policy satisfies that envelope, no adaptive-superiority claim is
allowed. The joint outcome-resource Pareto frontier is required as a secondary
analysis, but Pareto non-dominance by itself is not evidence that adaptation
created value.

### Strong liveness definition

An epoch is live only when both actions are admissible and produce different
physical/resource trajectories under exact CRN replay. Dispatch feasibility
alone is an implementation diagnostic, not the confirmatory G-B numerator.
The difference must appear in at least one of convoy availability/ETA, staged
inventory, inventory in transit, departures, or the next feasible action set.

## Promotion

Promote to observable policy only if all frozen thresholds in
`contracts/op7_op8_finite_convoy_v1.json` pass, including:

- at least 20% live epochs;
- each binary action optimal in at least 15% of states;
- clustered oracle ReT lower CI above zero and point delta at least 0.01;
- service-loss AUC reduction at least 5%;
- mass and convoy conservation;
- resource-dominance-constrained static comparison selected once on calibration;
- Pareto reporting is secondary and cannot substitute for the primary comparison;
- no tail or lost-order contradiction.

No threshold or physical parameter may be changed after calibration results to
rescue the family. If the pre-RL gate fails, emit
`STOP_NO_DYNAMIC_CONVOY_HEADROOM` and train no PPO.
