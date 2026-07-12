# Program F final verdict — 2026-07-12

## Terminal decision

`STOP_PROGRAM_F_SCREEN`

Program F executed its preregistered 24-cell learnability phase diagram and
terminated before confirmatory calibration. No admissible two-token cell passed
the conjunctive screen preselection rule. Seeds 940001+, 950001+ and 960001+
remain unopened. No tree was promoted, no PPO or lookahead learner was trained,
and retained learning was not authorized.

This result does not reopen or alter Program E, DRA-2b or any earlier terminal
verdict.

## Frozen genealogy

- Program F charter: `b0088ac`, tag `program-f-preregistered-2026-07-12`.
- Event-model freeze: `b1db8d1`, tag
  `program-f-implementation-freeze-2026-07-12`.
- Physical preflight: `47dff5f`, tag
  `program-f-physical-preflight-2026-07-12`.
- Phase design: `2b815d9`, tag
  `program-f-screen-design-freeze-2026-07-12`.
- Screen runner: `932c9fd`, tag
  `program-f-screen-runner-freeze-2026-07-12`.

The frozen Latin-hypercube design SHA-256 is
`300a29998bb5f51118caad5a8fc8818e0ae59a9a4730c5bb9755f2b873d298d3`.

## Physical preflight

Before the screen, 18 disposable tapes established:

- exact two-token accounting in the primary contract;
- zero mass residual;
- identical exogenous threat hashes across actions;
- manufacturing liveness and planned-downtime cost;
- transport protection liveness after eligible R22/R23 events;
- finite reserve issue and replenishment liveness.

No calibration tape was used for this preflight.

## Screen executed

- 24 cells spanning efficacy, signal accuracy, context dwell, token tightness,
  thesis-anchored risk amplitude and commitment;
- 12 dedicated screen tapes per cell: 288 tapes total, seeds 970001–970288;
- four prefix-balanced states per tape;
- every available fixed-budget action branched for four weeks;
- cross-fitted depth-3 tree evaluated by complete episodic rollout;
- all cells reported, including one- and three-token boundary cells;
- only the eight two-token cells were eligible for Program F v1 selection.

## Results

Across 24 cells:

- all three actuators were physically live in `24/24` cells;
- action diversity passed in `24/24` cells;
- material oracle headroom passed in `16/24` cells;
- observable-tree conversion passed in `0/24` cells;
- admissible passing cells: `0/8`;
- maximum mass residual: `0`;
- threat CRN identity: `24/24` cells.

The largest restricted oracle headroom was in `FSC-24`:

- oracle delta ReT: `0.022584`;
- tree conversion: `-1.283`.

Only four cells produced a positive tree delta, all outside the two-token
confirmatory contract. The largest conversion was `0.471` in `FSC-15`, below
the frozen `0.50` requirement; that cell also had oracle delta ReT `0.007942`,
below `0.01`. It therefore could not pass even if it had been admissible.

Among the eight admissible two-token cells, several had material oracle
headroom, but every tree rollout delta was negative. Consequently there was no
cell to select under the frozen lowest-amplitude rule. Selecting `FSC-24` for
its maximum oracle ReT would violate the preregistration and is forbidden.

## Allowed claim

> Risk-specific mitigation, a binding budget, persistent contexts and noisy
> lead signals created physical liveness, action-ranking diversity and frequent
> clairvoyant headroom across a broad preregistered phase diagram. However, no
> admissible two-token cell converted that headroom into positive out-of-sample
> rollout value with the observable depth-3 policy tree; Program F therefore
> stopped before confirmation or reinforcement learning.

This extends the diagnostic ladder:

`physical effect -> action diversity -> clairvoyant headroom -/-> observable rollout value`.

## Forbidden claims and actions

- RL is impossible in every alternative MFSC extension.
- The M/T/R actuators were inert.
- Program F had no oracle headroom.
- The largest-headroom cell may be promoted post hoc.
- The 0.471 conversion cell may be rounded into a pass.
- Signal accuracy, efficacy, dwell, risks or the 0.50 gate may be changed after
  seeing this screen.
- Seeds 940001+, 950001+ or 960001+ may be opened under Program F v1.
- PPO, recurrent PPO, lookahead or retained-learning arms may be trained under
  Program F v1.

## Publication consequence

Program F is a terminal seventh boundary result and a phase diagram of
RL-eligibility conditions. The paper no longer depends on a positive neural
result. Its central evidence is that specialization and clairvoyant headroom
remain insufficient without observable rollout conversion.
