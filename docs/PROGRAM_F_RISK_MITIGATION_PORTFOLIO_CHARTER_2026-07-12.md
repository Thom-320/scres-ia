# Program F — Risk-Conditioned Mitigation Portfolio

Status: **FROZEN BEFORE SEEDS 940001+ OR ANY PROGRAM F ENDPOINT**.

## Scientific boundary

Program E remains terminal at `STOP_PROGRAM_E_VALIDATION`. Program F neither
retunes Program E nor reinterprets its null. It tests a new decision class:
weekly allocation of a fixed readiness budget among three risk-specific,
physically consequential mitigations.

The lane is a Garrido-authorized researcher-imposed extension. It is not called
an exact reconstruction of transport, maintenance, intelligence, or reserve
doctrine absent from the thesis.

## Research question

Under persistent and partially observable risk contexts, can a policy allocate
two weekly readiness tokens among manufacturing maintenance (M), transport
protection (T), and finite reserve response (R) to outperform every constant
allocation of the same budget?

## Why this can create headroom

Frequency or impact escalation alone changes the state distribution but need
not change the best action. Program F requires the full package:

1. risk-specific mitigation;
2. a binding two-token budget;
3. contexts lasting longer than action activation;
4. imperfect but informative lead signals;
5. current actions changing future damage, condition or finite stock;
6. materially different optimal allocations across reachable states.

Threat occurrence, base damage and demand remain exogenous and common across
policies. Mitigation changes realized damage or response capacity; it never
deletes a threat or demand event from the tape.

## Primary physical contract

The action is one of the six integer allocations summing to two tokens:
`200`, `020`, `002`, `110`, `101`, `011` in M/T/R order. Decisions are weekly,
activate one week later and remain in force for at least one week.

### Manufacturing readiness

Op5–Op7 receives a persistent condition index. Wear is exogenous and event
keyed. Maintenance consumes 0/12/24 planned production hours for 0/1/2 tokens,
reduces next-week condition, and attenuates realized R11 downtime. It cannot
remove or reschedule the exogenous R11 event.

### Transport readiness

Protection commits for the week. It reduces realized R22/R23 closure duration
by frozen factors `1.00/0.65/0.40`, subject to a one-hour minimum. Threat onset,
target and unmitigated duration are unchanged across policies.

### Reserve readiness

Every policy begins with the same finite, costed 10,000-ration forward reserve.
R tokens control how much may be issued per R24 event: 0/2,500/5,000. Issued
stock is depleted, reaches demand after 24 hours, and is replenished only by
moving existing Op9 stock across Op10–Op12 with a 168-hour administrative lead.
No inventory is created by an action.

## Context and information contract

The latent context is equipment pressure, interdiction campaign or mission
surge. Dwell is sampled uniformly from four to eight weeks. The policy never
observes the latent label. One week before activation it receives three noisy
scores whose frozen correct-class probability is 0.75. It also observes only
realized condition, attacks, downtime, demand, inventories, backlog, mitigation
in force, pending mitigation and switching history.

## Data and gates

- Calibration: 60 new tapes, 20 per dominant starting context, seeds 940001+.
- Observable holdout: 40 new tapes, seeds 950001+.
- Virgin learner test: 60 reserved tapes, seeds 960001+.

Before any learner, enumerate all six constants, branch 480 states across all
six actions at four- and eight-week horizons, and enumerate all `6^3=216`
three-decision sequences on a frozen 60-state subset. Fit a depth-3 tree and a
frozen hysteresis rule with grouping by tape; classification accuracy is only a
diagnostic. Promotion depends on complete-rollout value.

All promotion requirements in the JSON contract are conjunctive. In particular:

- all three actuators must be physically live;
- at least two allocations must each be optimal in at least 15% of states;
- no allocation may dominate more than 85%;
- oracle delta ReT must be at least 0.01 with CI95 lower bound above zero;
- service-loss reduction must be at least 5%;
- the observable policy must capture at least half the oracle headroom and beat
  the best constant on holdout;
- the direction must be stable across horizons;
- lost orders and tail risk may not deteriorate.

If any requirement fails, emit `STOP_PROGRAM_F_PRELEARNER` and do not train a
learner. Parameters may not be changed after reading the failed endpoint.

## Claim boundary

A passing prelearner gate establishes observable adaptive-control eligibility,
not retained learning. Only a subsequent out-of-sample policy win can authorize
persistent/reset/frozen comparisons and only an order-history experiment can
support path dependence.
