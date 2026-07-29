# Program G — Structured Spatial Headroom

Status: **CHARTER FROZEN BEFORE ENVIRONMENT BUILD OR PROGRAM G TAPES**.

Primary question:
Which realistic combinations of operational-tempo variation, advance
information, and shared-resource scarcity create observable adaptive headroom
in the MFSC?

Environment selection rule:
No RL algorithm may be trained or consulted during environment selection.

Primary pre-RL estimands:
`H_PI`, `H_obs_tree`, action diversity, and resource-adjusted service/ReT.

Non-rescue boundary:
Program E and Program F remain terminal; none of their tapes, thresholds,
weights, selected cells or claims are reused.

## New decision class

Program G is not another risk multiplier or a renamed Program F. It tests a
spatial commitment problem absent from DRA-1:

- CSSU A and B have separate stock, backlog, demand tempo and route status;
- one finite downstream convoy is shared between them;
- dispatch to one destination makes the convoy unavailable to the other;
- unused lift is not automatically reassigned after commitment;
- convoy location and return time persist into future decisions;
- a finite reserve moves only with the convoy and is never created by action;
- actions are selected before the relevant tempo is fully realized, using an
  imperfect operational lead signal.

The lane is a Garrido-informed structural extension, not an exact recovery of
vehicles, intelligence or two-theatre doctrine from the thesis.

## Minimal action contract

Weekly action:

`(convoy_destination, assembly_shift)`

- destination: `A`, `B`, or `HOLD`;
- shift: `S1` or `S2`;
- six actions, exhaustively enumerable;
- no independent inventory-creation or risk-suppression action;
- S2 has explicit labor-hours and one-week activation delay;
- dispatch is masked when the convoy is unavailable, the route is closed, or
  no physical load exists.

Every static comparator receives the same six actions and physical constraints.

## Factorial structural identification

Before choosing a final environment, execute the following nested arms:

| Arm | Persistent tempo | Advance signal | Shared spatial resource | Cost/expiry |
|---|---:|---:|---:|---:|
| Base | no | no | no | no |
| T | yes | no | no | no |
| TR | yes | no | yes | no |
| TS | yes | yes | no | no |
| TRS | yes | yes | yes | no |
| TRSC | yes | yes | yes | yes |

The purpose is component attribution, not selecting the arm where PPO wins.
All arms are published. Environment selection occurs entirely from exact
branching and observable-policy rollout.

## Tempo and information

Each CSSU has an independent latent tempo in `{low, routine, surge}` with
multi-week persistence. The policy never observes the latent labels. Candidate
signals are operational quantities such as planned troop levels, deployment
calendar pressure and route-threat intelligence. Their lead, sensitivity and
false-positive rate must be frozen as a domain envelope before tapes are
generated.

Signal value must be tested against shuffled and delayed placebos. A signal that
arrives after the convoy commitment does not qualify as advance information.

## Physical invariants

1. A/B demand summed together preserves the corresponding aggregate demand
   tape.
2. Convoy count, location, load, outbound travel and return travel conserve the
   vehicle resource.
3. Reserve movement conserves rations and consumes convoy capacity.
4. Threat occurrence and demand remain exogenous and event keyed.
5. Actions may change exposure or response, never erase a threat or demand.
6. Same prefix plus same action produces identical trajectory hashes.
7. Swapping A/B labels and mirrored actions preserves aggregate outcomes.

## Pre-RL sequence

1. Domain-envelope freeze and physical preflight.
2. Six-action same-contract static frontier.
3. Exact CRN single-action branching.
4. Exact four- and eight-week action sequences.
5. Perfect-information restricted oracle.
6. Cross-fitted depth-3 tree and hysteresis rule using only deployable signals.
7. Signal-versus-shuffled/delayed placebo.
8. Resource-envelope comparison and independent holdout.
9. Only then: rollout/MPC, contextual bandit and MaskablePPO.

## Learner-eligibility gate

RL is authorized only if all conditions hold in at least two adjacent plausible
parameter cells:

- oracle minus full-contract static delta ReT at least `0.02`, CI95 lower > 0;
- service-loss reduction at least `5%`;
- at least two actions optimal in at least `15%` of states each;
- no action optimal in more than `85%`;
- observable tree captures at least `30%` of oracle headroom;
- tree minus static CI95 lower > 0 on independent holdout;
- signal beats shuffled and delayed placebos;
- no extra convoy-hours, shift-hours or reserve-time outside the frozen envelope;
- lost orders and tail risk do not deteriorate.

The 30% threshold authorizes a learner; it does not make the tree a managerial
solution. A tree requires at least 50% conversion for that stronger claim.

## Stop rule

If no component arm and no adjacent pair of plausible cells satisfies the gate,
emit `STOP_PROGRAM_G_NO_OBSERVABLE_SPATIAL_HEADROOM`. Do not change signal
quality, tempo persistence, fleet size, service MCID or action semantics after
reading the result.

## Relationship to learning retention

Program G can establish adaptive-control eligibility. It cannot establish
`L_(e-1)`. Persistent/reset/frozen is authorized only after a policy beats the
best full-contract static and interpretable policies on virgin tapes.
