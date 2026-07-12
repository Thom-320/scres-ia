# Program D DRA-2 calibration final verdict

Status: **CLOSED AT PRE-RL GATE — `STOP_DRA2_PRE_RL_GATE`**.

DRA-2 was executed once under the contract frozen at tag
`dra2-precalibration-2026-07-12`. The physical lane is disclosed as a stylized,
researcher-imposed finite-convoy extension authorized under PI-delegated modeling
autonomy. It is not an exact operational reconstruction of the thesis.

## Audit trail

- 60 calibration tapes: 15 per family, seeds 850001–850060.
- Nine same-contract static threshold/wait policies.
- 240 exact one-action branch states: four policy-prefixes per tape.
- Restricted seven-day sequence oracle: one state from every tape.
- Exact ten-day sufficiency sensitivity: 12 states, three per family.
- Resource comparator: nine static continuations from the identical 60 states.
- No holdout or virgin tapes opened.
- No observable tree fitted.
- No PPO trained.

## Integrity and liveness

- CRN identity: PASS.
- Prefix identity: PASS.
- Mass conservation: PASS.
- Convoy-slot conservation: PASS.
- Strong-live fraction: 1.00 (PASS; threshold 0.20).

The earlier 0.493 figure is only dispatch-feasibility frequency in the full static
episodes. It is not used as the G-B result.

## State-contingent diversity

Across the 240 one-action states, using the frozen positive-margin tie rule:

| Supported action | States | Fraction |
|---|---:|---:|
| HOLD | 139 | 57.9% |
| DISPATCH_NOW | 96 | 40.0% |
| tie/zero | 5 | 2.1% |

Both actions exceed the preregistered 15% diversity threshold.

## Resource-adjusted oracle result

The restricted seven-day future-information sequence oracle used on average:

- 12.55 departures;
- 592.4 convoy-unavailable hours.

The calibration-selected resource-dominance comparator is
`threshold_5000__wait_72h`, using:

- 12.25 departures;
- 574.0 convoy-unavailable hours.

It lies inside the oracle candidate's joint resource envelope. Paired by tape:

| Endpoint | Estimate | CI95 | Gate |
|---|---:|---:|---|
| ReT improvement | +0.02662 | [+0.01650, +0.03732] | PASS (minimum +0.01) |
| Service-loss reduction | 5.174% | [3.272%, 7.045%] | PASS (minimum 5%; CI > 0) |
| Lost-order change | 0.0 | — | PASS |

Thus DRA-2 differs materially from DRA-1: a restricted future-information oracle can
exploit the convoy timing decision under comparable resources.

## Failed promotion gate: horizon sufficiency

The preregistered 7d/10d sensitivity produced:

- first-action agreement: 91.67% (passes the 90% subcriterion);
- headroom stability: FAIL;
- no-sign-change criterion: FAIL;
- overall sequence-sufficiency gate: **FAIL**.

The raw headroom increased materially when the exact open-loop decision window was
extended from seven to ten days; two states changed from zero to positive headroom.
Therefore seven days cannot be treated as a stable decision horizon, and the oracle
labels are not sufficiently frozen to train or evaluate an observable policy without
changing the preregistered design after seeing calibration.

## Binding decision

```text
observable_tree_authorized = false
holdout_opened = false
ppo_authorized = false
ppo_trained = false
virgin_tapes_opened = 0
```

The allowed claim is:

> A stylized finite-convoy extension exposes material resource-adjusted oracle
> headroom and action-ranking diversity, but the preferred decision and estimated
> headroom were not stable across the preregistered planning horizons. The study
> therefore does not establish observable adaptive-control value and does not
> authorize PPO.

The prohibited claims remain:

- PPO beats the best static policy;
- DRA-2 provides a deployable dynamic policy;
- retained learning improves resilience;
- the convoy contract reproduces Garrido's original operational transport physics.

Machine verdict:
`results/program_d/dra2_exact_branching_calibration/resource_gate_verdict.json`.
