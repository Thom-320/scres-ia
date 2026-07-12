# DRA-1 Gate C prefix-balanced corrective audit

Status: **FINAL CORRECTIVE RESULT**.

This audit corrects the prefix-induced state imbalance identified in verification
V3. It does not change the DRA-1 physics, action set, metric, static comparator,
calibration tapes, risk calendars, or guardrails. It opens no virgin tape and
trains no PPO.

## Design

Each of the same 60 calibration tapes generates one live state under each
`SPT_FULL` prefix:

- allocation A = 0.25: select an A-stressed state;
- allocation A = 0.50: select the most balanced live state;
- allocation A = 0.75: select a B-stressed state.

All nine actions are replay-branched for one epoch from each of the resulting
180 states. Every branch then returns to the frozen best static continuation
`0.25 / SPT_FULL`. Inference averages the three states within tape and
bootstraps the 60 tapes. Allocation is also expressed as share sent to the
currently stressed node. Exact ties with zero improvement do not count toward
action diversity.

## Balance and invariants

- prefix counts: 60 / 60 / 60;
- stressed-node counts: A = 66, B = 65, balanced = 49;
- categories: joint scarcity = 45, localized R24 = 45, post-hit recovery = 90;
- exact replay-prefix identity: PASS;
- mass conservation: PASS;
- all guardrails: PASS;
- virgin tapes opened: 0;
- PPO trained: false.

## Result

Only 11 of 180 states exhibited strictly positive oracle headroom. The
clustered oracle contrast was:

```text
mean delta ReT = 0.000087895
CI95 = [0.000027576, 0.000165914]
```

Positive normalized action weights were far below the preregistered 15% of the
131 stressed states:

```text
share_to_stressed 0.25: weight 3.0
share_to_stressed 0.50: weight 2.5
share_to_stressed 0.75: weight 3.5
required per promoted level: 19.65
```

Therefore `diversity_pass = false` and the final Gate C verdict is:

```text
STOP_NO_DYNAMIC_ORACLE_HEADROOM
```

The earlier raw result “allocation A=0.25 is optimal in 58/60 states” must not
be used because it was conditioned on a 0.25 prefix. The corrected conclusion
is narrower: after balancing prefix histories and A/B stress direction, dynamic
oracle gains remain extremely sparse and do not expose a sufficiently diverse
action-ranking surface for an observable policy, tree, heuristic, or PPO.

Authoritative artifact:
`results/program_d/dra1_prefix_balanced_branching/verdict.json`.
