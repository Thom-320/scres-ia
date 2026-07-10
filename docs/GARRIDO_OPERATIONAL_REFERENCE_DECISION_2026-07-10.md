# Decision: operational reference replaces exact-replication training gate

## Binding decision

The project will not spend further time attempting to infer unavailable Simulink
event-to-order attribution code from the Garrido workbooks. The failed endogenous
distribution gate remains part of the evidence record, but it is no longer a
prerequisite for adaptive-control research.

New research training is authorized under
`garrido_operational_reference_v1`.

## Why this is scientifically preferable

The thesis and workbooks identify the topology, operations, risks, controls,
inventory/capacity configurations, and ReT evaluation lineage. They do not
uniquely identify the original risk-to-order assignment algorithm. Multiple
tested mechanisms reproduce different subsets of the workbook behavior, and the
uniform hybrid candidate failed on odd Cf cases. Continuing to tune attribution
would overfit an unidentifiable legacy implementation.

The operational reference therefore freezes one transparent convention:

- causal Op1–Op2 material flow and route-aware replenishment;
- mass-conserving Op1–Op9 physical flow;
- linked order fulfillment with daily Op9 release;
- thesis-window endogenous risks;
- event-overlap attribution plus a bounded 168 h R24 convention;
- elapsed recovery period;
- Garrido Excel ReT as the primary evaluation metric.

The convention was chosen for parsimony, physical auditability, and aggregate ReT
fidelity. It is not described as recovered Simulink semantics.

## Publication language

Allowed:

> We reconstruct a Garrido-informed military food supply-chain DES using the
> thesis topology, risk families, inventory and capacity controls, and ReT
> evaluation lineage. Because the original Simulink event-attribution code was
> unavailable, we pre-specify a transparent operational attribution convention
> and report the workbook distribution mismatch as a model boundary.

Prohibited:

- “exact replication of Garrido-Ríos (2017)”;
- “recovered original Simulink logic”;
- “validated against every Cf distribution”;
- hiding the failed warm-up, CT, RP, or risk-share gates.

## Conditions on new RL results

Authorization is not permission to reuse favorable historical comparisons. A
new confirmatory lane must:

1. retrain policies on the frozen operational reference;
2. construct a new full-contract static frontier on calibration tapes only;
3. evaluate with common random numbers on virgin tapes;
4. keep Garrido Excel ReT primary and report the full metric panel;
5. label old Track B checkpoints as legacy-physics evidence only;
6. retain all failed fidelity and attribution audits in the repository.

This decision removes the unresolvable blocker without converting a failed
replication claim into a success.
