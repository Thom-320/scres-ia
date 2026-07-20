# Paper 2 alignment with Garrido-Ríos (2017), Garrido et al. (2024), and v0

## Verdict

The integrated Paper 2 is a direct empirical continuation of Garrido et al.
(2024) at the level of **closing the open DES loop**. It is not yet a test of
their stronger organizational-learning interpretation, which requires
knowledge retained across campaigns and therefore belongs to Paper 3.

## What is preserved

- The thesis supplies the 13-operation MFSC structure, risk taxonomy, product
  and capacity context, and ReT continuity metric.
- Garrido et al. (2024) identifies the absence of learning between DES outputs
  and subsequent decisions and proposes neural networks, reinforcement
  learning, and simulation optimization as candidate bridges.
- Paper 2 implements that bridge as observable feedback: the DES remains the
  physical authority and the controller changes only explicit decision rights.
- ReT remains the primary continuity endpoint, with physical service, product
  balance, lost demand, resources, and tail outcomes retained as guardrails.

## What must change in the v0 draft

| v0 element | Paper 2 decision |
|---|---|
| “Deep Learning Algorithms” in the title | Replace with the single learning-augmented event-triggered MPC method. |
| Predictive accuracy in the main RQ | Remove as a headline. Belief calibration is an auxiliary diagnostic. |
| ANN/RNN/RL/KAN menu | Remove. Use one frozen method after the pre-learner gates. |
| Track-A quantity/ROP/shift contract | Remove from the main experiment; that bounded lane is historical. |
| Cobb-Douglas sequential reward | Remove. Use canonical terminal ReT and explicit constraints. |
| Weekly intervention only | Extend prospectively to product mix plus a scarce next-review decision. |
| H1 shorter recovery | Replace with static-search and adaptive-value hypotheses. TTR remains secondary. |
| H2 improvement over successive disruptions | Move to Paper 3 retained-learning design. |
| H3 variance reduction | Replace with prospectively frozen product-service and CVaR guardrails. |
| H4 path dependency / L(t-1) | Move to Paper 3; do not retrofit it onto Program Q. |
| DKA/KAN architecture | Remove from Paper 2 unless a future independent contract directly tests it. |
| Exact reproduction language | Use “Garrido-informed, thesis-grounded reconstruction and researcher extension.” |

## New Paper 2 hypotheses

1. Simulator-budgeted search approaches the exact static optimum without
   reading the enumerated answer matrix.
2. Observable feedback beats the complete static frontier.
3. Endogenous review timing adds value beyond the best fixed-cadence adaptive
   control with the same four-review budget. The unconstrained weekly oracle is
   reported only as an upper ceiling.
4. The complete hybrid improves over strengthened MPC and pure RL without
   violating product-service constraints.

## Claim limits

The source literature does not license “first AI for SCRES,” universal neural
superiority, organizational learning observed in a real supply chain, or an
exact digital twin. If successful, Paper 2 supports the best tested method in a
full-DES researcher extension. Paper 3 is required for the causal effect of
retaining knowledge between disruption campaigns.
