# Program L(e-1): claim and identification contract

**Status:** implementation contract; no retained-learning result is claimed.

## Primary claim under test

Under identical physical initial conditions, disruption tapes, information,
resource accounting, and campaign-wise update budgets, retaining actor-critic
weights across disruption campaigns improves held-out supply-chain resilience
relative to restoring the same calibrated initial weights before every campaign.

The claim is promoted only if the confirmatory `persistent_weights - reset_local`
contrast simultaneously satisfies all of the following:

1. the two-way seed-by-tape CI95 for Garrido Excel ReT is wholly above zero;
2. service-loss AUC falls by at least 5%, with its CI95 supporting the reduction;
3. extra shift-hours remain within +/-2% or the persistent policy is Pareto
   non-dominated in ReT, service loss, and resources;
4. the result survives a carrier definition that excludes optimizer and
   normalization state.

## Treatment and counterfactual

- **Treatment:** actor and critic weights learned through campaigns `1..e-1`
  are used to initialize campaign `e`.
- **Counterfactual:** the same actor and critic are restored to `theta_0` before
  campaign `e` and learn only from that campaign.
- **Held constant:** DES version, physical reset distribution, buffer, shock
  tape, observations, reward, environment steps, gradient-update budget,
  normalization, and evaluation probes.
- **Primary carrier:** actor and critic weights only.
- **Technical ablation:** weights plus optimizer state.
- **Never a carrier:** inventory, WIP, backlog, RNG state, risk regime, future
  events, or observation-normalization statistics.

## Unit of analysis and outcomes

Unit: `learner seed x checkpoint x probe tape x strategic buffer`.

Co-primary outcomes:

1. Garrido Excel ReT (`ret_excel` / Excel-formula mean, higher is better);
2. post-promise service-loss AUC in ration-hours (lower is better).

Secondary outcomes: total backlog AUC, fill rate, CVaR10 of Excel ReT,
system-level TTR, RPj distribution, extra shift-hours, switches, and resource
frontiers. Weeks inside a trajectory are not independent experimental units.

## Hypotheses and claim ladder

- **H1 retained learning:** `persistent_weights > reset_local` on common probes.
- **H2 learning curve:** the persistent-minus-reset probe contrast grows with
  campaign exposure.
- **H3 downside resilience:** persistence improves CVaR10/service-loss tails
  without degrading mean ReT or consuming more resources.
- **H4 path dependence:** different orders of the same campaign multiset yield
  reproducibly different final policies or probe outcomes. H4 is non-directional.
- **M1 managerial value:** persistent control is compared separately with the
  best calibration-selected static and observable heuristic policies.

Interpretation:

| Evidence | Permitted conclusion |
|---|---|
| No observable branching headroom | No deployable adaptive headroom in the tested buffer-shift surface |
| PPO beats static but persistent approximately equals reset | Adaptive control, not retained learning |
| Persistent beats reset but not static/heuristic | Retained computational learning with limited managerial value |
| Persistent beats reset and is Pareto useful | Retained learning with operational value |
| Same multiset, different order, reproducibly different outcomes | Computational path dependence |

## Nonclaims

This study does not claim predictive accuracy, prevention, anticipation,
organizational learning in a real supply chain, KAN/RNN/algorithmic superiority,
an exact Garrido/Simulink reproduction, or the first DES-RL study. It uses a
"thesis-grounded reconstruction with audited physical causality and a disclosed
attribution proxy."

## Stop rules

- Stop before PPO if the observable branching policy fails to reduce
  service-loss AUC by 5% at equivalent resources in at least two non-extreme
  buffers while keeping ReT within a 1% non-inferiority margin.
- Stop the retained-learning headline if either co-primary confirmatory gate
  fails.
- Report H4 as null if order effects do not reproduce; H1 may remain supported.
- Never alter reward, observations, normalization, tape universes, or primary
  analysis after opening virgin probes.


---
Cross-reference: the frozen pre-registration companion (runtime freeze,
promotion rules, claim ladder, acceptance tests) is
`docs/CLAIM_AND_IDENTIFICATION_2026-07-11.md`. Both documents govern jointly;
on conflict the stricter rule applies.
