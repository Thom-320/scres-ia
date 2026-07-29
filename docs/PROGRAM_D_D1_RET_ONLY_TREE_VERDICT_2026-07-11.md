# Program D D1-ReT observable-tree verdict

Date: 2026-07-11

Preregistration commit: `079f41a`

Verdict: **`STOP_D1_RET_NOT_OBSERVABLY_CONVERTIBLE`**

## Design executed

Five depth-3 decision trees were cross-fitted with `GroupKFold` by tape using
the 600 already-open D1 branch states. Each tree was then executed daily over
the 12 tapes excluded from its training, for 60 paired sequential tape effects.
Comparator: frozen `spt_flat`. Primary endpoint:
`ret_excel_visible_v1`; required sensitivity: visible ReT clipped to [0,1].
Virgin tapes opened: 0. PPO trained: no.

## Results

| Estimand | Result |
|---|---:|
| Sequential tree minus static ReT | -0.010709 |
| CI95 | [-0.014877, -0.006853] |
| Clipped ReT delta | -0.010709 |
| Clipped CI95 | [-0.014689, -0.006816] |
| Tapes with positive ReT delta | 0/60 |
| Cross-fitted branch headroom captured | -86.6% |
| Lost-order relative degradation | +23.25%, CI95 [+18.74%, +27.69%] |
| Service-loss relative degradation | -0.013%, CI95 [-0.028%, +0.001%] |
| Backlog-AUC relative degradation | -2.44%, CI95 [-3.05%, -1.86%] |

Mass conservation and exogenous identity passed. Mean classification accuracy
was 76.2%, but accuracy was not a promotion endpoint. The trees predicted
`spt_contingent` for 512/600 sampled states and used it for 40,491 daily epochs;
the locally optimal one-day branch labels therefore induced a near-constant
sequential policy that was worse than `spt_flat`. The positive clairvoyant
one-day oracle was not deployably convertible and the distribution shift was
harmful, particularly for lost orders.

## Claim boundary

D1 is now closed under both questions:

1. original multi-endpoint D1: no state-dependent service headroom;
2. Garrido's narrower criterion: no observable dynamic policy beat the best
   static rule in ReT.

No alternative tree depth, class weighting, threshold, epoch, rule subset or
RL algorithm may be tuned inside D1. The next authorized adaptive experiment is
DRA-1-v2, whose prospective ReT-primary hierarchy was frozen in commit
`079f41a` before this result.
