# Program D D1-v2 branching verdict (2026-07-11)

**Terminal verdict:** `STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM`.

The risks-on constant frontier legitimately promoted D1 to branching. The
locked admissible comparator was `spt_flat`; on validation it improved Excel
ReT by `+0.010339`, CI95 `[+0.005325,+0.015697]`, while service-loss did not
improve. This authorized branching but not a managerial conclusion.

The exact-prefix experiment evaluated 600 preregistered states × six rules ×
two horizons = 7,200 branch rows across 60 materialized calibration tapes.
All replay and exogenous-identity assertions passed. No virgin tape was opened.

## Gate result

- Optimal rule shares: SPT-contingent 72.67%; SPT-flat 18.67%; LPT-contingent
  5.67%; FIFO-contingent 1.50%; age-threshold 1.17%; FIFO-flat 0.33%.
- Oracle − locked constant Excel ReT: `+0.00001410`, CI95
  `[+0.00000730,+0.00002229]`.
- Relative service-loss reduction: `−0.0000761` (−0.0076%), CI95
  `[−0.0005490,+0.0004086]`.
- Relative lost-order increase: `+0.001084`, CI95
  `[+0.0001515,+0.0023214]`.

Confidence intervals bootstrap the 60 tapes after averaging the ten sampled
state effects within each tape; states are not treated as independent units.

Action variability passed, but the binding 5% service-loss criterion failed by
orders of magnitude and its CI included zero. Therefore no observable tree may
be fitted, no confirmatory/virgin loader may open, and no PPO/retained-learning
experiment is authorized. Per preregistration, D1 is closed; the next optional
decision family is DRA-1 (explicit CSSU-A/B allocation) under a new frozen
structural-extension contract.

Primary machine artifact:
`results/program_d/d1_branching/verdict.json`.
