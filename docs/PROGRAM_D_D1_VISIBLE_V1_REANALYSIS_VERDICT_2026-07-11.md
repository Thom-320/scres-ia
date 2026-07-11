# D1 reanalysis under `ret_excel_visible_v1` (2026-07-11)

**Verdict remains terminal:** `STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM`.
No virgin tapes opened; no tree or PPO trained.

## Static frontier

The same 60 materialized selection/validation tapes were rerun. `spt_flat`
remained the locked best admissible constant. Validation delta versus thesis SPT:

- visible Excel ReT: `+0.010497`, CI95 `[+0.005443,+0.015865]`;
- relative service-loss reduction: `-0.000146`, CI95
  `[-0.000330,+0.000037]`.

This authorized branching under the frozen OR rule but did not support a
managerial service claim.

## Exact-prefix branching

600 states × six rules × two horizons = 7,200 rows, with inference clustered by
the 60 tapes. The visible-ledger contract changed the magnitude, not the decision:

- oracle visible ReT delta: `+0.001095`, CI95
  `[+0.000422,+0.001977]`;
- clipped-[0,1] sensitivity: `+0.001095`, CI95
  `[+0.000430,+0.002011]`;
- relative service-loss reduction: `+0.0000055`, CI95
  `[-0.000529,+0.000563]` — effectively zero, far below the required 5%;
- relative lost-order increase: `+0.004822`, CI95
  `[+0.002628,+0.007433]`.

Action variability passed (SPT-contingent 75%, SPT-flat 16%), but the binding
service-loss gate failed. D1 is now closed under both the historical full-ledger
metric and the exact workbook-visible metric.

Artifacts: `results/program_d/d1_v3_visible_frontier/` and
`results/program_d/d1_v3_visible_branching/`.
