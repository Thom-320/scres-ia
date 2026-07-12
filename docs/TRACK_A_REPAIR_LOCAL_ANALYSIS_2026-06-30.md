# Track A Repair Local Analysis 2026-06-30

## Verdict

- Dynamic Excel ReT: `0.155247248`
- Best static Excel ReT: `0.155254241` (`op30_op50.1_op90_S2`)
- Raw ReT delta: `-0.000006992`
- Raw ReT win: `False`
- Pareto non-dominated: `False`
- Dynamic resource: `0.072026` vs best-static resource `0.266527`
- Best static at or below dynamic resource: `op30_op50_op90.25_S1` Excel `0.155247248`, resource `0.042366`

## Interpretation

- The repaired PPO did not collapse. All three seeds converged to the same held-out Excel value.
- The miss is microscopic (`7e-6` ReT), but it is still not a raw ReT win.
- The current evidence does **not** prove that the problem is only a missing fraction between 0.10 and 0.15. The best held-out static is `op5=0.10, S2`, while the dynamic mean is closer to `op9=0.25, S1`.
- A finer grid is worth testing only if the static frontier is densified at the same time; otherwise we would create another unfair comparison.

## Action Means

| policy | op3 | op5 | op9 | shift_signal |
|---|---:|---:|---:|---:|
| dynamic | 0.0080 | 0.0121 | 0.2499 | -0.8946 |
| best static | 0.0000 | 0.1000 | 0.0000 | 0.0000 |
| best <= dynamic resource | 0.0000 | 0.0000 | 0.2500 | -1.0000 |

## Seed Outcomes

| seed | Excel | delta vs best static | resource | action mean |
|---:|---:|---:|---:|---|
| 1 | 0.155247248 | -0.000006992 | 0.049939 | [0.0046, 0.003, 0.2495, -0.9744] |
| 2 | 0.155247248 | -0.000006992 | 0.054184 | [0.0098, 0.0046, 0.2426, -0.9573] |
| 3 | 0.155247248 | -0.000006992 | 0.111956 | [0.0096, 0.0288, 0.2575, -0.7521] |

## Dominating Statics

- Dominating static count: `10`
- First 10 dominators:

| label | Excel | resource | action |
|---|---:|---:|---|
| `op30_op50_op90.25_S1` | 0.155247248 | 0.042366 | [0.0, 0.0, 0.25, -1.0] |
| `op30_op50.05_op90.1_S1` | 0.155247248 | 0.025210 | [0.0, 0.05, 0.1, -1.0] |
| `op30_op50.05_op90.25_S1` | 0.155247248 | 0.050629 | [0.0, 0.05, 0.25, -1.0] |
| `op30_op50.1_op90.1_S1` | 0.155247248 | 0.033473 | [0.0, 0.1, 0.1, -1.0] |
| `op30_op50.1_op90.25_S1` | 0.155247248 | 0.058893 | [0.0, 0.1, 0.25, -1.0] |
| `op30_op50.25_op90.1_S1` | 0.155247248 | 0.058263 | [0.0, 0.25, 0.1, -1.0] |
| `op30.05_op50_op90.25_S1` | 0.155247248 | 0.050629 | [0.05, 0.0, 0.25, -1.0] |
| `op30.05_op50.05_op90.1_S1` | 0.155247248 | 0.033473 | [0.05, 0.05, 0.1, -1.0] |
| `op30.05_op50.05_op90.25_S1` | 0.155247248 | 0.058893 | [0.05, 0.05, 0.25, -1.0] |
| `op30.05_op50.1_op90.1_S1` | 0.155247248 | 0.041737 | [0.05, 0.1, 0.1, -1.0] |

## Core Metrics Dynamic vs Best Static

| metric | dynamic | best static | delta | rel delta |
|---|---:|---:|---:|---:|
| `backorder_qty_final` | 39352.4 | 39929.2 | -576.778 | -1.445% |
| `ctj_p90` | 1429.33 | 1405.33 | +24 | +1.708% |
| `ctj_p99` | 4371.56 | 4400 | -28.4444 | -0.646% |
| `delivered_rations` | 780139 | 779369 | +770.185 | +0.099% |
| `demanded_rations` | 851283 | 851090 | +193.148 | +0.023% |
| `flow_fill_rate` | 0.931839 | 0.931099 | +0.000739661 | +0.079% |
| `lost_orders` | 12.3333 | 12.3333 | +0 | +0.000% |
| `lost_rate` | 0.0395299 | 0.0395299 | +0 | +0.000% |
| `ret_continuous` | 0.399658 | 0.399762 | -0.000103913 | -0.026% |
| `ret_excel` | 0.155247 | 0.155254 | -6.99242e-06 | -0.005% |
| `ret_thesis` | 0.0551706 | 0.0551776 | -6.99242e-06 | -0.013% |
| `rpj_p90` | 513.911 | 513.911 | +0 | +0.000% |
| `rpj_p99` | 1296 | 1300.47 | -4.46667 | -0.343% |
| `service_loss_auc_per_order` | 1.75047e+06 | 1.75464e+06 | -4171.68 | -0.238% |
| `service_loss_auc_ration_hours` | 5.46146e+08 | 5.47448e+08 | -1.30156e+06 | -0.238% |
