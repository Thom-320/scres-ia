# Track B Q1 Stats Audit

Primary dense static comparator: `S2_op10_2.00_op12_1.50` selected by `order_ret_excel`.

## Primary Result

| Metric | PPO | Best dense static | Delta | CI95 | Cohen d paired |
|---|---:|---:|---:|---:|---:|
| Excel ReT | 0.005893 | 0.005466 | +0.000426 | [+0.000389, +0.000463] | 2.87 |

## Pareto Verdict

- PPO non-dominated vs dense static frontier on ReT/cost/tail/flow: `True`.
- Figures use `order_ret_excel` as the y-axis resilience metric.
- PPO is not claimed as resource-efficient on assembly cost alone; dispatch-inclusive cost is reported as sensitivity.

## Seed-Level Primary Inference

- Seed count: `5`.
- All seed mean deltas positive: `True`.
- Seed-clustered bootstrap CI95 for mean delta: [+0.000414, +0.000438].

## Unified Tail Metric

- CVaR05 definition: conditional mean of the lowest 5% of per-episode order_ret_excel; PPO `0.005645` vs static `0.005110`, delta `+0.000535`.

## Comparator Scope

- The dense static frontier varies `shift x op10 x op12`; it is not an 8D static frontier.
- Inventory-dimension constants require a separate bounded grid before any best-8D-static wording.

## Effect Size Table

| metric | ppo_mean | static_mean | raw_delta_ppo_minus_static | directional_gain | directional_gain_ci95_low | directional_gain_ci95_high | oriented_paired_cohens_d | ci95_directional_win |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| order_ret_excel | 0.00589265 | 0.00546626 | 0.000426397 | 0.000426397 | 0.000388649 | 0.000463043 | 2.87055 | True |
| order_level_ret_mean | 0.00566602 | 0.00525112 | 0.000414905 | 0.000414905 | 0.000377956 | 0.000450425 | 2.89364 | True |
| flow_fill_rate | 0.96132 | 0.668667 | 0.292653 | 0.292653 | 0.280594 | 0.304346 | 6.17159 | True |
| terminal_rolling_fill_rate_4w | 0.996037 | 0.908664 | 0.0873732 | 0.0873732 | 0.0340019 | 0.153788 | 0.365544 | True |
| ret_garrido2024_sigmoid_mean | 0.938526 | 0.753501 | 0.185025 | 0.185025 | 0.175769 | 0.19539 | 4.87115 | True |
| assembly_cost_index | 0.682051 | 0.666667 | 0.0153846 | -0.0153846 | -0.0478659 | 0.0147436 | -0.126696 | False |
| order_service_loss_auc_per_order | 113170 | 1.12548e+06 | -1.01231e+06 | 1.01231e+06 | 932980 | 1.10034e+06 | 3.03322 | True |
| order_backorder_qty_final | 309.363 | 10929.2 | -10619.9 | 10619.9 | 6204.43 | 15747.3 | 0.549739 | True |
| order_lost_rate | 0 | 0.000155763 | -0.000155763 | 0.000155763 | 0 | 0.000389408 | 0.184141 | False |
| order_ctj_p99 | 1206.94 | 8112.92 | -6905.98 | 6905.98 | 6537.6 | 7276.34 | 4.72264 | True |
| order_rpj_p99 | 328.174 | 2047.65 | -1719.47 | 1719.47 | 1600.97 | 1846.62 | 3.43346 | True |
| order_dpj_p99 | 1206.94 | 8112.92 | -6905.98 | 6905.98 | 6537.6 | 7276.34 | 4.72264 | True |

## Artifacts

- `effect_sizes.csv`
- `pareto_points.csv`
- `pareto_ret_cost.png`
- `pareto_ret_tail_ctj.png`
- `pareto_ret_flow.png`
- `pareto_summary.json`
- `seed_level_inference.csv`
- `top12_static_robustness.csv`
- `cvar05_effect.csv`
- `dispatch_cost_sensitivity.csv`
- `comparator_scope.json`
