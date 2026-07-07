# Track B branch-shift / exposure mechanism - 2026-07-06

## Finding

The Excel branch can change because of the policy, not only because the
scenario changes.

Mechanism: an order is marked as risk-touched when a risk event overlaps the
order window from `OPTj` to `OATj`. `OATj` is the actual delivery time, and it is
policy-dependent. A faster dispatch policy can close the order window before a
risk event overlaps it, moving that order into the Excel `fill_rate` branch
instead of a risk/recovery branch.

Code basis: `supply_chain/supply_chain.py`, `_set_order_ret_indicators`.

Relevant logic:

- point events: `order.OPTj <= event.start_time <= order.OATj`;
- duration events: `max(event.start_time, order.OPTj) < min(event.end_time, order.OATj)`;
- if no risk indicators are set, the order returns through the fill-rate case.

## Evidence from current artifacts

Source:

`outputs/experiments/track_b_riskcond_metric_check_2026-07-06/garrido_downstream_cherry/v7_no_forecast/episode_metrics.csv`

Scenario: `garrido_downstream_cherry`, enabled risks `R22/R23/R24`, same seeds,
same protocol.

Percent of orders in the Excel `fill_rate` branch:

| Policy | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---:|---:|---:|---:|
| PPO | 80.093 | 80.293 | 80.312 | 80.233 |
| s2_d2.00 | 76.855 | 77.090 | 77.444 | 77.130 |
| s3_d1.50 | 76.818 | 77.053 | 77.406 | 77.092 |

PPO consistently moves about 3.1 percentage points more orders into the
no-risk/fill-rate branch than the best static alternatives.

It also improves the risk-conditional ReT:

| Policy | ReT Excel | Risk-conditional ReT |
|---|---:|---:|
| PPO | 0.606324 | 0.117377 |
| s2_d2.00 | 0.592212 | 0.102439 |
| s3_d1.50 | 0.592218 | 0.102972 |

So the mechanism is not only branch composition: PPO also improves the orders
that remain risk-touched.

## Boundary

This is not evidence of preventive anticipation. It is evidence of exposure
reduction through faster adaptive dispatch.

In the all-risk Garrido case, the branch-shift effect disappears:

`outputs/experiments/track_b_riskcond_metric_check_2026-07-06/garrido_all_current/v7_no_forecast/episode_metrics.csv`

PPO and the best static policies all show:

- `fill_rate` branch: 0.0%;
- `recovery` branch: about 99.684%;
- `unfulfilled` branch: about 0.316%.

Interpretation: with very frequent risks such as R11/R13/R14 active, almost no
order window can avoid risk overlap. The exposure-reduction mechanism is visible
only when risks are sparse enough for dispatch speed to matter.

## Suggested paper wording

Use:

"In downstream-only stress tests, the dynamic policy reduces exposure to risk by
closing order windows faster: more orders avoid overlap with disruption windows
and remain in the Excel fill-rate branch, while risk-touched orders also receive
higher risk-conditional ReT."

Avoid:

- "The policy anticipates disruptions."
- "The branch shift proves prevention."
- "Downstream-only absolute ReT is comparable to the all-risk headline."

## Next reporting panel

For Case C and per-risk headroom, report:

1. `order_ret_excel_mean`;
2. `order_ret_excel_cvar05_mean`;
3. `order_ret_excel_risk_conditional_mean_mean`;
4. Excel branch percentages;
5. resource cost;
6. PPO-vs-best-static deltas within the same scenario.
