# Track B CVaR metric note - 2026-07-06

## Definition

For Track B reporting, the canonical CVaR metric is:

`order_ret_excel_cvar05_mean`

It is computed episode-by-episode from the per-order Garrido Excel ReT values:

1. Exclude warm-up orders using the same treatment window as the rest of the
   order-derived metric panel.
2. Compute `ReT_excel_j` for each order with the Garrido Excel piecewise formula.
3. Sort those per-order values ascending.
4. Take the worst 5% of orders.
5. Average that lower tail.

In formula form:

`CVaR_0.05(ReT_excel) = mean({lowest 5% of ReT_excel_j})`

Because this is a resilience score, **higher is better**.

## Implementation

The implementation already exists in:

- `supply_chain/episode_metrics.py`
- function: `_tail_mean(values, frac=0.05, lower_tail=True)`
- emitted metric: `ret_excel_cvar05`
- Track B summary field: `order_ret_excel_cvar05_mean`

The same panel also reports:

- `order_ret_excel_p05_mean`
- `order_ret_excel_p10_mean`
- `order_ret_excel_p25_mean`
- `order_ret_excel_rolling_4w_min_mean`

## Interpretation

Mean ReT Excel answers: "How resilient is the average order?"

CVaR05 answers: "How bad is the resilience of the worst 5% of orders?"

For headroom/prevention experiments, a useful candidate should ideally improve:

1. mean `order_ret_excel_mean`;
2. lower-tail `order_ret_excel_cvar05_mean`;
3. branch-aware interpretation (`excel_case_pct_*`);
4. resource cost, or at least not hide the gain behind always-max capacity.

For downstream-only scenarios (`R22,R23,R24`), absolute ReT and CVaR values are
not directly comparable to all-risk Garrido/adaptive Track B because the Excel
formula branch composition changes. Compare CVaR within the same scenario or
against that scenario's best static baseline.
