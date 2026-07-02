# Track B Q1 final stats bundle (2026-07-02)

This directory is the manuscript-facing source-of-truth bundle after E1-E6. It mirrors the zero-compute Q1 stats bundle and adds the completed E1/E2/E3/E4/E6 artifacts. Numeric manuscript claims should cite rows from these files rather than older draft values.

Key added files:

- `e1_verdict.json`, `e1_go_no_go_comparisons.csv`, `e1_gap_decomposition.csv`
- `e2_masked_policy_summary.csv`, `e2_masked_seed_metrics.csv`, `e2_masked_comparison_table.csv`
- `e3_cross_regime_horizon_matrix.csv`, `e3_h52_risk_overview.csv`, `e3_h104_risk_overview.csv`
- `e4_ablation_summary.csv` and per-arm policy summaries
- `upstream_bound_3x3_policy_summary.csv`, `upstream_bound_3x3_seed_metrics.csv`,
  `upstream_bound_3x3_summary.json`, `upstream_bound_3x3_verdict.json`
- `manifest.json` with source artifact paths

Comparator-scope note: the headline 147-cell static frontier is dense over
shift x Op10 x Op12, not the full 8D Track B action contract. The added
3x3 upstream bound varies Op3 and Op9 quantity multipliers at the canonical
best downstream cell. PPO remains above the best bound policy by
`+0.0000540` order-level ReT with seed-paired CI95
`[+0.0000424, +0.0000656]`.
