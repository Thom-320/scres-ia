# Track B DMLPA Sidecar Verdict -- 2026-07-03

## Artifact

`outputs/experiments/track_b_dmlpa_sidecar_2026-07-03/full8d_5seed_60k_h104/`

## Protocol Checked

- Architecture: PPO + DMLPA transformer-over-history feature extractor
- Seeds: 1..5
- Training timesteps: 60,000
- Evaluation episodes: 12 per seed
- Horizon: h104
- Risk regime: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Observation: `v7`
- Action contract: `track_b_v1` full 8D
- Raw-material flow: `bom_total_units_order_up_to`
- Training vectorization: `n_envs=1`, `n_steps=1024`

## Result

DMLPA is a real positive Track B learner, but it is not the architecture winner.

From `policy_summary.csv`, PPO+DMLPA obtains:

- `order_level_ret_mean_mean = 0.005505` (the sidecar's configured primary metric)
- `order_ret_excel_mean = 0.005721` (Garrido/Excel formula metric)
- `flow_fill_rate_mean = 0.825775`
- `assembly_cost_index_mean = 0.602457`

The best static by primary metric in this bundle is `s2_d1.50`:

- `order_level_ret_mean_mean = 0.005214`
- `order_ret_excel_mean = 0.005428`

Verified gap versus best static:

- `+0.000291` on `order_level_ret_mean_mean`
- `+0.000293` on `order_ret_excel_mean`

Against the current canonical PPO+MLP 10-seed anchor on the same Garrido/Excel metric
(`order_ret_excel ~= 0.005898`), DMLPA is lower:

- DMLPA `order_ret_excel_mean = 0.005721`
- Gap vs canonical PPO+MLP anchor: `-0.000177`

Do not cite `0.005721` as `order_level_ret_mean`; it is the Excel-formula row.
Conversely, do not compare `order_level_ret_mean=0.005505` directly to the
canonical `order_ret_excel=0.005898` headline without labeling that as a cross-metric
diagnostic rather than a manuscript-safe gap.

## Interpretation

DMLPA confirms that a history/Transformer architecture can exploit the Track B action surface, so the positive Track B result is not restricted to a plain MLP. However, this run does not justify replacing PPO+MLP as the Paper 1 headline architecture.

Paper-safe wording:

> A DMLPA history sidecar also beats static policies under the corrected Track B contract, but it does not exceed the canonical PPO+MLP anchor. The result supports architecture robustness rather than architectural superiority.

## Caveats

`summary.json` and `comparison_table.csv` disagree on the printed best-static label (`s2_d1.50` vs `s3_d1.50`). Recomputing directly from `policy_summary.csv` identifies `s2_d1.50` as the best static by the primary metric. This does not change the conclusion.
