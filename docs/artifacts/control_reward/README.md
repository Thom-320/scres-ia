# Reward Artifact Bundles

This directory stores tracked, auditable bundles for benchmark runs cited in the manuscript or meeting briefs.

Repository-level benchmark defaults are frozen in:

- `docs/REPOSITORY_SOURCE_OF_TRUTH.md`
- `docs/REPRODUCIBILITY.md`

Each bundle contains:

- `comparison_table.csv`
- `policy_summary.csv`
- `summary.json`
- `manifest.json`

The manifest records:

- benchmark command
- git commit hash
- benchmark date
- source benchmark directory
- copied file paths

Current manuscript source map:

- Table 6: `outputs/benchmarks/table6_minimal/policy_summary.csv`
- Table 7: `outputs/benchmarks/delta_sweep_static/delta_transition.csv`
- Table 8: `outputs/benchmarks/ret_ablation_static/transition_summary.csv`
- Table 9: `outputs/benchmarks/ret_case_diagnostics/case_summary.csv`
- Table 10: `outputs/benchmarks/control_reward/comparison_table.csv` and `outputs/benchmarks/control_reward/policy_summary.csv`

Long-run source-of-truth bundles for the historical `control_v1` lane:

- `docs/artifacts/control_reward/control_reward_500k_increased_stopt`
- `docs/artifacts/control_reward/control_reward_500k_severe_stopt`

Those bundles correspond to the historical comparator configuration:

- `reward_mode=control_v1`
- `observation_version=v1`
- `risk_level in {increased, severe}`
- `stochastic_pt=True`
- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- `500,000` PPO timesteps
- `5` seeds

The repo default has now moved to `ReT_seq_v1` with `ret_seq_kappa=0.20`.
Use these `control_v1` bundles as a legacy comparator, not as the primary
training contract.

Seed-level inferential summary for those runs:

- `docs/artifacts/control_reward/control_reward_500k_seed_inference/seed_inference.json`
- `docs/artifacts/control_reward/control_reward_500k_seed_inference/seed_inference.md`
