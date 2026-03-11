**Control Reward Artifact Bundles**

This directory stores tracked, auditable bundles for control-reward benchmarks cited in the manuscript or meeting briefs.

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
