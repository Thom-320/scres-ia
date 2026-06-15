# Garrido Results Status - 2026-06-15

This note records the current post-fix evidence after rechecking the thesis,
local runners, and Kaggle state. It is the working status for which results are
currently usable and which older results are superseded.

## Current Valid Results

### Thesis Table/Contract Reports

Regenerated locally on 2026-06-15:

- `outputs/benchmarks/table_6_10_reproduction/kit_equivalent_order_up_to_codex/TABLE_6_10_REPRODUCTION.md`
- `outputs/benchmarks/thesis_bom_semantics/current_codex/THESIS_BOM_SEMANTICS.md`
- `outputs/benchmarks/thesis_decision_tables/current_codex/THESIS_DECISION_TABLES.md`
- `outputs/benchmarks/thesis_risk_tables/current_codex/THESIS_RISK_TABLES.md`
- `outputs/benchmarks/thesis_operations_table/current_codex/THESIS_OPERATIONS_BACKBONE.md`
- `outputs/benchmarks/thesis_design_matrix/current_codex/THESIS_DESIGN_MATRIX.md`
- `outputs/benchmarks/thesis_ret_schema/current_codex/THESIS_RET_SCHEMA.md`

Table 6.10 remains valid under `raw_material_flow_mode=kit_equivalent_order_up_to`:
Python production is `738,432` rations/year, with RMSE vs thesis ECS `61,013.6`,
below the thesis reported RMSE `87,918`.

### Known Open Fidelity Gap

`outputs/benchmarks/thesis_risk_frequency/current_codex/THESIS_RISK_FREQUENCY.md`
still fails Table 6.11 frequency fidelity. This is a real DES process-semantics
gap, not a missing-constant issue: Table 6.12 constants are present, but the
current renewal/event process does not reproduce Table 6.11 frequencies.

### Post-Fix Static Gates

The post-fix inventory contract is:

- `raw_material_flow_mode=kit_equivalent_order_up_to`
- internal canonical mode `bom_total_units_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`

Current local post-fix gates are usable:

- H1 thesis-horizon matched-only gate:
  `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h1_matched_only_cf31_90_thesis_horizon_1rep_codex/`
- H2 thesis-horizon inventory gate:
  `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h2_inventory_cf31_60_thesis_horizon_3reps_codex/`
- H3 thesis-horizon capacity gate:
  `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h3_capacity_cf61_90_thesis_horizon_3reps_codex/`
- 260-week full-profile smoke:
  `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_full_cf31_90_260w_3reps_codex/`

The post-processor was rerun against these directories and rewrote
`FIDELITY_GATE_ANALYSIS.md` / `fidelity_gate_analysis.json`.

### Post-Fix Confirmatory Static Local Rerun

Completed locally with exit code `0`:

- `outputs/benchmarks/confirmatory_static_ladder/postfix_confirmatory_static_full_20260615T224417Z/`

Contract:

- panel `Cf31-90`
- `10` replications
- `260` weekly steps
- `reward_mode=ReT_thesis`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- common-seed paired panel, not strict CRN

Summary:

| policy | fill | ReT | reward | n |
|---|---:|---:|---:|---:|
| `crossed_uniform_I504_S3` | 0.9691 | 0.8460 | 209.60 | 600 |
| `pure_inventory_I672_S1` | 0.9687 | 0.8450 | 240.82 | 600 |
| `per_node_I1344_I504_I504_S3` | 0.9681 | 0.8436 | 209.38 | 600 |
| `pure_capacity_I0_S3` | 0.9620 | 0.7720 | 202.46 | 600 |
| `garrido_matched_DOE_baseline` | 0.9583 | 0.7785 | 226.17 | 600 |

Key scenario-level contrasts:

- `crossed_uniform_I504_S3` vs `garrido_matched_DOE_baseline`: fill `+0.0108`
  with CI `[+0.0048, +0.0183]`, Wilcoxon `p=0.00002`; ReT `+0.0674` with
  CI `[+0.0321, +0.1065]`, Wilcoxon `p=0.000001`.
- `pure_inventory_I672_S1` vs `garrido_matched_DOE_baseline`: fill `+0.0104`,
  Wilcoxon `p=0.00279`; ReT `+0.0665`, Wilcoxon `p=0.000002`.
- `crossed_uniform_I504_S3` vs `pure_inventory_I672_S1`: fill `+0.0004` with
  CI crossing zero and Wilcoxon `p=0.49378`; ReT `+0.0009` with CI crossing zero
  and Wilcoxon `p=0.49358`.
- `per_node_I1344_I504_I504_S3` vs `crossed_uniform_I504_S3`: fill `-0.0010`
  and ReT `-0.0024`, both non-significant.

Interpretation: after the inventory repair, the durable static result remains
small and conservative. Non-zero buffering beats the matched DOE baseline, but
crossing inventory with S3 does not significantly beat the simple thesis-pure
`I672,S1` buffered policy; per-node granularity does not beat the uniform crossed
policy. The old pre-fix headline should stay withdrawn.

### Kaggle State

- Post-fix rerun code is published on branch `codex/garrido-postfix-reruns`
  at commit `3c7f7eef1e4c803420b451ef9e99ae6a20edcb6a`.
- `thomaschisica/scresia-garrido-fidelity-postfix` is still running. Do not
  claim the full Cf31-90 x 5-profile x 3-rep thesis-horizon panel complete until
  its artifacts are downloaded and post-processed.
- `thomaschisica/scresia-garrido-fidelity-h2-thesis` version 1 completed and
  its log reports 900/900 episodes with:
  - inventory family: `I672-I0` fill `+0.0679`, ReT `+0.3029`;
  - capacity family: `S3-S1` fill `+0.0594`, ReT `+0.2533`.
  However, version 1 did not expose the expected run directory in downloadable
  output, so it is log evidence only.
- Version 2 of `thomaschisica/scresia-garrido-fidelity-h2-thesis` has been
  pushed with explicit export into `scres-ia/kaggle_outputs/...` and is running.
- `thomaschisica/scresia-confirmatory-static-postfix` has been launched from
  the post-fix branch and is running.
- `thomaschisica/scresia-confirmatory-ppo-postfix` version 1 failed because the
  expected PPO dataset mount was not found. Version 2 found the mount but failed
  because Kaggle datasets are read-only and the runner tried to rebuild SB3
  `.zip` files in `/kaggle/input`. Version 3 repackages those model directories
  under `/tmp/scresia_ppo_zips` and is running. It is evaluation-only and uses
  the `thomaschisica/scresia-ppo-bestshot-artifacts` dataset when mounted.

## Superseded Results

Treat these as superseded for inventory-lever or headline claims:

- pre-fix L1b/per-node nulls;
- pre-fix unified evaluation tables;
- pre-fix PPO/static comparisons that evaluate inventory effects before
  `kit_equivalent_order_up_to`;
- any claim that strict CRN was achieved across different policies.

They can still be used as historical debugging artifacts, but not as current
scientific evidence about whether inventory, per-node granularity, I x S
crossing, or PPO works in the repaired thesis-inventory environment.

## Rerun Readiness

The following rerunners now expose post-fix raw-material semantics:

- `scripts/run_unified_thesis_evaluation.py`
- `scripts/run_confirmatory_static_ladder.py`
- `scripts/run_confirmatory_ppo_ladder.py`

Smoke artifacts generated under `kit_equivalent_order_up_to`:

- `outputs/benchmarks/unified_evaluation/postfix_unified_smoke_kit_equiv/`
- `outputs/benchmarks/confirmatory_static_ladder/postfix_smoke_kit_equiv/`
- `outputs/benchmarks/confirmatory_ppo_ladder/postfix_ppo_smoke_kit_equiv/`

Local full confirmatory static has completed. Local full confirmatory PPO is
running in tmux session `scres_ppo_postfix_230801`, output directory:

- `outputs/benchmarks/confirmatory_ppo_ladder/postfix_confirmatory_ppo_full_20260615T230801Z/`

Last observed local PPO progress: `1000/4800` evaluations completed. It has
loaded three 500k PPO best-shot models and is still running; do not interpret
its partial CSV as a final PPO result until `CONFIRMATORY_PPO_LADDER.md` and
`exit_code.txt` are present.

These are readiness checks, not final scientific results. The serious rerun
order is:

1. finish/download/post-process the full Kaggle static thesis-horizon panel;
2. rerun confirmatory static under the repaired raw-material mode;
3. only then evaluate PPO against the repaired static baseline, without treating
   pre-fix trained PPO as evidence of post-fix learning unless clearly labeled.

## Verification Run

Local verification passed:

- `pytest tests/test_thesis_faithful_lane.py`
- `pytest tests/test_thesis_faithful_lane.py tests/test_run_garrido_static_fidelity_stress.py`
- `ruff check` on the modified rerunner/Kaggle wrapper files
