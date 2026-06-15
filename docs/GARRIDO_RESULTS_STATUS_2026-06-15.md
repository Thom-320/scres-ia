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

### Kaggle State

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
