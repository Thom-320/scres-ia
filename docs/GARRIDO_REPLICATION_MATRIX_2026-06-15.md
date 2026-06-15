# Garrido Replication Matrix - 2026-06-15

This matrix tracks the current one-to-one fidelity status against
Garrido-Rios (2017). It is intentionally conservative: a row is marked as
passed only when there is a committed test, committed runner, or local output
artifact that directly exercises the claim under the current code.

## Active Scientific Contract

The post-fix thesis-faithful inventory contract is explicit and opt-in:

- Historical/default lane: `raw_material_flow_mode=legacy_validated`
- Thesis-inventory repair lane: `raw_material_flow_mode=kit_equivalent_order_up_to`
- Canonical internal mode: `bom_total_units_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`

The repair is not "remove x12." It preserves BOM-consistent raw-material
semantics and adds order-up-to replenishment so the inventory buffers can bind.
The mode must pass both deterministic production fidelity and inventory
moderation before it can be used for final Garrido comparisons.

## Evidence Matrix

| Thesis item | Current status | Evidence | Interpretation |
|---|---|---|---|
| Table 6.1 BOM semantics | Implemented as explicit mode | `supply_chain/supply_chain.py` scales raw targets by `NUM_RAW_MATERIALS` and consumes `_raw_units_per_ration`; `tests/test_thesis_faithful_lane.py::test_bom_total_units_mode_scales_raw_buffer_targets_only` | One ration is represented as a kit-equivalent draw over 12 raw-material components. |
| Op1-Op13 DES backbone, Table 6.4 demand, and Figure 6.2 downstream ranges | Passed locally | `outputs/benchmarks/thesis_operations_table/current_codex/THESIS_OPERATIONS_BACKBONE.md`; `tests/test_thesis_faithful_lane.py::test_thesis_operations_table_reporter_writes_match_artifacts` | All 85 checked backbone rows match the extracted thesis constants: operation PT/Q/ROP/risks/units, demand process, downstream Q ranges, and core time constants. |
| Cf1-Cf90 thesis design matrix | Passed locally | `outputs/benchmarks/thesis_design_matrix/current_codex/THESIS_DESIGN_MATRIX.md`; `tests/test_thesis_faithful_lane.py::test_thesis_design_matrix_reporter_writes_match_artifacts` | All 90 configurations match the expected family, source-Cf mapping, risk overrides, inventory period, shift level, initial buffers, and thesis horizon. |
| Table 6.25 SDM schema and Eq. 5.5 ReT cases | Passed locally | `outputs/benchmarks/thesis_ret_schema/current_codex/THESIS_RET_SCHEMA.md`; `tests/test_thesis_faithful_lane.py::test_thesis_ret_schema_reporter_writes_match_artifacts` | All 12 SDM columns, thesis ReT weights, and five canonical Eq. 5.5 cases match the implemented order-level ReT module. |
| Table 6.16 inventory buffers | Passed locally | `outputs/benchmarks/thesis_decision_tables/current_codex/THESIS_DECISION_TABLES.md`; `tests/test_thesis_faithful_lane.py::test_thesis_decision_tables_reporter_writes_match_artifacts` | All 15 Op3/Op5/Op9 buffer values match the extracted thesis table. |
| Table 6.20 capacity by shifts | Passed locally | `outputs/benchmarks/thesis_decision_tables/current_codex/THESIS_DECISION_TABLES.md`; `tests/test_thesis_faithful_lane.py::test_thesis_decision_tables_reporter_writes_match_artifacts` | All 24 capacity/ROP/batch-size fields for S1/S2/S3 match the extracted thesis table. |
| Table 6.12 risk distributions | Passed locally | `outputs/benchmarks/thesis_risk_tables/current_codex/THESIS_RISK_TABLES.md`; `tests/test_thesis_faithful_lane.py::test_thesis_risk_tables_reporter_writes_match_artifacts` | Current and increased risk distributions for R11-R14, R21-R24, and R3 match the extracted thesis table. |
| Table 6.10 deterministic production | Passed locally | `tests/test_thesis_faithful_lane.py::test_bom_order_up_to_mode_passes_table_6_10_production_gate`; alias test also passes | The repair does not break the 738,432 rations/year deterministic production gate. |
| Table 6.10 year-by-year comparison | Passed locally | `outputs/benchmarks/table_6_10_reproduction/kit_equivalent_order_up_to_codex/TABLE_6_10_REPRODUCTION.md`; `tests/test_thesis_faithful_lane.py::test_table_6_10_reporter_writes_year_by_year_artifacts` | Python Cf0 produces 738,432 rations/year; RMSE vs thesis ECS is 61,013.6, below the thesis reported RMSE 87,918. |
| Historical reproducibility | Preserved | `tests/test_thesis_faithful_lane.py::test_legacy_validated_mode_preserves_raw_buffer_targets` | The default legacy path remains available for pre-fix artifact comparison. |
| H1 risk degradation | Passed locally under thesis horizons | `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h1_matched_only_cf31_90_thesis_horizon_1rep_codex/GARRIDO_STATIC_FIDELITY_STRESS.md` | Matched DOE fill/ReT decline from `current` to `severe_extended`, while disruption hours increase. |
| H2 inventory moderation | Passed locally under thesis horizons | `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h2_inventory_cf31_60_thesis_horizon_3reps_codex/GARRIDO_STATIC_FIDELITY_STRESS.md` | `I672 > I0` in 30/30 scenarios by ReT and 25/30 by fill; mean ReT delta `+0.3029`. |
| H3 capacity moderation | Passed locally under thesis horizons | `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h3_capacity_cf61_90_thesis_horizon_3reps_codex/GARRIDO_STATIC_FIDELITY_STRESS.md` | `S3 > S1` in 27/30 scenarios by ReT and 18/30 by fill; mean ReT delta `+0.2533`. |
| Full Cf31-90 x 5-profile thesis-horizon panel | Pending cloud artifact | Kaggle kernel `thomaschisica/scresia-garrido-fidelity-postfix` | Running at last check; do not claim complete until artifacts are downloaded and inspected. |
| Cf31-90 thesis-pattern 3-rep H2/H3 panel | Pending cloud artifact | Kaggle kernel `thomaschisica/scresia-garrido-fidelity-h2-thesis` | Running at last check; local fragmented H2/H3 gates already support the direction checks. |
| Pre-fix L1b/unified/PPO nulls | Superseded for inventory claims | `docs/GARRIDO_FIDELITY_POST_FIX_2026-06-15.md` | Treat as pre-fix artifacts because the inventory lever was inert before the structural repair. |

## Current Non-Claims

- Do not claim final full replication of all thesis scenario tables until the
  Kaggle thesis-horizon artifacts are downloaded and summarized.
- Do not claim PPO/RL beats Garrido on the repaired thesis-inventory lane until
  static post-fix baselines are complete and RL is evaluated against them.
- Do not compare pre-fix unified evaluation tables against post-fix static
  gates; the action semantics changed in the inventory lever.

## Next Gate

To regenerate the deterministic Table 6.10 artifact:

```bash
python scripts/report_table_6_10_reproduction.py \
  --label kit_equivalent_order_up_to_codex \
  --raw-material-flow-mode kit_equivalent_order_up_to
```

To regenerate the decision-table constants artifact:

```bash
python scripts/report_thesis_decision_tables.py --label current_codex
```

To regenerate the risk-table constants artifact:

```bash
python scripts/report_thesis_risk_tables.py --label current_codex
```

To regenerate the operations-backbone artifact:

```bash
python scripts/report_thesis_operations_table.py --label current_codex
```

To regenerate the design-matrix artifact:

```bash
python scripts/report_thesis_design_matrix.py --label current_codex
```

To regenerate the SDM/ReT schema artifact:

```bash
python scripts/report_thesis_ret_schema.py --label current_codex
```

When either Kaggle kernel finishes:

```bash
kaggle kernels output thomaschisica/scresia-garrido-fidelity-postfix \
  -p outputs/kaggle_garrido_fidelity_postfix_latest

kaggle kernels output thomaschisica/scresia-garrido-fidelity-h2-thesis \
  -p outputs/kaggle_garrido_fidelity_h2_thesis_latest
```

Then compare the downloaded `GARRIDO_STATIC_FIDELITY_STRESS.md` and
`episode_metrics.csv` against the local H1/H2/H3 gates in this matrix. Use the
post-processor so the cloud artifacts are summarized by the same gate logic:

```bash
python scripts/analyze_garrido_fidelity_outputs.py \
  outputs/kaggle_garrido_fidelity_postfix_latest \
  outputs/kaggle_garrido_fidelity_h2_thesis_latest
```

Only after that should the repaired environment be used as the baseline for
any post-fix PPO or decision-ladder claims.
