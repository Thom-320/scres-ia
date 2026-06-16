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
- `outputs/benchmarks/thesis_risk_frequency/thesis_periodic_codex/THESIS_RISK_FREQUENCY.md`
- `outputs/benchmarks/thesis_risk_frequency/legacy_renewal_codex/THESIS_RISK_FREQUENCY.md`
- `outputs/benchmarks/thesis_operations_table/current_codex/THESIS_OPERATIONS_BACKBONE.md`
- `outputs/benchmarks/thesis_design_matrix/current_codex/THESIS_DESIGN_MATRIX.md`
- `outputs/benchmarks/thesis_ret_schema/current_codex/THESIS_RET_SCHEMA.md`

Table 6.10 remains valid under `raw_material_flow_mode=kit_equivalent_order_up_to`:
Python production is `738,432` rations/year, with RMSE vs thesis ECS `61,013.6`,
below the thesis reported RMSE `87,918`.

### Risk-Frequency Fidelity

Table 6.11 now has an explicit mode gate:

- `risk_occurrence_mode=legacy_renewal` preserves the historical/default DES
  process and intentionally fails the Table 6.11 frequency audit.
- `risk_occurrence_mode=thesis_periodic` passes the Table 6.11 frequency audit,
  including the separate R13 fix from monthly to weekly supplier-delivery checks.

Evidence:

- Legacy gap artifact:
  `outputs/benchmarks/thesis_risk_frequency/legacy_renewal_codex/THESIS_RISK_FREQUENCY.md`
- Thesis-frequency artifact:
  `outputs/benchmarks/thesis_risk_frequency/thesis_periodic_codex/THESIS_RISK_FREQUENCY.md`

Interpretation: prior post-inventory reruns that did not set
`risk_occurrence_mode=thesis_periodic` are still useful for inventory debugging
and legacy comparability, but they are superseded for final thesis-frequency
risk claims.

### Post-Fix Static Gates

The post-fix inventory contract is:

- `raw_material_flow_mode=kit_equivalent_order_up_to`
- internal canonical mode `bom_total_units_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`
- for final thesis-frequency risk claims:
  `risk_occurrence_mode=thesis_periodic`

Current local post-fix gates are usable as inventory-repair evidence, but they
were run before the Table 6.11 occurrence-mode repair and should be rerun before
being used as final thesis-frequency risk evidence:

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

The local full static panel equivalent to the long Kaggle fidelity panel is:

- `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_full_cf31_90_260w_3reps_codex/`

Contract:

- panel `Cf31-90`
- profiles `thesis_pattern,current,increased,severe,severe_extended`
- `3` replications
- `260` weekly steps
- `policy_set=minimal`
- `4500` local episode rows plus header
- raw-material mode recorded as internal canonical
  `bom_total_units_order_up_to` with multiplier `2.0`

Local full static H1 risk degradation passes:

| family | profile path | fill | ReT | disruption hours |
|---|---|---|---|---|
| capacity | current -> increased -> severe -> severe_extended | 0.9843 -> 0.9064 -> 0.7991 -> 0.7048 | 0.7947 -> 0.6321 -> 0.5601 -> 0.5401 | 3808.1 -> 14771.2 -> 27644.4 -> 32339.0 |
| inventory | current -> increased -> severe -> severe_extended | 1.0000 -> 0.9371 -> 0.8379 -> 0.7496 | 0.9796 -> 0.7838 -> 0.7198 -> 0.6929 | 3999.9 -> 14397.6 -> 27897.1 -> 32037.8 |

Local full static H2/H3 direction checks are positive across all five profiles
and both families. Thesis-pattern means:

| family | H2 `I672-I0` fill | H2 `I672-I0` ReT | H3 `S3-S1` fill | H3 `S3-S1` ReT |
|---|---:|---:|---:|---:|
| inventory | +0.0774 | +0.3540 | +0.0660 | +0.2710 |
| capacity | +0.0739 | +0.3432 | +0.0656 | +0.2707 |

Interpretation: locally, the repaired static panel supports the basic mechanism:
more severe risk profiles degrade matched-DOE performance, while simple
inventory and capacity interventions improve fill/ReT directionally. This does
not rescue the older pre-fix PPO or per-node headline claims.

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

### Post-Fix Confirmatory PPO Local Rerun

Completed locally with exit code `0`:

- `outputs/benchmarks/confirmatory_ppo_ladder/postfix_confirmatory_ppo_full_20260615T230801Z/`

Contract:

- panel `Cf31-90`
- `10` replications
- `260` weekly steps
- `reward_mode=ReT_thesis`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- common-seed paired panel, not strict CRN
- evaluation-only loading of three pretrained 500k PPO best-shot models

Summary:

| policy | kind | fill | ReT | reward | n |
|---|---|---:|---:|---:|---:|
| `pure_inventory_I672_S1` | static | 0.9687 | 0.8834 | 243.54 | 600 |
| `ppo500k_seed202` | ppo | 0.9687 | 0.8833 | 213.10 | 600 |
| `ppo500k_seed303` | ppo | 0.9687 | 0.8833 | 237.39 | 600 |
| `per_node_I1344_I504_I504_S3` | static | 0.9687 | 0.8833 | 212.32 | 600 |
| `crossed_uniform_I504_S3` | static | 0.9687 | 0.8833 | 212.33 | 600 |
| `ppo500k_seed101` | ppo | 0.9686 | 0.8833 | 242.48 | 600 |
| `pure_capacity_I0_S3` | static | 0.9623 | 0.8295 | 206.24 | 600 |
| `garrido_matched_DOE_baseline` | matched_doe | 0.9590 | 0.8386 | 231.28 | 600 |

Primary PPO contrasts:

- PPO seeds tie `pure_inventory_I672_S1` and `crossed_uniform_I504_S3` on fill
  and ReT at practical scale.
- Against `pure_inventory_I672_S1`, PPO fill deltas are `-0.0001` to `0.0000`;
  ReT deltas are near zero; PPO reward is lower for all three seeds.
- Against `crossed_uniform_I504_S3`, PPO fill/ReT remains tied; reward varies by
  seed because the reward surface weights the static policies differently from
  the loaded PPO actions.
- PPO policies beat `garrido_matched_DOE_baseline` on fill/ReT, but the same is
  already true for simple static buffered policies.

Interpretation: the repaired local PPO comparison does not support a PPO
improvement claim. The current usable claim is narrower: post-fix PPO evaluation
does not outperform simple static buffering on fill or ReT; any paper or
notebook wording should avoid presenting these pretrained PPO models as evidence
of learned superiority in the repaired inventory environment.

### Kaggle State

- Post-fix rerun code is published on branch `codex/garrido-postfix-reruns`.
- Thesis-periodic risk rerun wrappers are published through commit `ea299bc`.
  New Kaggle versions were launched after the Table 6.11 fix:
  - `thomaschisica/scresia-garrido-fidelity-h2-thesis` version 3: `COMPLETE`
  - `thomaschisica/scresia-garrido-fidelity-postfix` version 3: `RUNNING`
  - `thomaschisica/scresia-confirmatory-static-postfix` version 2: `COMPLETE`
  - `thomaschisica/scresia-confirmatory-ppo-postfix` version 5: `RUNNING`
  These are the current Kaggle reruns for thesis-frequency risk timing.
- The Kaggle H2/H3 thesis-periodic gate v3 was downloaded to:
  `outputs/kaggle_garrido_fidelity_h2_thesis_v3_latest/kaggle_outputs/kaggle_h2_thesis_20260616T022622Z/`
  It has `900` rows, records `risk_occurrence_mode=thesis_periodic`, and keeps
  the H2/H3 direction checks positive:
  - inventory family H2 (`I672-I0`): fill `+0.0517`, ReT `+0.3288`;
  - capacity family H2 (`I672-I0`): fill `+0.0511`, ReT `+0.3318`;
  - inventory family H3 (`S3-S1`): fill `+0.0498`, ReT `+0.2753`;
  - capacity family H3 (`S3-S1`): fill `+0.0500`, ReT `+0.2785`.
- The Kaggle confirmatory static thesis-periodic v2 was downloaded to:
  `outputs/kaggle_confirmatory_static_postfix_v2_latest/kaggle_outputs/kaggle_confirmatory_static_postfix_20260616T022622Z/`
  The panel is nearly saturated under faithful risk timing:
  - `crossed_uniform_I504_S3`: fill `0.9982`, ReT `0.9459`, reward `218.50`;
  - `per_node_I1344_I504_I504_S3`: fill `0.9981`, ReT `0.9453`, reward `218.28`;
  - `pure_inventory_I672_S1`: fill `0.9978`, ReT `0.9431`, reward `249.41`;
  - `pure_capacity_I0_S3`: fill `0.9954`, ReT `0.8481`, reward `207.55`;
  - `garrido_matched_DOE_baseline`: fill `0.9891`, ReT `0.8583`, reward `232.68`.
  Interpretation: after both the inventory and risk-frequency repairs, simple
  static buffering is very strong and there is little headroom for adaptive PPO
  in the `[6,3]` thesis-factorized action contract.
- A local severe-extended probe was run after exposing the same fidelity modes in
  `scripts/run_thesis_decision_ppo_smoke.py` (commit `6868988`):
  `outputs/benchmarks/thesis_decision_ppo_smoke/probe_severe_extended_thesis_periodic_30k_codex/`.
  Contract: `risk_level=severe_extended`,
  `risk_occurrence_mode=thesis_periodic`,
  `raw_material_flow_mode=kit_equivalent_order_up_to`, `30k` PPO timesteps,
  `3` eval episodes, `80` weekly steps. Result:
  - PPO MLP: fill `0.5639`, ReT `0.2543`, reward `67.68`;
  - best static grid by fill, `I672,S3`: fill `0.6097`, ReT `0.2922`, reward `61.40`;
  - delta PPO minus best static: fill `-0.0458`, ReT `-0.0378`, reward `+6.29`.
  Interpretation: the earlier PPO `+0.04` fill signal from the legacy-risk probe
  does not survive under faithful risk timing. `severe_extended` creates
  headroom, but the short PPO probe still loses to the best static grid on
  fill/ReT.
- `thomaschisica/scresia-garrido-fidelity-postfix` version 1 stayed `RUNNING`
  for hours without logs, files, or downloadable filtered artifacts. Version 2
  was pushed on 2026-06-16 at about `00:11 UTC` using the artifact-only export
  wrapper from commit `e12ea05`. Do not claim the full Cf31-90 x 5-profile x
  3-rep thesis-horizon panel complete until the version 2 artifacts are
  downloaded and post-processed. Because this run predates the
  `risk_occurrence_mode=thesis_periodic` fix, treat it as a legacy-risk artifact
  even if it finishes; it cannot be the final thesis-frequency risk panel.
- `thomaschisica/scresia-garrido-fidelity-h2-thesis` version 2 completed and its
  exported artifacts were downloaded to:
  `outputs/kaggle_garrido_fidelity_h2_thesis_v2_latest/scres-ia/kaggle_outputs/kaggle_h2_thesis_20260615T223853Z/`
- H2/H3 Kaggle thesis-pattern gate has 900 downloaded episode rows and
  `FIDELITY_GATE_ANALYSIS.md` was regenerated locally against the downloaded
  artifact:
  - inventory family H2 (`I672-I0`): fill `+0.0679`, ReT `+0.3029`;
  - capacity family H2 (`I672-I0`): fill `+0.0633`, ReT `+0.2947`;
  - capacity family H3 (`S3-S1`): fill `+0.0594`, ReT `+0.2533`;
  - inventory family H3 (`S3-S1`): fill `+0.0609`, ReT `+0.2586`.
  This is now downloadable Kaggle evidence, not log-only evidence. It is still a
  thesis-pattern minimal gate, not the full multi-profile fidelity panel.
- `thomaschisica/scresia-confirmatory-static-postfix` completed and its
  exported artifacts were downloaded to:
  `outputs/kaggle_probe_confirmatory_static_latest/scres-ia/kaggle_outputs/kaggle_confirmatory_static_postfix_20260615T224733Z/`
- The Kaggle confirmatory static rerun has `3000` scenario rows plus header and
  matches the local confirmatory static result:
  - `crossed_uniform_I504_S3`: fill `0.9691`, ReT `0.8460`, reward `209.60`;
  - `pure_inventory_I672_S1`: fill `0.9687`, ReT `0.8450`, reward `240.82`;
  - `per_node_I1344_I504_I504_S3`: fill `0.9681`, ReT `0.8436`, reward `209.38`;
  - `pure_capacity_I0_S3`: fill `0.9620`, ReT `0.7720`, reward `202.46`;
  - `garrido_matched_DOE_baseline`: fill `0.9583`, ReT `0.7785`, reward `226.17`.
  The same interpretation applies: buffering beats the matched DOE baseline, but
  crossed uniform and per-node policies do not materially beat the simpler
  `pure_inventory_I672_S1` static policy.
- `thomaschisica/scresia-confirmatory-ppo-postfix` version 1 failed because the
  expected PPO dataset mount was not found. Version 2 found the mount but failed
  because Kaggle datasets are read-only and the runner tried to rebuild SB3
  `.zip` files in `/kaggle/input`. Version 3 repackaged those model directories
  under `/tmp/scresia_ppo_zips`, but remained `RUNNING` without logs, files, or
  downloadable filtered artifacts. Version 4 was pushed on 2026-06-16 at about
  `00:11 UTC` using the artifact-only export wrapper from commit `e12ea05`.
  Version 4 completed and its exported artifacts were downloaded to:
  `outputs/kaggle_confirmatory_ppo_postfix_v4_latest/kaggle_outputs/kaggle_confirmatory_ppo_postfix_20260616T001201Z/`
- The Kaggle confirmatory PPO rerun has `4800` scenario rows plus header and
  matches the local confirmatory PPO result. It is evaluation-only, uses the
  `thomaschisica/scresia-ppo-bestshot-artifacts` dataset when mounted, and does
  not support a PPO superiority claim under the repaired inventory contract:
  PPO ties simple static buffered policies on fill/ReT and does not beat
  `pure_inventory_I672_S1` at practical scale.

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
also complete, output directory:

- `outputs/benchmarks/confirmatory_ppo_ladder/postfix_confirmatory_ppo_full_20260615T230801Z/`

It wrote `4800` scenario rows plus header and `exit_code.txt` is `0`.

These are readiness checks, not final scientific results. The serious rerun
order is:

1. finish/download/post-process the full Kaggle static thesis-horizon panel;
2. treat the completed local static/PPO and Kaggle static/PPO reruns above as current
   evidence, while labeling the loaded PPO models as pretrained pre-existing
   artifacts evaluated under the repaired contract.

## Verification Run

Local verification passed:

- `pytest tests/test_thesis_faithful_lane.py`
- `pytest tests/test_thesis_faithful_lane.py tests/test_run_garrido_static_fidelity_stress.py`
- `ruff check` on the modified rerunner/Kaggle wrapper files
