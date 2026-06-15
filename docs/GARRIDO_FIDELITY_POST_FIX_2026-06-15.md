# Garrido Fidelity Post-Fix Status - 2026-06-15

This note records the first post-fix evidence after adding the explicit raw
material flow modes. It supersedes the pre-fix interpretation of L1b/unified
decision-ladder nulls, but it does not replace a thesis-horizon replication.

## Scope

Mode evaluated:

- `raw_material_flow_mode=bom_total_units_order_up_to`
- public alias: `raw_material_flow_mode=kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`
- `reward_mode=ReT_thesis`
- `action_space_mode=thesis_factorized`
- `inventory_period_mode=thesis_strict`

This mode is opt-in. The default `legacy_validated` path remains unchanged for
historical reproducibility. Use the public alias `kit_equivalent_order_up_to`
when describing the scientific contract; it canonicalizes internally to
`bom_total_units_order_up_to`.

## Gates Passed

### Gate 1 - Table 6.10 Production

The new mode passes the deterministic Table 6.10 production gate in
`tests/test_thesis_faithful_lane.py::test_bom_order_up_to_mode_passes_table_6_10_production_gate`.

The important point is that the fix is not "remove the x12"; the accepted mode
keeps a BOM-consistent total raw-material consumption and order-up-to
replenishment. The naive `bom_total_units` mode remains available for audit, but
it does not pass the production gate.

### Gate 2 - H1/H2/H3 Direction Checks

Full-panel 260-week static smoke:

```bash
python scripts/run_garrido_static_fidelity_stress.py \
  --label bom_order_up_to_full_cf31_90_260w_3reps_codex \
  --output-root outputs/benchmarks/garrido_static_fidelity_stress \
  --panel-cfis 31-90 \
  --profiles thesis_pattern,current,increased,severe,severe_extended \
  --policy-set minimal \
  --replications 3 \
  --horizon-mode fixed \
  --max-steps 260 \
  --reward-mode ReT_thesis \
  --raw-material-flow-mode bom_total_units_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --progress-every 250
```

Output:

- `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_full_cf31_90_260w_3reps_codex/GARRIDO_STATIC_FIDELITY_STRESS.md`
- `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_full_cf31_90_260w_3reps_codex/episode_metrics.csv`

Observed direction checks on Cf31-90, 3 replications, 260 weekly steps:

| Check | Evidence |
|---|---|
| H1 risk degradation | Matched DOE fill/ReT decline monotonically from `current` to `severe_extended`; disruption hours increase. |
| H2 inventory moderation | `pure_inventory_I672_S1 - pure_inventory_I0_S1` is positive for fill and ReT in all risk profiles/families; scenario-level positives are 58-60 / 60 depending on metric/profile. |
| H3 capacity moderation | `pure_capacity_I0_S3 - pure_capacity_I0_S1` is positive for fill and ReT in all risk profiles/families; scenario-level positives are 55-60 / 60 depending on metric/profile. |

Representative full-panel contrasts:

| profile | family | I672-I0 fill | I672-I0 ReT | S3-S1 fill | S3-S1 ReT |
|---|---|---:|---:|---:|---:|
| thesis_pattern | inventory | 0.0774 | 0.3540 | 0.0660 | 0.2710 |
| thesis_pattern | capacity | 0.0739 | 0.3432 | 0.0656 | 0.2707 |
| increased | inventory | 0.0875 | 0.3600 | 0.0846 | 0.2991 |
| increased | capacity | 0.0896 | 0.3588 | 0.0786 | 0.2891 |
| severe_extended | inventory | 0.1458 | 0.3602 | 0.1440 | 0.2988 |
| severe_extended | capacity | 0.1458 | 0.3686 | 0.1431 | 0.3042 |

## Interpretation

The post-fix environment now has a live inventory lever again. This explains why
pre-fix L1b/per-node and unified-evaluation nulls should be treated as
pre-fix artifacts, not final claims about the thesis decision space.

The 260-week full-panel smoke supports moving to thesis-horizon fidelity runs.
It is not itself a final replication of Garrido's 10/20-year design horizons.

## Next Required Gate

Run a thesis-horizon static replication with the same mode. A first local
single-rep thesis-pattern gate has been completed; the full Kaggle run remains
active.

### Thesis-Horizon Interim Gate

Local command:

```bash
python scripts/run_garrido_static_fidelity_stress.py \
  --label bom_order_up_to_cf31_90_thesis_pattern_thesis_horizon_1rep_codex \
  --output-root outputs/benchmarks/garrido_static_fidelity_stress \
  --panel-cfis 31-90 \
  --profiles thesis_pattern \
  --policy-set minimal \
  --replications 1 \
  --horizon-mode thesis \
  --reward-mode ReT_thesis \
  --raw-material-flow-mode bom_total_units_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --progress-every 50
```

Output:

- `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_cf31_90_thesis_pattern_thesis_horizon_1rep_codex/GARRIDO_STATIC_FIDELITY_STRESS.md`

Result on Cf31-90, thesis horizons (`480` weekly decisions for 10-year rows and
`960` for 20-year rows), one replication:

| Check | Scenario-level evidence |
|---|---|
| H2 fill | `I672 > I0` in 42 / 60 scenarios; mean delta `+0.0657`. |
| H2 ReT | `I672 > I0` in 59 / 60 scenarios; mean delta `+0.2953`. |
| H3 fill | `S3 > S1` in 41 / 60 scenarios; mean delta `+0.0593`. |
| H3 ReT | `S3 > S1` in 56 / 60 scenarios; mean delta `+0.2565`. |

This is stronger for ReT than fill because the longer horizon saturates fill in
some rows. It supports the post-fix thesis-horizon direction checks, but it is
still a one-rep interim gate.

### Full Thesis-Horizon Kaggle Gate

The full 3-rep thesis-horizon run is launched as a Kaggle kernel:

```bash
kaggle kernels status thomaschisica/scresia-garrido-fidelity-postfix
```

When it finishes, download artifacts with:

```bash
kaggle kernels output thomaschisica/scresia-garrido-fidelity-postfix \
  -p outputs/kaggle_garrido_fidelity_postfix_latest
```

The intended full command inside Kaggle is:

```bash
python scripts/run_garrido_static_fidelity_stress.py \
  --label bom_order_up_to_full_cf31_90_thesis_horizon_minimal \
  --output-root outputs/benchmarks/garrido_static_fidelity_stress \
  --panel-cfis 31-90 \
  --profiles thesis_pattern,current,increased,severe,severe_extended \
  --policy-set minimal \
  --replications 3 \
  --horizon-mode thesis \
  --reward-mode ReT_thesis \
  --raw-material-flow-mode bom_total_units_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --progress-every 250
```

If this is too slow locally, run it on Kaggle before resuming PPO/RL claims.

Because the full 5-profile x minimal-policy thesis-horizon matrix is expensive,
a smaller H2/H3 thesis-pattern gate is also launched:

```bash
kaggle kernels status thomaschisica/scresia-garrido-fidelity-h2-thesis
```

It runs Cf31-90, `thesis_pattern`, `policy-set=minimal`, 3 replications,
`horizon-mode=thesis`, and `raw_material_flow_mode=kit_equivalent_order_up_to`.
Download with:

```bash
kaggle kernels output thomaschisica/scresia-garrido-fidelity-h2-thesis \
  -p outputs/kaggle_garrido_fidelity_h2_thesis_latest
```

### H1 Matched-Only Thesis-Horizon Gate

For a faster risk-degradation check, the runner now supports
`--policy-set matched_only`. This evaluates only
`garrido_matched_DOE_baseline`, which is enough to test the thesis H1 direction
before evaluating the full static policy set.

Local command:

```bash
python scripts/run_garrido_static_fidelity_stress.py \
  --label bom_order_up_to_h1_matched_only_cf31_90_thesis_horizon_1rep_codex \
  --output-root outputs/benchmarks/garrido_static_fidelity_stress \
  --panel-cfis 31-90 \
  --profiles thesis_pattern,current,increased,severe,severe_extended \
  --policy-set matched_only \
  --replications 1 \
  --horizon-mode thesis \
  --reward-mode ReT_thesis \
  --raw-material-flow-mode kit_equivalent_order_up_to \
  --raw-material-order-up-to-multiplier 2.0 \
  --progress-every 50
```

Output:

- `outputs/benchmarks/garrido_static_fidelity_stress/bom_order_up_to_h1_matched_only_cf31_90_thesis_horizon_1rep_codex/GARRIDO_STATIC_FIDELITY_STRESS.md`

Result: H1 passes in both families under thesis horizons. From `current` to
`severe_extended`, fill and ReT decline monotonically while disruption hours
increase monotonically.

| family | current fill | increased fill | severe fill | severe_ext fill | current ReT | severe_ext ReT | current disr h | severe_ext disr h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| inventory | 1.0000 | 0.9379 | 0.8389 | 0.7502 | 0.9895 | 0.6998 | 8373.0 | 60963.2 |
| capacity | 0.9867 | 0.9164 | 0.8049 | 0.7083 | 0.8574 | 0.5686 | 8179.8 | 62349.5 |
