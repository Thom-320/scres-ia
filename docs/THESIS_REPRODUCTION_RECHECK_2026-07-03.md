# Thesis Reproduction Recheck — 2026-07-03

Purpose: answer the current concern about whether the project is still reproducing the Garrido
(2017) thesis/workbook evidence after the Track A/B hardening work. This note separates three
claims that should not be collapsed:

1. Excel formula reproduction.
2. Thesis/workbook replay fidelity under the frozen thesis-faithful configuration.
3. Endogenous DES behavioral fidelity without replaying the workbook tapes.

## Source Files Checked

- `/Users/thom/Downloads/Rsult_1.xlsx`
- `/Users/thom/Downloads/Raw_data1+Re.xlsx`
- `/Users/thom/Downloads/Raw_data2+Re.xlsx`
- `/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf`
- `outputs/audits/freeze_check_replicate_garrido_2026-06-26/replication_audit.json`
- `docs/THESIS_FAITHFUL_ENV_FREEZE_2026-06-26.md`
- `docs/E6_FIDELITY_MODE_RECONCILIATION_2026-07-02.md`

## Verdict

The project reproduces the Garrido workbook formula and the thesis-scale workbook replay lane well
enough to use as the validation anchor, but it should not claim full behavioral equivalence to every
endogenous thesis simulation mode.

Manuscript-safe wording:

> We reproduce the Garrido workbook resilience formula exactly on the intended raw-order rows and
> calibrate a thesis-faithful DES lane to the workbook scale under the frozen configuration. The
> validation anchor is strongest for Excel ReT scale and H2/H3 ReT sign behavior; fill-rate signs,
> the H1 lane, and some endogenous tail behavior remain disclosed limitations.

## Excel Formula Check

The workbook formula identified in `replication_audit.json` is:

`IF(AVERAGE(risk_cols)>0, IF(APj>0, APj/LT, 0.5*(1/RPj)), 1-((sumBt+sumUt)/j))`

The frozen formula audit reports:

- Total intended raw rows: 47,546.
- Total mismatches: 0.
- Maximum absolute difference: 0.0.

Direct workbook inspection on 2026-07-03 confirms the same scale. The scan detects the header row
per sheet because `CF2` starts on row 2 rather than row 1.

| Workbook | Family | Weighted mean Excel ReT | Simple CF mean | Rows inspected |
|---|---:|---:|---:|---:|
| `Raw_data1+Re.xlsx` | CF1-CF10 / R1 | 0.006253 | 0.006282 | 25,853 |
| `Raw_data2+Re.xlsx` | CF11-CF20 / R2 | 0.201916 | 0.200742 | 21,693 |

The direct scan intentionally used cached Excel values. The repository's frozen audit remains the
authoritative formula-mismatch check because it applies the intended row filters and records zero
mismatches over the thesis raw-order rows.

## Frozen Thesis-Faithful Replay Check

`outputs/audits/freeze_check_replicate_garrido_2026-06-26/replication_audit.json` reports:

| Family | Target ReT mean | Sim ReT mean | Absolute mean gap |
|---|---:|---:|---:|
| R1 | 0.006282 | 0.006302 | 0.000019 |
| R2 | 0.200742 | 0.205146 | 0.004592 |
| Overall | 0.103512 | 0.105724 | 0.002306 |

This is strong evidence for thesis-scale replay fidelity under the frozen configuration, especially
for Excel ReT scale and workbook branch shares.

The frozen configuration includes:

- `year_basis=thesis`
- `horizon_hours=161280`
- `warmup_trigger=op9_arrival`
- `downstream_q_source=figure_6_2`
- `r14_defect_mode=thesis_strict_op6`
- `risk_occurrence_mode=thesis_window`
- `raw_material_flow_mode=kit_equivalent_order_up_to`
- `raw_material_order_up_to_multiplier=2.0`
- `stochastic_pt=false`
- `demand_on_hand_fulfillment_delay=54.0`

## E6 Fidelity-Gate Status

The June 28 `legacy_validated` run should not be used as the paper-facing validation anchor. E6
reran the gate under the requested faithful flow mode and canonicalized internally to the current
faithful implementation.

Paper-facing interpretation:

- H2/H3 ReT sign behavior passes strongly for R1 and R2.
- R3 remains weaker: H2 is borderline and H3 does not pass by the same sign criterion.
- Fill-rate sign behavior is not a strong validation claim.
- H1 was not covered by the thesis-pattern rerun.

Therefore the validation claim should remain scoped to a ReT-sign fidelity/moderation gate, not a
blanket claim that every thesis metric and family is reproduced.

## Relation To Track A And Track B

Track A/B results should be described as experiments in a Garrido-grounded DES testbed. Track B is a
DES-preserving operational extension, not a thesis-faithful replication of the exact 2017 decision
surface.

Track A v2 confirms a useful diagnostic boundary: even after replacing the old per-op buffer top-up
contract with a conservation-respecting action contract, PPO did not convert the available static
oracle headroom into a held-out win. That strengthens the negative boundary result; it does not
weaken the thesis reproduction anchor.

## Current Actionable Limits

- Do not write "full thesis reproduction" without qualifiers.
- Do write "Excel ReT formula reproduced exactly on intended rows."
- Do write "frozen thesis-faithful replay matches R1/R2 ReT scale closely."
- Do disclose that E6 is a ReT-sign gate and that R3/fill/H1 are weaker or incomplete.
- Keep CVaR secondary/descriptive; the thesis/workbook anchor remains Excel ReT.
