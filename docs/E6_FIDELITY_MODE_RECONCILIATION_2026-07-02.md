# E6 Fidelity Flow-Mode Reconciliation - 2026-07-02

## Verdict

The June 28 fidelity gate was not paper-facing evidence because its manifest used
`raw_material_flow_mode=legacy_validated`. A corrected 300-episode rerun was
completed under requested `kit_equivalent_order_up_to`, canonicalized internally
to `bom_total_units_order_up_to`.

Artifact:
`outputs/benchmarks/garrido_static_fidelity_stress/paired_h2_h3_full_cf1_30_thesis_1rep_2026_07_02_e6_kit_equiv/`

## Comparison Against Legacy Gate

Both gates used 300 episodes, `profiles=thesis_pattern`, `panel_cfis=1-30`,
`policy_set=thesis_static`, `replications=1`, and `stochastic_pt=true`.

| Gate | Family | H2 ReT positives | H2 binom p | H3 ReT positives | H3 binom p |
|---|---|---:|---:|---:|---:|
| legacy_validated | risk_r1 | 10/10 | n/a | 10/10 | n/a |
| legacy_validated | risk_r2 | 10/10 | n/a | 10/10 | n/a |
| legacy_validated | risk_r3 | 9/10 | n/a | 7/10 | n/a |
| kit_equivalent_order_up_to | risk_r1 | 10/10 | 0.00098 | 10/10 | 0.00098 |
| kit_equivalent_order_up_to | risk_r2 | 10/10 | 0.00098 | 10/10 | 0.00098 |
| kit_equivalent_order_up_to | risk_r3 | 8/10 | 0.05469 | 6/10 | 0.37695 |

## Interpretation

The corrected faithful-mode gate supports the ReT-sign moderation story for
risk_r1 and risk_r2. It weakens the risk_r3 claim: H2 is borderline and H3 is
not a sign-test pass. Fill-sign checks remain weak or absent, so the manuscript
must describe this as a ReT-sign fidelity/moderation gate, not broad fill-rate
fidelity.

H1 remains structurally missing in this artifact because only the
`thesis_pattern` profile was run; risk-degradation validation needs a separate
multi-profile run.
