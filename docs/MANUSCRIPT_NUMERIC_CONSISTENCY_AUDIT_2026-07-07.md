# Manuscript Numeric Consistency Audit - 2026-07-07

Scope: surgical table-by-table pass after the Reviewer #2 cold read. This
audit checks the current Elsevier manuscript under
`docs/manuscript_current/submission/elsevier/` against the frozen evidence
bundles already in `docs/`.

## Verdict

No blocking numeric inconsistency remains in the edited manuscript. The only
material issue found during this pass was the CVaR row source: the first
available CVaR row came from the older 5-seed bundle, while Table 4 is a
10-seed / 120-pair table. The manuscript row was corrected to the 10-seed
bundle before compile.

## Table-by-table check

| Manuscript object | Current protocol/wording | Evidence source | Status |
|---|---|---|---|
| Appendix top-12 robustness table | 60 CRN pairs, 5-seed canonical top-12 static robustness | `docs/track_b_q1_stats_2026-07-02/top12_static_robustness.csv` | OK. Caption explicitly says 60 pairs; not mixed with the 10-seed headline. |
| Table 1 related-work positioning | Conceptual positioning matrix, not systematic coding | manuscript text only | OK. Caption already avoids systematic-review language. |
| Methodology risk/action/observation/evidence-map tables | Design/provenance tables | manuscript construction | OK. No numeric claim conflict found. |
| Track A boundary table | PPO/BC-PPO 0.155247 vs static 0.155254; oracle headroom +0.000176 to +0.000296 | `docs/TRACK_A_REPAIR_LOCAL_ANALYSIS_2026-06-30.md`, `docs/TRACK_A_HEADROOM_SEARCH_2026-06-29.md` | OK. Framed as boundary evidence, not impossibility proof. |
| Track B main result table | 10 seeds x 12 episodes = 120 paired CRN evaluations | `docs/track_b_q1_stats_2026-07-02_final_10seed/effect_sizes.csv` | OK. Excel ReT, order-level ReT, flow, rolling fill, CTj p99, service-loss AUC, and cost match the 10-seed bundle after rounding. |
| Track B CVaR row | Excel ReT CVaR05 lower tail, 120 pairs | `docs/track_b_q1_stats_2026-07-02_final_10seed/cvar05_effect.csv` | OK after correction. Current row: PPO 0.005673, static 0.005098, delta +0.000575, CI95 [0.000526, 0.000613]. |
| Dispatch-inclusive cost table | 120 paired CRN evaluations, negative delta favors PPO | `docs/track_b_q1_stats_2026-07-02_final_10seed/dispatch_cost_sensitivity.csv` | OK. Rows and CIs match after manuscript rounding. |
| Action-space ablation table and figure caption | 5 seeds x 60k, each arm vs its own best evaluated comparator | `docs/track_b_q1_stats_2026-07-02_final/e4_ablation_summary.csv` and E4 summaries | OK. Wording softened from conclusive/sufficient to supportive/consistent. |
| Privileged-observation table | 5-seed canonical pool; explicitly different from 10-seed headline | `docs/track_b_q1_stats_2026-07-02_final/e1_gap_decomposition.csv`, `e2_masked_comparison_table.csv` | OK. Caption discloses smaller 5-seed pool. |
| Cross-regime stress table | Frozen PPO vs best in-cell static under stress-screen convention | `docs/track_b_q1_stats_2026-07-02_final/e3_cross_regime_horizon_matrix.csv` | OK. Caption scopes the table to order-level ReT and keeps severe-risk boundary. |

## Language changes made in the same pass

- Abstract shortened and de-densified.
- CVaR is now defined once as lower-tail Excel ReT CVaR05, with higher values
  better.
- Main result table now reports CVaR05 using the 10-seed bundle.
- Ablation language now says the pattern is "consistent with" downstream
  dispatch being the strongest observed lever, not proof that dispatch is
  universally sufficient.
- Discussion and conclusion now avoid universal bottleneck-law wording.

## Remaining known scope boundaries

- Top-12 robustness remains a 5-seed appendix robustness check; the manuscript
  explicitly distinguishes it from the 10-seed headline.
- The static frontier is dense over downstream dispatch, not exhaustive over
  all 8 Track B dimensions. The manuscript still relies on the local 3x3
  upstream bound plus Track A boundary evidence for the non-dispatch dimensions.
- The paper should continue to avoid preventive/anticipatory claims. The
  recent Ruta B counterfactual gate audit invalidated that stronger claim under
  the corrected gate.
