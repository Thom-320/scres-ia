# Q1 Evidence Synthesis - 2026-07-02

This is the manuscript-facing evidence contract after the E1-E6 recovery plan
and the completed 10-seed Track B expansion. Headline Track B numbers now trace
to `docs/track_b_q1_stats_2026-07-02_final_10seed/`; mechanism, E1-E4, E6, and
appendix robustness numbers trace to the cited experiment directories and the
5-seed final bundle where noted. Older manuscript numbers are superseded.

## Source of Truth

Primary stats bundle:
`docs/track_b_q1_stats_2026-07-02_final_10seed/`

Supporting 5-seed mechanism/robustness bundle:
`docs/track_b_q1_stats_2026-07-02_final/`

Core artifact roots:

- E1 regime-table and heuristic go/no-go:
  `outputs/experiments/track_b_e1_confirmatory_2026-07-02/` and
  `outputs/audits/track_b_e1_go_no_go_2026-07-02/`
- E2 masked-observation retrain:
  `outputs/experiments/track_b_e2_obs_masked_confirm_vps_2026-07-02/`
- E3 cross-regime/horizon matrix:
  `outputs/experiments/track_b_e3_cross_regime_horizon_matrix_vps_2026-07-02/`
- E4 final action-space ablation:
  `outputs/experiments/track_b_ablation_8d_final_2026-07-01/`
- E5 zero-compute inference/statistics:
  `docs/track_b_q1_stats_2026-07-02_final_10seed/` for the headline
  comparison; `docs/track_b_q1_stats_2026-07-02_final/` for E1-E4 support
  tables and the top-12 appendix.
- E6 fidelity flow-mode reconciliation:
  `outputs/benchmarks/garrido_static_fidelity_stress/paired_h2_h3_full_cf1_30_thesis_1rep_2026_07_02_e6_kit_equiv/`

## Verdicts

### E1 - Privileged Regime Table

**Verdict: GO.** PPO beats the best zero-learning comparator with CI95 > 0.

From `e1_verdict.json`:

- Best zero-learning comparator: fitted five-regime static lookup table.
- Comparator mean: `0.0054939463`.
- PPO mean: `0.0058926522`.
- Delta: `+0.0003987059`.
- CI95: `[+0.0003693285, +0.0004302702]`.

Allowed claim: a non-learning policy with direct access to the true regime
signal cannot explain the PPO result. The regime table improves only modestly
over the best common static and remains far below PPO.

### E2 - Masked Privileged Observation

**Verdict: supported.** PPO retains most of its advantage after masking the
five true-regime one-hot fields and two true-transition forecast fields.

From `e2_masked_policy_summary.csv`:

- Masked PPO order-level ReT mean: `0.0056130959` on OVH.
- Best comparator in the masked run: `heur_hysteresis`, order-level ReT
  `0.005245` by policy summary; Kaggle fetched episode means give masked PPO
  `0.005605`.
- The masked run remains above the static and heuristic comparator family.

Allowed claim: privileged observation fields help slightly, but they do not
drive the result. The policy still wins when the privileged fields are zeroed.

### E3 - Scope and Generalization

**Verdict: supported for current/increased, negative for severe.**

From `e3_per_cell_seed_ci.csv`, using the conservative comparator convention
now adopted by the manuscript (best in-cell static by the primary metric,
order-level ReT):

| Horizon | Risk level | PPO ReT | Best static ReT | Delta |
|---|---|---:|---:|---:|
| h52 | current | -- | -- | +0.0003594 |
| h52 | increased | -- | -- | +0.0005375 |
| h52 | severe | -- | -- | -0.0000604 |
| h104 | current | 0.005648 | 0.005405 | +0.0002436 |
| h104 | increased | 0.003660 | 0.003037 | +0.0006234 |
| h104 | severe | -- | -- | -0.0000747 |

Allowed claim: the result generalizes beyond the designed
`adaptive_benchmark_v2` cell to current and increased Garrido-native risk
levels at h52 and h104. Severe is the boundary regime and must be reported
as negative at both tested horizons under the primary-metric comparator
convention. Do not cite the older h104/severe marginal-positive value or
the older majority-win count; those came from selecting the per-cell
comparator by a secondary criterion and are superseded.

Dense-frontier follow-up (2026-07-02/03): both canonical-horizon
current/increased transfer cells have now been re-run against the full
147-cell shift x Op10 x Op12 static dispatch frontier at the same CRN plan.
Artifact:
`outputs/experiments/track_b_e3_h104_per_cell_dense_frontier_2026-07-02/`.
For `current/h104`, the best dense static is
`S3_op10_2.00_op12_1.50` with order-level ReT `0.005405`; frozen PPO remains
higher at `0.005648` (`Delta=+0.000244`, seed-clustered CI95
`[+0.000209,+0.000278]`, 5/5 seeds positive), also improving flow fill
`0.984` vs `0.840` and shift-utilization cost `0.638` vs `1.000`.
For `increased/h104`, the best dense static is
`S2_op10_2.00_op12_1.25` with order-level ReT `0.003037`; frozen PPO remains
higher at `0.003660` (`Delta=+0.000623`, seed-clustered CI95
`[+0.000584,+0.000663]`, 5/5 seeds positive), with flow fill `0.941` vs
`0.613`. The dense follow-up strengthens, rather than weakens, the
current/increased h104 transfer claims; the fixed-comparator caveat now
applies only to h52 cells and the severe boundary cells.

### E4 - Action-Space Ablation

**Verdict: downstream bottleneck access is the mechanism evidence.**

From `e4_ablation_summary.csv`:

| Arm | PPO order-level ReT | Best comparator | Delta |
|---|---:|---|---:|
| joint | 0.0055874 | `heur_tuned` | +0.0003665 |
| downstream_only | 0.0056777 | `heur_s1_max_downstream` | +0.0004290 |
| shift_only | 0.0056071 | `heur_forecast_threshold` | +0.0003765 |

The arm-specific best comparators include heuristics. Earlier static-only
deltas are larger; manuscript wording should specify which comparator family
is being used. The safe mechanism claim is that downstream-dispatch access
is sufficient to recover the strongest ablation performance and that adding
more controllable dimensions is not the explanation.

### E5 - Main Inference and Robustness

From `effect_sizes.csv`, `seed_level_inference.csv`,
`dispatch_cost_sensitivity.csv`, and `summary.json` in the 10-seed bundle,
plus the 5-seed `top12_static_robustness.csv` appendix check:

- Main Excel ReT: PPO `0.0058977` vs best static
  `S2_op10_2.00_op12_1.50` at `0.0054596`, delta `+0.0004382`,
  pooled CI95 `[+0.0004094, +0.0004676]`, seed-clustered CI95
  `[+0.0004207, +0.0004582]`.
- Order-level ReT: PPO `0.0056727` vs static `0.0052465`, delta
  `+0.0004262`, CI95 `[+0.0003985, +0.0004546]`.
- Seed-level deltas are all positive across seeds 1-10:
  `0.000434`, `0.000408`, `0.000447`, `0.000429`, `0.000414`,
  `0.000408`, `0.000424`, `0.000429`, `0.000495`, `0.000494`;
  each seed has `12/12` positive episode pairs.
- Flow fill: PPO `0.960453` vs static `0.665418`, delta `+0.295035`.
- CTj p99: PPO `1201.65h` vs static `8396.17h`, reduction `7194.52h`.
- Shift-utilization cost index is not a primary win: PPO `0.664690` vs
  static `0.666667`; CI crosses zero and `ci95_directional_win=False`.
- Dispatch-inclusive cost sensitivity is nominally cheaper for PPO even at
  `lambda=0` in the 10-seed pool but not significantly so; it becomes
  significantly cheaper once dispatch charge reaches `0.025` per
  multiplier-step. This is a sensitivity result, not a headline cost claim.
- CVaR05 of Excel ReT: PPO `0.0056449` vs static `0.0051103`, delta
  `+0.0005346`, CI95 `[+0.0004696, +0.0005978]`.
- Local 3x3 upstream static bound at the best downstream cell:
  best bound policy `S2_op3_1.00_op9_1.25_op10_2.00_op12_1.50`,
  order-level ReT `0.0056120` vs PPO `0.0056660`, delta `+0.0000540`,
  seed-paired CI95 `[+0.0000424, +0.0000656]`. This does not prove an
  exhaustive eight-dimensional static frontier, but it closes the nearest
  op3/op9 static-bound attack.

### E6 - Fidelity Flow-Mode Reconciliation

**Verdict: usable with narrowed wording.** The corrected flow mode
`kit_equivalent_order_up_to` canonicalizes to `bom_total_units_order_up_to`.
The faithful rerun supports ReT-sign moderation for `risk_r1` and `risk_r2`,
but weakens `risk_r3`.

From `e6_fidelity_gate_analysis.json`:

- H2 ReT positives: r1 `10/10` (p=0.00098), r2 `10/10` (p=0.00098),
  r3 `8/10` (p=0.05469).
- H3 ReT positives: r1 `10/10` (p=0.00098), r2 `10/10` (p=0.00098),
  r3 `6/10` (p=0.37695).
- Fill-sign checks remain weak.

Allowed claim: the model passes a corrected ReT-sign fidelity/moderation
gate for r1/r2 and partially for r3. Do not claim broad fill-rate fidelity.

## Allowed Manuscript Claims

- PPO improves adaptive recovery in the designed sustained-disruption Track B
  cell under common-random-number paired evaluation against dense static
  comparators.
- A local upstream op3/op9 static bound at the best downstream-dispatch cell
  does not overturn the PPO advantage; the safe comparator phrase remains
  "dense downstream-dispatch grid plus local upstream bound," not "full 8D
  static frontier."
- The main Track B result is an Excel/order-level ReT, service-continuity,
  backlog, and recovery-tail win, not a strict assembly-cost win.
- The privileged-observation confound is closed from both directions:
  masked PPO still wins, and a regime-conditioned lookup table still cannot
  match PPO.
- Generalization holds for current/increased risk levels at both h52 and h104,
  but severe is negative at both tested horizons and defines the scope
  boundary.
- The ablation supports downstream dispatch/bottleneck access as the key
  mechanism. It does not prove a universal "RL wins iff bottleneck is
  controllable" law.
- The fidelity gate is a corrected ReT-sign validation/moderation gate, not a
  broad claim of fill-rate validation.

## Claims That Must Not Appear

- PPO achieves a perfect-fill/zero-backorder headline.
- PPO is a strict assembly-hours/cost win at zero dispatch charge.
- The action space uses the retired seven-dimensional description; the
  canonical Track B contract is 8D.
- Cross-scenario generalization is universally closed across all regimes.
- The agent "anticipates" disruptions. Use "adaptive response" unless a
  separate lead/lag mechanism audit supports anticipation.
- The result proves that RL wins if and only if the action space reaches the
  bottleneck.
- The June 28 `legacy_validated` fidelity gate is paper-facing evidence.

## Submission Framing

Defensible IJPR framing:

> PPO improves adaptive recovery in a thesis-grounded military supply-chain
> DES when the control interface reaches the downstream dispatch bottleneck.
> The result survives regime-table and masked-observation controls, holds
> across current/increased risk regimes, and fails narrowly under severe
> service-floor stress.

This should be the claim boundary for the manuscript rewrite and response
package.
