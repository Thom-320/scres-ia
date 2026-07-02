# Q1 Evidence Synthesis - 2026-07-02

This is the manuscript-facing evidence contract after the E1-E6 recovery plan.
All numeric claims below trace to `docs/track_b_q1_stats_2026-07-02_final/`
or the cited experiment directories. Older manuscript numbers are superseded.

## Source of Truth

Primary stats bundle:
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
  `docs/track_b_q1_stats_2026-07-02_final/`
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

**Verdict: supported for current/increased, mixed for severe.**

From `e3_cross_regime_horizon_matrix.csv`:

| Horizon | Risk level | PPO ReT | Best static ReT | Delta |
|---|---|---:|---:|---:|
| h52 | current | 0.0053155 | 0.0048324 | +0.0004831 |
| h52 | increased | 0.0032588 | 0.0025172 | +0.0007416 |
| h52 | severe | 0.0001193 | 0.0001797 | -0.0000604 |
| h104 | current | 0.0056481 | 0.0054393 | +0.0002089 |
| h104 | increased | 0.0036601 | 0.0030952 | +0.0005649 |
| h104 | severe | 0.0001282 | 0.0001188 | +0.0000094 |

Allowed claim: the result generalizes beyond the designed
`adaptive_benchmark_v2` cell to current and increased Garrido-native risk
levels at h52 and h104. Severe is a boundary case and must be reported as
mixed, not a clean win.

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

From `effect_sizes.csv`, `seed_level_inference.csv`, `top12_static_robustness.csv`,
`cvar05_effect.csv`, and `dispatch_cost_sensitivity.csv`:

- Main Excel ReT: PPO `0.005893` vs best static
  `S2_op10_2.00_op12_1.50` at `0.005466`, delta `+0.000426`,
  CI95 `[+0.000389, +0.000463]`.
- Order-level ReT: PPO `0.005666` vs static `0.005251`, delta `+0.000415`,
  CI95 `[+0.000378, +0.000450]`.
- Seed-level deltas are all positive: seeds 1-5 =
  `0.000434`, `0.000408`, `0.000447`, `0.000429`, `0.000414`.
- Flow fill: PPO `0.961320` vs static `0.668667`, delta `+0.292653`.
- CTj p99: PPO `1206.94h` vs static `8112.92h`, reduction `6905.98h`.
- Assembly cost index is not a primary win: PPO `0.682051` vs static
  `0.666667`; CI crosses zero and `ci95_directional_win=False`.
- Dispatch-inclusive cost sensitivity becomes directionally cheaper for PPO
  once dispatch charge reaches `0.025` per multiplier-step and has CI95 below
  zero from `0.075` upward. This is a sensitivity result, not a headline cost
  claim.
- CVaR05 of Excel ReT: PPO `0.0056449` vs static `0.0051103`, delta
  `+0.0005346`, CI95 `[+0.0004696, +0.0005978]`.

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
- The main Track B result is an Excel/order-level ReT, service-continuity,
  backlog, and recovery-tail win, not a strict assembly-cost win.
- The privileged-observation confound is closed from both directions:
  masked PPO still wins, and a regime-conditioned lookup table still cannot
  match PPO.
- Generalization holds for current/increased risk levels at both h52 and h104,
  but severe is mixed and should define the scope boundary.
- The ablation supports downstream dispatch/bottleneck access as the key
  mechanism. It does not prove a universal "RL wins iff bottleneck is
  controllable" law.
- The fidelity gate is a corrected ReT-sign validation/moderation gate, not a
  broad claim of fill-rate validation.

## Claims That Must Not Appear

- PPO achieves perfect fill or zero backorders.
- PPO is 57% cheaper or a strict assembly-hours/cost win.
- The action space is 7D; the canonical Track B contract is 8D.
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
> across current/increased risk regimes, and is mixed under severe stress.

This should be the claim boundary for the manuscript rewrite and response
package.
