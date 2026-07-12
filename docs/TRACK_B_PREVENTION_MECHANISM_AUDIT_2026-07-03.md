# Track B Prevention Mechanism Audit

Date: 2026-07-03

## Purpose

Garrido's prevention question is not only whether PPO+MLP or Real-KAN wins on
ReT. The mechanism question is whether a frozen policy prepares before risk,
reacts after damage, or combines both behaviors.

This audit is intentionally evaluation-only. It does not retrain any policy.
It replays the same Track B CRN episodes and evaluates frozen PPO+MLP and
Real-KAN policies under the same seed/episode/eval-seed plan.

## Implemented Runner

Script:

```bash
scripts/audit_track_b_prevention_mechanism.py
```

Outputs:

- `step_ledger.csv`: one row per weekly decision.
- `event_study.csv`: average action around the risk event window `t=-4...+8`.
- `counterfactual_reset.csv`: `R_full - R_reset` by pre/event/post window.
- `forecast_ablation.csv`: full vs forecast zeroed vs forecast scrambled.
- `feature_attribution.csv`: occlusion sensitivity by feature group and action dimension.
- `lead_lag_tests.csv`: Granger-style lead-lag OLS/F-tests by episode and lag.
- `lead_lag_summary.csv`: policy-level summary of directional lead-lag tests.
- `cross_correlation.csv`: lagged cross-correlations by episode.
- `cross_correlation_summary.csv`: policy-level lagged correlation summary.
- `policy_episode_metrics.csv`: ReT, fill rate, service loss, cost, event anchor.
- `policy_classification.csv`: PAI/RRI/REI and final label.
- `summary.json` and `verdict.md`.

## Classification Rule

We do not call a policy preventive unless both conditions hold:

1. It increases action intensity before the event.
2. The pre-event window has positive value under `R_full - R_reset`.

Interpretation:

- `R_full`: final episode ReT under the frozen learned policy.
- `R_reset(w)`: same episode/seed, but actions in window `w` are replaced by a neutral/static action.
- `R_full - R_reset(w) > 0`: actions in that window added value.

Labels:

- `preventiva`: positive pre-risk activation and positive pre-risk contribution.
- `reactiva`: post-risk activation and post-risk contribution.
- `mixta`: both conditions hold.
- `sin señal clara`: insufficient evidence under the strict rule.

## Smoke Test

Command run:

```bash
.venv/bin/python scripts/audit_track_b_prevention_mechanism.py \
  --output-dir outputs/experiments/track_b_prevention_mechanism_audit_2026-07-03_smoke \
  --policies ppo_mlp real_kan \
  --seeds 1 \
  --eval-episodes 2 \
  --max-steps 104 \
  --attribution-samples 8
```

Result: smoke passed and generated all required outputs.

Row counts:

- `step_ledger`: 416
- `event_study`: 22
- `counterfactual_reset`: 12
- `forecast_ablation`: 6
- `feature_attribution`: 8

The smoke is not a final mechanism claim. It only verifies that the audit
plumbing works end to end with frozen PPO+MLP and Real-KAN policies.

## Recommended Full Run

The paired 5-seed audit has now been run:

```bash
.venv/bin/python scripts/audit_track_b_prevention_mechanism.py \
  --output-dir outputs/experiments/track_b_prevention_mechanism_audit_2026-07-03_full5 \
  --policies ppo_mlp real_kan \
  --seeds 1 2 3 4 5 \
  --eval-episodes 12 \
  --max-steps 104 \
  --attribution-samples 64
```

Output:

```text
outputs/experiments/track_b_prevention_mechanism_audit_2026-07-03_full5/
```

Row counts:

- `step_ledger`: 12,480
- `event_study`: 26
- `counterfactual_reset`: 360
- `forecast_ablation`: 6
- `feature_attribution`: 40
- `lead_lag_tests`: 1,920
- `lead_lag_summary`: 8
- `cross_correlation`: 8,160
- `cross_correlation_summary`: 136

## 5-Seed Descriptive Result

| Policy | PAI | RRI | REI | Delta pre | Delta post | Forecast reliance scrambled | Classification |
|---|---:|---:|---:|---:|---:|---:|---|
| PPO+MLP | 0.000078 | 0.036008 | 0.000026 | -0.000004 | 0.000026 | -0.000011 | reactiva |
| Real-KAN | -0.019267 | 0.032087 | 0.000047 | 0.000020 | 0.000047 | -0.000006 | reactiva |

**Status after sanity check:** treat these labels as provisional/descriptive,
not as final mechanism claims. The main audit suggests more post-event than
pre-event value for both learned policies, but the automatic classification rule
has not yet passed the known-reference validation below.

## Known-Reference Sanity Check

We ran `scripts/audit_track_b_prevention_sanity_check.py` against three controls:

| Reference policy | Expected | Actual | Match |
|---|---|---|---|
| `static_s2_d1.50` | sin señal clara | sin señal clara | yes |
| `heur_forecast_threshold` | preventiva | sin señal clara | no |
| `heur_downstream_reactive` | reactiva | sin señal clara | no |

The failure is informative. `heur_forecast_threshold` does show pre-event
activation (`PAI=0.088759`), but its pre-event counterfactual contribution is
near zero (`Delta pre=-0.000001`). Under the current strict rule, "preventive"
means both (1) acting before the event and (2) adding measurable ReT value in
that pre-event window. The heuristic satisfies the first condition but not the
second. The reactive heuristic also fails the value condition. Therefore the
classifier is currently better read as a "valuable preventive/reactive action"
screen, not as a complete behavioral taxonomy.

### Window-local ReT correction check

We also tested a stricter local-value correction in
`scripts/audit_track_b_prevention_sanity_check.py`: instead of using the final
episode ReT delta, it computes ReT only for orders placed inside the intervened
window. This avoids diluting a 4- or 8-week intervention over a 104-week
episode.

That correction **did not fix the sanity check**:

| Reference policy | Expected | Windowed actual | Match |
|---|---|---|---|
| `static_s2_d1.50` | sin señal clara | sin señal clara | yes |
| `heur_forecast_threshold` | preventiva | sin señal clara | no |
| `heur_downstream_reactive` | reactiva | sin señal clara | no |

Windowed deltas were still negative for both heuristic controls
(`heur_forecast_threshold`: `Delta pre=-0.0000189`,
`Delta post=-0.0000192`; `heur_downstream_reactive`:
`Delta pre=-0.0000872`, `Delta post=-0.0001109`). The lesson is not simply
"episode ReT was too diluted." The current rule detects **value-positive
interventions**, not behavior labels. A forecast-triggered heuristic can be
behaviorally preventive but still fail the value test if its early surge does
not improve ReT under the tested risk/evaluation protocol.

Decision: do not present PPO+MLP or Real-KAN as definitively "reactive" to
Garrido yet. The safe wording is: **the current audit finds adaptive learning
and post-event value, while the preventive/reactive classifier remains under
calibration.**

Interpretation: both frozen policies create value mainly after the event window.
Neither meets the strict preventive rule because neither combines clear
pre-event activation with positive pre-event contribution. Forecast ablation also
does not degrade performance in this 5-seed audit, so the current defensible
phrase is `régimen-responsivo/adaptativo`, not `predictivo`.

Lead-lag evidence points the same way. For PPO+MLP, backlog/service lagged
signals predict later action strongly (`share p<0.05 = 0.775`, median
`p = 0.0052`), while action-leading-risk and risk-phase-leading-action are weak
and sporadic. Real-KAN also shows more reactive than preventive evidence:
backlog-to-action is the strongest directional screen (`share p<0.05 = 0.417`),
while forecast-to-action and action-leading-risk are not stable enough to call
prediction. This reinforces the wording: adaptive recovery, not proven
anticipation.

Lagged cross-correlations are consistent with the lead-lag tests. The strongest
PPO+MLP association is backlog-to-action (`best lag = 8`, mean correlation
`0.615`), while forecast-to-action is small and negative (`-0.099`). Real-KAN's
strongest association is also backlog-to-action (`best lag = 5`, mean
correlation `0.234`), while forecast-to-action remains small (`-0.042`). This
is exactly the pattern expected for reactive recovery rather than forecast-led
prevention.

`statsmodels==0.14.6` is installed in the project venv for future independent
Granger validation; the runner currently uses a small built-in OLS/F-test
implementation with `scipy` so the audit remains dependency-light and explicit.

## Coverage Map

Implemented now:

- Causal/value: windowed `R_full - R_reset(w)`.
- Causal/value baseline: learned policy vs static frontier.
- Temporal association: event study, Granger-style lead-lag tests, lagged
  cross-correlation.
- Observation dependence: forecast zeroing and forecast scrambling at inference.
- Feature dependence: grouped occlusion sensitivity.
- Descriptive indices: PAI, RRI, REI and final behavior label.

Not yet implemented:

- Leave-one-step-out action ablation. This is the most granular causal test, but
  much more expensive and noisy than the windowed reset.
- Integrated Gradients/SHAP for PPO+MLP. Occlusion is already present and is the
  first robust pass.
- pyKAN `plot()`, `attribute()`, `auto_symbolic()` on the trained Real-KAN
  policy. The installed pyKAN API supports all three; this should be a focused
  Real-KAN interpretability pass, not mixed into the mechanism runner.
- RUDDER-style return decomposition. Lower priority because the reward is already
  dense by step, while RUDDER is most useful for sparse/delayed reward settings.
- Shifted risk-calendar generalization and unseen-regime transfer. These are
  separate robustness experiments, not explanation diagnostics.

If Garrido wants this as a formal mechanism result, extend the same audit to
seeds 6-10 and add bootstrap intervals around PAI/RRI/REI.

## Methodological Anchors

- RUDDER / return decomposition: temporal credit under delayed rewards.
- Integrated Gradients and SHAP: feature-attribution motivation.
- Attention-is-not-explanation caution: DMLPA attention maps are auxiliary, not
  sufficient proof of mechanism.
