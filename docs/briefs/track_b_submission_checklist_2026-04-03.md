# Track B Submission Checklist

## Must Have

- Finish the `downstream_only` 500k ablation in [track_b_ablation_500k_production](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/benchmarks/track_b_ablation_500k_production) and fold it into the main causal story.
- Finish the observation ablation in [track_b_observation_ablation_smoke_20260403Tbogota](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/outputs/benchmarks/track_b_observation_ablation_smoke_20260403Tbogota) or rerun it at publication-grade budget.
- Run the frozen PPO forecast-sensitivity check so we can say whether explicit forecasts matter at all.
- Keep `Track A` in the paper as the negative control and `Track B` as the minimal diagnostic extension.
- Report `current / increased / severe / severe_extended` so PPO is shown both in its strong regime and its failure regime.
- Add a limitations paragraph: overcapacity, forecast contrast, partially Markovized observation design, and no “first AI” claim.

## Strongly Recommended

- Run anticipation analysis across all seeds, not only one seed.
- Add a full cross-scenario table for the expanded static DOE, not only the shortlisted manuscript rows.
- Decide whether cost stays as a secondary proxy (`AE`, assembly hours) or is removed from the main claim.
- If `v7_no_forecast` degrades materially, run `RecurrentPPO` only in that reduced-information setting.

## Nice To Have

- Train a PPO specialist directly under `severe` and compare it with the adaptive_v2-trained PPO.
- Add a forecast-scrambling evaluation for the recurrent policy after the PPO result is known.
- Add a short appendix section on why the step-level thesis proxy is diagnostic only.

## Do Not Claim

- “First AI/ANN/NN approach for supply chain resilience.”
- “The model predicts risk autonomously.”
- “Recurrence or belief-state modeling is already superior.”
- “Track B proves PPO is universally best.”

## Claim Instead

- RL effectiveness in this DES depends critically on action-space alignment with the active bottleneck.
- Resilience-aligned rewards can be trained online and audited offline.
- The Garrido 2017 order-level metric remains useful as an audit target, not as an online reward.
- Track B is a minimal research extension built to test the diagnostic failure revealed by Track A.
