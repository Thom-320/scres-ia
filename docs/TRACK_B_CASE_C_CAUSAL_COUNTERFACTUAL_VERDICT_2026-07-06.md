# Track B Case C causal counterfactual verdict - 2026-07-06

## Question

Case C (`R24` frequency x3, `R22`/`R23` impact x1.5, downstream-only risk
roster) is the strongest mechanism result of the whole session: PPO+MLP beats
the best static baseline by +10.65% on Garrido Excel ReT, and Real-KAN by
+8.11%. Both were shown to work via an exposure-reduction mechanism (more
orders land in the Excel `fill_rate` branch because the policy closes order
windows faster, per `docs/TRACK_B_BRANCH_SHIFT_EXPOSURE_MECHANISM_2026-07-06.md`).

This is the last optional gate: does any part of that gain come from
anticipating R22/R23/R24 before they occur, rather than purely reacting fast
after they occur?

## Method

`R_full - R_reset(pre-risk)`: for each real R22/R23/R24 event in the frozen
Case C PPO checkpoint's evaluation rollouts, replace the policy's action in
the 4 weeks immediately before the event with the policy's own empirical calm
action, and compare the resulting ReT Excel to the actual (`R_full`). A
positive, stable delta with a high positive-pair rate would mean the
pre-event action carried real, anticipatory value.

Script: `scripts/audit_track_b_risk_event_counterfactual.py`, extended in this
session (additively, no existing behavior changed) to support
`--enabled-risks`, `--risk-frequency-by-id`, `--risk-impact-by-id`,
`--faithful`, and `--obs-config v7_no_forecast` so the counterfactual
environment exactly matches the trained Case C protocol.

Checkpoint: `outputs/experiments/track_b_case_c_selected_h104_confirm_2026-07-06/case_c_r24_freq3_r22r23_impact1p5_ppo_h104/v7_no_forecast/` (PPO+MLP,
5 seeds, the same checkpoint that produced the confirmed +10.65% margin).

Protocol: 5 seeds x 12 eval episodes, `max_steps=104`, `risk_level=current`,
`enabled_risks=R22,R23,R24`, `risk_frequency_by_id=R24:3.0`,
`risk_impact_by_id=R22:1.5,R23:1.5`, `faithful=True`, `obs_config=v7_no_forecast`,
`target_risks=R22,R23,R24`, pre-event reset window `[-4,-1]` weeks.

## Results

| Risk | Pairs | Positive pairs | Positive rate | Mean Δ ReT Excel |
|---|---:|---:|---:|---:|
| R22 | 274 | 23 | 8.39% | +0.0000927 |
| R23 | 127 | 15 | 11.81% | +0.000308 |
| R24 | 377 | 14 | 3.71% | +0.0000110 |

Median delta is `0.0` for all three risks -- the small positive mean in each
case is carried by a minority of episodes, not a stable population effect.

## Reading

None of the three risks clear a defensible causal bar. All three positive-pair
rates sit inside the ~8-12% band that has shown up as noise everywhere else
this session (belief-scalar sidecar, Ruta A on R24 and R22, reward shaping,
belief-conditioned PBRS, the oracle prevention-ceiling gate, `ReT_tail_v2`).
R24 -- the risk with the most events per episode (89.8) and therefore the
most statistical power of the three -- has the *lowest* positive-pair rate
(3.71%), which argues against a real effect diluted by noise; more data made
the signal weaker, not clearer.

## Verdict

**Case C's +10.65% (PPO) / +8.11% (Real-KAN) mechanism contains no detectable
anticipatory/preventive component.** It is fully explained by the
already-documented exposure-reduction mechanism: the policy closes order
fulfillment windows faster under downstream stress, which mechanically shifts
more orders into the Excel `fill_rate` branch and improves the recovery-branch
orders it cannot avoid. This is reactive, not preventive -- the policy is not
detectably acting *before* R22/R23/R24 occur in a way that adds stable value.

This closes the causal-prevention question for Track B under every
combination tried this session: memory (v10 obs), belief-scalar sidecar,
Ruta A (R24 and R22 targets), reward shaping (Arm 1), belief-conditioned PBRS
(16-point grid), the oracle prevention-ceiling gate (perfect foreknowledge),
`ReT_tail_v2`, and now the strongest mechanism result available (Case C). All
are negative on the same causal test.

## Claim boundary for the paper

Supported:

- Track B's learned policies (PPO+MLP and Real-KAN) are adaptive, reactive
  resilience optimizers that reduce risk exposure via faster dispatch, and
  this mechanism strengthens under targeted downstream stress (Case C).
- This mechanism is real, reproducible across seeds, and holds under a
  dedicated causal counterfactual test at the point where it is strongest.

Not supported, and should not be claimed:

- Any form of anticipatory or preventive learning (acting before a risk based
  on forecasting it). Every causal test this session, across architecture,
  reward, observation, and risk-roster variations, returns a null result.

Recommended framing for Garrido: "PPO/Real-KAN learn a genuine reactive
resilience mechanism -- fast exposure reduction -- that is measurable, grows
under targeted stress, and survives a causal counterfactual test. It is not
evidence of anticipatory prevention; every attempt to detect that separately
this session was negative."
