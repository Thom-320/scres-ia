# Track B Ruta B Counterfactual Gate Audit - 2026-07-07

## Verdict

The earlier Ruta B preventive-learning claim is not currently supported.

The Ruta B policy still looks useful as an adaptive/low-cost Case C policy, but
the counterfactual gate used in the 2026-07-06 verdict is not specific to
predictive prevention. A reactive PPO baseline clears the same gate at an even
higher rate when evaluated apples-to-apples.

Therefore, do not cite the 74.1% positive-pair result as confirmed preventive
learning.

## Why This Audit Was Needed

The Ruta B confirmatory run used an inlined `R_full - R_reset(pre-risk)` gate in
`scripts/run_track_b_ruta_b_sidecar.py`. That gate defined the replacement
action as the median action over the whole episode.

Under selected Case C, R24 is frequent (`R24` frequency x3). The full-episode
median can be contaminated by nearby reactive behavior rather than representing
a genuinely calm posture. A copied apples-to-apples gate on the frozen reactive
PPO baseline then produced high positive rates too, contradicting the earlier
separate audit result.

## Controls

Primary metric remains Garrido Excel ReT. Positive-pair rate is from
`delta_ret_excel = R_full - R_reset(pre-risk)`.

| Run | Gate | ReT | Cost | Positive Pairs | Positive Rate | Mean Delta |
|---|---:|---:|---:|---:|---:|---:|
| Ruta B confirm, R22+R24 lead 2, lambda 0.25 | naive inlined | 0.481086 | 0.396 | 317/428 | 0.741 | +0.000729 |
| Permuted-label Ruta B control | naive inlined | 0.485030 | 0.423 | 116/171 | 0.678 | +0.000596 |
| Case B Ruta B generalization | naive inlined | 0.606339 | 0.381 | 27/180 | 0.150 | +0.000076 |
| Case A Ruta B generalization | naive inlined | 0.005874 | 0.418 | 67/182 | 0.368 | +0.000006 |
| Real-KAN Ruta B Case C screen | naive inlined | 0.485700 | 0.689 | 72/171 | 0.421 | +0.000253 |
| Reactive PPO Case C | corrected calm-action gate | n/a | n/a | 377/428 | 0.881 | +0.001940 |
| Ruta B Case C | corrected calm-action gate | n/a | n/a | 363/428 | 0.848 | +0.001264 |

## Corrected Gate Details

`scripts/audit_ruta_b_gate_corrected.py` ports the calm-action logic from the
established counterfactual auditor:

- exclude a `(-4, +8)` halo around target-risk events;
- if no non-event candidate steps remain, use the policy's own lowest
  action-intensity quartile.

For selected Case C, R24 is frequent enough that the exclusion halo covers the
entire episode in every seed. Both reactive PPO and Ruta B therefore used the
fallback:

`own_lowest_action_intensity_quartile`, with `0/1248` non-event candidate steps
available per seed.

That makes the gate a test of whether replacing pre-risk actions with a much
calmer low-intensity posture hurts ReT. It does not distinguish anticipatory
prevention from ordinary reactive/adaptive control.

## Per-Seed Corrected Gate

Reactive PPO:

| Seed | Pairs | Positive Rate | Mean Delta |
|---:|---:|---:|---:|
| 1 | 85 | 0.929 | +0.002318 |
| 2 | 86 | 0.872 | +0.002006 |
| 3 | 86 | 0.767 | +0.001332 |
| 4 | 85 | 0.906 | +0.002245 |
| 5 | 86 | 0.930 | +0.001805 |

Ruta B:

| Seed | Pairs | Positive Rate | Mean Delta |
|---:|---:|---:|---:|
| 1 | 85 | 0.694 | +0.000601 |
| 2 | 86 | 0.907 | +0.001942 |
| 3 | 86 | 0.849 | +0.001225 |
| 4 | 85 | 0.906 | +0.001346 |
| 5 | 86 | 0.884 | +0.001202 |

Reactive PPO is stronger than Ruta B under this corrected gate.

## Interpretation

The live auxiliary Ruta B mechanism may still be useful, but the current
counterfactual evidence does not prove prevention. Three checks undermine the
claim:

1. A permuted-label control passes the naive gate.
2. A reactive PPO baseline passes the corrected gate even more strongly than
   Ruta B.
3. Case A, Case B, and Real-KAN Ruta B screens do not reproduce the original
   high causal signal under the naive screen.

The defensible statement is now:

> Ruta B is an adaptive Case C variant with low cost and competitive ReT. The
> previously reported pre-risk reset signal is not specific enough to support a
> preventive-learning claim.

## Artifacts

- Naive Ruta B confirm:
  `outputs/experiments/track_b_ruta_b_r22r24_l2_confirm_5seed_60k_2026-07-06`
- Permuted-label control:
  `outputs/experiments/track_b_ruta_b_permuted_label_control_3seed_30k_2026-07-07`
- Case A generalization:
  `outputs/experiments/track_b_ruta_b_case_a_generalization_3seed_30k_2026-07-07`
- Case B generalization:
  `outputs/experiments/track_b_ruta_b_case_b_generalization_3seed_30k_2026-07-07`
- Real-KAN Ruta B screen:
  `outputs/experiments/track_b_ruta_b_real_kan_case_c_screen_3seed_30k_2026-07-07`
- Corrected reactive PPO gate:
  `outputs/experiments/track_b_corrected_gate_reactive_ppo_case_c_5seed_2026-07-07`
- Corrected Ruta B gate:
  `outputs/experiments/track_b_corrected_gate_ruta_b_case_c_5seed_2026-07-07`

## Next Step

If we still want to search for prevention, the next gate must avoid frequent
event overlap and reactive contamination. Candidate designs:

1. Use sparse target events with non-overlapping pre-windows.
2. Match event episodes and compare action timing before the first event only.
3. Measure lead-lag action uplift before event onset, then require that uplift
   to predict improved post-event ReT.
4. Use a holdout intervention where only the belief label is disrupted while
   the policy action stream and exogenous event calendar stay otherwise fixed.

Until such a gate is built, Track B should be framed as adaptive resilience and
exposure/cost control, not confirmed anticipatory prevention.
