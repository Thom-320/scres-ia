# Track B Ruta B Live-Auxiliary Case C Confirmatory Verdict - 2026-07-06

## 2026-07-07 Erratum

This verdict is superseded for the prevention claim by
`docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md`.

The reported Ruta B ReT and cost remain valid, but the pre-risk counterfactual
gate used here is not specific enough to support confirmed preventive learning.
A permuted-label Ruta B control and a reactive PPO baseline also clear related
versions of the gate. Do not cite the 74.1% positive-pair result as confirmed
prevention.

## Status

Superseded by the 2026-07-07 gate audit. The original result showed a strong
counterfactual signal, but later controls showed the gate is not specific to
predictive prevention.

The original audit result was not just higher adaptive ReT: the event-aligned
`R_full - R_reset(pre-risk)` audit was positive at 5 seeds x 60k. That audit is
now treated as non-specific rather than confirmatory.

## Protocol

- Scenario: selected Case C
  - enabled risks: `R22,R23,R24`
  - `R24` frequency multiplier: `3.0`
  - `R22/R23` impact multiplier: `1.5`
- Observation: `v10_no_forecast`
  - v10 memory retained
  - explicit forecast fields masked
- Label channel: raw under `VecNormalize`
  - `label_channel_raw_under_vecnormalize = true`
- Policy: PPO+MLP with live auxiliary belief head
- Auxiliary objective:
  - target risks: `R22 R24`
  - lead: `2` weeks
  - lambda: `0.25`
- Seeds: `1,2,3,4,5`
- Train timesteps: `60000`
- Eval episodes: `12`
- Horizon: `104` weeks
- Batch size: `64`
- n_steps: `1024`
- Reward: `control_v1`

Primary metric: Garrido Excel ReT (`ruta_b_ret_excel_mean`).

Preventive gate: fixed-RNG `R_full - R_reset(pre-risk)`.

## Confirmatory Result

| Metric | Value |
|---|---:|
| Ruta B ReT Excel | 0.481086 |
| Ruta B ReT CI95 by seed | [0.479535, 0.482637] |
| Ruta B cost | 0.396 |
| Cost CI95 by seed | [0.388, 0.404] |
| Best static | `s2_d2.00` |
| Best static ReT Excel | 0.437133 |
| Delta vs best static | +0.043953 |
| Relative delta vs best static | +10.05% |
| Counterfactual pairs | 428 |
| Positive pairs | 317 |
| Positive-pair rate | 0.741 |
| Positive-pair-rate CI95 by seed | [0.648, 0.833] |
| Mean counterfactual delta | +0.000729 |
| Mean-delta CI95 by seed | [+0.000244, +0.001212] |

Per-seed causal audit:

| Seed | Pairs | Positive Rate | Mean Delta |
|---:|---:|---:|---:|
| 1 | 85 | 0.635 | +0.000110 |
| 2 | 86 | 0.837 | +0.001166 |
| 3 | 86 | 0.709 | +0.000707 |
| 4 | 85 | 0.765 | +0.000746 |
| 5 | 86 | 0.756 | +0.000910 |

All five seeds have positive mean counterfactual delta and positive-pair rate
above 0.5.

## Comparison To Prior Case C Baseline

Final Case C PPO baseline:

- PPO+MLP, `v7_no_forecast`, 5 seeds x 60k: ReT `0.481160`, cost `0.719`

Ruta B:

- ReT `0.481086`, cost `0.396`

Ruta B keeps essentially the same Garrido Excel ReT as the final PPO Case C
baseline while spending substantially less resource and, unlike the baseline,
passes the event-aligned preventive counterfactual.

This matters: previous lanes improved training or adaptive exposure, but their
pre-risk reset counterfactuals remained flat or negative. Ruta B changes that.

## Interpretation

The mechanism is consistent with the diagnosis:

- Ruta A failed because the pretrained belief representation could be
  repurposed during PPO fine-tuning.
- Ruta B keeps the future-risk prediction loss active throughout PPO updates.
- The joint `R22+R24` label is necessary; the R22-only screen failed the causal
  gate.

The supported claim is narrow but real:

> In the selected downstream stress Case C, no-forecast PPO with a live
> auxiliary R22/R24 belief objective learns a preventive component: resetting
> pre-risk actions reduces Garrido Excel ReT in most event-aligned pairs.

## Claim Boundary

Supported:

- Preventive learning in selected Case C.
- No explicit forecast leakage (`v10_no_forecast`).
- Effect survives 5 seeds x 60k and 428 event-aligned counterfactual pairs.
- The effect is causal under the implemented `R_full - R_reset(pre-risk)` gate.

Not yet supported:

- Generalization to all-risk Case A.
- Real-KAN Ruta B.
- Claiming the whole Track B headline is preventive. The correct paper framing
  is: standard PPO is adaptive/exposure-reducing; Ruta B adds a confirmed
  preventive component under the controlled Case C stress design.

## Recommendation

Promote Ruta B to the prevention panel, not the main Track B headline.

Use the main paper spine as:

1. Case A/B/C no-forecast PPO/Real-KAN: adaptive resilience and exposure
   reduction.
2. Ruta B Case C: controlled evidence that a live auxiliary belief objective
   can turn the same environment into preventive behavior.

If compute is available, the next optional tests are:

1. Run Ruta B on Case B to see if prevention survives without extra Case C
   headroom.
2. Run a Real-KAN Ruta B sidecar only after preserving this PPO result.
3. Add branch/exposure traces around the positive counterfactual anchors for a
   reviewer-facing mechanism figure.

## Artifacts

- Confirmatory output:
  `outputs/experiments/track_b_ruta_b_r22r24_l2_confirm_5seed_60k_2026-07-06`
- Summary:
  `outputs/experiments/track_b_ruta_b_r22r24_l2_confirm_5seed_60k_2026-07-06/summary.json`
- Counterfactual rows:
  `outputs/experiments/track_b_ruta_b_r22r24_l2_confirm_5seed_60k_2026-07-06/counterfactual_rows.csv`
- Screen verdict:
  `docs/TRACK_B_RUTA_B_LIVE_AUX_CASE_C_VERDICT_2026-07-06.md`
