# Track B Preventive-Tail PBRS Case C Verdict - 2026-07-06

## Verdict

The belief-conditioned/tail PBRS lane improves training, but it does **not**
isolate a preventive mechanism strongly enough to spend the next causal gate on
this branch.

The reason is adversarial rather than cosmetic: all four variants cluster
tightly, and the "tail/preventive" terms do not separate meaningfully from the
belief-conditioned control without tail terms.

## Protocol

- Scenario: selected Case C
  - enabled risks: `R22,R23,R24`
  - `R24` frequency multiplier: `3.0`
  - `R22/R23` impact multiplier: `1.5`
- Observation: `v10_no_forecast`
  - v10 memory retained
  - explicit forecast fields masked for both policy input and belief input
- Policy: PPO+MLP
- Seeds: `1,2,3`
- Training: `30000` timesteps
- Evaluation: `8` episodes
- Horizon: `104` weeks
- Batch size: `64`
- Base reward: `control_v1`
- Auxiliary trigger: R22 4-week belief head

## Results

| Variant | ReT Excel | Static Delta | Cost |
|---|---:|---:|---:|
| belief_v2_control | 0.484793848 | +10.122% | 0.740 |
| preventive_conservative | 0.484806345 | +10.125% | 0.769 |
| preventive_balanced | 0.484592525 | +10.076% | 0.805 |
| preventive_aggressive | 0.484904166 | +10.147% | 0.806 |

Reference from final Case C confirmatory package:

- PPO+MLP Case C final: ReT `0.481160397`, cost `0.719`

## Interpretation

The lane has a real training signal: all PBRS variants are slightly above the
5-seed final Case C PPO baseline in absolute ReT. But the incremental effect of
the risk-conditioned tail terms is tiny:

- `preventive_aggressive` vs. `belief_v2_control`: +0.000110 ReT
- cost increases from `0.740` to `0.806`
- reported `order_ret_excel_cvar05_mean` is `0.0` for all cells in this
  downstream-only branch regime, so it does not distinguish variants here

That pattern is more consistent with a stronger expensive adaptive posture than
with a clean preventive mechanism.

## Decision

Do **not** launch a dedicated PBRS counterfactual from this grid yet. The
next causal compute should go to Ruta B, because the smoke test for Ruta B
already produced a small but direct counterfactual signal:

- smoke only: `2/2` positive pairs
- mean counterfactual delta: `+0.0011565`

That smoke is not evidence by itself, but it is enough to justify a 3-seed Ruta
B grid before spending more causal audit time on PBRS.

## Artifacts

- Grid output:
  `outputs/experiments/track_b_preventive_tail_grid_case_c_3seed_30k_2026-07-06`
- Implementation note:
  `docs/TRACK_B_PREVENTIVE_POLICY_IMPLEMENTATION_2026-07-06.md`
