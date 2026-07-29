# Controlled-Risk Reward Diagnosis (2026-06-29)

## Question

The controlled-risk probes did not produce a win. Is the failure caused by the reward, the action contract, or the environment/frontier?

## Findings

### CF13

For CF13 (`R21/R22/R24 increased, R23 current`), the static surface is nearly flat on the evaluation metrics. Many policies have identical Excel ReT/CVaR. All tested rewards (`ReT_excel_delta`, `ReT_excel_plus_cvar`, `ReT_tail_v2`, `control_v1`) select low-resource S1 variants, while the Excel/CVaR surface also has equivalent S2 points.

Diagnosis: **not primarily a reward problem**. There is little decision headroom in this CF subset.

### CF20

For CF20 (`R21/R22/R23/R24 increased`), the reward surface is informative:

| reward | best by train return | best by Excel ReT | best by tail/CVaR |
|---|---|---|---|
| `ReT_excel_delta` | `f0.05_S1` | `f0.05_S1` | `f0_S2` |
| `ReT_excel_plus_cvar` | `f0.05_S1` | `f0.05_S1` | `f0_S2` |
| `ReT_tail_v2` | `f0.125_S1` | `f0.05_S1` | `f0_S2` |
| `control_v1` | `f0.05_S1` | `f0.05_S1` | `f0_S2` |

So for Excel ReT, **the reward is mostly aligned**. The learned PPO policy nevertheless collapsed to `frac=0`, `S2`.

The more detailed CF20 surface shows a narrow ridge:

| policy | Excel ReT | train return | resource |
|---|---:|---:|---:|
| `f0.075_S1` | 0.26878 | 84.04 | 0.0375 |
| `f0.05_S1` | 0.26139 | 80.86 | 0.0250 |
| `f0_S1` | 0.23347 | 71.06 | 0.0000 |
| `f0_S2` | 0.21696 | 65.95 | 0.2500 |
| `f0.125_S1` | 0.21696 | 65.98 | 0.0625 |

The optimum is therefore a very narrow low-buffer/S1 band. PPO's deterministic actions in the failed run were:

- `action_frac = 0.0` throughout,
- `action_shift_signal` stayed inside the S2 threshold band,
- applied policy = constant `f0_S2`.

Diagnosis: **mostly optimization/action-contract failure, not reward failure**.

## Mechanism

The current continuous Track-A action has two hard features:

1. Buffer optimum is close to the lower boundary (`frac≈0.05-0.075`), so small policy-mean errors collapse to `0`.
2. Shift is continuous only superficially: the wrapper maps `shift_signal` through hard thresholds:
   - `< -0.33` -> S1
   - `< 0.33` -> S2
   - else S3

PPO initialized near the middle tends to remain in S2 unless the policy crosses a discontinuous threshold. That makes `f0.05_S1` hard to discover even though the reward ranks it correctly.

## Claim Boundary

The controlled-risk lane is still useful as a thesis-faithful diagnostic, but the first probes do **not** show a win.

The failure should not be summarized as "reward bad." A better diagnosis is:

> In CF20 the reward identifies the correct low-buffer/S1 sweet spot, but vanilla PPO on the continuous-threshold Track-A contract fails to discover it. In CF13 the frontier is nearly flat, so there is little headroom regardless of reward.

## Next Tests If We Continue This Lane

Cheap, targeted tests:

1. **Warm-start / behavior-clone near `f0.05_S1` or `f0.075_S1`** and ask whether PPO can improve or only drift away.
2. **Use a factorized/discrete shift action** instead of a continuous threshold for shift.
3. **Fine low-buffer static frontier** around `frac ∈ [0.025, 0.10]` before any RL.
4. **Per-op buffer** for CF20 only, because the common buffer may hide an Op9-specific sweet spot.
5. **Reward objective split:** Excel-primary (`ReT_excel_delta` / `ReT_excel_plus_cvar`) versus tail-primary (`ReT_tail_v2`) because CF20 has Excel/tail tension (`f0.075_S1` vs `f0_S2`).
