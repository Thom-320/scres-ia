# Real-KAN Sidecar Last-Chance Verdict -- 2026-07-03

## Artifact

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_3seed_30k_h104/`

## Protocol Verified

- Architecture: PPO + official `pykan` `kan.KAN` via `RealKANFeaturesExtractor`
- Seeds: `1..3`
- Training timesteps: `30,000`
- Evaluation episodes: `12` per seed
- Horizon: `h104`
- Risk regime: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Observation: `v7`
- Action contract: `track_b_v1`
- Raw-material flow: requested `kit_equivalent_order_up_to`, canonicalized to
  `bom_total_units_order_up_to`

This is a sidecar screen, not a paper-facing confirmation. The intended
5-seed/60k confirmation is already running separately at:

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`

## Result

Real-KAN is viable in the corrected Track B PPO loop and clearly beats the
static/heuristic family in this 3-seed screen.

From `policy_summary.csv`:

- PPO+Real-KAN `order_level_ret_mean_mean = 0.005703`
- Best static (`s2_d1.50`) `order_level_ret_mean_mean = 0.005236`
- Gap: `+0.000467`
- PPO+Real-KAN `order_ret_excel_mean = 0.005931`
- Best static (`s2_d1.50`) `order_ret_excel_mean = 0.005451`
- Gap: `+0.000479`
- PPO+Real-KAN `flow_fill_rate_mean = 0.969388`
- Best static (`s2_d1.50`) `flow_fill_rate_mean = 0.672992`
- PPO+Real-KAN `assembly_cost_index_mean = 0.913462`
- Best static (`s2_d1.50`) `assembly_cost_index_mean = 0.666667`

Seed-level signs are consistent:

| Seed | Delta `order_level_ret_mean` | Delta `order_ret_excel` | Delta flow fill | Delta cost index |
|---:|---:|---:|---:|---:|
| 1 | +0.000438 | +0.000452 | +0.290338 | +0.333333 |
| 2 | +0.000472 | +0.000483 | +0.300085 | +0.333333 |
| 3 | +0.000490 | +0.000503 | +0.298766 | +0.073718 |

## Interpretation

This is the first real-KAN online-control result in the repo: not an RBF proxy,
not a supervised static-frontier surrogate, and not a linear-skip feature
extractor. It answers Garrido's feasibility question positively: an official
KAN policy/value approximator can be connected to the corrected Track B DES and
can learn a policy that beats strong static comparators.

It does **not** yet replace PPO+MLP as the paper headline architecture:

- It is only `3` seeds and `30k` training steps, while the canonical PPO+MLP
  headline is `10` seeds at the canonical Track B scale.
- It uses more shift capacity than the best static comparator (`0.913` vs
  `0.667` shift-utilization cost index).
- The same-metric Excel ReT value (`0.005931`) is slightly above the current
  canonical PPO+MLP headline anchor (`~0.005898`), but this screen is too small
  to claim architectural superiority.

## Decision

Promote Real-KAN to a full confirmation, not to the manuscript spine. The
confirmation is already running:

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`

Paper-safe wording before the confirmation lands:

> A preliminary real-KAN sidecar using the official pykan implementation is
> feasible and positive in Track B, but remains under confirmation. It is
> treated as architecture-sidecar evidence, not as the manuscript's main claim.
