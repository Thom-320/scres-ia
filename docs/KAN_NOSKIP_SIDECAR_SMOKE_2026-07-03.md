# KAN No-Skip Sidecar Smoke -- 2026-07-03

## Artifact

`outputs/experiments/track_b_kan_noskip_sidecar_2026-07-03/smoke_seed1_10k_h104/`

## Purpose

This smoke test checks whether the KAN-style RBF extractor still trains when the
full linear skip path is disabled. The earlier KAN sidecar was positive, but the
skip connection made it unsafe to claim that the KAN-style basis itself carried
the result.

## Protocol

- Architecture: PPO + `RBFKANFeaturesExtractor`
- Linear skip: disabled (`--kan-no-linear-skip`)
- Seeds: 1
- Training timesteps: 10,000
- Evaluation episodes: 2
- Horizon: h104
- Risk regime: `adaptive_benchmark_v2`
- Reward: `control_v1`
- Observation: `v7`
- Action contract: `track_b_v1`
- Raw-material flow: `bom_total_units_order_up_to`

## Result

The no-skip KAN-style smoke ran cleanly and beat the best static in this tiny
bundle:

- PPO-KAN no-skip `order_level_ret_mean = 0.005580`
- Best static (`s3_d1.00`) `order_level_ret_mean = 0.005123`
- Gap: `+0.000458`
- PPO-KAN no-skip `order_ret_excel = 0.005849`
- Best static `order_ret_excel = 0.005370`

## Interpretation

This is a useful sign that removing the linear skip path does not immediately
break the Track B PPO-KAN sidecar. It is not a confirmatory architecture result:
it uses one seed, 10k training steps, and two evaluation episodes. Treat it as
Garrido-facing evidence that a cleaner KAN-style PPO policy is feasible, not as
a Paper 1 headline or a claim that KAN is superior to PPO+MLP.
