# Overnight Handoff 2026-06-27

## Status

This note records the autonomous overnight execution state after freezing the
headroom frontier and launching the confirmatory jobs.

## Frozen Headroom Frontier

Primary Track-A memory profile:

- `envb_frontier_v2`
- `risk_frequency_multiplier=1.0`
- `risk_impact_multiplier=1.5`
- `stochastic_pt=false`
- reward gate winner for this cell: `control_v1`
- outcome: Garrido Excel ReT (`excel_ret`)

Rationale: the full-horizon static grid found this cell non-saturated and
corner-free, while the reward alignment gate found `control_v1` most aligned
with Excel ReT among the tested training rewards for this frontier.

Contract artifacts:

- `docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md`
- `supply_chain/data/headroom_env_contract_v2_2026-06-26.json`
- `outputs/experiments/headroom_calibration_2026-06-26/summary.json`
- `outputs/benchmarks/reward_alignment_static_surface_frontier_v2_2026_06_26/20260627T042241Z/summary.json`

## Completed Confirmatory: DQN Frontier Ladder

Kaggle kernel:

- `thomaschisica/scresia-garrido-dqn-frontier-ladder`
- output directory: `outputs/kaggle/garrido_dqn_frontier_ladder_remote_v3/`

Result:

- MEMORY retained-reset: `-0.000041`
- CI95: `[-0.000085, +0.000003]`
- TOTAL retained-frozen: `+0.000014`
- CI95: `[-0.000054, +0.000082]`

Interpretation: no positive retained-memory result in this DQN frontier cell.
This is a clean null for the retained-vs-reset memory claim, not a dynamic
policy dominance result.

## Completed Sensitivities: Local DQN Ladders

Aggressive Excel route:

- label: `nightly_envb_aggr_g24_raw_ladder_10seed_24block_2026-06-26`
- reward: `ReT_garrido2024_raw`
- `phi=2.0`, `psi=1.5`
- MEMORY retained-reset: `+0.0000015`
- CI95: `[-0.0000321, +0.0000351]`
- TOTAL retained-frozen: `+0.0000133`
- CI95: `[-0.0000151, +0.0000416]`

Conservative Pareto/resource route:

- label: `nightly_envb_cons_control_v2_ladder_10seed_24block_2026-06-26`
- reward: `control_v2`
- `phi=1.0`, `psi=1.25`
- MEMORY retained-reset: `-0.0000344`
- CI95: `[-0.0000817, +0.0000130]`
- TOTAL retained-frozen: `+0.0000302`
- CI95: `[-0.0000502, +0.0001106]`

Interpretation: all DQN retained-transfer ladders are null under the frozen
Excel-ReT outcome. The paper should not lead with retained memory in Track A.

## Running

PPO dynamic-vs-static confirmatory:

- Kaggle kernel: `thomaschisica/scresia-garrido-envb-confirmatory`
- first long run completed, but its kernel exported the full repo and nested
  payloads, making the output impractical to retrieve cleanly.
- kernel was patched to copy the repo to `/kaggle/temp/scres-ia`; clean rerun
  pushed as version 5 and is running.
- candidates:
  - `envb_aggr_g24_raw_ppo`
  - `envb_aggr_g24_raw_recurrent`
  - `envb_cons_control_v2_ppo`

## Current Scientific Read

The strongest remaining path for a paper is now the PPO dynamic-vs-static
confirmatory. The retained-memory DQN result is null in the most defensible
frontier cell and its two sensitivity cells, so the paper should not lead with
a memory-retention win.

Potential claims after PPO:

- Excel-only win: dynamic policy beats all static policies on Garrido Excel ReT.
- Pareto/efficiency win: dynamic policy is Excel-noninferior and improves
  resources/C-D/service metrics.
- Null: if PPO also fails, the honest result is that the thesis-faithful and
  frozen headroom Track-A benchmark is still dominated by simple static policies.
