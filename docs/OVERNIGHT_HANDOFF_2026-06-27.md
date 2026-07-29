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

## Completed Confirmatory: PPO Dynamic-vs-Static Env B

Kaggle kernel:

- `thomaschisica/scresia-garrido-envb-confirmatory`
- first long run completed, but its kernel exported the full repo and nested
  payloads, making the output impractical to retrieve cleanly.
- kernel was patched to copy the repo to `/kaggle/temp/scres-ia`; clean rerun
  pushed as version 5.
- after the clean rerun was launched, the local Kaggle CLI began returning
  `Permission 'kernels.get' was denied` for both the PPO and DQN kernels, even
  though the same authenticated session had just pushed/downloaded kernels
  earlier in the night. This looks like a Kaggle auth/session problem, not a repo
  or kernel-code failure.
- fallback: the same confirmatory profile was run locally to completion.

Local output directory:

- `outputs/kaggle/garrido_envb_confirmatory/`
- `confirmatory_decision.json`
- `confirmatory_summary.csv`
- `confirmatory_all_statics.csv`
- `confirmatory_report.md`
- per-candidate folders under `runs/`

Profile:

- seeds: `8501-8510`
- `eval_episodes=5`
- `max_steps=52`
- `train_timesteps=65536`
- regime: `severe`

Decision:

- complete-win labels: none
- partial Excel-win labels: none
- partial Pareto/resource labels vs frozen efficient frontier: none
- all three candidates dominate the weak `S3_I1344` resource baseline under the
  relaxed report criterion, but none beats the efficient frontier.

Candidate details:

1. `envb_aggr_g24_raw_ppo`
   - reward: `ReT_garrido2024_raw`
   - `phi=2.0`, `psi=1.5`
   - lane: Cobb-Douglas same-bar; C-D is the resilience metric for this lane, and Excel ReT is a
     continuity / non-inferiority check.
   - vs frozen efficient `static_S2_I168`:
     - Excel delta: `+0.0000035`, CI95 `[-0.0000316, +0.0000386]`
     - C-D delta: `+0.001778`, CI95 `[-0.003074, +0.006630]`
     - resource delta: `+5,854,874` (uses more resource)
   - strongest live candidate, but only a near-tie / weak positive against the
     efficient frontier, not a paper-grade win.

2. `envb_aggr_g24_raw_recurrent`
   - reward: `ReT_garrido2024_raw`
   - `phi=2.0`, `psi=1.5`
   - vs frozen efficient `static_S2_I168`:
     - Excel delta: `-0.0000689`, CI95 `[-0.0001183, -0.0000196]`
     - C-D delta: `+0.001626`, CI95 `[-0.003505, +0.006757]`
     - resource delta: `+3,700,343`
   - recurrent history does not help; it is worse than PPO simple on the primary
     Excel metric.

3. `envb_cons_control_v2_ppo`
   - reward: `control_v2`
   - `phi=1.0`, `psi=1.25`
   - service-heavy weights:
     - fill `1.2`
     - service `6.0`
     - lost `4.0`
     - inventory `0.04`
     - shift `0.06`
     - switch `0.01`
   - vs frozen efficient `static_S2_I168`:
     - Excel delta: `-0.000171`, CI95 `[-0.000312, -0.000030]`
     - C-D delta: `-0.001890`, CI95 `[-0.007562, +0.003781]`
     - resource delta: `+752,116`
   - does not deliver the Pareto/resource win against the efficient frontier.

## Current Scientific Read

The strongest result is now negative but useful:

- Track-A retained memory is a clean null across DQN frontier and sensitivity
  cells.
- Recurrent PPO is not a useful rescue path.
- PPO with `ReT_garrido2024_raw` is closest to a live Cobb-Douglas same-bar result, because it is
  slightly positive on C-D and Excel-noninferior against the frozen efficient frontier, but it spends
  more resources and does not beat all static policies.
- `control_v2` did not produce a Pareto/resource win.

The paper should not claim that RL beats Garrido's efficient static frontier
under the current Track-A action surface. The honest paper-facing path is either:

1. A strong fidelity + benchmark paper: the thesis/Excel environment is
   reconstructed, audited, and shown to leave very little headroom for Track-A
   RL once the efficient static frontier is computed.
2. A next control-surface paper: move to a richer action surface (Track B or
   continuous operational control), where RL can control the actual bottleneck
   instead of only choosing buffer and shifts.

Operational note: no local training/download processes from this overnight run
were left running after the local fallback confirmatory completed.
