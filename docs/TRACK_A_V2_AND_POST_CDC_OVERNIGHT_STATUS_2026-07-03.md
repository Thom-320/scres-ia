# Track A v2 / Post-CDC Overnight Status (2026-07-03)

## Track A v2 conservation gate

Track A is reopened as a scientific question, not yet as a manuscript-facing
positive result.

The old Track A headline used the `per_op_buffer` contract, which can update
buffer targets through `_top_up_inventory_buffer` without checking upstream
availability. The conservation-respecting replacement uses
`MFSCGymEnvShifts(action_contract="track_a_v1")` and treats the effective action
surface as 5D: `op3_q`, `op9_q`, `op3_rop`, `op9_rop`, and `shift`.
`op5_q` remains inert under the standard construction unless `op5_rm` is added
to `initial_buffers`, which would reintroduce the top-up mechanism flagged in
D10/D11.

### Landed 3D screen

Artifact:
`outputs/experiments/track_a_v2_conservation_gate_2026-07-03/`

This first screen varies `op3_q`, `op9_q`, and `shift`, while holding
`op3_rop` and `op9_rop` at Garrido baseline. It is a 3D screen on the 5D
contract, not a full 5D bound.

Result:

- `opening_real=True`
- oracle minus best single static: `+0.0019149` Excel ReT
- best action changes across risk regimes: `True`
- candidates: `108`
- seed count: `1`

Interpretation: this is the first meaningful signal that Track A may have
dynamic headroom under a conservation-respecting contract. It is not sufficient
to launch a paper claim or rewrite the manuscript.

### Landed 5D gate

The real 5D gate finished on the VPS:
`outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03/`.

Result:

- `opening_real=True`
- oracle minus best single static: `+0.0041566` Excel ReT
- best action changes across risk regimes: `True`
- candidates: `192`
- grid scope: genuine 5D (`op3_q`, `op9_q`, `op3_rop`, `op9_rop`, `shift`)

Interpretation: the 5D gate clears the pre-PPO promotion rule. Track A still is
not a paper-facing win; it is now justified for confirmatory PPO training.

### PPO confirmatory landed

Artifact:
`outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/`.

Verdict note:
`docs/TRACK_A_V2_CONSERVATION_PPO_VERDICT_2026-07-03.md`.

Result:

- teacher: `oracle_by_regime`
- held-out best static: `op3_1_op9_1_rop3_1_rop9_0.5_S2`
- held-out best static Excel ReT: `0.174422`
- PPO mean Excel ReT: `0.168105`
- PPO delta vs held-out best static: `-0.006316`
- positive seeds: `0/5`
- raw ReT win: `False`

Interpretation: Track A v2 is not a positive RL result. The overnight reopening
test found real conservation-respecting oracle headroom, but PPO did not convert
it. This strengthens Track A as the boundary/null case and preserves Track B as
the Paper 1 positive spine.

## Track B post-CDC screen

Artifact:
`outputs/experiments/track_b_ablation_post_cdc_screen_2026-07-02/`

This screen freezes the CDC/Op3 controls and leaves authority over downstream
dispatch and shift decisions. It completed at small scale (`2` seeds, `30k`
timesteps, `8` eval episodes).

Key screen result:

- PPO order-level ReT: `0.005578`
- best static order-level ReT in runner: `0.005192`
- PPO gap vs best static: `+0.000386`
- `RawWin=True` in run log
- `promote_to_long_run=False` in `comparison_table.csv`

Interpretation: the Track B win does not appear to depend on controlling Op3/CDC
in this small screen. Keep this as sidecar evidence until promoted to a full
5-seed/60k run.

### Full post-CDC promotion landed

Local Mac process PID `37561` finished and wrote:

- Artifact: `outputs/experiments/track_b_ablation_8d_final_2026-07-01/post_cdc_only/`
- Log: `outputs/experiments/track_b_ablation_8d_final_2026-07-01/post_cdc_only_run.log`
- Verdict note: `docs/TRACK_B_POST_CDC_FULL_VERDICT_2026-07-03.md`

Verification:

- `5` seeds x `12` eval episodes x `16` policies = `960` episode rows
- all `60` CRN keys contain all `16` policies
- best static in the runner decision: `s2_d1.50`
- PPO gap vs best static on `order_level_ret_mean`: `+0.0003967`
- paired episode signs: `60/60` positive
- seed signs: `5/5` positive

Interpretation: Track B's positive result survives freezing Op3/CDC controls.
This is manuscript-safe as a mechanism/robustness check, not a universal claim
that CDC decisions never matter.

## Track A old-contract lead-time robustness

The baseline replicate arm of the old `per_op_buffer` robustness check landed:

- Artifact: `outputs/experiments/track_a_repair_leadtime_robustness_2026-07-02/arm_a_baseline_replicate/`
- Best held-out static: `op30_op50.1_op90_S2`, Excel `0.155254`
- Dynamic PPO Excel: `0.154993`
- Delta: `-0.000262`
- Raw ReT win: `False`
- Pareto non-dominated: `False`

Interpretation: the old magical-top-up contract remains a null/negative lane.
It does not decide the conservation-respecting Track A v2 question, which is
still waiting on the separate PPO confirmatory run.

The remaining friction arms also landed. See
`docs/TRACK_A_BUFFER_FRICTION_ROBUSTNESS_VERDICT_2026-07-03.md`.

- Arm B (`lead_time=168`, `holding_cost=0`): PPO Excel `0.155042` vs best static
  `0.156288`, delta `-0.001246`; Pareto non-dominated but no primary ReT win.
- Arm C (`lead_time=168`, `holding_cost=0.05`): PPO Excel `0.154690` vs best
  static `0.156288`, delta `-0.001599`; dominated.

Interpretation: replenishment friction does not revive the retired Track A
buffer contract.

## Capacity policy

At the time of this note the overnight gating jobs have landed:

- Track B post-CDC full promotion: positive
- Track A old-contract baseline replicate: negative
- Track A old-contract friction arms B/C: no primary ReT win
- Track A v2 conservation PPO: negative vs held-out best static

No automatic relaunch is warranted. Kaggle CLI is available, but VPS/local
remain the reliable compute path unless a fresh Kaggle smoke proves runtime
package installation and output retrieval.
