# Track B RecurrentPPO/LSTM History Sidecar Verdict — 2026-07-03

## Verdict

RecurrentPPO/LSTM learns a real Track B policy and beats the best static
comparator, but it does **not** beat canonical PPO+MLP on the same Garrido/Excel
ReT scale.

This closes the narrow hypothesis that recurrent memory alone explains the
Track B gain. It does not close DKANA, which also adds structured/matricial
state encoding plus local/global attention.

## Artifact

- Run directory:
  `outputs/experiments/track_b_recurrent_ppo_2026-07-03/confirm_3seed_30k_h104/`
- Runner: `scripts/run_track_b_smoke.py --algo recurrent_ppo`
- Seeds: `1 2 3`
- Train timesteps: `30000`
- Eval episodes: `12`
- Horizon: `max_steps=104`
- Reward: `control_v1`
- Observation: `v7`
- Risk: `adaptive_benchmark_v2`
- Action contract: `track_b_v1`
- Raw-material flow mode: requested `kit_equivalent_order_up_to`, canonical
  `bom_total_units_order_up_to` (the code-level alias for the corrected mode).

## Main Numbers

From `policy_summary.csv`:

| Policy | Order-level ReT | Excel ReT | Shift mix S1/S2/S3 |
|---|---:|---:|---:|
| best static (`s2_d1.50`) | 0.005236 | 0.005451 | 0.0 / 100.0 / 0.0 |
| `recurrent_ppo` | 0.005630 | 0.005857 | 0.4 / 89.6 / 10.0 |

RecurrentPPO's Excel ReT gain over the best static comparator is approximately
`+0.000406`.

Against the canonical PPO+MLP episode ledger in
`docs/track_b_q1_stats_2026-07-02_final_10seed/ppo_episode_metrics_10seed.csv`,
restricted to the same seeds 1--3:

| Seed | RecurrentPPO Excel ReT | PPO+MLP Excel ReT | Delta |
|---:|---:|---:|---:|
| 1 | 0.005857 | 0.005921 | -0.000064 |
| 2 | 0.005875 | 0.005906 | -0.000031 |
| 3 | 0.005838 | 0.005920 | -0.000082 |

Mean same-seed delta: `-0.0000588`; signs: `0/3` positive.

## Interpretation

The result is useful but not architecture-promoting:

- It supports the claim that richer architectures can learn the Track B
  bottleneck-aligned control problem.
- It does not support a claim that memory alone improves on the conservative
  PPO+MLP baseline.
- Compared with DMLPA and Real-KAN, the LSTM result suggests that any observed
  alternative-architecture lift should be attributed to the specific attention
  or spline mechanism, not merely to having a recurrent hidden state.

No Paper 1 headline should change from this sidecar.
