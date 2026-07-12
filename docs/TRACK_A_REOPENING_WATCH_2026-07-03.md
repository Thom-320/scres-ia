# Track A Reopening Watch Note — 2026-07-02/03

## Current status

Track A was reopened as a scientific question and is now resolved as a cleaner
boundary/null result, not as a paper-facing positive RL result.

A conservation-respecting Track A contract exists via `MFSCGymEnvShifts` with
`action_contract="track_a_v1"`. The useful action surface is 5 effective dims:
`op3_q`, `op9_q`, `op3_rop`, `op9_rop`, and `shift`. The nominal `op5_q` dim is
inert under standard construction unless `initial_buffers["op5_rm"]` is set; doing
that would reintroduce the exogenous top-up mechanism flagged in D10/D11.

## Landed preliminary evidence

- 3D conservation screen (`op3_q`, `op9_q`, `shift`; ROP fixed) finished on VPS:
  `outputs/experiments/track_a_v2_conservation_gate_2026-07-03/`.
- Gate result: `opening_real=True`; oracle minus best single static = `+0.0019149`
  Excel ReT; best action changes across regimes.
- Boundary: this is 1 seed and only a 3D screen. It is evidence to continue, not a
  manuscript-facing Track A win.

Track B post-CDC screen also finished:
`outputs/experiments/track_b_ablation_post_cdc_screen_2026-07-02/`.
It reports `post_cdc_only` positive in the screen (`PPO order-level ReT =
0.005578` vs best static `0.005192`, `Delta=+0.000386`; run log
`RawWin=True`), suggesting Track B does not depend on controlling Op3/CDC.
The runner still marks `promote_to_long_run=False`, so this remains sidecar
evidence until promoted to full scale.

See also: `docs/TRACK_A_V2_AND_POST_CDC_OVERNIGHT_STATUS_2026-07-03.md`.

## Landed confirmatory evidence

- Real 5D Track A v2 conservation gate finished positive on VPS:
  `outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03/`
  (`opening_real=True`, oracle minus best single static `+0.0041566` Excel ReT).
- PPO confirmatory training finished:
  `outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/`.
- Verdict note:
  `docs/TRACK_A_V2_CONSERVATION_PPO_VERDICT_2026-07-03.md`.

PPO did not convert the 5D oracle headroom:

- held-out best static Excel ReT: `0.174422`
- PPO mean Excel ReT: `0.168105`
- PPO delta vs held-out best static: `-0.006316`
- positive seeds: `0/5`
- raw ReT win: `False`

Lead-time/holding-cost baseline replicate for the old `per_op_buffer` contract
also landed negative and remains defensive, not a replacement for the
conservation-respecting contract.

## Rule

Do not reframe Paper 1 around Track A. Track B remains the paper spine. Track A
may be described as a stronger boundary case: even with a conservation-respecting
action surface and measurable oracle headroom, this standard PPO run did not beat
the held-out best static comparator.
