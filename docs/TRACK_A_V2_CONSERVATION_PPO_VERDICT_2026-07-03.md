# Track A v2 Conservation PPO Verdict (2026-07-03)

## Verdict

Track A is reopened as a useful diagnostic lane, but not as a positive
manuscript-facing RL result.

The conservation-respecting 5D gate showed real static-oracle headroom, but the
5-seed PPO confirmatory run did not convert that headroom. PPO lost to the
held-out best static in every seed.

## Artifact

`outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/`

Run log:

`outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/run.log`

## Protocol Check

- Environment: `MFSCGymEnvShifts(action_contract="track_a_v1")`
- Effective action surface: conservation-respecting Track A v2
- Effective dimensions: `op3_q`, `op9_q`, `op3_rop`, `op9_rop`, `shift`
- `op5_q` is not interpreted as a live dimension because it is inert unless
  `op5_rm` is introduced through `initial_buffers`, which would reintroduce the
  top-up mechanism flagged in the decision ledger.
- Gate source:
  `outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03/`
- Teacher: `oracle_by_regime`
- Seeds: `1..5`
- Training: `40k` PPO timesteps per seed, with BC teacher warm-start
- Eval horizon: `max_steps=52`
- Selection/eval seeds: `8000` / `9000` blocks
- Intended primary metric: Excel ReT

## Gate Context

The 5D static/oracle gate was positive:

- `opening_real=True`
- oracle minus best single static: `+0.0041566` Excel ReT
- best action changes across regimes: `True`

This justified PPO training. It did not guarantee that a learner could convert
the oracle headroom.

## Confirmatory PPO Result

Held-out best static:

- candidate: `op3_1_op9_1_rop3_1_rop9_0.5_S2`
- Excel ReT: `0.174422`

PPO:

- mean Excel ReT: `0.168105`
- delta vs held-out best static: `-0.006316`
- positive seeds: `0/5`
- raw ReT win: `False`

Seed-level deltas:

| Seed | PPO Excel ReT | PPO - held-out static |
|---:|---:|---:|
| 1 | `0.166634` | `-0.007788` |
| 2 | `0.165035` | `-0.009386` |
| 3 | `0.172252` | `-0.002170` |
| 4 | `0.171516` | `-0.002906` |
| 5 | `0.165090` | `-0.009331` |

## Interpretation

This is not a Track A revival in the strong sense. It is a better negative
result:

1. The old Track A null was potentially confounded by the `per_op_buffer`
   top-up mechanism.
2. The new Track A v2 contract respects inventory conservation.
3. Under that cleaner contract, the static/oracle gate shows that dynamic
   headroom exists.
4. PPO still fails to convert that headroom.

That strengthens the Paper 1 framing: action-space coverage and oracle headroom
are not sufficient by themselves. Track B remains the positive bottleneck-aligned
case; Track A remains the boundary case where a standard learner does not
convert available headroom.

## Manuscript-Safe Wording

Use:

> A conservation-respecting Track A v2 audit found positive oracle headroom, but
> PPO did not convert it: the learned policy lost to the held-out best static in
> all five seeds. This reinforces the paper's boundary claim that controllable
> headroom is necessary but not sufficient for learned SCRES improvement.

Avoid:

> Track A is positive.

Avoid:

> PPO cannot learn Track A.

The present result is specific to this learner, horizon, reward, seed block, and
training budget. It closes the overnight reopening test; it does not prove a
universal impossibility.

## Next-Step Recommendation

Do not reframe Paper 1 around Track A. Keep Track A as a stronger boundary/null
case and mention the conservation-respecting audit only if space allows or if a
reviewer asks about the buffer-replenishment mechanism.

If Track A becomes important for a future paper, the next clean step is not more
ad hoc PPO. It is a pre-registered learner sweep or imitation/optimization study
specifically targeted at converting the 5D oracle headroom.
