# Track B RecurrentPPO History Smoke Preregistration — 2026-07-03

## Purpose

This sidecar tests the narrow question raised by David's DKANA proposal: does a
minimal recurrent memory mechanism help Track B beyond the current feed-forward
PPO baseline?

It does not implement DKANA. DKANA combines a relational/matricial state
encoding, local causal attention within each state matrix, and global attention
over a sequence of states. This smoke only tests the cheaper first-order
hypothesis: whether sequential memory by itself is promising enough to justify a
larger history architecture.

## Protocol

- Environment: Track B, `track_b_v1`, corrected `kit_equivalent_order_up_to`
  material-flow mode inherited through `make_track_b_env`.
- Algorithm: `sb3-contrib` `RecurrentPPO` with `MlpLstmPolicy`.
- Observation: `v7`, no explicit frame stacking.
- Reward: `control_v1`.
- Risk: `adaptive_benchmark_v2`.
- Horizon: `max_steps=104`.
- Smoke scale: seed 1, 10k training timesteps, 4 eval episodes.

## Decision Rule

This is not a promotion gate. It can only produce one of three sidecar labels:

- `promising_history_signal`: RecurrentPPO beats the best static comparator in
  the smoke and lands near the canonical PPO+MLP scale.
- `learns_but_no_architecture_win`: RecurrentPPO beats static but remains below
  canonical PPO+MLP.
- `no_history_signal`: RecurrentPPO does not beat the best static comparator.

Any positive smoke must be followed by a same-protocol 5-seed run before it is
cited as evidence. DKANA remains a separate engineering task because it tests
structured relational attention, not merely recurrent memory.
