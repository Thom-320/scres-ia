# K3 PPO Freeze

K3 confirmed observable adaptive ReT headroom on `6800001-6800120`. This
authorizes one learner study without altering the confirmed MPC claim.

- Training tapes: `6710001-6710120` only.
- Learner test: `6900001-6900120`, opened once after this freeze.
- PPO: MLP `[64,64]`, tanh, 120,000 steps, gamma 0.99, learning rate 3e-4,
  n_steps 1024, batch 256, six seeds.
- Observation: on-hand, pipeline, backlog, remaining budget, weeks remaining,
  noisy next-week forecast.
- Action: seven order levels; feasibility layer enforces exact `10·D0` total.
- Dense reward: served rations / D0 minus 0.25 backlog / D0; terminal bonus is
  canonical order ReT. Evaluation remains ReT, never training reward.
- Frozen comparators: `(s,S)=(2,3)` and MPC `(1.25,0,1.5)`.

PPO has incremental neural value only if at least four of six seeds beat the
frozen MPC with positive paired CI95 lower bounds, while quantity-ReT, lost
orders and exact resources pass. Otherwise MPC is the Paper 2 policy and
retained neural learning remains unauthorized.
