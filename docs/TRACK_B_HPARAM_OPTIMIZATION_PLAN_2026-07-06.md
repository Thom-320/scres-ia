# Track B hyperparameter optimization plan - 2026-07-06

## Current evidence

Primary metric: Garrido Excel ReT, `order_ret_excel_mean`.

Tail metric: `order_ret_excel_cvar05_mean`.

Mechanism metrics:

- `order_ret_excel_risk_conditional_mean_mean`;
- Excel branch mix;
- resource cost.

## PPO+MLP

Existing batch-size evidence already covers the main no-forecast settings:

`docs/TRACK_B_ENV_HPARAM_TUNING_VERDICT_2026-07-05.md`

Current reading:

- Case A / Garrido all risks: `batch_size=256` is the best 3-seed/30k cell by
  ReT Excel, but the margin over `64` is tiny.
- Case B / downstream-only: `64`, `128`, and `256` are close; `64` has the best
  learned-policy ReT, `128` the largest delta vs static, and `256` the lowest
  cost.
- Adaptive-v2 no-forecast: `64` is the best current cell.

Gamma/reward-normalization evidence:

`docs/TRACK_B_GAMMA_REWARDNORM_MINIGRID_2026-07-05.md`

Current reading:

- Keep `gamma=0.99`.
- Keep `gae_lambda=0.95`.
- Do not use reward normalization for the current confirmatory lane.

## Real-KAN

The confirmed Real-KAN results use `batch_size=256`.

Current reading:

- `batch_size=256` is the Real-KAN protocol to use for confirmatory runs.
- Real-KAN `batch_size=64` should be treated as non-canonical unless a dedicated
  Real-KAN batch-size grid proves otherwise.

We do not yet have a dedicated Real-KAN batch-size sweep by scenario. Because
Real-KAN is slower and already has a confirmed positive protocol at `256`, this
is lower priority than completing the final A/B/C confirmation.

## Recommended optimization sequence

### Stage 1: finish current final runs

Do not launch more local grids until these land:

1. A/B h104 confirmation: PPO+MLP and Real-KAN.
2. Case C per-risk headroom grid.
3. Horizon screen.

Reason: they are already consuming CPU and directly answer the current paper
question.

### Stage 2: PPO targeted hparam grid, only if needed

Run only on the scenario/horizon that will be used in the paper.

Small screen:

- seeds: `1,2,3`;
- train timesteps: `30000`;
- eval episodes: `8`;
- horizon: selected horizon, likely `104` unless horizon screen says otherwise;
- observation: `v7_no_forecast`;
- reward: `control_v1`.

Grid:

| Parameter | Values |
|---|---|
| batch_size | scenario-dependent candidates from prior screen |
| learning_rate | `1e-4`, `3e-4`, `6e-4` |
| ent_coef | `0.0`, `0.003`, `0.01` |
| clip_range | `0.1`, `0.2` |
| n_steps | `512`, `1024` |

Keep fixed unless a future result justifies changing:

- `gamma=0.99`;
- `gae_lambda=0.95`;
- `norm_reward=False`.

Use a staged screen rather than a full Cartesian product:

1. choose batch size by existing grid;
2. cross `learning_rate x ent_coef`;
3. test `clip_range` and `n_steps` only on the top 2 cells.

### Stage 3: Real-KAN batch sanity grid, optional

Only if Real-KAN becomes a central result rather than an architecture sidecar.

Small screen:

- batch sizes: `128`, `256`, `512`;
- same scenario/horizon as PPO;
- seeds: `1,2,3`;
- train timesteps: `30000`;
- eval episodes: `8`.

Promotion rule:

- keep `256` unless another batch size improves ReT Excel and CVaR without
  increasing cost or instability.

### Stage 4: confirmatory run

Promote at most one configuration per architecture to:

- seeds: at least `1..5`;
- train timesteps: `60000`;
- eval episodes: `12`;
- same horizon/scenario;
- same static frontier comparator.

## Current provisional settings

Until the current final A/B/C evidence says otherwise:

| Architecture | Setting |
|---|---|
| PPO+MLP | `batch_size=64` for continuity/adaptive no-forecast; consider `256` for Case A all-risk only if final A/B supports it |
| Real-KAN | `batch_size=256` |
| learning_rate | `3e-4` |
| n_steps | `1024` |
| n_epochs | `10` |
| gamma | `0.99` |
| gae_lambda | `0.95` |
| ent_coef | `0.0` |
| norm_reward | `False` |

## Decision boundary

Do not claim "best hyperparameters" globally. Claim:

"Under the tested Track B no-forecast protocol, the selected hyperparameters
were chosen by a staged screen optimizing Garrido Excel ReT, CVaR05, cost, and
scenario-specific static-frontier deltas."
