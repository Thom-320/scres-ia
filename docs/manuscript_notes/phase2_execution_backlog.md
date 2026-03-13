# Phase 2 Execution Backlog

This note converts the strategy decision into an implementation backlog. It is
decision-complete on purpose: the implementer should not need to guess the next
steps, benchmark backbone, or baseline definitions.

## Benchmark backbone

Keep the current benchmark family as the common backbone for all phase-2 work:

- `reward_mode="control_v1"`
- scenarios:
  - `increased + stochastic_pt=True`
  - `severe + stochastic_pt=True`
- `step_size_hours=168.0`
- reporting seeds:
  - `[11, 22, 33, 44, 55, 66, 77, 88, 99, 111]`
- development seeds for smoke and heuristic tuning:
  - `[11, 22, 33]`

The current `v1` baseline stays frozen for comparability. New work should
prefer `observation_version="v2"` unless a specific ablation requires `v1`.

## Workstream 1: RecurrentPPO

Implement a new learned-policy lane based on `sb3-contrib`.

### Dependency and API

- Add `sb3-contrib` as a required dependency for the recurrent benchmark path.
- Use `RecurrentPPO` with `MlpLstmPolicy`.
- Integrate it into the same control benchmark family rather than creating an
  unrelated script.

### Initial configuration

Use these defaults for the first comparable runs:

- `learning_rate=3e-4`
- `n_steps=1024`
- `batch_size=64`
- `n_epochs=10`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `lstm_hidden_size=128`
- `n_lstm_layers=1`
- `device="cpu"`

### Required comparisons

Run these four conditions first:

1. `PPO + MLP`, `v1`, `frame_stack=1`
2. `PPO + MLP`, `v2`, `frame_stack=1`
3. `PPO + MLP`, `v1`, `frame_stack=4`
4. `RecurrentPPO + LSTM`, `v2`

If time permits after those four, add:

5. `RecurrentPPO + LSTM`, `v1`

### Acceptance criterion

Treat `RecurrentPPO` as the main improvement only if it beats the frozen
`PPO + MLP` baseline on reward under `severe + stochastic_pt` while maintaining
comparable fill rate and backorder rate.

## Workstream 2: Tuned heuristic baseline

Implement a single tuned heuristic baseline before adding more algorithms.

### Policy definition

Use a hysteresis shift policy with neutral inventory multipliers:

- inventory-control action dimensions remain fixed at `0.0`
- only the fifth action dimension changes the shift decision

The heuristic state inputs are:

- `fill_rate`
- `backorder_rate`
- `assembly_line_down`
- `any_location_down`

### Exact decision rule

Maintain an internal shift state in `{1, 2, 3}`.

At each step:

- if `assembly_line_down == 1` or `any_location_down == 1`, increase shift by
  one level up to `S3`
- else if `backorder_rate >= tau_up` or `fill_rate <= fr_low`, increase shift
  by one level up to `S3`
- else if `backorder_rate <= tau_down` and `fill_rate >= fr_high`, decrease
  shift by one level down to `S1`
- else, keep the current shift

Map shifts back to the fifth action dimension as:

- `S1 -> -1.0`
- `S2 -> 0.0`
- `S3 -> 1.0`

### Tuning grid

Tune on development seeds only, under `increased + stochastic_pt=True`, with:

- `tau_up in {0.18, 0.22, 0.26}`
- `tau_down in {0.08, 0.12, 0.16}`
- `fr_low in {0.80, 0.84}`
- `fr_high in {0.90, 0.93}`

Select the best heuristic by:

1. highest mean `reward_total`
2. tie-breaker: higher `fill_rate`
3. tie-breaker: lower `backorder_rate`

After selection, freeze that heuristic and evaluate it on the full reporting
seed set under both paper scenarios.

## Workstream 3: Statistics and reporting discipline

Use the seed as the primary statistical unit.

Keep:

- mean
- standard deviation
- `CI95`

Add:

- paired seed-mean difference vs best static baseline
- paired seed-mean difference vs tuned heuristic
- exact sign-flip or bootstrap-based interval note, using the same cautious
  interpretation style already used in the repo

Do not use:

- `statistically significant`
- `solves the problem`

unless the inference section is intentionally upgraded later.

## Workstream 4: PBRS gate

Do not start PBRS before the following are done:

- `RecurrentPPO` comparison is complete
- tuned heuristic baseline is implemented
- 10-seed benchmark outputs exist for the frozen scenarios

If those conditions are met and the reward-service tradeoff still looks weak,
PBRS becomes the next method upgrade.

PBRS scope for later:

- new reward mode, separate from `control_v1`
- benchmarked on the same scenarios
- reported as a methodological extension, not as a replacement for the current
  benchmark story
