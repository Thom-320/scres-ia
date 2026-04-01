# Track B Minimal Spec

Updated: 2026-03-30

## Purpose

Define the smallest repo-level extension that gives RL real leverage over the
active MFSC bottleneck without mutating the frozen Family A benchmark.

Track B is a **new research lane**, not a repair of Family A.

Family A remains frozen around:

- `reward_mode=ReT_seq_v1`
- `observation_version=v1`
- `year_basis="thesis"`
- `risk_level="increased"`
- `step_size_hours=168`

Track B exists to answer a different question:

> What is the minimal DES+RL contract in this repo under which adaptive control
> can plausibly outperform fixed static policies because the agent can act on
> the active transport/distribution constraint?

---

## Diagnosis

The current Track A/Family A action contract controls upstream production and
SB dispatch, while downstream transport remains effectively fixed.

Current mutable control:

- `op3_q`
- `op9_q_min`
- `op9_q_max`
- `op3_rop`
- `op9_rop`
- `assembly_shifts`

Current fixed downstream transport in the DES:

- `Op10` reads static `OPERATIONS[10]["q"]` and `op10_rop`
- `Op12` reads static `OPERATIONS[12]["q"]` and `op12_rop`

Relevant code:

- [external_env_interface.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/external_env_interface.py)
- [supply_chain.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py)

This means RL mostly controls upstream inventory and capacity while the
distribution bottleneck remains outside the action space.

The repo already records that positive action headroom above `S2` is small and
downside from poor inventory settings is large:

- [PAPER_FINDINGS_REGISTRY.md](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/PAPER_FINDINGS_REGISTRY.md)

Track B therefore should not start with another reward redesign. It should
start by moving control toward the active bottleneck.

---

## Non-Goals

Track B minimal does **not**:

- replace Family A as source of truth
- retroactively change any Family A bundle
- clear backlog by default after warmup
- silently mutate `observation_version=v6`
- reuse `paper_*` naming

Track B outputs must live in a separate namespace and separate documentation
lane.

---

## Recommendation on Backlog Clearing

Do **not** use `clear pending_backorders after warmup` as the main fix.

Why:

- it changes the underlying Garrido-style operating problem too aggressively
- it removes a real state variable instead of giving the agent more leverage
- it can make RL look better by construction rather than by better control

Allowed use:

- only as an explicit ablation such as `reset_backlog_mode="cleared"`
- never as the default Track B contract

Preferred alternative:

- keep backlog dynamics
- expose better short-horizon observables
- randomize post-priming operational state within a realistic envelope

---

## Track B Minimal Contract

### Environment identity

Create a separate contract:

- `env_variant="track_b_adaptive_control"`
- `action_contract="track_b_v1"`
- `observation_version="v7"`
- `risk_level="adaptive_benchmark_v2"`
- `year_basis="thesis"`
- `step_size_hours=168`
- `stochastic_pt=True`

This avoids contaminating Family A or mutating the existing Track A contract.

### Training reward

Default Track B training reward:

- `reward_mode="ReT_seq_v1"`
- `ret_seq_kappa=0.20`

Reason:

- it preserves the strongest reward=resilience alignment currently available in
  the repo
- it avoids opening a new reward-design thread before testing the MDP fix

Comparator lane:

- `reward_mode="control_v1"` remains a comparator only

Shared comparison metrics:

- `fill_rate`
- `backorder_rate`
- `order_level_ret_mean`
- shift mix
- new downstream control usage metrics

---

## Action Contract: `track_b_v1`

### Principle

Keep the existing 5D contract and add only the smallest downstream controls
needed to touch the active bottleneck.

### New action dimension

Track B minimal uses **7** action dimensions:

1. `op3_q_multiplier_signal`
2. `op9_q_multiplier_signal`
3. `op3_rop_multiplier_signal`
4. `op9_rop_multiplier_signal`
5. `assembly_shift_signal`
6. `op10_q_multiplier_signal`
7. `op12_q_multiplier_signal`

Bounds:

- all dimensions remain in `[-1.0, 1.0]`

Mapping:

- reuse the existing inventory mapping
- `multiplier = 1.25 + 0.75 * signal`
- downstream dimensions scale both `q_min` and `q_max`

Minimal mutable downstream parameters added to the DES:

- `op10_q_min`
- `op10_q_max`
- `op12_q_min`
- `op12_q_max`

Deliberately deferred from `track_b_v1`:

- `op10_rop`
- `op12_rop`
- transport PT control
- explicit routing/prioritization policies

Reason:

- the first Track B step should test whether downstream quantity control alone
  is enough to restore meaningful RL headroom before expanding the action
  space again

Escalation rule:

- if `track_b_v1` still shows trivial headroom in static DOE, then `track_b_v2`
  should add `op10_rop` and `op12_rop`

---

## Observation Contract: `v7`

### Principle

Do not mutate `v6`. Create `v7` as `v6 + minimal bottleneck observability`.

`v6` already contributes:

- regime state
- 48h / 168h disruption forecasts
- maintenance debt
- backlog age
- theatre cover days

### New `v7` fields

Append these fields to `v6`:

1. `op10_down`
2. `op12_down`
3. `op10_queue_pressure_norm`
4. `op12_queue_pressure_norm`
5. `rolling_fill_rate_4w`
6. `rolling_backorder_rate_4w`

Definitions:

- `op10_down`, `op12_down`:
  Current binary disruption flags for the two downstream transport stages.
- `op10_queue_pressure_norm`:
  normalized pressure at SB dispatch, based on `rations_sb_dispatch` relative
  to current downstream dispatch capacity.
- `op12_queue_pressure_norm`:
  normalized pressure at CSSU dispatch, based on `rations_cssu` relative to
  current theatre dispatch capacity.
- `rolling_fill_rate_4w`:
  order-level fill rate over the last 4 simulated weeks.
- `rolling_backorder_rate_4w`:
  backorder rate over the last 4 simulated weeks.

Reason:

- keep cumulative thesis metrics for audit
- give the agent short-horizon state signals that are more sensitive to recent
  control

Deliberately deferred from `v7`:

- full route graph features
- per-node lead-time forecasts
- explicit convoy availability state

---

## Risk Contract: `adaptive_benchmark_v2`

### Principle

Use the existing adaptive benchmark lane as the base, but make downstream
transport risk a first-class source of controllable stress.

Track B minimal should add:

- more frequent transport disruption exposure at `Op10` and `Op12`
- stronger persistence/correlation between upstream strain and downstream
  transport stress
- demand surges that occasionally exceed the static `~2500/day` delivery pace

Minimal additions:

1. downstream transport risks for `Op10`
2. downstream transport risks for `Op12`
3. correlated regime uplift so severe upstream states raise downstream risk
4. surge-demand scaling tied to disrupted/pre-disruption regimes

Deliberately deferred:

- dynamic route reconfiguration
- network topology changes
- supplier recruitment/repair actions

Those belong to a later non-minimal Track B, closer to a Ding-style network
reconfiguration problem.

---

## Reset Contract

### Default rule

Keep thesis warmup and priming. Do not wipe backlog.

### New Track B reset behavior

Add `reset_profile="track_b_operational_v1"`:

- start from a post-warmup, post-priming operational state
- sample from a bounded envelope of realistic operational states
- preserve nonzero backlog, inventory in transit, and theatre stock

Randomized variables:

- theatre inventory
- CSSU inventory
- SB dispatch inventory
- pending backorder quantity
- backlog age
- adaptive regime state

The sampled state must still satisfy:

- theatre inventory present
- minimum operational fill threshold met
- no impossible negative/infeasible inventory states

Reason:

- avoid startup determinism dominating every episode
- preserve realism better than clearing backlog to zero

Optional ablation only:

- `reset_backlog_mode="cleared"`

---

## Benchmark Plan

Before any long Track B run, execute this sequence.

### Stage 0: Static headroom check

Run a small DOE over fixed policies in the new Track B environment:

- `S1/S2/S3`
- downstream low/neutral/high dispatch
- existing Garrido-style baselines carried over where meaningful

Success criterion:

- best fixed policy should beat neutral `S2` by more than the current Track A
  ~1% headroom

### Stage 1: 100k smoke

Run:

- PPO + MLP, `ReT_seq_v1`
- PPO + MLP, `control_v1`
- same seeds
- same Track B contract

Success criterion:

- learned lane should at least separate from `static_s2` in service metrics or
  short-horizon rolling metrics

### Stage 2: 500k production only if Stage 0 and Stage 1 are promising

Do not launch long Track B production runs unless the static DOE and 100k smoke
already show meaningful action headroom.

---

## Files to Change

These are the exact repo files that should change when Track B coding starts.

### Core DES

- [supply_chain/supply_chain.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/supply_chain.py)

Required changes:

- make `Op10` and `Op12` read mutable `self.params` values for dispatch ranges
- add `op10_q_min`, `op10_q_max`, `op12_q_min`, `op12_q_max` to mutable params
- add rolling 4-week service metrics
- add `v7` observation helpers
- add Track B reset-envelope sampling
- add `adaptive_benchmark_v2` downstream risk logic

### Environment wrapper

- [supply_chain/env_experimental_shifts.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/env_experimental_shifts.py)

Required changes:

- add `observation_version="v7"`
- add `action_contract="track_b_v1"` support
- expand action mapping from 5D to 7D under Track B
- export new Track B diagnostics in `info`
- keep Family A path unchanged

### External contract

- [supply_chain/external_env_interface.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/external_env_interface.py)

Required changes:

- add `OBSERVATION_FIELDS_V7`
- add `ACTION_FIELDS_TRACK_B_V1`
- add `ACTION_BOUNDS_TRACK_B_V1`
- add `get_track_b_env_spec()`
- keep `get_shift_control_env_spec()` frozen for Track A

### Config

- [supply_chain/config.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/supply_chain/config.py)

Required changes:

- add Track B defaults
- add downstream risk configuration
- add reset-envelope config
- add rolling-window config

### Training / benchmarking

- [train_agent.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/train_agent.py)
- [scripts/benchmark_control_reward.py](/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/scripts/benchmark_control_reward.py)

Required changes:

- allow selecting Track B contract explicitly
- keep Track A defaults unchanged
- write Track B contract metadata into bundle outputs

### New helper scripts

Recommended new files:

- `scripts/run_track_b_smoke.py`
- `scripts/run_track_b_doe.py`

Reason:

- avoid overloading the paper-facing Track A launcher with experimental Track B
  switches

---

## Tests to Add When Coding Starts

Not part of this spec patch, but required during implementation:

1. mutable downstream parameter test
2. `track_b_v1` action-shape and mapping test
3. `v7` observation-shape and field-order test
4. Track B reset-envelope validity test
5. downstream-risk activation test under `adaptive_benchmark_v2`
6. static DOE smoke test showing nontrivial headroom

---

## Decision

Track B minimal should begin with:

- downstream quantity control at `Op10` and `Op12`
- `v7` observation with downstream state and rolling service signals
- adaptive downstream risk under a separate `adaptive_benchmark_v2`
- randomized operational resets without backlog wiping

Do **not** start with:

- another reward redesign
- action-space reduction
- backlog clearing as the default

Those are either already tested, too weak, or too confounded to answer the
real MDP question.
