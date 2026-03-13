# DKANA Integration Guide

## Quick Start

```python
from supply_chain.external_env_interface import make_shift_control_env, run_episodes
import numpy as np

# Option 1: Standard Gymnasium loop
env = make_shift_control_env(reward_mode="control_v1", risk_level="increased",
                              w_bo=4.0, w_cost=0.02, w_disr=0.0)
obs, info = env.reset(seed=42)
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

# Option 2: Run any custom policy with one call (recommended for DKANA)
def my_policy(obs, info):
    return np.zeros(5, dtype=np.float32)  # replace with your model

results = run_episodes(
    my_policy,
    n_episodes=10,
    seed=42,
    env_kwargs={"reward_mode": "control_v1", "risk_level": "increased",
                "w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0},
    policy_name="dkana_v1",
)
# results[0]["reward_total"], results[0]["fill_rate"], etc.
```

This is a standard Gymnasium environment. `run_episodes()` accepts any callable
`(obs, info) -> action` and returns per-episode metrics matching the benchmark format.

---

## Observation Space: 15 dimensions

| Index | Field                      | Range    | Description                                   |
|-------|----------------------------|----------|-----------------------------------------------|
| 0     | raw_material_wdc_norm      | [0, inf) | WDC raw material level / 1e6                  |
| 1     | raw_material_al_norm       | [0, inf) | Assembly line raw material / 1e6              |
| 2     | rations_al_norm            | [0, inf) | Rations at assembly / 1e5                     |
| 3     | rations_sb_norm            | [0, inf) | Rations at supply battalion / 1e5             |
| 4     | rations_cssu_norm          | [0, inf) | Rations at CSSUs / 1e5                        |
| 5     | rations_theatre_norm       | [0, inf) | Rations at theatre / 1e5                      |
| 6     | fill_rate                  | [0, 1]   | Fraction of orders fulfilled (not backordered)|
| 7     | backorder_rate             | [0, 1]   | Backorders / total orders                     |
| 8     | assembly_line_down         | {0, 1}   | 1 if any assembly op (5/6/7) is disrupted     |
| 9     | any_location_down          | {0, 1}   | 1 if any LOC (4/8/10/12) is disrupted         |
| 10    | op9_down                   | {0, 1}   | 1 if supply battalion is disrupted            |
| 11    | op11_down                  | {0, 1}   | 1 if CSSU is disrupted                        |
| 12    | time_fraction              | [0, 1]   | Current sim time / horizon                    |
| 13    | pending_batch_fraction     | [0, 1]   | Rations in assembly / batch size              |
| 14    | contingent_demand_fraction | [0, inf) | Contingent demand qty / normal daily demand   |

---

## Action Space: 5 dimensions, all in [-1, 1]

| Index | Field                        | Mapping                                    |
|-------|------------------------------|--------------------------------------------|
| 0     | op3_q_multiplier_signal      | Multiplier = 1.25 + 0.75 * action[0]       |
| 1     | op9_q_multiplier_signal      | Same mapping as [0], applied to Op9 qty     |
| 2     | op3_rop_multiplier_signal    | Same mapping, applied to Op3 reorder point  |
| 3     | op9_rop_multiplier_signal    | Same mapping, applied to Op9 reorder point  |
| 4     | assembly_shift_signal        | < -0.33 → S=1, [-0.33, 0.33) → S=2, >= 0.33 → S=3 |

Actions [0-3] control inventory policy. Each maps [-1, 1] to a multiplier
in [0.5, 2.0] applied to the baseline parameter.

Action [4] controls the number of assembly shifts (production capacity).
S=1 = 8h/day (~2,564 rations/day), S=2 = 16h/day (~5,128), S=3 = 24h/day (~7,692).

---

## Reward

**Primary reward**: `control_v1` (used for benchmarking and publication):

```
reward = -(w_bo * service_loss + w_cost * shift_cost + w_disr * disruption_fraction)
```

Default weights: `w_bo=4.0, w_cost=0.02, w_disr=0.0`. Components available in `info` after each step.

**Legacy reward**: `ReT_thesis` (retained as a reporting-only metric, not used for training).

Reward components are available in `info` after each step (`service_loss_step`, `shift_cost_step`, `disruption_fraction_step`, `ret_thesis_corrected_step`).

---

## Episode Structure

- **Step size**: 168 hours (1 week)
- **Episode length**: ~955 steps (20 years minus warm-up)
- **Warm-up**: ~839 hours (automatic, handled internally)
- **Risks**: Enabled by default at "current" level

---

## For DKANA Integration

David's architecture expects `(batch, sequence, features)`. Three approaches:

### Option A: Use `run_episodes()` with your trained DKANA model

The simplest path for evaluation. Wrap your DKANA model in a callable:

```python
from supply_chain.external_env_interface import run_episodes

class DKANAWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, obs, info):
        # Your inference logic here
        dist = self.model.predict(obs)
        return dist.mean.detach().numpy()

results = run_episodes(
    DKANAWrapper(trained_model),
    n_episodes=10,
    seed=42,
    env_kwargs={
        "reward_mode": "control_v1",
        "risk_level": "increased",
        "w_bo": 4.0, "w_cost": 0.02, "w_disr": 0.0,
        "step_size_hours": 168, "max_steps": 260,
        "stochastic_pt": True,
    },
    policy_name="dkana_v1",
    collect_trajectories=True,  # optional: captures per-step data
)
```

### Option B: Export to numpy for offline training

```bash
python scripts/export_trajectories_for_david.py \
    --episodes 100 --risk-level increased \
    --reward-mode control_v1 --observation-version v2 \
    --output-dir outputs/data_export
```

### Option C: Build DKANA-ready causal windows inside this repo

The repo now includes a starter implementation of the DKANA input pipeline:

```bash
python scripts/export_trajectories_for_david.py --episodes 100 --output-dir outputs/data_export
python scripts/build_dkana_dataset.py --input-dir outputs/data_export --window-size 12
```

This writes:

- `dkana_row_matrices.npy` with shape `(N, window, rows, 3)`
- `dkana_config_context.npy` with shape `(N, window, config_dim)`
- `dkana_action_targets.npy` with shape `(N, 5)`
- `dkana_time_mask.npy` with shape `(N, window)`

The PyTorch starter policy lives in `supply_chain/dkana.py` as `DKANAPolicy`.
It implements:

- row-wise MLP projection for each symbolic triplet
- a local causal self-attention block inside each state matrix
- a global causal self-attention block across state history
- a distributional decoder that returns Gaussian action parameters

This is a starter integration layer, not yet a publishable benchmark by itself.
To compare against PPO fairly, train and evaluate DKANA with the same seeds,
reward mode, risk level, and metrics used by the benchmark scripts.

---

## Machine-Readable Spec

```python
from supply_chain.external_env_interface import get_shift_control_env_spec, spec_to_dict
import json

spec = get_shift_control_env_spec()
print(json.dumps(spec_to_dict(spec), indent=2))
```

---

## State Constraint Context (22-dim, for external models)

The `state_constraint_context` provides per-step operational state signals beyond
the 15/18-dim observation. These flow automatically through the export pipeline
into `state_constraint_context.npy` and DKANA `row_matrices`.

| Index | Field | Description |
|-------|-------|-------------|
| 0-6 | inventory levels | Raw material and rations at each location |
| 7 | total_inventory | Sum of all inventory |
| 8-10 | dispatch capacities | Op3/Op9 dispatch limits |
| 11-14 | availability flags | Assembly/location/op availability |
| 15-19 | operational rates | fill_rate, backorder_rate, time/batch/demand fractions |
| **20** | **cumulative_backorder_qty** | Total backorder quantity accumulated in episode |
| **21** | **cumulative_disruption_hours** | Total op-hours of disruption accumulated in episode |

Fields 20-21 are cumulative (non-stationary, grow monotonically). They provide
episode-level memory that the 15-dim observation does not capture.

---

## Comparison Benchmark

PPO baseline results are in:
```
outputs/benchmarks/control_reward/
```

Use `run_episodes()` with the same `env_kwargs` to get directly comparable metrics.
David's model should report: `reward_total`, `fill_rate`, `pct_steps_S1/S2/S3` per seed.
