# DKANA Integration Guide

## Quick Start

```python
from supply_chain.external_env_interface import make_shift_control_env

env = make_shift_control_env()
obs, info = env.reset(seed=42)

# obs is a numpy array of shape (15,)
# action must be a numpy array of shape (5,), each value in [-1, 1]
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)
```

This is a standard Gymnasium environment. Any RL algorithm that works with
continuous observation and action spaces can plug in directly.

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

Default mode is `ReT_thesis` — approximation of Garrido (2017) Eq. 5.5:

```
Reward = ReT_step - delta * (shifts - 1)
```

Where `ReT_step` is in [0, 1] based on four thesis resilience cases:
- **fill_rate_only**: No disruption active → ReT = fill_rate
- **autotomy**: Disruption active but fill rate >= 0.95 → ReT = 1 - disruption_frac
- **recovery**: Disruption active, fill rate < 0.95 → ReT = 1/(1 + disruption_frac)
- **non_recovery**: High disruption + low fill rate → ReT = 0

The `delta * (shifts - 1)` term penalizes higher shift counts to prevent
trivial S=3 solutions. Default delta = 0.06 (DOE-calibrated).

Reward components are available in `info["ret_components"]` after each step.

---

## Episode Structure

- **Step size**: 168 hours (1 week)
- **Episode length**: ~955 steps (20 years minus warm-up)
- **Warm-up**: ~839 hours (automatic, handled internally)
- **Risks**: Enabled by default at "current" level

---

## For DKANA Integration

David's architecture expects `(batch, sequence, features)`. Two approaches:

### Option A: Single-step (standard RL)
Use the env directly. Each `step()` returns one obs of shape `(15,)`.
Wrap with frame stacking if you want a history window:

```python
from gymnasium.wrappers import FrameStack
env = FrameStack(make_shift_control_env(), num_stack=10)
# obs shape becomes (10, 15)
```

### Option B: Collect trajectories, train offline
Run episodes, collect `(obs, action, reward)` tuples, then train DKANA
on batches of trajectories:

```python
env = make_shift_control_env()
trajectories = []
for seed in range(100):
    obs, _ = env.reset(seed=seed)
    episode = []
    done, truncated = False, False
    while not (done or truncated):
        action = env.action_space.sample()  # or your policy
        next_obs, reward, done, truncated, info = env.step(action)
        episode.append((obs, action, reward, info))
        obs = next_obs
    trajectories.append(episode)
```

### Option C: Export to numpy for offline use
```python
import numpy as np
env = make_shift_control_env()
obs_list, reward_list = [], []
obs, _ = env.reset(seed=42)
done, truncated = False, False
while not (done or truncated):
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
    obs_list.append(obs)
    reward_list.append(reward)

np.save("observations.npy", np.array(obs_list))   # shape (T, 15)
np.save("rewards.npy", np.array(reward_list))      # shape (T,)
```

---

## Machine-Readable Spec

```python
from supply_chain.external_env_interface import get_shift_control_env_spec, spec_to_dict
import json

spec = get_shift_control_env_spec()
print(json.dumps(spec_to_dict(spec), indent=2))
```

---

## Comparison Benchmark

PPO baseline results will be in:
```
outputs/benchmarks/ppo_shift_control_ret_thesis/benchmark_summary.json
```

David's model should report the same metrics (mean reward, std, CI95, per-seed)
for a fair comparison.
