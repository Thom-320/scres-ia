# David DKANA Thesis-Faithful Tutorial

This is the simple Colab-friendly path for David. It stays on the Garrido thesis-faithful lane:

- Action contract: `thesis_faithful_dkana_v1`
- Action vector: 18D thesis decision vector
- Default observation: 19D = realized 18D thesis decision vector + reward
- Optional richer observation: environment state + Table 6.25-style history + reward
- No Track B Op10/Op12 actions are exposed

## 1. Colab Setup

```python
import shutil, sys

shutil.rmtree("/content/scresia", ignore_errors=True)
!git clone https://github.com/Thom-320/scres-ia.git /content/scresia
sys.path.insert(0, "/content/scresia")
```

```bash
%cd /content/scresia
!pip install -r requirements.txt
```

## 2. Smoke Check The 18D/19D Contract

```bash
!python scripts/david_dkana_thesis_faithful_smoke.py \
  --observation-mode decision_reward \
  --reward-mode ReT_seq_v1 \
  --risk-level increased \
  --max-steps 2
```

Expected contract:

- `action_shape: (18,)`
- `observation_shape: (19,)`
- `action_contract: thesis_faithful_dkana_v1`

Minimal Python usage:

```python
import numpy as np
from supply_chain.external_env_interface import (
    get_dkana_thesis_faithful_env_spec,
    make_dkana_thesis_faithful_env,
)

spec = get_dkana_thesis_faithful_env_spec(
    reward_mode="ReT_seq_v1",
    observation_mode="decision_reward",
)
env = make_dkana_thesis_faithful_env(
    reward_mode="ReT_seq_v1",
    risk_level="increased",
    observation_mode="decision_reward",
    max_steps=2,
    stochastic_pt=True,
)

obs, info = env.reset(seed=42)
action = np.zeros(18, dtype=np.float32)
action[2] = 1.0    # Op3 inventory period I504,1
action[7] = 1.0    # Op5 inventory period I504,1
action[12] = 1.0   # Op9 inventory period I504,1
action[16] = 1.0   # S2

next_obs, reward, terminated, truncated, info = env.step(action)
print(obs.shape, next_obs.shape, reward)
env.close()
```

## 3. Export Offline Trajectories For DKANA

Classic 19D observation handoff:

```bash
python scripts/export_trajectories_for_david.py \
  --episodes 20 \
  --seed-start 0 \
  --reward-mode ReT_seq_v1 \
  --risk-level increased \
  --action-contract thesis_faithful_dkana_v1 \
  --thesis-observation-mode decision_reward \
  --stochastic-pt \
  --max-steps 260 \
  --output-dir outputs/data_export_thesis_dkana_19d
```

Richer research observation handoff, recommended for learning experiments:

```bash
python scripts/export_trajectories_for_david.py \
  --episodes 20 \
  --seed-start 0 \
  --reward-mode ReT_seq_v1 \
  --risk-level increased \
  --observation-version v5 \
  --action-contract thesis_faithful_dkana_v1 \
  --thesis-observation-mode env_sdm_history_reward \
  --stochastic-pt \
  --max-steps 260 \
  --output-dir outputs/data_export_thesis_dkana_sdm
```

Important files in the export directory:

- `observations.npy`
- `actions.npy` with shape `(N, 18)`
- `rewards.npy`
- `episode_ids.npy`
- `env_spec.json`
- `metadata.json`
- `action_fields.json`

## 4. Build DKANA Windows

```bash
python scripts/build_dkana_dataset.py \
  --input-dir outputs/data_export_thesis_dkana_sdm \
  --window-size 12 \
  --relation-mode temporal_delta \
  --include-prev-reward
```

This writes:

- `dkana_row_matrices.npy`
- `dkana_config_context.npy`
- `dkana_action_targets.npy`
- `dkana_time_mask.npy`
- `dkana_reward_targets.npy`
- `metadata.json`

## 5. What The 18D Action Means

The first 15 entries are inventory-period scores from Garrido Table 6.16:

- Op3: indices `0..4`
- Op5: indices `5..9`
- Op9: indices `10..14`

The last 3 entries are capacity scores from Garrido Table 6.20:

- `S1`: index `15`
- `S2`: index `16`
- `S3`: index `17`

By default, `thesis_strict` mode selects one common dominant inventory period across Op3, Op5, and Op9. This matches the Garrido factorial design more closely than allowing each node to choose a separate period.

## 6. What David Should Not Use For This Lane

Do not use Track B commands if the goal is thesis-faithful DKANA. Track B has 7D actions and adds Op10/Op12 downstream controls. That lane works well experimentally, but it is not the same Garrido decision surface.

Use Track B only if the experiment is explicitly about extended action control. For the thesis-faithful paper lane, keep `--action-contract thesis_faithful_dkana_v1`.
