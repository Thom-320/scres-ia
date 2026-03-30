# DKANA Contributor Handoff

This document tells you everything you need to implement and evaluate DKANA in this repository. You should NOT need to modify the environment, reward function, or benchmark infrastructure.

## Your Task

Train a DKANA policy that **outperforms PPO+MLP** on the same environment, using the same reward and evaluation metrics. The environment, reward, baselines, and evaluation protocol are frozen.

## Quick Start (5 minutes)

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Generate training data (offline trajectories)
python scripts/export_trajectories_for_david.py \
  --episodes 200 \
  --reward-mode ReT_seq_v1 \
  --observation-version v1 \
  --risk-level increased \
  --output-dir outputs/data_export

# 3. Build DKANA-ready windows
python scripts/build_dkana_dataset.py \
  --input-dir outputs/data_export \
  --window-size 12

# 4. Evaluate your trained model
python -c "
from supply_chain.external_env_interface import run_episodes
import numpy as np

def your_policy(obs, info):
    # Replace with your trained DKANA model
    return np.zeros(5, dtype=np.float32)

results = run_episodes(
    your_policy,
    n_episodes=50,
    seed=42,
    env_kwargs={
        'reward_mode': 'ReT_seq_v1',
        'risk_level': 'increased',
        'stochastic_pt': True,
    },
    policy_name='dkana_v1',
)
for r in results[:3]:
    print(f'reward={r[\"reward_total\"]:.2f}  fill_rate={r[\"fill_rate\"]:.3f}  shifts=S1:{r[\"pct_steps_S1\"]:.0f}/S2:{r[\"pct_steps_S2\"]:.0f}/S3:{r[\"pct_steps_S3\"]:.0f}')
"
```

## Environment Contract

### Observation Space

Default: **v1 (15 dimensions)**. Also available: v2 (18), v3 (20), v4 (24).

| Index | Field | Range | Description |
|-------|-------|-------|-------------|
| 0-5 | Inventory levels | [0, ~20] | Raw materials and rations at 6 locations (normalized) |
| 6 | fill_rate | [0, 1] | Fraction of demand satisfied |
| 7 | backorder_rate | [0, 1] | Fraction of demand backordered |
| 8-11 | Disruption flags | {0, 1} | Assembly line, LOC, Op9, Op11 down |
| 12 | time_fraction | [0, 1] | Simulation progress |
| 13 | pending_batch_norm | [0, ~5] | Pending batch / batch size |
| 14 | contingent_demand_norm | [0, 5] | Contingent demand / 2600 |

### Action Space (5 dimensions, continuous [-1, 1])

| Dim | Control | Mapping |
|-----|---------|---------|
| 0 | op3_q (warehouse dispatch) | multiplier = 1.25 + 0.75 × signal |
| 1 | op9_q (battalion dispatch) | multiplier = 1.25 + 0.75 × signal |
| 2 | op3_rop (warehouse reorder point) | multiplier = 1.25 + 0.75 × signal |
| 3 | op9_rop (battalion reorder point) | multiplier = 1.25 + 0.75 × signal |
| 4 | shifts (assembly capacity) | < -0.33 → S1, [-0.33, 0.33) → S2, ≥ 0.33 → S3 |

### Reward Function

**Primary:** `ReT_seq_v1` (κ=0.20) — Cobb-Douglas resilience metric.

```
r_t = SC_t^0.60 × BC_t^0.25 × AE_t^0.15
```

This is both the training reward AND the resilience metric. Higher = better.

### Step Configuration

- Step size: 168 hours (1 week)
- Episode length: 260 steps (20 years)
- Warmup: ~838h + ~2000h priming at S=2
- Risk level: `increased` (primary), also evaluate `current` and `severe`

## PPO Baseline (your target to beat)

Results from PPO+MLP (64,64) trained for 500k steps:

| Metric | PPO | static_s2 | garrido_cf_s2 |
|--------|-----|-----------|---------------|
| ReT_seq_v1 reward | ~133 | ~133 | ~131 |
| Fill rate | ~0.79 | ~0.79 | ~0.79 |
| Shift mix S1/S2/S3 | variable | 0/100/0 | 0/100/0 |

*Note: PPO results are preliminary. Updated numbers will be provided once the production run completes.*

## Evaluation Protocol

To be comparable with PPO results, your evaluation MUST use:

```python
env_kwargs = {
    "reward_mode": "ReT_seq_v1",
    "risk_level": "increased",      # primary scenario
    "stochastic_pt": True,
    "step_size_hours": 168,
    "max_steps": 260,
    "year_basis": "thesis",
    "observation_version": "v1",     # or v4 if you use richer obs
    "ret_seq_kappa": 0.20,
}
```

**Metrics to report:**
1. `reward_total` (ReT_seq_v1 sum over episode)
2. `fill_rate` (terminal cumulative)
3. `backorder_rate` (terminal cumulative)
4. `pct_steps_S1/S2/S3` (shift distribution)
5. `ret_thesis_corrected_total` (thesis audit metric)

**Cross-scenario evaluation:**
- Train on `increased`
- Evaluate on: `current`, `increased`, `severe`

**Seeds:** Use seeds 11, 22, 33, 44, 55 (or more) for reproducibility.

## Key Files

| File | Purpose |
|------|---------|
| `supply_chain/env_experimental_shifts.py` | Gymnasium environment with all reward modes |
| `supply_chain/external_env_interface.py` | Stable external API: `make_shift_control_env()`, `run_episodes()` |
| `supply_chain/dkana.py` | DKANA architecture starter: `DKANAPolicy`, `DKANADataset`, `build_dkana_windows()` |
| `scripts/export_trajectories_for_david.py` | Export trajectories as .npy for offline training |
| `scripts/build_dkana_dataset.py` | Convert trajectories to DKANA-ready windows |
| `supply_chain/config.py` | All simulation parameters from Garrido thesis |

## DKANA Architecture (already implemented)

`supply_chain/dkana.py` contains a complete starter implementation:

```
DKANAPolicy(config_dim=14, action_dim=5, latent_dim=128)
  ├── Row Encoder (MLP: 3 → latent)
  ├── Config Encoder (MLP: 14 → latent)
  ├── Local Attention (causal, within-state rows)
  ├── Global Attention (causal, across time steps)
  └── Decoder (latent → Normal(mean, std) over 5 actions)
```

Input: `DKANADataset` from `build_dkana_windows()`:
- `row_matrices`: (batch, seq, rows, 3) — [var_id, relation_id, value] triplets
- `config_context`: (batch, seq, 14) — control params + previous actions
- `time_mask`: (batch, seq) — validity mask for padded sequences

## What NOT to Change

- Do NOT modify the environment (`env_experimental_shifts.py`)
- Do NOT modify the reward function
- Do NOT modify the benchmark infrastructure
- Do NOT change the static baseline implementations

Your contribution is the DKANA model training and evaluation only.
