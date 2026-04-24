# DKANA Contributor Handoff

## Why Track B, Not Track A

We discovered empirically that **Track A (thesis-faithful, 5D actions) has ~1% adaptive headroom** — static S=2 is already near-optimal because the agent controls upstream production while the real bottleneck is downstream distribution. PPO, RecurrentPPO, and every reward variant we tested fail to beat S2 in Track A. The problem is structural, not algorithmic.

**Track B extends the action space to include downstream control (Op10, Op12)**, opening 2-3pp of real headroom. In a 100k smoke, PPO achieved fill_rate=1.000 vs best static=0.987. This is where DKANA can demonstrate value.

Your task: train a DKANA policy on Track B and show it outperforms PPO+MLP under the same contract.

## Quick Start

```bash
# 1. Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify Track B env works
python -c "
from supply_chain.external_env_interface import make_dkana_track_b_env, get_track_b_env_spec
spec = get_track_b_env_spec()
print(f'Obs: {len(spec.observation_fields)} dims, Action: {len(spec.action_fields)} dims')
env = make_dkana_track_b_env(dkana_window_size=12, relation_mode="temporal_delta")
obs, info = env.reset(seed=42)
print(f'Obs shape: {obs.shape}, sample: {obs[:5]}')
print(f'DKANA rows: {info["dkana_row_matrices"].shape}')
print(f'DKANA config: {info["dkana_config_context"].shape}')
print(f'DKANA mask: {info["dkana_time_mask"]}')
"

# 3. Export Track B trajectories for offline training
python scripts/export_trajectories_for_david.py \
  --episodes 200 \
  --reward-mode ReT_seq_v1 \
  --observation-version v7 \
  --risk-level adaptive_benchmark_v2 \
  --action-contract track_b_v1 \
  --stochastic-pt \
  --output-dir outputs/data_export_track_b_v7

# 4. Build DKANA-ready context windows with {=,<,>} temporal relations
python scripts/build_dkana_dataset.py \
  --input-dir outputs/data_export_track_b_v7 \
  --window-size 12 \
  --relation-mode temporal_delta

# 5. Train the starter DKANA behavior-cloning policy
python scripts/train_dkana_behavior_clone.py \
  --dataset-dir outputs/data_export_track_b_v7/dkana_seq_w12 \
  --output-dir outputs/dkana_track_b_v7_bc \
  --epochs 25

# 6. Evaluate DKANA against the same Track B static baselines
python scripts/evaluate_dkana_track_b.py \
  --checkpoint outputs/dkana_track_b_v7_bc/dkana_policy.pt \
  --output-dir outputs/dkana_track_b_v7_eval \
  --episodes 20
```

## Track B Contract

Use `make_track_b_env()` or `get_track_b_env_spec()` from `external_env_interface.py`.

| Parameter | Value |
|-----------|-------|
| `action_contract` | `track_b_v1` |
| `observation_version` | `v7` |
| `reward_mode` | `ReT_seq_v1` |
| `risk_level` | `adaptive_benchmark_v2` |
| `step_size_hours` | `168` |
| `max_steps` | `260` |
| `stochastic_pt` | `True` |
| `year_basis` | `thesis` |

### Observation Space: v7 (46 dimensions)

| Dim | Field | Description |
|-----|-------|-------------|
| 0 | `raw_material_wdc_norm` | WDC raw material inventory |
| 1 | `raw_material_al_norm` | Assembly line raw material inventory |
| 2 | `rations_al_norm` | Finished rations at assembly |
| 3 | `rations_sb_norm` | Rations at Supply Battalion |
| 4 | `rations_cssu_norm` | Rations at Combat Service Support Units |
| 5 | `rations_theatre_norm` | Rations at Theatre of Operations |
| 6 | `fill_rate` | Cumulative fill rate (lagging indicator) |
| 7 | `backorder_rate` | Cumulative backorder rate |
| 8 | `assembly_line_down` | Binary: assembly line disrupted |
| 9 | `any_location_down` | Binary: any location disrupted |
| 10 | `op9_down` | Binary: Supply Battalion disrupted |
| 11 | `op11_down` | Binary: CSSU disrupted |
| 12 | `time_fraction` | Elapsed time / total horizon |
| 13 | `pending_batch_fraction` | Pending batch progress |
| 14 | `contingent_demand_fraction` | Contingent demand level |
| 15-17 | `prev_step_*` | Previous step demand/backorder/disruption |
| 18-19 | `cum_*` | Cumulative backorder rate and downtime |
| 20 | `rations_sb_dispatch_norm` | SB dispatch rate |
| 21 | `assembly_shifts_active_norm` | Current shift level (1/2/3 normalized) |
| 22-23 | `op1_down`, `op2_down` | Procurement disruption flags |
| 24-25 | `op1/op2_cycle_phase_norm` | Procurement cycle phase (anticipatory) |
| 26-29 | `workweek/workday_phase_*` | Calendar cycle signals |
| 30-34 | `regime_*` | Operational regime indicators (nominal/strained/pre_disruption/disrupted/recovery) |
| 35-36 | `risk_forecast_*` | Short-term risk forecasts (48h, 168h) |
| 37 | `maintenance_debt_norm` | Accumulated S3 maintenance debt |
| 38 | `backlog_age_norm` | Age of oldest pending backorder |
| 39 | `theatre_cover_days_norm` | Days of theatre inventory coverage |
| **40** | **`op10_down`** | **Binary: Op10 transport disrupted (NEW)** |
| **41** | **`op12_down`** | **Binary: Op12 transport disrupted (NEW)** |
| **42** | **`op10_queue_pressure_norm`** | **Op10 queue pressure (NEW)** |
| **43** | **`op12_queue_pressure_norm`** | **Op12 queue pressure (NEW)** |
| **44** | **`rolling_fill_rate_4w`** | **4-week rolling fill rate (NEW, not lagging)** |
| **45** | **`rolling_backorder_rate_4w`** | **4-week rolling backorder rate (NEW)** |

The NEW fields (40-45) are what make Track B different from Track A. They give the agent visibility into the downstream bottleneck.

### Action Space: 7 dimensions (Track B)

| Dim | Field | Mapping |
|-----|-------|---------|
| 0 | `op3_q_multiplier_signal` | `multiplier = 1.25 + 0.75 * signal` |
| 1 | `op9_q_multiplier_signal` | `multiplier = 1.25 + 0.75 * signal` |
| 2 | `op3_rop_multiplier_signal` | `multiplier = 1.25 + 0.75 * signal` |
| 3 | `op9_rop_multiplier_signal` | `multiplier = 1.25 + 0.75 * signal` |
| 4 | `assembly_shift_signal` | `< -0.33 -> S1, [-0.33, 0.33) -> S2, >= 0.33 -> S3` |
| **5** | **`op10_q_multiplier_signal`** | **`multiplier = 1.25 + 0.75 * signal` (NEW)** |
| **6** | **`op12_q_multiplier_signal`** | **`multiplier = 1.25 + 0.75 * signal` (NEW)** |

Dims 5-6 control downstream dispatch quantities. These are the actions that touch the active bottleneck.

### Reward

```
ReT_seq_v1 (Cobb-Douglas sequential resilience):
  r_t = SC_t^0.60 x BC_t^0.25 x AE_t^0.15
  with kappa=0.20 shift cost penalty
```

## Baselines to Compare Against

| Baseline | Description |
|----------|-------------|
| `s2_d1.00` | Static S=2, downstream dispatch at 1.0x (default) |
| `s3_d1.00` | Static S=3, downstream dispatch at 1.0x |
| `s3_d2.00` | Static S=3, downstream dispatch at 2.0x |
| `random` | Uniform random across all 7 dims |
| **PPO+MLP** | **PPO baseline from smoke/500k benchmarks** |

Smoke 100k results (your target to beat):

| Policy | Reward | Fill Rate | Order ReT |
|--------|--------|-----------|-----------|
| PPO+MLP | 250.17 | 1.000 | 0.927 |
| s3_d2.00 | 170.34 | 0.987 | 0.454 |
| s2_d1.00 | 177.78 | 0.965 | 0.495 |

## Evaluation Protocol

```python
env_kwargs = {
    "reward_mode": "ReT_seq_v1",
    "risk_level": "adaptive_benchmark_v2",
    "stochastic_pt": True,
    "step_size_hours": 168,
    "max_steps": 260,
    "year_basis": "thesis",
    "observation_version": "v7",
    "action_contract": "track_b_v1",
}
```

Primary metrics:
1. `reward_total`
2. `fill_rate`
3. `backorder_rate`
4. `order_level_ret_mean`
5. `rolling_fill_rate_4w` (end-of-episode)
6. Shift mix: `pct_steps_S1`, `pct_steps_S2`, `pct_steps_S3`

Seeds: `11, 22, 33, 44, 55`

## Why DKANA Matters Here

PPO+MLP already achieves fill_rate=1.000 in the smoke — but with 78% S1 shifts. A DKANA architecture with temporal attention could potentially:

1. **Anticipate disruptions** using regime signals (dims 30-34) and risk forecasts (dims 35-36) to pre-position inventory
2. **Coordinate upstream and downstream** actions jointly (the 7D action space has cross-dependencies that MLP may not capture)
3. **Learn from disruption sequences** across episodes (the L_{t-1} in your paper's R_t = f(S_t, D_t, L_{t-1}) formulation)
4. **Reduce S1 usage** while maintaining fill rate — proving that anticipatory control is more efficient than conservative excess capacity

## Offline Contract

Use `export_trajectories_for_david.py` with Track B settings:

```bash
python scripts/export_trajectories_for_david.py \
  --episodes 200 \
  --reward-mode ReT_seq_v1 \
  --observation-version v7 \
  --risk-level adaptive_benchmark_v2 \
  --action-contract track_b_v1 \
  --stochastic-pt \
  --output-dir outputs/data_export_track_b_v7
```

Each export includes: `observations.npy`, `actions.npy`, `rewards.npy`, `reward_terms.npy`, `constraint_context.npy`, `state_constraint_context.npy`, `env_spec.json`, `metadata.json`.

Then build DKANA windows:

```bash
python scripts/build_dkana_dataset.py \
  --input-dir outputs/data_export_track_b_v7 \
  --window-size 12 \
  --relation-mode temporal_delta
```

This writes `dkana_row_matrices.npy` as `(N, window, rows, 3)`, `dkana_config_context.npy` as `(N, window, config_dim)`, and `dkana_time_mask.npy` as `(N, window)`. With `--relation-mode temporal_delta`, the MRC emits both equality rows and temporal relation rows where the relation index maps `{ "=": 0, "<": 1, ">": 2 }`.

## Key Files

| File | Purpose |
|------|---------|
| `supply_chain/env_experimental_shifts.py` | DES-backed Gymnasium env (Track A and B) |
| `supply_chain/external_env_interface.py` | `make_track_b_env()`, `make_dkana_track_b_env()`, `get_track_b_env_spec()` |
| `supply_chain/dkana_env.py` | DKANA environment wrapper that emits the context window in `info` |
| `supply_chain/dkana.py` | DKANA preprocessing, temporal relation MRC, starter architecture, and online context-window adapter |
| `scripts/export_trajectories_for_david.py` | Offline trajectory export |
| `scripts/build_dkana_dataset.py` | DKANA window construction |
| `scripts/train_dkana_behavior_clone.py` | Starter DKANA training from DKANA-ready tensors |
| `scripts/evaluate_dkana_track_b.py` | Online Track B DKANA evaluation against static baselines and optional PPO |
| `scripts/run_track_b_smoke.py` | Track B benchmark script |
| `docs/TRACK_B_MINIMAL_SPEC.md` | Track B design rationale |
| `docs/PAPER_FINDINGS_REGISTRY.md` | Why Track A doesn't work (F2, F11) |

## What Not to Change

- Do not modify the DES engine (`supply_chain.py`) or risk distributions
- Do not modify the reward function parameters
- Do not alter the observation field order or action mapping
- Do not change the benchmark baseline definitions

Your contribution is the DKANA training pipeline and policy architecture under the frozen Track B contract.

## Where the Context Window Lives

The context window `SMS = [S_{k+1}, ..., S_{k+n}]` (paper's Enumeration + Matricial Representation block) is now available directly through the DKANA environment wrapper. This keeps the base PPO env unchanged, but gives DKANA a one-call environment entry point.

- **Environment entry point for David**: `make_dkana_track_b_env()` returns Track B with the DKANA context already attached to every `info`.
- **Offline training**: `build_dkana_windows()` in [dkana.py](../supply_chain/dkana.py) produces fixed-length windows with left-padding and a `time_mask` over exported trajectories.
- **Online inference**: use `info["dkana_row_matrices"]`, `info["dkana_config_context"]`, and `info["dkana_time_mask"]` directly. `DKANAOnlinePolicyAdapter` still exists for older code that uses the normal Track B env.

Minimal online usage:

```python
import torch
from supply_chain.external_env_interface import make_dkana_track_b_env
from supply_chain.dkana import DKANAPolicy

env = make_dkana_track_b_env(dkana_window_size=12, relation_mode="temporal_delta")
obs, info = env.reset(seed=42)

row_matrices = torch.from_numpy(info["dkana_row_matrices"][None]).float()
config_context = torch.from_numpy(info["dkana_config_context"][None]).float()
time_mask = torch.from_numpy(info["dkana_time_mask"][None])

# model = DKANAPolicy(...)
# dist = model(row_matrices, config_context, time_mask)
# action = dist.mean.squeeze(0).detach().numpy().clip(-1.0, 1.0)
# obs, reward, terminated, truncated, info = env.step(action)
```

Shapes with the frozen Track B contract and `dkana_window_size=12`:

- `obs`: `(46,)`
- `info["dkana_row_matrices"]`: `(12, 182, 3)`
- `info["dkana_config_context"]`: `(12, 16)`
- `info["dkana_time_mask"]`: `(12,)`

### Activating `<, =, >` relations

For the environment, call `make_dkana_track_b_env(relation_mode="temporal_delta")`. For offline datasets, pass `--relation-mode temporal_delta` to `scripts/build_dkana_dataset.py`. The mapping is `{"=": 0, "<": 1, ">": 2}`.

### Unified benchmark with statics + heuristics + PPO + DKANA

`scripts/run_track_b_smoke.py` accepts an optional `--dkana-checkpoint PATH`. When set, DKANA is evaluated alongside the other lanes under the same seeds/CI95 aggregation. Example:

```bash
python scripts/run_track_b_smoke.py \
  --seeds 11 22 33 \
  --train-timesteps 100000 \
  --eval-episodes 10 \
  --dkana-checkpoint outputs/dkana_track_b_v7_bc/dkana_policy.pt
```

The resulting `policy_summary.csv` includes a `dkana` row with the same metrics (reward_total, fill_rate, backorder_rate, order_level_ret_mean, rolling_fill_rate_4w, shift mix, downstream multiplier percentiles, assembly cost index) as statics, heuristics, and PPO.

## Track A (Legacy Reference)

Track A is frozen as a benchmark with honest negative results. If you need Track A data for comparison:
- `reward_mode="control_v1"`, `observation_version="v4"`, `risk_level="increased"`
- PPO does NOT beat S2 in Track A (documented in PAPER_FINDINGS_REGISTRY.md)
- Track A evidence is in `outputs/paper_benchmarks/`
