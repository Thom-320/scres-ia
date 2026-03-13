# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Summary

Python rebuild of a 13-operation Military Food Supply Chain (MFSC) discrete-event simulation (originally MATLAB/Simulink from Garrido-Rios' 2017 PhD thesis) using SimPy. Wrapped in Gymnasium for RL training with Stable-Baselines3 (PPO). The repo is being developed toward a Q1/Q2 journal publication (target: IJPR, IEEE TAI, or EJOR) with the contribution framed as reward design and auditing for operational resilience control under disruptions, not architectural novelty.

## Commands

```bash
# Setup
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run simulation baselines
python run_static.py --det-only --year-basis thesis     # Deterministic (Phase 1)
python run_static.py --sto-only --year-basis thesis     # Stochastic (Phase 2)
python run_static.py --year-basis thesis                # Both

# Validation report
python validation_report.py --official-basis thesis

# RL training (default env is shift_control with ReT_thesis reward)
python train_agent.py --timesteps 20000 --n-envs 1 --seed 42 --year-basis thesis
python train_agent.py --timesteps 500000 --n-envs 4 --reward-mode ReT_thesis --env-variant shift_control
python train_agent.py --timesteps 100000 --env-variant base --reward-mode rt_v0 --rt-alpha 8

# Benchmarks
python scripts/benchmark_control_reward.py --seeds 1 --train-timesteps 2000 --eval-episodes 2 --step-size-hours 24 --max-steps 20 --risk-level increased --w-bo 2.0 --w-cost 0.06 --w-disr 0.0 --stochastic-pt   # smoke
python scripts/benchmark_delta_sweep_static.py    # delta parameter sensitivity
python scripts/benchmark_minimal_shift_control.py # multi-seed PPO vs static
python scripts/benchmark_ret_ablation_static.py   # ReT formula variant ablation
python scripts/formal_evaluation.py               # formal eval of pre-trained models

# DKANA data pipeline
python scripts/export_trajectories_for_david.py   # export trajectory .npy files
python scripts/build_dkana_dataset.py              # convert to DKANA-ready windows

# Tests
pytest tests/                                    # Full suite
pytest tests/test_env.py                         # Single file
pytest tests/test_env.py::test_gym_step_returns  # Single test

# Quality
black .
ruff check . --fix
mypy supply_chain/
```

## Architecture

### Data flow: DES -> Gym -> RL

```
config.py (thesis params) -> MFSCSimulation (SimPy DES engine)
    -> MFSCGymEnv / MFSCGymEnvShifts (Gymnasium wrappers)
        -> train_agent.py (PPO via SB3, VecNormalize, Monitor)
            -> outputs/ (models, curves, JSON, CSV)
```

The simulation runs at hourly granularity internally. The Gym envs call `sim.step(action_dict, step_hours=168)` to advance 1 week per RL step, then read `sim.get_observation()` for a 15-dim state vector.

### Core package: `supply_chain/`
- **`config.py`** -- Single source of truth for all simulation parameters. Every constant comes from thesis tables (6.4, 6.10, 6.12, 6.16, 6.20, 6.25). Never hardcode numbers elsewhere. Contains three risk-level dicts: `RISKS_CURRENT`, `RISKS_INCREASED`, `RISKS_SEVERE`.
- **`supply_chain.py`** -- SimPy DES engine (`MFSCSimulation`). 13 operations at hourly granularity. Supports deterministic (Phase 1) and stochastic risk (Phase 2) modes. Key API: `step(action, step_hours)` returns `(obs, reward, terminated, info)`, `get_observation()` returns 15-dim list. Mutable params in `self.params` dict allow RL to adjust inventory quantities, reorder points, and assembly shifts at runtime. Also exposes `get_state_constraint_context()` for DKANA export.
- **`env.py`** -- `MFSCGymEnv`: base Gymnasium wrapper. 15-dim obs, 4-dim action (policy multipliers in [-1,1] mapped to [0.5, 2.0] via `1.25 + 0.75 * action`). Actions control: op3_q, op9_q, op3_rop, op9_rop.
- **`env_experimental_shifts.py`** -- `MFSCGymEnvShifts`: extended env adding a 5th action dimension for assembly shift control (1/2/3 shifts via tri-level thresholds at -0.33/+0.33). Implements four reward modes: `rt_v0`, `ReT_thesis`, `control_v1`, and `control_v1_pbrs`. The `control_v1` reward (`-w_bo*service_loss - w_cost*shift_cost - w_disr*disruption`) is the primary benchmarking reward; `control_v1_pbrs` adds Potential-Based Reward Shaping (Ng et al. 1999) with a target-deficit service potential `Φ(s) = -α·max(0, τ-FR)/τ`. Two PBRS variants: `cumulative` (uses obs[6] fill_rate) and `step_level` (uses obs[16] prev_step_backorder_qty_norm, requires v2). `ReT_thesis` is retained as a reporting-only metric.
- **`external_env_interface.py`** -- `ExternalEnvSpec` dataclass and factory functions (`make_shift_control_env`, `get_shift_control_env_spec`) for external model integration. Defines the stable machine-readable contract: obs fields (15-dim), action fields (5-dim), control context (9-dim), state constraint fields (22-dim, includes `cumulative_backorder_qty` and `cumulative_disruption_hours`), and reward term fields. Also provides `run_episodes(policy_fn, ...)` for evaluating any callable policy without depending on benchmark internals. Used by the DKANA export pipeline.
- **`dkana.py`** -- DKANA-compatible input pipeline and policy architecture. Contains `DKANADataset` dataclass, `DKANAPolicy` (PyTorch nn.Module with row-wise MLP encoder, local/global causal self-attention, distributional Gaussian decoder). Bridges exported trajectories to DKANA training. Helper functions: `build_mfsc_relational_state()`, `build_previous_action_context()`, `build_dkana_windows()`.

### Reward modes by environment variant

| Env variant      | Class              | Actions | Supported rewards                    |
|------------------|--------------------|---------|--------------------------------------|
| `base`           | `MFSCGymEnv`       | 4-dim   | `proxy`, `rt_v0`                     |
| `shift_control`  | `MFSCGymEnvShifts` | 5-dim   | `rt_v0`, `ReT_thesis`, `control_v1`, `control_v1_pbrs` |

- **`ReT_thesis`**: Garrido Eq. 5.5 approximation at step level with linear shift cost `delta*(S-1)`. Four cases: fill_rate_only, autotomy, recovery, non_recovery. **Now reporting-only** -- kept for audit/comparison but NOT used as the training objective.
- **`control_v1`**: Operational control reward (`-w_bo*service_loss - w_cost*shift_cost - w_disr*disruption`). **Primary benchmarking reward.** Configurable weights enable service-cost tradeoff analysis and Pareto front approximation.
- **`control_v1_pbrs`**: control_v1 + PBRS shaping bonus `F = γΦ(s') - Φ(s)`. Target-deficit potential `Φ(s) = -α·max(0, τ-FR)/τ` (cumulative variant) or `Φ(s) = -α·prev_step_backorder_norm` (step_level variant, requires v2 obs). Params: `pbrs_alpha` (scale), `pbrs_tau` (target, default 0.95), `pbrs_gamma` (discount, must match SB3). Preserves optimal policy (Ng et al. 1999).
- **`rt_v0`**: Legacy weighted sum of recovery/holding/service loss.

### DKANA data pipeline

```
export_trajectories_for_david.py (rollouts → .npy)
    → outputs/data_export/ (observations, actions, episode_ids, constraint_context, state_constraint_context, rewards, env_spec.json)
        → build_dkana_dataset.py (sliding windows + left-padding)
            → outputs/data_export/dkana_seq_w{size}/ (row_matrices, config_context, action_targets, time_mask, metadata.json)
                → DKANAPolicy training (PyTorch, distributional Gaussian output)
```

### Benchmark framework

Four evaluation lanes, each with its own script:
1. **Control reward benchmark** (`scripts/benchmark_control_reward.py`) -- Multi-seed PPO/SAC vs static S1/S2/S3 + heuristic baselines under `control_v1` (or `control_v1_pbrs`). Supports weight-grid screening ("survivors") before training, cross-scenario evaluation (`--eval-risk-levels`), heuristic tuning (`--tune-heuristic`), frame-stacking (`--frame-stack`), and RecurrentPPO. Three heuristic policies: `HeuristicHysteresis` (deadband shift control), `HeuristicDisruptionAware` (reactive shift+inventory), `HeuristicTuned` (combined, grid-searchable). Outputs metrics CSV with CI95.
2. **Delta sweep** (`scripts/benchmark_delta_sweep_static.py`) -- Sweeps `rt_delta` shift-cost parameter over static policies to find S1/S2 optimality transition points.
3. **ReT ablation** (`scripts/benchmark_ret_ablation_static.py`) -- Tests formula variants (`default`, `autotomy_equals_recovery`, `merged_recovery_formula`) and autotomy thresholds.
4. **Minimal shift control** (`scripts/benchmark_minimal_shift_control.py`) -- Quick multi-seed PPO vs static comparison.
5. **Formal evaluation** (`scripts/formal_evaluation.py`) -- End-to-end evaluation of pre-trained models from `outputs/benchmarks/`, comparing trained/random/default policies with CI95 aggregation.

### Entry points (root)
- **`run_static.py`** -- Deterministic/stochastic baselines, validates against thesis.
- **`train_agent.py`** -- PPO training pipeline. Default: `shift_control` env with `ReT_thesis`. Outputs models, normalization stats, learning curves, CSVs, JSON to `outputs/`.
- **`validation_report.py`** -- Dual-basis validation tables to `outputs/validation/`.

### Other directories
- `scripts/` -- Benchmark scripts (see above), trajectory export (`export_trajectories_for_david.py`), DKANA dataset builder (`build_dkana_dataset.py`), and diagnostic scripts.
- `tests/` -- Pytest suite. `conftest.py` adds project root to `sys.path`. Tests cover env contracts, reward components, benchmark aggregation logic, DKANA pipeline, delta sweep, ReT ablation, and formal evaluation.
- `legacy/` -- Deprecated code kept for reference.
- `docs/` -- Thesis PDF, planning docs, integration guide (`DKANA_INTEGRATION_GUIDE.md`), research briefs (`docs/briefs/`), manuscript notes (`docs/manuscript_notes/`), and benchmark artifacts (`docs/artifacts/`).
- `outputs/` -- Generated artifacts (gitignored).

## Key Conventions

- **Thesis notation is mandatory**: variable names must follow Garrido Chapter 6 (Opk,j, PT, Q, ROP, S, It,S, Rcr, lambda=320.5).
- **Hourly granularity**: assembly line (Op5-Op7) runs hourly -- never convert to daily (masks sub-day risks like R11 at ~2.2h).
- **Year basis**: two modes -- `thesis` (336 days, 8064 hrs) and `gregorian` (365 days, 8760 hrs). Default is `thesis`.
- **Warm-up**: simulation warm-up ends when first 5000-ration batch reaches Op9 (~838.8 hrs deterministic).
- **Buffers use `simpy.Container`** (continuous quantities), not `simpy.Store`.
- **Gym step returns**: always `(observation, reward, terminated, truncated, info)`.
- **Risk levels**: `current` (Table 6.12 '-'), `increased` ('+'), `severe` ('++' extrapolated). Base env supports current/increased only; shift_control supports all three.
- **`control_v1` is the primary benchmarking reward**. `ReT_thesis` is kept as a reporting-only metric for audit. Do not train on `ReT_thesis` for publication results -- it induces misaligned incentives (collapses to S1).
- **Benchmark reproducibility**: all benchmark scripts accept `--seeds`, output manifest JSON with commit hash, and support separate train/eval seed offsets.
- Python 3.11 recommended for SB3 compatibility. Formatting: black. Linting: ruff. Types: mypy.

## Preliminary Results (frozen findings)

- **PPO + ReT_thesis**: Misaligned objective -- agent collapses to S1 (cheapest shift, poor service). Not suitable as training reward.
- **PPO + control_v1 (50k steps)**: Weight-sensitive; narrow region where PPO outperforms static.
- **PPO + control_v1 (500k steps)**: Competitive with best static under `increased` risk; slight advantage under `severe` (p~0.19).
- **Three-part bottleneck identified**: reward alignment → reward sensitivity → regime dependence.
- **Next experimental steps**: SAC vs PPO comparison, frame-stacking (VecFrameStack n=4), then RecurrentPPO (LSTM) for POMDP. DKANA justified only after these comparisons.
