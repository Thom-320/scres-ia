# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Current Paper Status

Before launching long experiments or reframing the manuscript, read:

- [PAPER_FINDINGS_REGISTRY.md](docs/PAPER_FINDINGS_REGISTRY.md) — 11 audited findings
- [FAMILY_A_DECISION_NOTE.md](docs/FAMILY_A_DECISION_NOTE.md) — Track A freeze rationale
- [TRACK_B_MINIMAL_SPEC.md](docs/TRACK_B_MINIMAL_SPEC.md) — Track B design

### Track A (Thesis-Faithful Benchmark) — CLOSED

Track A is a benchmark with honest negative results. **No RL configuration beats static S=2.**

- Frozen Family A contract: `ReT_seq_v1`, `v1`, `168h`, `increased`, `thesis` year basis
- Valid evidence: `paper_benchmarks/paper_ret_seq_k020_500k`, `paper_control_v1_500k`, `paper_ret_seq_k010_500k`
- **INVALID** (pre-audit DES): `control_reward_500k_*_stopt`, `section4_3_*`
- RecurrentPPO 500k completed: fill=0.751 vs S2=0.794 — **LOSES**
- Root cause: downstream distribution bottleneck (F11), 1% action headroom (F2)
- Paper framing: "when does RL help?" + mechanistic explanation of structural limitations

### Track B (Extended Action Space) — ACTIVE

Track B adds downstream control (Op10/Op12) to the action space, opening real headroom.

- Contract: `action_contract="track_b_v1"`, `v7` (46 dims), `7D actions`, `ReT_seq_v1`, `adaptive_benchmark_v2`
- Smoke 100k: PPO fill=1.000 vs best static=0.987 — **PPO WINS**
- 500k x 5 seeds: **VALIDATED** — PPO fill=1.000 beats all baselines (see `scripts/analyze_track_b_500k.py`)
- DKANA handoff: [DKANA_CONTRIBUTOR_HANDOFF.md](docs/DKANA_CONTRIBUTOR_HANDOFF.md)

### Key Rules

- Do **not** add new reward modes.
- Do **not** use `control_reward_500k_*_stopt` as evidence — pre-audit DES, now `historical_artifact`.
- Do **not** mix Track A and Track B evidence.
- Track A valid runs use `year_basis="thesis"`, `observation_version="v1"`.
- Track B uses `action_contract="track_b_v1"`, `observation_version="v7"`.

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

# RL training (default env is shift_control with ReT_seq_v1 κ=0.20)
python train_agent.py --timesteps 20000 --n-envs 1 --seed 42 --year-basis thesis
python train_agent.py --timesteps 500000 --n-envs 4 --reward-mode ReT_seq_v1 --ret-seq-kappa 0.20 --env-variant shift_control
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
- **`env_experimental_shifts.py`** -- `MFSCGymEnvShifts`: extended env adding a 5th action dimension for assembly shift control (1/2/3 shifts via tri-level thresholds at -0.33/+0.33). **Primary Track A reward: `control_v1`** (operational service-cost control signal). Also supports `control_v1_pbrs`, `ReT_seq_v1`, `ReT_unified_v1`, `ReT_thesis` (audit-only piecewise approximation), and `rt_v0` (legacy).
- **`external_env_interface.py`** -- `ExternalEnvSpec` dataclass and factory functions (`make_shift_control_env`, `get_shift_control_env_spec`) for external model integration. Defines the stable machine-readable contract: obs fields (15-dim), action fields (5-dim), control context (9-dim), state constraint fields (22-dim, includes `cumulative_backorder_qty` and `cumulative_disruption_hours`), and reward term fields. Also provides `run_episodes(policy_fn, ...)` for evaluating any callable policy without depending on benchmark internals. Used by the DKANA export pipeline.
- **`dkana.py`** -- DKANA-compatible input pipeline and policy architecture. Contains `DKANADataset` dataclass, `DKANAPolicy` (PyTorch nn.Module with row-wise MLP encoder, local/global causal self-attention, distributional Gaussian decoder). Bridges exported trajectories to DKANA training. Helper functions: `build_mfsc_relational_state()`, `build_previous_action_context()`, `build_dkana_windows()`.

### Reward modes by environment variant

| Env variant      | Class              | Actions | Observation | Supported rewards                    |
|------------------|--------------------|---------|-------------|--------------------------------------|
| `base`           | `MFSCGymEnv`       | 4-dim   | v1 (15)     | `proxy`, `rt_v0`                     |
| `shift_control`  | `MFSCGymEnvShifts` | 5-dim   | v1-v6       | `control_v1`, `ReT_seq_v1`, `ReT_unified_v1`, etc. |
| **`track_b_v1`** | `MFSCGymEnvShifts` | **7-dim** | **v7 (46)** | `ReT_seq_v1` (primary), `control_v1` |

- **`ReT_seq_v1`** (TRACK B PRIMARY, TRACK A FAMILY A): Sequential operational resilience extending Garrido Eq. 5.5. C-D form: `SC^0.60 × BC^0.25 × AE^0.15`.
- **`control_v1`** (TRACK A COMPARATOR): Linear operational control reward (`-w_bo*service_loss - w_cost*shift_cost`). Frozen weights: `w_bo=4.0`, `w_cost=0.02`, `w_disr=0.0`. PPO does NOT beat S2 with this reward on the corrected DES.
- **`ReT_thesis`**: Piecewise step-level approximation of Eq. 5.5. **Audit-only** -- NOT suitable as training objective (collapses to S1).
- **`action_mode`**: `"full"` (5D, default), `"shift_only"` (1D), `"shift_q9"` (2D) — experimental action space reductions.

### Track B action space (7D)

Track B extends the 5D shift_control actions with downstream dispatch control:
- dims 0-4: same as shift_control (op3_q, op9_q, op3_rop, op9_rop, shift)
- **dim 5**: `op10_q_multiplier_signal` — controls Op10 transport dispatch quantity
- **dim 6**: `op12_q_multiplier_signal` — controls Op12 transport dispatch quantity

These target the downstream distribution bottleneck that limits Track A (see F11 in PAPER_FINDINGS_REGISTRY.md). Use `make_track_b_env()` from `external_env_interface.py`.

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
1. **Control reward benchmark** (`scripts/benchmark_control_reward.py`) -- Multi-seed PPO/SAC vs static S1/S2/S3 + heuristic baselines. Default reward is now `control_v1` for the frozen Track A contract. Also supports `control_v1_pbrs`, `ReT_seq_v1`, and `ReT_unified_v1` for historical/exploratory lanes. Supports cross-scenario evaluation (`--eval-risk-levels`), heuristic tuning (`--tune-heuristic`), frame-stacking (`--frame-stack`), and RecurrentPPO. Outputs metrics CSV with CI95.
2. **Delta sweep** (`scripts/benchmark_delta_sweep_static.py`) -- Sweeps `rt_delta` shift-cost parameter over static policies to find S1/S2 optimality transition points.
3. **ReT ablation** (`scripts/benchmark_ret_ablation_static.py`) -- Tests formula variants (`default`, `autotomy_equals_recovery`, `merged_recovery_formula`) and autotomy thresholds.
4. **Minimal shift control** (`scripts/benchmark_minimal_shift_control.py`) -- Quick multi-seed PPO vs static comparison.
5. **Formal evaluation** (`scripts/formal_evaluation.py`) -- End-to-end evaluation of pre-trained models from `outputs/benchmarks/`, comparing trained/random/default policies with CI95 aggregation.

### Entry points (root)
- **`run_static.py`** -- Deterministic/stochastic baselines, validates against thesis.
- **`train_agent.py`** -- PPO training pipeline. Default: `shift_control` env with `ReT_seq_v1` (κ=0.20). Outputs models, normalization stats, learning curves, CSVs, JSON to `outputs/`.
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
- **Warmup priming**: After DES warmup (~838h), the env runs an additional priming phase at S=2 (~2000h) to stabilize fill_rate before RL starts. Without this, ~80k pending backorders from warmup drown all reward signals. See `_prime_after_warmup()` in `env_experimental_shifts.py`.
- **Observation space bounded**: `high=20.0` (not inf). VecNormalize further normalizes during training. R24 contingent demand capped at 5x2600.
- **`control_v1` is the frozen Track A training reward.** `ReT_seq_v1`, `ReT_unified_v1`, and `ReT_garrido2024` remain available as resilience-oriented audit or exploratory lanes. Do not train on `ReT_thesis` -- it collapses to S1.
- **Garrido 2024 C-D family**: `ReT_garrido2024_raw` (5-var raw product), `ReT_garrido2024` (sigmoid eval index), `ReT_garrido2024_train` (cost-excluded training variant). Calibration JSON at `supply_chain/data/ret_garrido2024_calibration.json`. These provide the theoretical bridge to Garrido et al. (2024, IJPR) but do NOT train as well as ReT_seq_v1 due to intermediate-variable bias (spare capacity phi always favors S3).
- **Reward function zoo**: Many modes exist from iterative development. For the paper backbone, only `control_v1` (train) and the audit metrics (`ReT_garrido2024`, `ret_thesis_corrected`, optional `ReT_unified_v1`) matter. The rest are ablation/history.
- **Adding new reward modes**: Must wire into 4 files: `env_experimental_shifts.py` (REWARD_MODE_OPTIONS + compute + step dispatch + info), `benchmark_control_reward.py` (choices + reward_family + weight_combos + env_kwargs), `train_agent.py` (SHIFT_ENV_REWARD_MODES), `run_paper_benchmark.py` (choices).
- **Known env issue**: R14 defects are recycled to raw materials instead of discarded. Affects all policies equally. Documented as known simplification.
- **Benchmark reproducibility**: all benchmark scripts accept `--seeds`, output manifest JSON with commit hash, and support separate train/eval seed offsets.
- Python 3.11 recommended for SB3 compatibility. Formatting: black. Linting: ruff. Types: mypy.

## Results Summary (2026-03-31)

Full findings with evidence sources: [PAPER_FINDINGS_REGISTRY.md](docs/PAPER_FINDINGS_REGISTRY.md)

### Track A (thesis-faithful) — No RL beats S2

- **PPO + control_v1 (500k)**: fill=0.782 vs S2=0.792. **PPO loses.** [paper_control_v1_500k]
- **PPO + ReT_seq_v1 κ=0.20 (500k)**: fill=0.788 vs S2=0.792. **PPO loses.** [paper_ret_seq_k020_500k]
- **RecurrentPPO + control_v1 (500k)**: fill=0.751 vs S2=0.794. **PPO loses badly.** [final_recurrent_ppo_v4_control_500k]
- **PPO + ReT_thesis**: S1 collapse (99.99% S1). Not suitable as reward.
- **C-D resilience rewards**: All fail as training objectives — agents exploit weakest dimension (F1, F8).
- **Root cause**: Downstream distribution bottleneck (F11) + 1% action headroom (F2).
- **AUDIT WARNING**: Runs named `control_reward_500k_*_stopt` used pre-audit DES with bugs. They are `historical_artifact`, not valid evidence.

### Track B (extended downstream control) — PPO wins

- **PPO + ReT_seq_v1 (100k smoke)**: fill=1.000 vs best static=0.987. **PPO wins.** [track_b_smoke_initial]
- **500k x 5 seeds**: **VALIDATED** — PPO fill=1.000 beats all baselines across seeds.
- **Why it works**: Track B adds Op10/Op12 dispatch actions, giving the agent control over the active bottleneck.

### Next steps

- Write the paper with dual-track framing: "Track A shows structural limitations, Track B shows RL works when the agent controls the right constraint."
- DKANA handoff ready for David: [DKANA_CONTRIBUTOR_HANDOFF.md](docs/DKANA_CONTRIBUTOR_HANDOFF.md)
