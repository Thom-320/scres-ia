# MFSC Simulation (Hybrid DES + Neural Learning)

Python rebuild of the Military Food Supply Chain (MFSC) discrete-event simulation with SimPy, based on Garrido-Rios (2017). Includes a hybrid simulation-neural framework for studying supply chain resilience (SCRES) as a learning-dependent capability.

## Project Structure

```text
supply_chain/          Core package (config, DES engine, Gymnasium environments)
scripts/               Diagnostic & analysis scripts
tests/                 Pytest test suite
docs/                  Reference PDFs, thesis, papers, planning docs
outputs/               Generated artifacts (plots, models, logs, reports)
legacy/                Deprecated code (kept for reference)
```

## Recommended Python

- Python `3.11` recommended for widest package compatibility.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run simulation baselines

```bash
# Deterministic baseline (Phase 1)
python run_static.py --det-only --year-basis thesis

# Stochastic baseline (Phase 2)
python run_static.py --sto-only --year-basis thesis --seed 42

# Combined comparison
python run_static.py --year-basis thesis
```

## Validation report (dual year basis)

```bash
python validation_report.py --official-basis thesis
```

Artifacts are written to `outputs/validation/validation_table_dual_basis.csv`.

## Hybrid model training

```bash
# Quick smoke check with the paper-facing defaults:
# shift_control + ReT_seq_v1 (kappa=0.20) + observation_version=v1
python train_agent.py --timesteps 20000 --n-envs 1 --seed 42 --year-basis thesis

# Frozen benchmark backbone used by the current manuscript lane
python train_agent.py \
  --timesteps 500000 \
  --n-envs 4 \
  --env-variant shift_control \
  --reward-mode ReT_seq_v1 \
  --ret-seq-kappa 0.20 \
  --observation-version v1 \
  --risk-level increased \
  --stochastic-pt \
  --w-bo 4.0 --w-cost 0.02 --w-disr 0.0
```

Artifacts (model, normalization stats, curves, csv, json) are saved under `outputs/`.

For the paper-facing benchmark lane and artifact map, see:

- `docs/REPOSITORY_SOURCE_OF_TRUTH.md`
- `docs/REPRODUCIBILITY.md`
- `docs/artifacts/control_reward/README.md`

## Diagnostic scripts

```bash
python scripts/diagnostic_doe_alpha.py          # DOE alpha sensitivity
python scripts/diagnostic_reward_spread.py      # Reward spread analysis
python scripts/diagnostic_doe_decompose.py      # Reward decomposition
python scripts/diagnostic_action_impact.py      # Action impact verification
```

## Tests

```bash
pytest tests/
```

## Quality checks

```bash
black .
ruff check . --fix
mypy supply_chain/
```

## Current benchmark defaults

The current repository defaults are aligned with the frozen resilience-reward contract:

- Training reward: `ReT_seq_v1`
- Frozen kappa: `0.20`
- Historical comparator: `control_v1`
- Thesis-aligned audit metric: `ret_thesis_corrected`
- Shift-control env: enabled by default
- Observation version: `v1` for the frozen benchmark; `v2` is preferred for new ablations
- Main paper scenarios:
  - `increased + stochastic_pt`
  - `severe + stochastic_pt`
