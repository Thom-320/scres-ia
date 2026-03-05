# MFSC Simulation (DES + RL)

Python rebuild of the Military Food Supply Chain (MFSC) simulation with SimPy and Gymnasium wrappers for RL experiments.

## Project Structure

```
supply_chain/          Core package (config, DES engine, Gym env)
scripts/               Diagnostic & analysis scripts
tests/                 Pytest test suite
docs/                  Reference PDFs, thesis, papers, planning docs
outputs/               Generated artifacts (plots, models, logs, reports)
legacy/                Deprecated code (kept for reference)
```

## Recommended Python

- Python `3.11` recommended for widest compatibility with `stable-baselines3`.

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

## RL training (PPO)

```bash
python train_agent.py --timesteps 20000 --n-envs 1 --seed 42 --year-basis thesis

# With R_t v0 reward
python train_agent.py --timesteps 500000 --n-envs 4 --reward-mode rt_v0 --rt-alpha 8
```

Artifacts (model, normalization stats, curves, csv, json) are saved under `outputs/`.

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
