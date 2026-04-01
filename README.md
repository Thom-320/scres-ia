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

## Research entry points

```bash
# Low-level env smoke via train_agent.py
# Defaults to the shift-control research env for quick iteration.
python train_agent.py --timesteps 20000 --n-envs 1 --seed 42 --year-basis thesis

# Primary paper-facing benchmark (Track B positive lane)
python scripts/run_track_b_benchmark.py \
  --label track_b_ret_seq_k020_500k_rerun1 \
  --reward-mode ReT_seq_v1 \
  --ret-seq-kappa 0.20

# Short Track B smoke benchmark
python scripts/run_track_b_smoke.py \
  --output-dir outputs/benchmarks/track_b_smoke_manual \
  --reward-mode ReT_seq_v1 \
  --ret-seq-kappa 0.20 \
  --train-timesteps 100000 \
  --eval-episodes 10 \
  --seeds 11 22 33

# Track A comparator family
python scripts/run_paper_benchmark.py \
  --label paper_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20
```

Artifacts (model, normalization stats, curves, csv, json) are saved under `outputs/`.

For the paper-facing benchmark lane and artifact map, see:

- `docs/REPOSITORY_SOURCE_OF_TRUTH.md`
- `docs/TRACK_B_SOURCE_OF_TRUTH.md`
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

## Current paper state

The current repository story is split into two benchmark families:

- `Track A`: thesis-faithful upstream-control comparator family. This is the negative result.
- `Track B`: downstream-control repair at `Op10/Op12`. This is the current positive paper lane.

The frozen primary paper backbone is:

- Environment: `track_b_adaptive_control`
- Reward: `ReT_seq_v1`
- `ret_seq_kappa=0.20`
- Observation contract: `v7`
- Action contract: `track_b_v1`
- Risk profile: `adaptive_benchmark_v2`
- Step size: `168` hours
- Year basis: `thesis`
- `stochastic_pt=True`

Important default/launcher note:

- `train_agent.py` remains a low-level research entry point.
- `scripts/run_track_b_benchmark.py` is the primary paper-facing launcher.
- `scripts/run_paper_benchmark.py` defaults to the historical `control_v1` comparator lane; use `--reward-mode ReT_seq_v1` when you want the paper-facing Track A rerun bundle.
