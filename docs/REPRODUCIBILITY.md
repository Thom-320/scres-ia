# Reproducibility Guide

This guide defines the shortest path to reproduce the current paper-facing benchmark story.

## Environment

Recommended Python:

- Python `3.11`

Install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-pinned.txt
```

If you only want the tested pinned stack, install `requirements-pinned.txt` directly.

## Quick verification

Run the baseline tests:

```bash
pytest tests/
ruff check .
```

## Deterministic and stochastic DES checks

```bash
python run_static.py --det-only --year-basis thesis
python run_static.py --sto-only --year-basis thesis --seed 42
python validation_report.py --official-basis thesis
```

Expected outputs:

- validation CSV under `outputs/validation/`
- deterministic throughput near the thesis reference with acceptable error band

## Canonical training command

The public training default now matches the paper-facing benchmark family:

```bash
python train_agent.py   --env-variant shift_control   --reward-mode ReT_seq_v1   --ret-seq-kappa 0.20   --observation-version v1   --risk-level increased   --stochastic-pt   --w-bo 4.0 --w-cost 0.02 --w-disr 0.0   --timesteps 500000   --year-basis thesis
```

This command reproduces the current primary benchmark configuration, but not necessarily the exact archived model weights unless the same seeds and artifact collection pipeline are used.

## Frozen artifact bundles

Reviewer-safe artifact references live under:

- `docs/artifacts/control_reward/control_reward_500k_increased_stopt`
- `docs/artifacts/control_reward/control_reward_500k_severe_stopt`
- `docs/artifacts/control_reward/control_reward_500k_seed_inference`

These should be cited instead of untracked local `outputs/` folders.

## Full publication benchmark path

To reproduce the multi-phase publication run:

```bash
bash scripts/run_publication_experiments.sh --preflight
bash scripts/run_publication_experiments.sh
```

The script covers:

1. heuristic tuning
2. PPO benchmark lane
3. SAC comparator
4. PPO + frame stacking
5. RecurrentPPO
6. publication summary analysis

## Statistical unit

Use the seed as the primary unit of inference.

Report at minimum:

- mean
- standard deviation
- CI95
- paired seed-mean difference against the best static baseline

## Scope of automated verification

The repository currently guarantees through tests:

- environment contracts
- core benchmark script functionality
- control reward diagnostics
- export utilities
- evaluation helpers

It does not guarantee by default that full 500k benchmark runs finish in CI; those are offline experiments.
