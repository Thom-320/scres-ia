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

## Canonical benchmark commands

Primary paper result:

```bash
python scripts/run_track_b_benchmark.py \
  --label track_b_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --ret-seq-kappa 0.20
```

Primary Track B smoke:

```bash
python scripts/run_track_b_smoke.py \
  --output-dir outputs/benchmarks/track_b_smoke_manual \
  --reward-mode ReT_seq_v1 \
  --ret-seq-kappa 0.20 \
  --train-timesteps 100000 \
  --eval-episodes 10 \
  --seeds 11 22 33
```

Track A comparator family:

```bash
python scripts/run_paper_benchmark.py \
  --label paper_ret_seq_k020_500k \
  --reward-mode ReT_seq_v1 \
  --kappa 0.20
```

Causal ablation (5D vs 7D, same v7/risk/reward):

```bash
# 5D (track_a actions, should LOSE to S2)
python train_agent.py --env-variant shift_control \
  --reward-mode ReT_seq_v1 --ret-seq-kappa 0.20 \
  --observation-version v7 --risk-level adaptive_benchmark_v2 \
  --timesteps 100000 --seed 42

# 7D (track_b actions, should WIN vs S2)
python train_agent.py --env-variant track_b \
  --reward-mode ReT_seq_v1 --ret-seq-kappa 0.20 \
  --timesteps 100000 --seed 42
```

Artifact: `outputs/track_b_ablation_5d_vs_7d.json`

## Auditable artifact bundles

Reviewer-safe artifact references live under:

- `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1` — Track B 500k primary
- `outputs/benchmarks/track_b_smoke_initial_2026-03-31` — Track B 100k smoke
- `outputs/track_b_ablation_5d_vs_7d.json` — Causal ablation (5D vs 7D)
- `outputs/paper_benchmarks/paper_ret_seq_k020_500k` — Track A primary
- `outputs/paper_benchmarks/paper_control_v1_500k` — Track A comparator
- `outputs/benchmarks/final_recurrent_ppo_v4_control_500k` — RecurrentPPO negative

Use these instead of the historical `*_stopt` bundles and incomplete Track B runs.

The old `control_reward_500k_*_stopt` bundles and their seed-inference note predate the March 2026 DES audit/alignment fixes and should not be cited as the primary evidence for the current repository state.

## Scope

The current paper story does **not** depend on reproducing every historical lane.
It depends on reproducing:

1. Track A negative comparator bundles
2. Track B smoke
3. Track B long run

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
