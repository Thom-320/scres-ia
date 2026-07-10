# Reproducibility Guide

This guide defines the shortest path to reproduce the current paper-facing
benchmark story: the **Track B** decision-contract result. The claim-by-claim
source of truth is `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`; every
headline number below traces to a dated artifact directory.

> Historical note (2026-07-10): earlier versions of this guide described the
> pre-Track-B `shift_control`/`ReT_seq_v1` lane as canonical. That lane is
> retired as primary evidence; its bundles under `outputs/paper_benchmarks/`
> are preserved for provenance only.

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

Expected outputs: a validation CSV under `outputs/validation/`, with
deterministic throughput near the thesis reference (reconstruction gap
$-4.43\%$ on the thesis basis; the thesis's own historical calibration
dispersion spans $-21.6\%$ to $+14.1\%$ — there is no formal $\pm 15\%$
acceptance threshold in the source).

## Canonical training command (Track B, the manuscript spine)

The manuscript's primary result trains PPO on the `track_b_v1` 8D contract
under the canonical protocol (`control_v1` reward, observation `v7`,
`adaptive_benchmark_v2` risk level, h104, thesis year basis, stochastic PT,
lr 3e-4):

```bash
python scripts/run_track_b_smoke.py \
  --reward-mode control_v1 \
  --observation-version v7 \
  --risk-level adaptive_benchmark_v2 \
  --max-steps 104 \
  --train-timesteps 60000 \
  --seeds 1 2 3 4 5 \
  --eval-episodes 12
```

Seeds 6-10 were trained identically in the seed-expansion run. Frozen
checkpoints:

- seeds 1-5: `outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104/models/`
- seeds 6-10: `outputs/experiments/track_b_seed_expansion_2026-07-02/track_b_seed_expansion_6_10_claude/models/`

Observation-provenance note: seeds 1-5 were trained on the 48-dim v7 (before
the four tail fields were appended); seeds 6-10 on the 52-dim v7. Field order
for dims 0-47 is unchanged; held-out evaluation slices the observation for
the older checkpoints (`scripts/run_track_b_crossed_eval.py`).

## Canonical evaluation and headline artifacts

- 10-seed paired dense-CRN stats bundle: `docs/track_b_q1_stats_2026-07-02_final_10seed/`
  (PPO `0.005898` vs static `0.005460` Excel ReT on the original 21-tape plan).
- **Fully crossed held-out evaluation** (all 10 checkpoints and the
  prespecified static comparator on 60 fresh tapes, eval seeds
  200001-200060, dependence-aware two-way inference):
  `outputs/experiments/track_b_crossed_eval_2026-07-09/` — delta `+0.000486`,
  two-way CI95 `[+0.000456, +0.000517]`, 10/10 checkpoints and 60/60 tapes
  positive.
- Dense 147-cell static frontier evaluation: `scripts/run_track_b_dense_crn_static.py`
  (see the claims registry entry C1 for the exact command).
- Corrected decision-contract factorial (mechanism gate):
  `scripts/run_track_b_contract_factorial.py` → `outputs/experiments/track_b_factorial_*_2026-07-09/`.

## Statistical unit

Primary inference treats the training seed as the unit, with tape dependence
handled explicitly: the crossed evaluation reports a two-way (checkpoint x
evaluation tape) cluster bootstrap alongside the per-checkpoint t-interval.
Report at minimum: mean, CI95 under both schemes, per-seed means, and
leave-one-out sensitivity over checkpoints and tapes.

## Primary metric

`ret_excel` (Garrido/Excel ReT, workbook-faithful; audited row-by-row against
the original workbooks with zero formula mismatches). Never substitute
`ret_thesis`. Secondary panel: `ret_excel_cvar05`, fill/flow-fill, backlog,
service-loss AUC, CTj/RPj/DPj tails, shift-utilization cost index.

## Scope of automated verification

Tests guarantee environment contracts, core benchmark script functionality,
control reward diagnostics, export utilities, evaluation helpers, and a
manuscript guard (`tests/test_manuscript_retired_claims.py`) that fails if
retired or unsafe claim language reappears. Full 60k training runs are
offline experiments and are not executed in CI.
