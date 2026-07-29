# Track B Final Consolidated Plan (2026-07-01)

## Source of Truth

Paper-facing claims, retired claims, and reviewer-defense gates are centralized
in `docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md`. If older documents mention
Track B as `7D`, "perfect fill", thesis-faithful, or strictly Pareto-dominant
on all metrics, treat those statements as historical until reconciled with the
claims registry.

## Framing

Track A is closed for the paper as a boundary characterization: with Garrido's
original buffer/shift decision family, dense static frontiers, CRN, and rich
metrics leave no publishable dynamic signal. It is not claimed mathematically
impossible; any revisit belongs in an appendix diagnostic only.

Track B `adaptive_benchmark_v2` is the primary positive claim. It is an
operational extension that exposes downstream dispatch control at the real
bottleneck Garrido left fixed. The claim is frontier-dependent learning:
neural control becomes valuable when the action space reaches the binding
downstream constraint.

Primary metric is Garrido/Excel ReT. CVaR/tail, Cobb-Douglas, service, backlog,
and resource are reported separately.

## Canonical Track B Baseline

- Action contract: `track_b_v1`
- Observation: `v7`
- Reward: `control_v1`
- Risk level: `adaptive_benchmark_v2`
- Horizon: `h104`
- Confirmatory scale: 5 seeds, 60k timesteps
- Baseline comparison: dense static frontier with CRN

Current confirmed result package:

```text
outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104/
```

The mechanism audit for this run is:

```text
outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104/mechanism_audit_v2/
```

## New v9 Observation Candidate

`v9 = v8 + 10 dims`:

- `backorder_queue_count_norm`
- `unattended_total_norm`
- `oldest_backorder_age_norm`
- `ewma_fill_rate`
- `ewma_backlog_growth`
- `delta_fill_rate`
- `delta_backlog_momentum`
- `prev_step_produced_norm`
- `prev_step_delivered_norm`
- `prev_step_available_assembly_hours_norm`

`v9` is a headroom candidate, not a replacement for the canonical `v7` claim
unless it wins under the same dense/CRN/rich-metric protocol.

## Reward/Observation Sweep

Use:

```bash
PYTHONUNBUFFERED=1 .venv/bin/python -u scripts/run_track_b_adaptive_sweep.py \
  --reward-modes control_v1,ReT_excel_plus_cvar,ReT_tail_v2,ReT_garrido2024_train \
  --observation-versions v7,v8,v9 \
  --cvar-alphas 0.05,0.1,0.2 \
  --seeds 1 2 \
  --train-timesteps 40000 \
  --eval-episodes 6 \
  --max-steps 104 \
  --n-envs 4 \
  --learning-rate 0.0001 \
  --output-dir outputs/experiments/track_b_adaptive_sweep_$(date -u +%Y%m%dT%H%M%SZ)
```

Promotion gate:

- Improve the current Excel ReT delta `+0.000415`, or
- Improve the current CVaR05 delta `+0.000506`, and
- Keep cost index `<= 0.70` unless tail improvement is very large.

Promote at most two candidates to confirmatory.

## Mechanism Audit

Use:

```bash
.venv/bin/python scripts/audit_track_b_mechanism.py <run_dir>
```

The claim is `adaptive recovery / backlog control` unless a future action-trace
lead/lag audit supports stronger anticipation language.

## Long-Run Discipline

Every long local or Kaggle run must have:

- payload/import preflight,
- unbuffered incremental logs,
- a live watcher that checks status and artifact existence,
- explicit error/finish alerting,
- CRN held-out eval seeds for final claims.

Do not claim a win from a watcher-only status. Trust live artifacts and direct
kernel status over stale watcher logs.
