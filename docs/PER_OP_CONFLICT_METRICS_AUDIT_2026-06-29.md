# Per-Op Conflict Campaign Metrics Audit

Date: 2026-06-29

## Context

The user asked for a broader metrics panel before promoting the per-op conflict
campaign signal. The prior 3-seed run showed a preliminary raw ReT/Pareto win,
but it only reported Excel ReT, CVaR service loss, and resource. I expanded
`scripts/run_per_op_conflict_campaign.py` to write:

- `metrics_panel.csv`
- `static_frontier_metrics.csv`
- full-static-frontier Pareto verdicts in `summary.json`

The richer rerun uses the same leading setup:

```bash
scripts/run_per_op_conflict_campaign.py \
  --gate-dir outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29 \
  --reward-mode ReT_excel_plus_cvar \
  --cvar-alpha 0.1 \
  --seeds 1,2,3 \
  --timesteps 40000 \
  --bc-epochs 50 \
  --n-envs 4 \
  --max-steps 52 \
  --eval-seed0 9000 \
  --holding-cost 0.0 \
  --eval-full-static-frontier
```

Artifact:
`outputs/experiments/per_op_conflict_campaign_pluscvar_a01_hc0_bc50_3seed_richmetrics_2026-06-29`

## Verdict

The richer 3-seed rerun does **not** confirm the preliminary win.

| policy | Excel ReT | CVaR service loss | resource | verdict |
|---|---:|---:|---:|---|
| Dynamic PPO | 0.151200 | 1.847e9 | 0.342 | loses |
| Robust static from gate | 0.154676 | 1.944e9 | 0.250 | dynamic beats CVaR only |
| Best full-grid static | 0.155254 | 1.842e9 | 0.267 | dominates dynamic |
| Best static at <= dynamic resource | 0.155254 | 1.842e9 | 0.267 | dominates dynamic |

Full-static verdict:

- Raw ReT win: `false`
- Pareto non-dominated: `false`
- Dominated by full-grid statics: `91`
- Dominates full-grid statics: `0`

## Broader Metrics

| metric | dynamic | best full-grid static | direction |
|---|---:|---:|---|
| Excel ReT | 0.151200 | 0.155254 | static better |
| Thesis ReT | 0.054271 | 0.055178 | static better |
| Continuous ReT | 0.395765 | 0.399762 | static better |
| flow_fill_rate | 0.916839 | 0.931099 | static better |
| lost_rate | 0.041904 | 0.039530 | static better |
| lost_orders | 13.07 | 12.33 | static better |
| backorder_qty_final | 50,021.85 | 39,929.22 | static better |
| service_loss_auc_per_order | 1,950,439.90 | 1,754,640.54 | static better |
| CVaR95 service loss | 1.470e9 | 1.433e9 | static better |
| ttr_mean | 184.48 | 169.25 | static better |
| CTj p90 | 1496.89 | 1405.33 | static better |
| CTj p99 | 4807.11 | 4400.00 | static better |
| RPj p90 | 561.91 | 513.91 | static better |
| RPj p99 | 1472.02 | 1300.47 | static better |
| DPj p99 | 5288.89 | 4832.00 | static better |
| delivered_rations | 767,983 | 779,369 | static better |
| demanded_rations | 851,043 | 851,090 | matched |
| resource | 0.342 | 0.267 | static better |

The dynamic still beats the weaker robust static on service-loss CVaR, but the
best full-grid static matches or beats it on essentially every important metric.

## Interpretation

The per-op conflict campaign remains scientifically useful because it revealed a
fragile, seed-dependent PPO signal and forced a richer full-grid audit. However,
with the current reward/BC/PPO setup it is **not yet a claim**. The correct
statement is:

> Per-op conflict plus BC warm-start produced an initial raw-ReT signal, but the
> signal did not survive a richer 3-seed metrics audit against the full-grid
> static frontier.

## Next Options

Only two Track-A follow-ups are still rational:

1. Improve PPO stability specifically, then repeat the rich metrics audit:
   stronger BC, lower learning rate, KL constraint, lower entropy after BC,
   and action penalty around the oracle table.
2. Stop Track A for claims and move weight to H4 retained-vs-reset or Track B,
   because the best full-grid static continues to absorb the available headroom.

