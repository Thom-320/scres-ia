# E3 h104 per-cell dense frontier — verdict (2026-07-02)

**Status: DONE, replicated.** Two independent runs on the OVH VPS:
- `outputs/experiments/track_b_e3_dense_frontier_2026-07-02/` (Claude; CRN
  plan replayed from the E3 PPO ledgers, 147 statics x 60 pairs per cell)
- `outputs/experiments/track_b_e3_h104_per_cell_dense_frontier_2026-07-02/`
  (Codex; same protocol, independently launched)

Both re-optimize the FULL 147-cell dense downstream-dispatch static grid
per cell (previously the E3 matrix used a fixed 9-cell static set — the
last "lighter comparator" reviewer objection for the transfer claims).

## Result (Claude run, CRN-paired to the E3 PPO ledgers; seed-clustered)

| Cell | Best dense static | PPO | Δ order-level ReT | CI95 | Seeds + |
|---|---|---:|---:|---|---|
| current/h104 | S3_op10_2.00_op12_2.00 (0.005439) | 0.005648 | +0.000209 | [+0.000171, +0.000247] | 5/5 |
| increased/h104 | S3_op10_1.00_op12_2.00 (0.003118) | 0.003660 | +0.000542 | [+0.000500, +0.000584] | 5/5 |

Verdicts (both cells, both runs): raw ReT win TRUE, tail (CVaR05) win
TRUE, Pareto ReT-cost TRUE. Codex's duplicate reproduces both wins with a
different eval plan (best-static identity shifts within noise:
S3_2.00_1.50 / S2_2.00_1.25; deltas +0.000244 / +0.000623) — the
conclusion is plan-invariant.

## What changes in the paper

The frozen policy's transfer to current and increased risk at the
canonical horizon is now validated against fully re-optimized per-cell
dense frontiers, not just the fixed 9-cell stress set. The remaining
fixed-comparator caveat applies only to the h52 cells and the severe
service-floor cells. Deltas are essentially unchanged vs the 9-cell
convention (+0.000209 → +0.000209; +0.000552 → +0.000542), which is
itself evidence the 9-cell screen was not inflating transfer claims.

Registry: upgrade C11's h104 rows to "dense-frontier confirmed."
