# Backbone Audit Notes (2026-03-25)

Key finding:

- The recent low-fill results are not explained by `v4`, `frame_stack`, PPO, or `ReT_seq_v1` alone.
- In HEAD, direct evaluation of `static_s2` reproduces the low-fill regime even with `v1` and even when `year_basis` is toggled between `thesis` and `gregorian`.
- That means the primary suspect is backbone drift in the DES / environment implementation relative to the frozen `control_reward_500k_increased_stopt` artifact.

Metric correction (same day):

- The benchmark helpers were also mixing two different service definitions under the same `fill_rate` label.
- `run_episodes()` and `benchmark_control_reward.py` were reconstructing episode service from cumulative step flows (`sum(new_backorder_qty) / sum(new_demanded)`), while the simulator state and thesis-facing interpretation use terminal/order-level service.
- This is now fixed:
  - `fill_rate` / `backorder_rate` are paper-facing terminal metrics from the DES
  - `flow_fill_rate` / `flow_backorder_rate` are retained separately for audit only
- Consequence:
  - old benchmark bundles should be treated as historical artifacts unless rerun under the corrected metric contract.

Working rule:

- Treat the frozen `500k` benchmark as a historical artifact, not as proof that HEAD should still produce the same static baseline.
- Before making new paper-facing claims about `ReT_seq_v1`, locate the first commit where `static_s2` drops from the `~0.84` regime to the `~0.42` regime.

Reproducible audit command:

```bash
python scripts/audit_backbone_regression.py --stochastic-pt
```

Outputs:

- `outputs/audits/backbone_regression/commit_backbone_regression.csv`
- `outputs/audits/backbone_regression/summary.json`
- `outputs/audits/backbone_regression/audit_report.md`

Paper-facing implication:

- Do not compare runs across different backbone tuples.
- At minimum, every benchmark comparison must freeze:
  - git commit
  - observation version
  - frame stack
  - year basis
  - risk level
  - stochastic PT flag
