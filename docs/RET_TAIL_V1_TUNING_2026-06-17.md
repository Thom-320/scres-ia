# ReT_tail_v1 Tuning Result - 2026-06-17

## Purpose

`ReT_tail_v1` replaces the Track A training reward that was too aligned with
mean service performance.  The acceptance rule is pre-training and static-only:
if a reward chooses a bad static policy on tail resilience, it must not be used
to train PPO.

## Backbone

- Risk occurrence: `thesis_periodic`
- Raw material flow: `kit_equivalent_order_up_to`
- Order-up-to multiplier: `2.0`
- Policy surface: `with_crossed`
- Profiles: `increased,severe`
- Panel: `Cf31-90`
- Replications: `1`
- Primary external metrics: `ret_p10_all`, `flow_fill_rate`,
  `stockout_week_pct`

## Search

The broad scout used `scripts/tune_ret_tail_reward.py` over recovery-dominant
Cobb-Douglas candidates.  The top three scout candidates were then confirmed on
the full panel with:

```bash
.venv/bin/python scripts/tune_ret_tail_reward.py \
  --candidates '0.30:0.60:0.10:0.40:0.25:0.0;0.10:0.80:0.10:0.10:0.25:1.0;0.20:0.70:0.10:0.10:0.25:2.0' \
  --panel-cfis 31-90 \
  --profiles increased,severe \
  --policy-set with_crossed \
  --replications 1
```

Full-panel output:

`outputs/benchmarks/ret_tail_reward_tuning/ret_tail_tuning_20260617T200729Z/ret_tail_tuning_summary.csv`

## Selected Defaults

The only candidate that passed both profiles was:

| parameter | value |
|---|---:|
| `ret_tail_w_sc` | 0.30 |
| `ret_tail_w_rc` | 0.60 |
| `ret_tail_w_ce` | 0.10 |
| `ret_tail_cap_kappa` | 0.40 |
| `ret_tail_inv_kappa` | 0.25 |
| `ret_tail_boost` | 0.00 |

Full-panel diagnostics for the selected candidate:

| profile | best-by-reward | p10 | p10 rank | top p10 | flow ratio | PASS |
|---|---|---:|---:|---:|---:|---|
| `increased` | `crossed_uniform_I168_S1` | 0.6834 | 3 | 0.7011 | 0.9866 | true |
| `severe` | `crossed_uniform_I168_S1` | 0.6667 | 1 | 0.6667 | 0.9972 | true |

The rejected finalists both selected `crossed_uniform_I168_S3` under
`increased`, which ranked 15th by `ret_p10_all`.  That means the extra recovery
boost / lower capacity cost over-pushed shifts without improving the tail.

## Interpretation

The selected reward does not maximize raw inventory or shifts.  It picks a
low-inventory, low-shift policy (`I168_S1`) that remains close to the best
available tail resilience on the full static surface.  This is the required
gate before running PPO/RecurrentPPO/DMLPA: the training reward now points
toward the same tail objective used for evaluation.

Training runs must still be judged by external metrics, not by `reward_total`.
