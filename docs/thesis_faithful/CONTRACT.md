# Thesis-Faithful Lane Contract

This lane exists to reproduce Garrido-Rios (2017) before training or comparing
RL policies. It is intentionally separate from the paper-facing RL benchmark.

## Protocol

- Protocol name: `thesis_1to1`
- Launcher: `python scripts/run_thesis_faithful.py`
- Output root: `outputs/thesis_faithful/`
- Source of truth: `WRAP_Theses_Garrido_Rios_2017.pdf` and `thesis.txt`
- Annualization: `year_basis="thesis"` (`8,064` hours/year)
- Horizon: `161,280` hours
- Warm-up trigger: first `Q=5,000` batch arrival at Op9
- Default downstream Q source: Figure 6.2 / Section 6.3.3 (`2,400-2,600`)
- Alternate downstream Q source: Table 6.20 (`2,000-2,500`)
- R14 defect mode: `thesis_strict_op6`
- Factorial launcher: `python scripts/run_thesis_factorial.py`

## Explicit Separation

`thesis_1to1` must not use:

- Gymnasium wrappers
- RL rewards or reward shaping
- post-warmup priming
- normalized action multipliers
- Gregorian annualization
- adaptive benchmark regimes

`paper_benchmark` may continue to use those mechanisms, but its outputs cannot
be described as a strict thesis reproduction.

## Gate Order

1. Run deterministic `Cf0` and compare post-warm-up annual delivery against
   Table 6.10 ECS.
2. Run at least one stochastic current-risk configuration and export order-level
   `OPTj/OATj/CTj/LTj`, backorder, unattended, and risk-event traces.
3. Run Cf1-Cf90 through the factorial launcher and analyze the replicated
   Apriori/Kruskal-Wallis/Wilcoxon evidence tables.
4. Run the no-training Gym static gate for `garrido_cf_s1/s2/s3`.
5. Only after these gates are documented, compare RL or AI models under the
   same audited backbone.

## Canonical Smoke Command

```bash
python scripts/run_thesis_faithful.py --label smoke_cf0 --scenario cf0
```

The launcher writes `command.txt`, `pid.txt`, `status.json`, `manifest.json`,
`summary.json`, one `orders_seed_*.csv`, and one `risk_events_seed_*.csv`.

## Factorial Smoke Command

```bash
python scripts/run_thesis_factorial.py --cfi 1 47 85 --dry-run --label smoke_factorial
```

The factorial launcher maps Cf1-Cf90 to thesis risk level matrices, inventory
periods, capacity shifts, risk subsets, and horizons. Use `--dry-run` to audit
the design matrix before running stochastic replications.

## Trainable Thesis-Aligned Gym Lane

The trainable lane is deliberately named `thesis_aligned_training`, not
`thesis_1to1`.

- Factory: `make_thesis_aligned_training_env()`
- Static gate: `python scripts/run_thesis_aligned_static_gate.py`
- Output root: `outputs/thesis_aligned_training/`
- Defaults: `year_basis="thesis"`, `warmup_trigger="op9_arrival"`,
  `downstream_q_source="figure_6_2"`, `r14_defect_mode="thesis_strict_op6"`,
  `priming_enabled=False`
- Action space: RL extension with 5 dimensions; use direct DES action dicts
  for exact static Garrido baselines.

This is the right environment for PPO/SAC/RecurrentPPO after the DES gate. It is
not a strict thesis reproduction because the agent can make dynamic decisions
that Garrido's scenario design did not allow.
