# Repository Source of Truth

This note freezes the current paper-facing repository story. Treat it as the primary reference for README examples, manuscript edits, benchmark commands, and reviewer-facing explanations unless a later note explicitly supersedes it.

## Canonical scientific story

The repository supports three distinct roles that must not be conflated:

- `ReT_seq_v1`: primary training reward for the paper-facing benchmark family.
- `control_v1` / `control_v1_pbrs`: historical operational comparators retained for legacy comparison.
- `ReT_thesis` / `ret_thesis_corrected_step`: thesis-aligned resilience metrics for reporting and audit.
- `rt_v0`: historical baseline retained for methodological comparison.

The paper contribution is therefore:

> A rigorous DES+RL benchmark for resilient control in a military food supply chain, with explicit treatment of reward alignment, partial observability, and stress-regime-dependent adaptive gains.

## Frozen benchmark backbone

Unless a new benchmark family is intentionally introduced, the current paper backbone is:

- Environment: `shift_control`
- Training reward: `ReT_seq_v1`
- Frozen `ret_seq_kappa`: `0.20`
- Historical comparator: `control_v1`
- Reporting resilience metric: `ReT_thesis` / `ret_thesis_corrected_step`
- Step size: `168` hours
- Year basis: `thesis`
- Benchmark observation version: `v1`
- Main scenarios:
  - `increased + stochastic_pt=True`
  - `severe + stochastic_pt=True`
- Official thesis-validation basis: `year_basis="thesis"`
- Gregorian annualization may still appear in diagnostics, but thesis-facing
  comparisons should use the thesis basis unless explicitly stated otherwise.
- Frozen paper-facing weights and resilience settings:
  - `w_bo = 4.0`
  - `w_cost = 0.02`
  - `w_disr = 0.0`
  - `ret_seq_kappa = 0.20`

Interpretation rule:

- `v1` remains the frozen benchmark contract for comparability with the existing 500k artifact bundles.
- `v2` is the preferred next-step observation contract for new ablations (`frame_stack`, `RecurrentPPO`, richer temporal context).
- Cross-mode reward totals remain non-comparable. Use `fill_rate`, `backorder_rate`, and `order_level_ret_mean` for `control_v1` vs `ReT_seq_v1` comparisons.

## Primary artifact bundles

The main auditable benchmark artifacts are:

- `outputs/paper_benchmarks/paper_ret_seq_k020_500k`
- `outputs/paper_benchmarks/paper_ret_seq_k010_500k`
- `outputs/paper_benchmarks/paper_control_v1_500k`
- `outputs/benchmarks/final_ret_seq_v1_500k`

Historical `control_reward_500k_*_stopt` bundles and the old seed-inference note remain useful only as legacy context. They were generated before the March 2026 DES audit/alignment fixes and must not be used as the primary evidence for the current repository state.

Current headline reading:

- The paper-trio comparison currently selects `ReT_seq_v1` with `κ=0.20` as the pragmatic leader against `static_s2` on cross-mode comparable metrics.
- `paper_control_v1_500k` remains the valid operational comparator for the current repo, but it is not the leading lane.
- `κ=0.10` remains a conservative ablation, not the repo default.
- `κ=0.30` is not a candidate default because it trends toward collapse-prone shift behavior.
- `final_ret_seq_v1_500k` is an auditable post-audit comparator, but it uses `year_basis="gregorian"` and should not be conflated with the thesis-basis paper bundle family.
- These results remain benchmark evidence, not a claim of universal superiority; use cautious inferential language.

## Public defaults

Public entry points should align with the benchmark story:

- `train_agent.py` default shift-control reward: `ReT_seq_v1`
- `train_agent.py` default `ret_seq_kappa`: `0.20`
- `train_agent.py` default observation version: `v1`
- `external_env_interface.make_shift_control_env()` default reward: `ReT_seq_v1`
- `external_env_interface.make_shift_control_env()` default observation version: `v1`

## ReT-Seq mapping

The current primary reward contract should be described as a sequential extension of Garrido-Rios (2017) Eq. 5.5:

- `SC_t` maps to `Re(FR_t)` from Eq. 5.4 and captures step-level service continuity.
- `BC_t` is the sequential recovery proxy tied to the recovery idea in Eq. 5.2 through pending backorders relative to cumulative demand.
- `AE_t` is the explicit cost-efficiency extension motivated by thesis Section 8.6.2, which calls for an optimum SCRes level that includes cost.
- Geometric aggregation is intentional because it reduces compensability across service, recovery, and efficiency dimensions.

## What is not the main paper lane

The following are valuable but secondary:

- `ReT_thesis` as the primary training reward
- `control_v1` as the primary training reward
- PBRS as the main claim (phase-2 extension only)
- DKANA / KAN / GNN as the main contribution
- `severe_training` as the main reported scenario

## Document hierarchy

Use the following hierarchy when documents disagree:

1. `docs/REPOSITORY_SOURCE_OF_TRUTH.md`
2. `docs/manuscript_notes/control_reward_500k_source_of_truth.md`
3. `docs/manuscript_notes/paper_strategy_decision_memo.md`
4. `docs/manuscript_notes/paper_writeup_backlog.md`
5. Historical reports and meeting notes

## Required language discipline

Preferred phrases:

- `POMDP-style control`
- `reporting-only resilience metric`
- `stress-regime-dependent gains`
- `competitive under moderate stress`
- `stronger under severe stress`

Avoid:

- `PPO solves the problem`
- `statistically significant`
- `novel architecture contribution`
- `ReT_thesis is the main training reward`
