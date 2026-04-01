# Repository Source of Truth

This note freezes the current paper-facing repository story. Treat it as the primary reference for README examples, manuscript edits, benchmark commands, and reviewer-facing explanations unless a later note explicitly supersedes it.

## Canonical scientific story

The repository now supports two paper-relevant benchmark families that must not be conflated:

- **Track A**: thesis-faithful upstream control only.
  The agent controls production/inventory decisions but not downstream transport.
  This family is the negative/diagnostic baseline.
- **Track B**: minimal MDP repair through downstream transport control at `Op10` and `Op12`.
  This family is the current positive result.

The paper contribution is therefore:

> On the thesis-faithful MFSC DES, RL does not beat strong static baselines when the action space controls the wrong bottleneck. Once the MDP is minimally repaired so the agent can act on the active downstream constraint, PPO produces clearly superior policies.

## Primary paper backbone

The current primary paper backbone is **Track B minimal**:

- Environment: `track_b_adaptive_control`
- Training reward: `ReT_seq_v1`
- Frozen `ret_seq_kappa`: `0.20`
- Observation contract: `v7`
- Action contract: `track_b_v1`
- Risk profile: `adaptive_benchmark_v2`
- Step size: `168` hours
- Year basis: `thesis`
- `stochastic_pt=True`
- Benchmark protocol: `track_b_minimal_v1`

Track A remains part of the paper, but as:

- a thesis-faithful comparator family,
- a negative result,
- and the mechanistic explanation for why Track B was introduced.

Interpretation rule:

- Within Track A and within Track B, `reward_total` is comparable only when `reward_mode` is shared.
- Across reward families, use `fill_rate`, `backorder_rate`, and `order_level_ret_mean`.
- Across Track A vs Track B, do **not** claim “same benchmark, better algorithm.” The correct claim is “different control contract, different MDP, different outcome.”

## Primary artifact bundles

The main auditable paper-facing artifact bundles are:

- `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1`
- `outputs/benchmarks/track_b_smoke_initial_2026-03-31`
- `outputs/paper_benchmarks/paper_ret_seq_k020_500k`
- `outputs/paper_benchmarks/paper_control_v1_500k`
- `outputs/benchmarks/final_recurrent_ppo_v4_control_500k`

Use them as follows:

- `track_b_ret_seq_k020_500k_rerun1`: primary positive result.
- `track_b_smoke_initial_2026-03-31`: preliminary Track B signal that motivated the long run.
- `paper_ret_seq_k020_500k` and `paper_control_v1_500k`: valid post-audit Track A comparators.
- `final_recurrent_ppo_v4_control_500k`: closes the “memory fixes Track A” hypothesis negatively.

Do **not** use as primary evidence:

- historical `control_reward_500k_*_stopt`
- mixed-family `section4_3_*`
- incomplete/failed Track B runs such as `track_b_ret_seq_k020_500k`

Current headline reading:

- Track A is closed as a valid negative benchmark family.
- Track B is the current repo winner.
- In `track_b_ret_seq_k020_500k_rerun1`, PPO beats `s2_d1.00` by `+3.40 pp` fill and beats the best static policy (`s3_d2.00`) by `+1.23 pp`.
- The Track B result is strong enough to support the manuscript’s main positive claim.

## Public entry points

Public entry points should now be described this way:

- `scripts/run_track_b_benchmark.py`: primary paper-facing launcher.
- `scripts/run_track_b_smoke.py`: short validation benchmark for the same Track B contract.
- `scripts/run_paper_benchmark.py`: Track A comparator launcher.
- `external_env_interface.make_track_b_env()`: primary research env constructor for the positive lane.
- `external_env_interface.make_shift_control_env()`: comparator env constructor for Track A.

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
- PBRS as the main claim
- DKANA / KAN / GNN as the main contribution
- raw Track A “RL fails” without the Track B follow-up
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
- `structural action-space limitation`
- `control contract`
- `MDP repair through downstream control`
- `Track A negative result`
- `Track B positive result`

Avoid:

- `PPO solves the problem`
- `statistically significant`
- `novel architecture contribution`
- `ReT_thesis is the main training reward`
- `Track A and Track B are directly interchangeable`
