# Consolidated Literature Strategy

This note consolidates the literature-facing strategy for the current MFSC
manuscript lane so the repo, manuscript drafts, and presentation material all
use the same story.

## Core contribution

The paper should be positioned as a benchmark-and-explanation contribution, not
as an architecture paper.

The defensible headline is:

> On the thesis-faithful MFSC DES, RL does not beat strong static baselines
> when the action space controls the wrong bottleneck. Once the control
> contract is minimally repaired so the agent can act on the active downstream
> transport constraint, PPO learns clearly superior policies.

This yields a dual contribution:

- `Track A` is a scientifically useful negative result because the failure is
  explained structurally rather than hidden.
- `Track B` is the positive result showing where adaptive control has real
  leverage in this benchmark.

## Literature buckets to cover

### 1. Garrido-Rios thesis as the benchmark foundation

Use Garrido-Rios (2017) as the operational source of truth for:

- DES structure
- parameter extraction
- throughput validation targets
- Eq. 5.5 resilience framing
- interpretation of the military food-supply operations

Repo evidence:

- `run_static.py`
- `validation_report.py`
- `outputs/validation/validation_table_dual_basis.csv`

### 2. Supply-chain resilience measurement

The manuscript should distinguish clearly between:

- resilience metrics used for reporting and audit
- reward functions used for RL training

`ReT_thesis` and `ret_thesis_corrected` belong in the measurement discussion.
`ReT_seq_v1` belongs in the training-reward discussion as a sequential,
trainable extension motivated by the thesis resilience logic.

The point is not that the thesis metric was "wrong". The point is that a
reporting metric and a training signal solve different problems.

### 3. RL for supply-chain control

Position this repo against a common pattern in RL-for-operations papers:

- many papers publish only the successful RL case
- fewer papers publish a matched negative result
- fewer still isolate the mechanism with a contract-level ablation

The distinctive value here is the combination of:

- validated DES benchmark
- strong static comparators
- negative `Track A` result
- positive `Track B` result
- matched 5D vs 7D ablation showing that downstream control matters

### 4. POMDP-style control and benchmark design

The weekly controller should be framed as a `POMDP` or augmented observed-state
control problem, not as a proved fully observable MDP.

The main methodological point is:

- control-contract design matters
- observability matters
- reward alignment matters
- algorithm choice is secondary to those structural choices

## Evidence map for the paper

Use the following artifacts as the main evidentiary backbone:

- DES validation:
  `outputs/validation/validation_table_dual_basis.csv`
- `Track A` negative family:
  `outputs/paper_benchmarks/paper_ret_seq_k020_500k`
- historical operational comparator:
  `outputs/paper_benchmarks/paper_control_v1_500k`
- `Track B` positive family:
  `outputs/track_b_benchmarks/track_b_ret_seq_k020_500k_rerun1`
- matched causality ablation:
  `outputs/track_b_ablation_5d_vs_7d.json`

## Reviewer-facing positioning

This story is strong when emphasizing:

- benchmark rigor
- operational realism
- negative-result honesty
- mechanistic explanation
- auditable reproducibility

It is weak when emphasizing:

- architectural novelty
- universal RL superiority claims
- cross-family reward comparisons as if they were directly interchangeable

## Claim discipline

Prefer:

- `Track A negative result`
- `Track B positive result`
- `structural action-space limitation`
- `MDP repair through downstream control`
- `reporting-only resilience metric`
- `POMDP-style control`

Avoid:

- `PPO solves the problem`
- `downstream control alone proves everything`
- `same benchmark, better algorithm`
- `statistically significant` without careful qualification

## Journal route

Current route:

- primary: `IJPR`
- stretch: `EJOR`

The repo already has enough science for submission. The remaining work is
consolidation:

- keep README/docs aligned with the frozen source of truth
- keep benchmark commands reproducible
- remove avoidable repo noise before merge to `main`
