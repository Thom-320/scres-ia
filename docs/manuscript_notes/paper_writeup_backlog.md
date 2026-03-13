# Paper Writeup Backlog

This note fixes the writing order and manuscript language for the current paper
route.

## Main claim

The paper should be framed as:

> A rigorous DES+RL benchmark for resilient control in a military food supply
> chain, with explicit treatment of reward alignment, partial observability,
> and stress-regime dependent adaptive gains.

This is the main contribution. The paper is not about architectural novelty.

## Journal route

Primary target:

- `IJPR`

Secondary target:

- `IEEE TAI`

Stretch target for a stronger later version:

- `EJOR`

## Writing order

Write in this order:

1. Section `4.2`:
   - frozen benchmark results
   - `PPO + control_v1`
   - interpretation by stress regime
2. Section `4.3`:
   - algorithm comparison
   - `PPO + MLP` vs `frame_stack` vs `RecurrentPPO`
   - tuned heuristic baseline
3. Section `5`:
   - discussion
   - limitations
   - future work

Do not spend time rewriting the introduction again before those sections exist.

## Required content by section

### Section 4.2

Must include:

- the benchmark backbone
- why `control_v1` is the training reward
- why `ReT_thesis` is reporting-only
- the `increased + stochastic_pt` result
- the `severe + stochastic_pt` result
- the statement that gains are regime-dependent, not uniformly dominant

### Section 4.3

Must include:

- why partial observability matters
- `frame_stack` as low-cost temporal context
- `RecurrentPPO` as the main POMDP-aware upgrade
- tuned heuristic baseline as an OR-respectable comparator
- if available, comparison against `SAC` as a secondary algorithmic check

### Section 5

Must include:

- limitations of the observed state
- why the current formulation is better described as a `POMDP`
- reward-design limitations and why `control_v1` was used
- why `DKANA`, `KAN`, and `GNN` were deferred
- future work path: `PBRS`, richer recurrent policies, and architecture studies

## Language rules

### Allowed language

- `POMDP-style control`
- `augmented observed state`
- `stress-regime dependent gains`
- `reporting-only resilience metric`
- `adaptive control advantage under severe stress`
- `competitive under moderate stress`

### Forbidden language

- `Markov property proven`
- `DKANA necessary`
- `PPO solves the problem`
- `novel architecture contribution`
- `statistically significant`

unless the inference section is intentionally expanded later.

## Figures and tables that must exist

Minimum paper set:

- one baseline table with `static`, `random`, heuristic, and learned policies
- one main results table for `increased` and `severe`
- one figure showing shift-mix behavior of learned vs static baselines
- one figure showing learning or comparative performance across algorithmic
  variants
- one short ablation table for `v1`, `v2`, `frame_stack`, and `RecurrentPPO`

## Future work wording

Use future work language like:

> A natural extension is to evaluate richer policy classes, including
> architecture-specific sequence models, against the frozen benchmark backbone
> established in this work.

Do not promise:

- `DKANA results`
- `GNN superiority`
- `KAN interpretability claims`

unless those results actually exist in the benchmark family.
