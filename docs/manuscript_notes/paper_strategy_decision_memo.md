# Paper Strategy Decision Memo

This note freezes the paper strategy for the current manuscript cycle. Treat it
as the source of truth for implementation priorities, manuscript framing, and
journal targeting unless a later note explicitly supersedes it.

## Decision

- Do not pursue architectural novelty as the central contribution.
- Optimize for a fast, defensible submission rather than a prestige-maximizing,
  high-risk route.
- Describe the weekly control problem as a `POMDP` or augmented observed-state
  control problem, not as a fully sufficient MDP.
- Keep `control_v1` as the base training reward for the main benchmark lane.
- Position `PBRS` as a phase-2 methodological upgrade, not as a blocker for the
  main paper route.
- Move `DKANA`, `KAN`, and `GNN` out of the main roadmap and into future work
  unless they later beat the frozen benchmark with credible evidence.

## What is already resolved in the repo

The current repo already supports the backbone of the paper story:

- The benchmark lane includes `static_s1`, `static_s2`, `static_s3`, `random`,
  learned policy evaluation, `frame_stack`, `observation_version`, manifests,
  and auditable artifacts.
- The reward contract and state-action-transition framing of the shift-control
  lane are already documented.
- The current benchmark reading is already frozen: `PPO + control_v1` is
  competitive under `increased + stochastic_pt` and stronger under
  `severe + stochastic_pt`, without supporting a claim of uniform dominance.

Supporting notes:

- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/briefs/shift_control_audit_note.md`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/briefs/preliminary_results_synthesis.md`
- `/Users/thom/Desktop/Universidad_Codigo/proyecto_grarrido_scres+ia/docs/manuscript_notes/phase2_algorithm_comparison_note.md`

## Official roadmap

The main roadmap is fixed in this order:

1. Implement `RecurrentPPO` via `sb3-contrib`.
2. Add a tuned heuristic baseline that an OR reviewer would recognize as a real
   comparator.
3. Expand the benchmark from `3-5` seeds to `10` reporting seeds.
4. Run controlled comparisons under:
   - `increased + stochastic_pt`
   - `severe + stochastic_pt`
5. Write Sections `4.2`, `4.3`, and `5` from that evidence.
6. Treat `PBRS` as an optional phase-2 extension that strengthens the paper but
   does not block the main submission route.

## What is explicitly out of scope for the main paper

- Any paper plan that depends on `DKANA` to be submission-ready
- Claims of novel neural architecture as the main scientific contribution
- Reopening the base reward discussion before the `RecurrentPPO` comparison is
  run
- Continuing to produce strategy documents instead of converting the current
  benchmark lane into results and manuscript prose

## Journal route

Primary target:

- `IJPR`

Secondary target:

- `IEEE TAI`

Stretch goal, only if the methodology and inference become stronger later:

- `EJOR`

## Preferred claim language

Use language like:

> The contribution of this paper is not architectural novelty, but a rigorous
> DES+RL control benchmark for a military food supply chain, with explicit
> treatment of reward alignment, partial observability, and stress-regime
> dependent adaptive gains.

Do not use language like:

- `DKANA is required`
- `PPO solves the problem`
- `Markov property is proven`
- `novel architecture is the main contribution`

## Default assumptions

- Priority: acceptance-fast and defensible
- Base benchmark reward: `control_v1`
- Problem formulation: `POMDP`
- `PBRS`: phase-2 extension
- `DKANA`: future work unless later evidence changes that decision
