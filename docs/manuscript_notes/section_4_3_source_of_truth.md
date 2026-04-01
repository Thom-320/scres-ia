# Section 4.3 Source of Truth

This note freezes the intended reading of the algorithm-comparison lane, but the artifact directories listed below are still pending rerun under the current post-audit DES. Treat this note as a plan, not as a citation source, until matched bundles exist.

## Purpose

Section 4.3 is the algorithmic and POMDP-aware comparison layer that sits on top of the frozen Section 4.2 benchmark backbone.

Section 4.2 answers:

- What happens under the paper-facing control benchmark?
- Is PPO + control_v1 competitive under increased stress and stronger under severe stress?

Section 4.3 answers:

- Does richer temporal context improve the learned policy?
- Is the current limitation mainly a memory / observability problem?
- How does the learned policy compare against a tuned OR-style heuristic?

## Frozen comparison family

The Section 4.3 comparison should be interpreted as a matched-budget algorithm comparison.

Core variants:

- `PPO + MLP`, `observation_version=v1`, `frame_stack=1`
- `PPO + MLP`, `observation_version=v2`, `frame_stack=1`
- `PPO + MLP`, `observation_version=v1`, `frame_stack=4`
- `RecurrentPPO + LSTM`, `observation_version=v2`, `frame_stack=1`

Common scenario backbone:

- `reward_mode=control_v1`
- `step_size_hours=168`
- `stochastic_pt=True`
- `w_bo=4.0`
- `w_cost=0.02`
- `w_disr=0.0`
- scenarios:
  - `increased`
  - `severe`

## Baseline discipline

The baseline hierarchy for this section is:

1. best fixed static policy
2. tuned heuristic baseline
3. learned policy variants

`random` may still appear in raw benchmark bundles, but it is not the main comparator for Section 4.3 prose.

## Claim discipline

Allowed language:

- `frame stacking provides low-cost temporal context`
- `RecurrentPPO tests the partial-observability hypothesis`
- `the tuned heuristic provides an OR-respectable baseline`
- `gains are regime-dependent`

Avoid:

- `RecurrentPPO solves the problem`
- `LSTM is universally better`
- `statistically significant`
- `DKANA is needed`

## Output artifacts expected

The run should produce:

- matched artifact bundles for the Section 4.3 variants
- one publication summary bundle for the algorithm-comparison table
- a baseline table
- an algorithm comparison table with seed-mean deltas vs best static and best heuristic

Planned directories:

- matched `section4_3` bundles for increased/severe PPO-MLP, PPO+frame-stack, and RecurrentPPO
- one publication summary bundle for the algorithm-comparison table

Audit status as of March 31, 2026:

- These planned `section4_3` bundles are not present as auditable directories in the current repo snapshot.
- Do not cite them as source-of-truth evidence until the matched reruns exist.

## Writing hook

Suggested prose anchor:

> Section 4.3 evaluates whether the limited gains of the baseline PPO-MLP policy are better explained by partial observability than by reward misalignment alone. We compare observation augmentation (v2), frame stacking, and recurrent policies against a tuned heuristic baseline under the same control-reward benchmark family.
