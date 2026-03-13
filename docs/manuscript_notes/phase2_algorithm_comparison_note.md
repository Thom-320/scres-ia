# Phase 2 Algorithm Comparison Note

Use this note to keep manuscript and meeting language aligned during the
algorithm-comparison phase.

## Current baseline reading

- `PPO + MLP` is not a failed baseline.
- It already shows a limited but real learning signal under the frozen
  `control_v1` benchmark.
- Under `increased + stochastic_pt`, PPO is competitive with the best static
  baseline but not superior in reward.
- Under `severe + stochastic_pt`, PPO achieves a reward advantage over the best
  fixed baseline while maintaining comparable service.

## Claim discipline

- Do not claim that an `MLP` "never converges."
- Do say that the current `PPO + MLP` baseline is not yet strong enough to
  settle the problem robustly across regimes.
- Frame the next step as a controlled comparison:
  - `SAC` tests whether the current limitation is algorithmic.
  - `PPO + frame stacking` tests whether short temporal context is missing.
  - `DKANA` is justified only if a richer sequential representation is still
    needed after those two comparisons.

## Frozen benchmark backbone

- `reward_mode="control_v1"`
- `risk_level="increased"` with `stochastic_pt=True`
- `risk_level="severe"` with `stochastic_pt=True`
- `ReT_thesis` remains reporting-only

## Decision rule for DKANA

- If `SAC` clearly beats `PPO` under `severe`, prioritize `SAC` as the stronger
  baseline and defer DKANA.
- If `frame_stack` improves PPO materially, interpret that as evidence that
  temporal context matters and use that to motivate DKANA.
- If neither improves materially, DKANA becomes a more justified structured
  sequence-model hypothesis rather than a rescue plan.
