# KAN Last-Chance Rationale -- 2026-07-03

## Reviewer-facing question

Garrido's concern is not only whether PPO+MLP works, but whether the architecture is
novel and justified. The cleanest answer is not to replace the whole method with a
different RL algorithm. It is to keep the RL algorithm fixed (PPO) and swap only the
function approximator:

- PPO+MLP: canonical, conservative baseline.
- PPO+DMLPA/history: David's transformer-over-history proposal.
- PPO+KAN: Garrido's Kolmogorov-Arnold-network proposal.

That isolates architecture. If PPO+KAN wins, the architecture matters. If PPO+KAN
does not win, the paper's current interpretation becomes stronger: Track B's value
comes primarily from action-space alignment with the downstream bottleneck, not from
a special network class.

## Why real KAN, not the older sidecar

The older `RBFKANFeaturesExtractor` is useful but not definitive because it includes
a full linear skip path. It can show that a KAN-style basis does not break the result,
but it cannot show that a real KAN is necessary or superior.

The last-chance test therefore uses `scripts/real_kan_extractor.py`, which wraps the
official `kan.KAN` class from `pykan` as an SB3 `BaseFeaturesExtractor`. This is the
literal KAN mechanism: learnable B-spline edge functions rather than fixed node
activations.

## Algorithm choice

Use PPO for the KAN sidecar. Do not switch to SAC/TD3 in this test. Changing both
algorithm and architecture would answer a different question and would make the
result harder to interpret. PPO is intentionally held constant so the comparison is:

> Given the same Track B environment and PPO optimization loop, does a KAN policy/value
> approximator outperform the MLP policy/value approximator?

## Current run

Canonical last-chance run:

`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_3seed_30k_h104/`

Protocol:

- Official `pykan` KAN via `RealKANFeaturesExtractor`
- `track_b_v1`, `v7`, `adaptive_benchmark_v2`, `control_v1`
- seeds `1..3`
- `train_timesteps=30000`
- `eval_episodes=12`
- `max_steps=104`

This is a small sidecar, not a Paper 1 gate. It is enough to answer whether real KAN
is viable in our corrected Track B loop and whether it is obviously superior. If it is
positive or close to PPO+MLP, then a 5-seed/60k confirmation is the next step before
any manuscript-facing claim.

## Decision rule

- If real KAN clearly beats PPO+MLP on Garrido/Excel ReT and does so consistently by
  seed, promote to a confirmatory 5-seed/60k architecture run.
- If real KAN beats statics but does not beat PPO+MLP, keep it as architecture
  robustness evidence.
- If real KAN fails to beat statics, document it honestly and keep PPO+MLP as the
  justified architecture for Paper 1.
