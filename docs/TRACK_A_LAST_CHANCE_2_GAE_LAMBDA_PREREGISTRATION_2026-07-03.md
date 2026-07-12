# Track A Last-Chance #2 Preregistration — GAE lambda retune (2026-07-03)

## Context

Two prior interventions failed the promotion threshold (beat held-out best static AND >=4/5
positive seeds), both under `docs/TRACK_A_LAST_CHANCE_PREREGISTRATION_2026-07-03.md`:

- Plain BC+PPO: 0/5 seeds, delta -0.006316.
- Critic-pretrained BC+PPO (LC1a): 0/5 seeds, delta -0.007127 (slightly worse).

The reward-noise audit (2026-07-03, reconciled between this session and Codex's independent audit)
ruled out retroactive rescoring as a bug, but surfaced a real, different mechanism: a meaningful
share of any given week's reward comes from orders placed in *earlier* weeks finally resolving —
delayed, multi-step credit (RPj p50 ~40h, p90 ~500-1400h; at a 168h step size that is roughly 0-8
steps of lag).

## Hypothesis

PPO's default GAE (`gae_lambda=0.95`) leans on the value function to bootstrap multi-step returns.
If the value function's estimate of "reward still owed from orders already in flight" is imprecise
(plausible given `critic-pretrain` only fit it on teacher rollouts, not on-policy data), GAE at
lambda=0.95 could produce noisy/biased advantage estimates specifically for states with many
in-flight orders — exactly the states that matter most for this delayed-credit structure. Raising
`gae_lambda` toward 1.0 shifts advantage estimation toward actual observed multi-step returns
(higher variance, lower bias, less dependent on value-function accuracy).

## Bias guardrail (same discipline as before)

- Primary metric: Garrido/Excel ReT (`ret_excel`), not reward total.
- Comparator: held-out best static from the same Track A v2 conservation gate.
- Promotion requires: PPO beats held-out best static on mean Excel ReT AND >=4/5 positive seed
  deltas. Anything less is a negative result, reported as such.

## Registered intervention

Single deliberate variable change vs the ORIGINAL (non-critic-pretrained) baseline run, so this is a
clean one-variable test, not stacked on top of the already-refuted critic-pretrain intervention:

- `--gae-lambda 0.98` (new flag, was hardcoded to SB3 default 0.95).
- `--gamma 0.99` (new flag, explicit even though it matches SB3's default — kept unchanged; the
  lag is mostly within ~8 steps, already well inside gamma=0.99's effective horizon of ~100 steps).
- `--critic-pretrain-epochs 0` (disabled) — isolates this test from the already-tested, refuted
  critic-lag hypothesis.
- All other hyperparameters identical to the original run (seeds 1-5, 40k timesteps, bc-epochs 150,
  lr 1e-4, clip 0.1, target-kl 0.02, teacher oracle_if_better).

## Rule

If this fails the promotion threshold too, Track A closes for this investigation cycle without
further retunes. Three independent, well-reasoned interventions (plain, critic-pretrain, GAE-lambda)
failing to convert confirmed static-oracle headroom into a PPO win is strong evidence the headroom is
not learnable by standard PPO on this contract, not that we haven't found the right knob yet.
