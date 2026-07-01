# PPO Discovery Failure Audit: CF20 Fine-Discrete Track A (2026-06-29)

## Question

Why does PPO fail to discover `f0.075_S1` in CF20, even though the reward surface ranks it as the best Excel-ReT policy?

## Verification: action application is correct

The fine-discrete wrapper was checked directly.

Requested action:

```text
f0.075_S1
```

DES info after step:

```text
fine_discrete_frac      = 0.075
fine_discrete_shift     = 1
continuous_its_frac     = 0.075
continuous_its_shift    = 1
resource_composite      = 0.0375
```

So the action translation is correct. This is not a bug where PPO chooses S1 and the simulator applies S2.

## Static truth

For CF20, the best static action in the fine grid is:

```text
f0.075_S1
Excel ReT = 0.268779
resource  = 0.0375
```

The learned bad policies score:

```text
Excel ReT = 0.216960
```

## PPO probability audit

The trained policy never becomes confident. At checkpoint 8192:

```text
top action prob        = 0.068
prob(f0.075_S1)        = 0.022
deterministic actions  = f0.1_S3 / f0.125_S1 mix
Excel ReT              = 0.216960
```

The target action probability starts near uniform:

```text
30 actions -> uniform prob ~= 0.033
prob(f0.075_S1) at t=0 = 0.034
```

Then it drifts down:

```text
t=512  -> 0.038
t=1024 -> 0.031
t=2048 -> 0.026
t=4096 -> 0.021
t=8192 -> 0.022
```

PPO is not learning a strong wrong optimum; it is staying weak/high-entropy and drifting around noisy actions.

## Why PPO does not discover the optimum

The CF20 optimum is not a simple one-step action reward. It is a **persistent policy**:

```text
hold low buffer around 0.05-0.075 AND keep S1 for many weeks
```

Random PPO exploration does not test static policies cleanly. It mixes actions week by week. A trajectory may choose `f0.075_S1` for a few weeks but then switch to S2/S3 or higher buffers. The episode-level Excel ReT does not credit the isolated good action clearly enough.

This creates a credit-assignment problem:

1. The good action must be repeated persistently.
2. The reward is delayed and path-dependent.
3. The action space has 30 discrete choices.
4. Many bad actions produce the same coarse Excel result (`0.216960`), so gradients are flat/noisy.
5. PPO's categorical policy remains near-uniform instead of collapsing onto the narrow low-buffer/S1 ridge.

## What PPO is learning

At this scale, PPO is mostly learning a weak, noisy preference over actions, not a stable resilience policy.

It learns neither:

- the static optimum `f0.075_S1`, nor
- a risk-conditioned adaptive policy.

The warm-start test confirms the distinction:

```text
Warm-start f0.075_S1 -> PPO maintains f0.075_S1 -> Excel = best static.
```

So PPO can execute the policy once initialized there. It cannot reliably discover it from scratch.

## Next logical tests

1. **Static-imitation pretraining / behavior cloning**
   - Train the policy to output `f0.075_S1` first.
   - Then allow PPO fine-tuning.
   - If it improves: dynamic headroom exists.
   - If it only maintains: best dynamic equals best static.

2. **Episode-level bandit / CEM / Bayesian search baseline**
   - Search over persistent static policies directly.
   - This is better matched to the problem than PPO when the optimum is a constant needle.

3. **Option-level RL**
   - Actions are not weekly raw actions, but options such as:
     - hold `f0.075_S1` for N weeks,
     - switch to `f0_S2`,
     - raise buffer temporarily.
   - This gives PPO temporally coherent actions.

4. **Use PPO only after adding a real dynamic frontier**
   - If the best policy is static, PPO is the wrong hammer.
   - Use RL where actions need to change over time, e.g. Track B or non-stationary campaigns with verified headroom.

## Artifact

`outputs/diagnostics/ppo_discovery_cf20_2026-06-29/summary.json`
