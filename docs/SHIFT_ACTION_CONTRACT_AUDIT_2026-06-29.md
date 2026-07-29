# Shift Action Contract Audit (2026-06-29)

## Question

The controlled-risk CF20 surface says the best Excel-ReT policy is a low-buffer S1 policy (`f0.05_S1` / `f0.075_S1`), but PPO learned `frac=0`, `S2`. Is this a reward problem, or is the current continuous shift action contract making the optimum hard to discover?

## Thesis Mechanism

In Garrido-Rios, `S` is a categorical manufacturing-capacity decision:

- `S1`: 1 shift/day.
- `S2`: 2 shifts/day.
- `S3`: 3 shifts/day.

The code faithfully implements Table 6.20 in `supply_chain/config.py::CAPACITY_BY_SHIFTS`:

- shifts change Op3/Op4 upstream raw-material quantities,
- shifts change Op7/Op8 batch/cadence,
- shifts do **not** change downstream Op9/Op10/Op12 dispatch caps.

Therefore, the thesis-faithful action representation for shifts is discrete/factorized, not a continuous scalar.

## Current Continuous Wrapper

`continuous_its_env.py` uses a continuous action:

```text
action[0] = buffer fraction
action[1] = shift_signal in [-1, 1]
```

Then it maps `shift_signal` through hard thresholds:

```text
shift_signal < -0.33 -> S1
shift_signal <  0.33 -> S2
else                  -> S3
```

This is valid as a convenience relaxation, but it is not the thesis-native mechanism. It creates an optimization cliff: PPO must push the mean far enough below `-0.33` to choose S1. In CF20, the optimum is a narrow low-buffer/S1 needle near the action-space boundary.

## Audit Results

### Static reward surface

For CF20, rewards are aligned with Excel ReT:

- `ReT_excel_delta`: best train policy = `f0.05_S1`.
- `ReT_excel_plus_cvar`: best train policy = `f0.05_S1`.
- Fine surface: `f0.075_S1` is even better than `f0.05_S1`.

So the reward is not blind.

### Continuous PPO failure

The continuous PPO policy collapsed to:

```text
frac = 0
shift_signal inside S2 band
applied policy = f0_S2
```

This loses Excel ReT against `f0.075_S1`.

### Fine-discrete shift/buffer probe

Implemented `scripts/run_controlled_risk_fine_discrete_probe.py`, where:

```text
action in Discrete(len(fracs) * 3) -> (buffer_frac, S1/S2/S3)
```

This is closer to the thesis for shifts because `S` is categorical.

Result without warm-start:

```text
CF20, 1 seed, 8k steps:
dynamic = f0.125_S3
Excel = 0.216960
best static = f0.075_S1, Excel = 0.268779
```

So discretizing shift alone did not solve exploration. PPO still picked a bad action.

Result with warm-start toward `f0.075_S1`:

```text
CF20, 1 seed, 4k steps:
dynamic = f0.075_S1
Excel = 0.268779
resource = 0.038
best static = f0.075_S1
```

PPO maintained the optimum but did not improve it.

## Diagnosis

The failure is not primarily the reward.

The more precise diagnosis is:

> In CF20, the correct policy is a narrow low-buffer/S1 needle. The reward ranks it correctly, but vanilla PPO does not reliably discover it from scratch. A factorized/discrete shift representation is more thesis-faithful, but still needs warm-start, imitation, or better exploration to find the needle.

## Implications

1. The current continuous threshold shift mechanism is acceptable for quick relaxed experiments, but not ideal for thesis-faithful controlled-risk learning.
2. Serious Track-A learning tests should use either:
   - discrete/factorized shift (`S1/S2/S3`) with continuous or fine-grid buffer, or
   - a hybrid policy with discrete shift head and continuous buffer head.
3. Warm-start/behavior-cloning from the dense static frontier is now the cleanest next test:
   - If PPO starts at `f0.075_S1` and improves, there is dynamic headroom.
   - If PPO only maintains it, the best dynamic policy is just the best static.
   - If PPO degrades, the learning setup is unsafe for confirmatory claims.

## Artifacts

- Continuous controlled-risk probe: `outputs/experiments/garrido_controlled_risk_probe_cf13_cf20_1seed_2026-06-29`
- Reward surface: `outputs/experiments/garrido_controlled_risk_probe_reward_surface_2026-06-29/static_reward_surface_summary.json`
- Fine-discrete probe: `outputs/experiments/controlled_risk_fine_discrete_cf20_1seed_2026-06-29`
- Warm-start fine-discrete probe: `outputs/experiments/controlled_risk_fine_discrete_cf20_warmstart_f0075_s1_1seed_2026-06-29`
