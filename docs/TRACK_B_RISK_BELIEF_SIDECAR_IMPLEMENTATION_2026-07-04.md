# Track B Risk-Belief Sidecar Implementation — 2026-07-04

## Question

Can a policy become more preventive if it receives a learned belief about an
upcoming frequent risk, instead of only receiving raw historical counts?

This is the first implementation of the idea:

> PPO/Real-KAN + observed risk memory + frozen auxiliary risk-prediction head,
> evaluated at the end with Garrido Excel ReT.

## Why this is the next step

The `v10` risk-memory smoke showed that appending raw historical risk features
alone is not enough:

- PPO+MLP v10 still beats static policies, but underperforms fixed-RNG v7.
- Real-KAN v10 is more robust than PPO+MLP v10, but still acts near the
  high-resource ceiling.

So the next hypothesis is not "more features"; it is "convert history into a
compact learned belief." The supervised Sprint 1 predictor found the cleanest
memory-only signal for R24 at 1-2 weeks, so this sidecar starts there.

## Implementation

New runner:

```text
scripts/run_track_b_risk_belief_sidecar.py
```

The runner:

1. Reads the supervised dataset:

```text
outputs/experiments/track_b_risk_belief_predictor_2026-07-04/risk_belief_dataset.csv
```

2. Trains two frozen logistic belief heads from the 12 observed v10 memory
   features:

- `P(R24 starts within 1 week)`
- `P(R24 starts within 2 weeks)`

3. Appends those two probabilities to the Track B `v10` observation.

The effective observation dimension is therefore:

```text
v10 = 101 raw observation fields
belief = 2 frozen probabilities
effective_dim = 103
```

4. Trains the selected PPO architecture normally, with the same reward and
evaluation protocol as Track B:

- action contract: `track_b_v1`
- reward: `control_v1`
- risk profile: `adaptive_benchmark_v2`
- final metric: Garrido Excel ReT (`order_ret_excel_mean`)

## Architectures

Two arms are supported:

- `ppo_mlp_belief`: standard PPO+MLP with the frozen belief features.
- `real_kan_belief`: PPO with the real pyKAN feature extractor plus the same
  frozen belief features.

This matters because the fair test is not "KAN gets prevention but MLP does
not." If the belief signal is useful, PPO+MLP should also be allowed to use it.

## Current smoke campaign

Launched on the VPS:

```text
outputs/experiments/track_b_risk_belief_ppo_3seed_30k_2026-07-04
outputs/experiments/track_b_risk_belief_real_kan_3seed_30k_2026-07-04
```

Protocol:

- seeds: 1, 2, 3
- train timesteps: 30,000 per seed
- eval episodes: 8 per seed
- horizon: 104 weeks
- observation: `v10 + belief(R24 1w, R24 2w)`

## Decision rule

Use `order_ret_excel_mean`, not `order_level_ret_mean`.

The belief sidecar is promising only if it improves ReT Excel against the
matched no-belief `v10` smoke:

- PPO+MLP v10 no-belief: `0.0058107`
- Real-KAN v10 no-belief: `0.0059148`

It becomes more interesting if it also approaches or beats the fixed-RNG v7
reference:

- PPO+MLP fixed-RNG v7: `0.0059206`
- Real-KAN fixed-RNG v7: `0.0059462`

Do not call this "preventive" yet. A ReT gain only says the learned belief
helped performance. Prevention still requires a mechanism audit showing that
pre-risk belief changes action timing and contributes positively to Garrido
Excel ReT.

## Interim result: PPO+MLP-belief landed

`PPO+MLP-belief` completed on the VPS and was fetched/verified locally:

```text
outputs/experiments/track_b_risk_belief_ppo_3seed_30k_2026-07-04
```

Protocol verified:

- observation: `v10 + belief(R24 1w, R24 2w)`
- effective observation dimension: 103
- seeds: 1, 2, 3
- train timesteps: 30,000
- eval episodes: 8
- max steps: 104
- reward: `control_v1`
- risk profile: `adaptive_benchmark_v2`
- action contract: `track_b_v1`

Result, using the primary metric `order_ret_excel_mean`:

| Lane | ReT Excel | Cost index | Reading |
|---|---:|---:|---|
| v7 fixed-RNG, 60k confirmatory | 0.0059206 | 0.700 | reference, larger budget |
| v7 fixed-RNG, 30k calibration | 0.0058432 | 0.734 | budget-matched calibration |
| v10 raw memory, 30k | 0.0058107 | 0.763 | raw memory alone |
| PPO+MLP-belief, 30k | 0.0057427 | 0.640 | frozen belief sidecar |

The 30k v7 calibration confirms that part of the apparent v10 drop comes from
the smaller training budget. However, at the same 30k budget, `v10` remains below
`v7`, and `PPO+MLP-belief` is below raw `v10` in all three seeds:

```text
seed1: belief 0.0057761 vs raw v10 0.0057967
seed2: belief 0.0056747 vs raw v10 0.0057381
seed3: belief 0.0057774 vs raw v10 0.0058974
```

Interim interpretation: this frozen logistic belief head is not enough to make
PPO+MLP use risk memory preventively. It reduces resource cost, but the primary
ReT Excel outcome gets worse. The next meaningful test is the still-running
Real-KAN-belief arm; after that, the stronger version would be an end-to-end
auxiliary loss rather than a frozen appended probability.
