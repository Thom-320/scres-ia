# PPO Tuning and Track A Gate Status (2026-06-29)

## Question

What remains to explore, and how should PPO be improved without overfitting to a lane that has no dynamic headroom?

## Dense-frontier gate result

The last surviving Track A candidate was:

```text
continuous_its
observation_version = v8
reward = ReT_excel_plus_cvar
alpha = 0.2
phi = 4
psi = 1.5
horizon = 104
n_fracs = 21
CRN eval seed0 = 9000
seeds = 1,2
timesteps = 30000
```

Artifact:

```text
outputs/experiments/v8_excel_plus_cvar_alpha02_dense_crn_gate_2seed_30k_2026-06-29/summary.json
```

Result:

| policy | Excel ReT | CVaR95 service loss | resource |
|---|---:|---:|---:|
| dynamic PPO | 0.00208 | 5.45e9 | 0.545 |
| dense static `f0.10_S1` | 0.00234 | 4.43e9 | 0.050 |
| dense static `f0.15_S1` | 0.00228 | 4.35e9 | 0.075 |

Both Pareto flags are false:

```text
Excel Pareto: dominated_by_static = true
CVaR Pareto:  dominated_by_static = true
```

The earlier v8 signal was therefore a coarse-frontier / weak-baseline signal, not a publishable Track A win.

## What PPO is learning

PPO is not failing to apply actions. The action propagation and CF20 fine-discrete gate show the simulator applies the chosen fraction and shift correctly.

The recurring failure mode is different:

1. Dense static policies find low-buffer/S1 sweet spots.
2. PPO explores weekly mixed sequences and often spends too much resource.
3. Reward shaping can make the policy adaptive, but it does not beat the dense static base-stock frontier when that frontier already captures the benefit.
4. Warm-start tests show PPO can maintain a static optimum but does not improve beyond it.

So the current Track A problem is mostly:

```text
static/persistent-policy search > sequential RL
```

unless a new environment exposes real time-varying headroom.

## What remains worth exploring

1. **H4 retained-vs-reset**: still the strongest learning claim, but current Kaggle outputs are micro/null and underpowered. Needs a runtime-safe one-seed-per-kernel design with partial writes before scaling.
2. **Track B**: downstream dispatch reaches the binding constraint. This is the most plausible route to a raw Excel ReT win, but it must be framed as an extension and the compressed fill/7D heuristic mismatch must be audited first.
3. **R2 fidelity/recovery audit**: continue the R2 endogenous-vs-Excel investigation, especially if controlled-risk experiments are kept.
4. **Mechanism ablations**: base vs risk_obs vs risk_obs+hazard vs shuffled hazard are still useful, but only for a candidate that survives dense static frontier.

## PPO improvement recipe

Do not add timesteps before checking the static frontier. If a lane survives that gate, use:

```text
n_envs = 4 or 8
reward normalization
curriculum h52 -> h104
low final entropy / stronger entropy decay
action smoothing or switch penalty
static/CEM warm-start
two-stage pi_init + pi_weekly
```

For Track A specifically, also prefer categorical or factorized shift handling over a continuous thresholded shift signal, because S1/S2/S3 are thesis categories.

## Reward tuning boundary

The following reward ideas are implemented or tested:

```text
ReT_excel_plus_cvar
ReT_excel_delta_bootstrap
ReT_excel_terminal_shaped / PBRS
ReT_tail_v2
```

On CF20, future-credit rewards did not solve discovery from scratch. They maintained the optimum only after warm-start. This means the bottleneck was not simply "bad reward"; it was lack of dynamic headroom plus hard persistent-policy discovery.

Use Lagrangian/budgeted rewards only on a lane with verified headroom. Otherwise they just train PPO to approximate a static budgeted policy that CEM finds faster.

## Decision

Track A should not be escalated further for a headline win unless a new structural gate first proves dynamic headroom against a dense static frontier.

Current priority order:

1. H4 retained-vs-reset with operationally safe compute.
2. Track B after compressed-fill and downstream-control audit.
3. R2 fidelity/recovery lane.
4. Mechanism/architecture ablations only after a surviving candidate exists.
