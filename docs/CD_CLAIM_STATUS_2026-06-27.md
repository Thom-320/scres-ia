# Cobb-Douglas Claim Status (2026-06-27)

## Verdict

There is **no current confirmable claim** that dynamic PPO beats the efficient
static frontier on the clean Cobb-Douglas lane.

The best current paper-safe statements are:

1. The Garrido Excel/fidelity lane is auditable and workbook-grounded.
2. The retained-memory Track-A lane is currently a null/weak-learning result.
3. The Cobb-Douglas same-bar lane exposes a real static decision frontier, but
   the current dynamic PPO pilots do not yet beat that frontier.
4. The `phi4/psi1.5` war-stress `WIN=YES` artifact is exploratory and should
   not be used as a headline result.

## Clean C-D Lane Result

Scaled run:

```text
output: outputs/benchmarks/garrido_dynamic_cd/scaled_phi2_train_8192_5seed_2026-06-27
env: phi=2.0, psi=1.0, deterministic PT
reward: ReT_garrido2024_train
primary_eval: cd_sigmoid_mean
seeds: 9201-9205
eval_episodes: 3
train_timesteps: 8192
```

Dynamic PPO versus best static C-D by regime:

| Regime | PPO C-D | Best Static | Static C-D | Delta |
| --- | ---: | --- | ---: | ---: |
| current | 0.641157 | `static_S1_I168` | 0.724610 | -0.083453 |
| increased | 0.621602 | `static_S1_I168` | 0.633714 | -0.012111 |
| severe | 0.577845 | `static_S2_I672` | 0.593483 | -0.015638 |

Interpretation: PPO is close in `increased`/`severe`, and sometimes saves
resources versus heavier statics, but it does **not** beat the C-D frontier.

## War `phi4/psi1.5` Check

The earlier `outputs/4cases/aggregated_results.csv` reported:

```text
case3_war_resilience: WIN=YES
case4_war_cd: WIN=YES
```

That result is not reliable because the original `war_cd_train.py` summary:

- read non-existent service keys and defaulted them to zero;
- compared PPO terminal-step C-D against static episode-mean C-D;
- used a weak robust-static reference rather than the true best static in the
  comparable run.

Component-decomposition rerun:

```text
output: outputs/experiments/cd_component_decomposition_phi4_psi1p5_raw_2026-06-27
env: phi=4.0, psi=1.5
reward: ReT_garrido2024_raw
seeds: 1,2
train_timesteps: 10000
```

Result:

```text
PPO cd_sigmoid_mean = 0.5887
best static = S2_I168
best static cd_sigmoid_mean = 0.6305
PPO - best static = -0.0418
```

Real service metrics were also worse for PPO:

| Metric | PPO | Best Static (`S2_I168`) |
| --- | ---: | ---: |
| flow_fill_rate | 0.4283 | 0.8090 |
| lost_rate | 0.5288 | 0.0593 |
| Excel ReT | 0.0007 | 0.0024 |

Interpretation: the apparent war win is not a defensible resilience claim.
`phi4/psi1.5` remains an exploratory stress cell only.

## Useful Next Work

The useful parts of the Freeze V2 plan are:

- align README and docs to the actual source-of-truth branch;
- add contract-alignment tests so runner defaults cannot drift from the paper
  contract;
- keep `control_v2_backlog` as a sensitivity reward if implemented;
- preserve the old retained-reset lane as a documented null, not the primary
  current route;
- use the existing Garrido order export machinery to produce audit-friendly
  ledgers when discussing Excel fidelity.

The less useful parts are:

- creating another large Freeze V2 before merging/exposing the current branch;
- restarting from `main` while the relevant work lives on
  `codex/garrido-replication-experiments`;
- treating `phi4/psi1.5` as primary;
- mixing the DQN retained-memory lane and the C-D dynamic-vs-static lane as one
  claim.

## Claim Boundary

Until a larger run reverses the current evidence, the paper should **not** claim:

```text
RL beats the efficient static frontier on Garrido/Excel or C-D resilience.
```

It may claim, with evidence:

```text
We rebuilt and audited the Garrido Excel/DES resilience lane, showed where the
faithful Track-A action space has limited learning headroom, and identified a
Cobb-Douglas same-bar frontier that remains unsolved by the current PPO pilots.
```

