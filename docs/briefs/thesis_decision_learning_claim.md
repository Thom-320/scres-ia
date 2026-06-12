# Thesis-Decision Learning Claim

Working note created after reviewing `v.0_neuralNet-scres(2).docx`.

## Core Paper Point

The paper should not be framed as "we added a neural network to a DES." The stronger claim is:

> Supply chain resilience can be operationalized as an adaptive, path-dependent learning capability when the DES state, disruption history, and thesis-valid decision variables are embedded in a learning loop.

In Garrido-Rios (2017), the thesis model evaluates static configurations of inventory policy and assembly capacity under disruption. In the v0 draft, the intended theoretical advance is to add accumulated learning, `L_{t-1}`, as an endogenous element of resilience performance. The empirical job of this repo is therefore to test whether a learned policy can improve resilience outcomes while keeping the same decision-variable surface as the thesis.

## Current Contract Correction

The main thesis-decision action contract is `thesis_factorized`, implemented as `MultiDiscrete([6,3])`: one common `I_{t,S}` level plus one `S` level. This is the contract to use when the paper says the learned policy uses the same decision variables as Garrido-Rios.

Two older surfaces remain useful only as ablations or declared extensions:

- `onehot_18d` is a compatibility representation of the realized thesis decision vector, not the clean RL action space.
- `factorized`, implemented as `MultiDiscrete([6,6,6,3])`, allows independent Op3/Op5/Op9 inventory periods and must be reported as a per-node inventory extension, not as thesis 1:1.

The paper-facing ladder is therefore: reproduce the static Garrido DOE, test a static `I x S` interaction with the same thesis decision variables, test the per-node static extension separately, then compare adaptive PPO only against the best static policy in the same action space.

## Theoretical Contribution

The contribution should be stated as:

1. Reframe SCRES from a static recovery outcome into an adaptive learning trajectory.
2. Preserve thesis-fidelity by using Garrido decision variables rather than an artificially enlarged action space.
3. Add a learning mechanism over repeated disruption episodes, so the policy conditions decisions on current state and historical disruption/order context.
4. Compare learned and static policies under paired Garrido Cf scenarios, not only under generic S1/S2/S3 baselines.

This is a better fit for reinforcement learning than a feedforward prediction-only ANN, because the paper needs sequential adaptation under recurring disruptions rather than one-shot policy prediction.

## Current Evidence

The thesis-decision lane now supports the needed fidelity fixes:

- Initial inventory and shifts can be chosen before warmup through `learn_initial_decision`.
- Replenishment can use thesis-strict common `It,S` periods.
- Observations can include compact Table 6.25-style SDM history through `env_sdm_history_reward`.
- Evaluation includes Garrido Cf31-Cf90 paired comparisons plus a static `It,S x S` grid.
- PPO+MLP can train on a Cf curriculum without changing the thesis action variables.

Current 100k x 5 evidence:

| Experiment | Paired wins vs Garrido Cf | Mean fill-rate delta | Positive Cf scenarios |
|---|---:|---:|---:|
| No Cf curriculum, PPO+MLP | 76 / 300 (25.3%) | +0.00995 | 19 |
| Cf curriculum, PPO+MLP medium | 90 / 300 (30.0%) | +0.01154 | 23 |
| Cf curriculum, PPO+MLP large | 86 / 300 (28.7%) | +0.01098 | 22 |
| Cf curriculum, PPO+MLP medium, `ReT_seq_v1` train reward | 86 / 300 (28.7%) | +0.01049 | 22 |
| Cf curriculum, PPO+MLP medium, v7 observation transfer | 87 / 300 (29.0%) | +0.01160 | 22 |

Short ablations:

| Experiment | Paired wins vs Garrido Cf | Mean fill-rate delta |
|---|---:|---:|
| 25k Cf curriculum, PPO+MLP medium | 53 / 180 (29.4%) | +0.01166 |
| 25k Cf curriculum, PPO+MLP large | 18 / 60 (30.0%) | +0.01373 |
| 25k Cf curriculum, RecurrentPPO | 15 / 60 (25.0%) | +0.00815 |

Interpretation: the learning setup is moving in the right direction, and Cf curriculum improves the evidence. However, the result is not yet a decisive "AI beats Garrido" claim across the full design space. It is currently a conditional claim: learning helps in a subset of disruption configurations, and improves mean paired fill rate, but it does not dominate Garrido's static designs scenario-by-scenario.

The 100k large-MLP ablation does not improve the main fill-rate result over the medium MLP. It slightly improves mean ReT delta (+0.07419 vs +0.07129) but loses paired fill-rate wins. For the paper path, this means network capacity is not the binding constraint at 100k; objective alignment and scenario curriculum matter more.

The 100k `ReT_seq_v1` ablation also does not improve paired fill-rate wins over `control_v1` in the thesis-decision lane. It keeps the result positive on mean paired fill delta, but the best current 100k setting remains `control_v1 + Cf curriculum + PPO medium`.

The 100k v7 observation-transfer ablation does not beat v5 on paired wins either. It slightly improves mean fill delta (+0.01160 vs +0.01154), but loses three paired wins. This suggests that observing downstream state helps interpretation more than control unless downstream actions are available.

The strongest current thesis-faithful signal is family-specific:

| Family | Paired wins | Mean fill-rate delta | Mean ReT delta |
|---|---:|---:|---:|
| Inventory Cf31-Cf60 | 25 / 150 (16.7%) | +0.00179 | -0.00115 |
| Capacity Cf61-Cf90 | 65 / 150 (43.3%) | +0.02130 | +0.14374 |

Interpretation: the learned policy is not meaningfully beating Garrido through inventory-period selection. It is learning adaptive capacity behavior. That aligns with the paper's theoretical claim better than an undifferentiated "AI beats all static policies" claim.

A capacity-focused training/evaluation gate over Cf61-Cf90 did not materially improve over the broader Cf31-Cf90 curriculum:

| Experiment | Capacity paired wins | Mean capacity fill-rate delta | Mean capacity ReT delta |
|---|---:|---:|---:|
| Train Cf31-Cf90, evaluate Cf61-Cf90 | 65 / 150 (43.3%) | +0.02130 | +0.14374 |
| Train Cf61-Cf90, evaluate Cf61-Cf90 | 64 / 150 (42.7%) | +0.02153 | +0.14249 |

Interpretation: the broader Cf curriculum should stay. Specializing to capacity scenarios does not improve win rate and slightly reduces ReT.

The 250k and 500k production gates of the best 100k lane also did not improve the result:

| Training length | Paired wins | Mean fill-rate delta | Mean ReT delta | Capacity wins | Capacity fill delta |
|---|---:|---:|---:|---:|
| 100k | 90 / 300 (30.0%) | +0.01154 | +0.07129 | 65 / 150 (43.3%) | +0.02130 |
| 250k | 85 / 300 (28.3%) | +0.01131 | +0.07197 | 64 / 150 (42.7%) | +0.02215 |
| 500k | 83 / 300 (27.7%) | +0.01038 | +0.07446 | 58 / 150 (38.7%) | +0.01868 |

Interpretation: longer PPO training does not improve paired fill-rate wins in this thesis-faithful lane. The 250k run slightly improves capacity fill delta, but loses global wins and does not beat the 100k capacity win rate. The best current paper-facing training horizon is therefore 100k, not 250k/500k.

## Track B Lessons For Thesis-Faithful Work

Track B worked because it changed the controllability of the system, not because of a larger neural network. The strongest Track B finding is that Op10/Op12 downstream dispatch control is necessary and sufficient for fill-rate improvement. That cannot be moved into the Garrido thesis-faithful lane if the action variables must remain `It,S` and `S`.

What can transfer without violating thesis fidelity:

- richer observations, especially v7 downstream disruption/queue visibility and rolling service metrics;
- regime/cycle/history features that let the policy condition on disruption context;
- Cf curriculum training and paired Cf evaluation;
- cost/service telemetry from Track B for policy interpretation;
- reward audit discipline: treat `ReT_thesis` as reporting-only, avoid reward modes that collapse to S1/S3, and keep `control_v1` as the operational training baseline unless a new ablation beats it.

The v7 observation transfer was tested and did not improve paired wins. The next active experiment is therefore a capacity-focused gate over Cf61-Cf90, where the current evidence says learning has real headroom.

## Publishable Claim To Test Next

The clean publishable claim is:

> Under thesis-faithful decision variables, a curriculum-trained learning policy improves average paired resilience performance across Garrido's disruption configurations, with gains concentrated in high-stress scenarios where static factorial settings under-adapt.

To upgrade this from a weak positive result to a strong paper result, the next experiment should test whether larger-capacity PPO or longer training increases the paired win rate without relaxing thesis-fidelity. If it does not, the honest contribution becomes diagnostic: Garrido's decision surface leaves limited adaptive headroom, and learning only helps where the static policy is mismatched to disruption history.

## Current Recommendation

Use PPO+MLP medium with `control_v1`, broad Cf31-Cf90 curriculum, VecNormalize, learnable initial decision, and paired Garrido Cf evaluation as the paper baseline. Leave the action surface unchanged. Do not switch to large MLP, RecurrentPPO, SAC, `ReT_seq_v1`, v7 observations, capacity-only training, 250k training, or 500k training as the primary lane unless later evidence beats the current paired-fill result. The paper should separate inventory and capacity families: the honest positive claim is adaptive capacity learning under thesis-valid decisions, while the inventory family is near-static and has little learnable headroom.
