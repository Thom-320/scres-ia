# Track A Frozen Experimental Protocol

This document freezes the next Track A artifact: reward, ReT, and thesis-replication audit before any long RL run. Track A uses the Garrido-Rios MFSC thesis as the experimental base. Strict thesis-reference audits remain separate from the trainable thesis-anchored Gym/RL extension. The current paper-facing contract is `docs/PAPER_CONTRACT_2026-06-24.md`.

## Research Question

Does retaining reinforcement-learning state across successive disruption blocks improve out-of-sample MFSC resilience relative to an otherwise identical reset-learning policy, under a thesis-derived inventory-shift action surface and held-out recurring disruption streams?

## Time and Replication Basis

- Internal DES resolution: 1 hour.
- Decision interval: 168 hours.
- Thesis year basis: 8,064 hours.
- Thesis horizon: 161,280 hours.
- Thesis warm-up trigger: first Op9 arrival.
- Frozen downstream-Q source for thesis replication and Track A training: `figure_6_2`.
- Explicit downstream-Q robustness lane: `table_6_20`.
- Paper-facing raw-material mode: `kit_equivalent_order_up_to` (canonical internal mode `bom_total_units_order_up_to`) with order-up-to multiplier `2.0`.
- Paper-facing risk occurrence mode: `thesis_window`.
- Current deterministic replication gate: Table 6.10 reproduction produced 738,432 rations/year, RMSE 61,013.6 against thesis ECS, below the thesis-reported RMSE of 87,918.
- Current risk-frequency gate: Table 6.11 must pass under `thesis_window`; `legacy_renewal` is retained only as an explicit historical negative-control lane.

## Lanes

The thesis-reference lane is not an RL environment. It exists to reproduce and audit Garrido-Rios constants, timing, risk tables, decision tables, design matrix, and the order-level ReT schema as closely as the available thesis evidence allows. The primary thesis ReT metric is the order-level implementation in `supply_chain/ret_thesis.py`.

The trainable Track A lane is a Gymnasium wrapper around a thesis-anchored SimPy environment. It is an extension: it combines inventory and capacity decisions, introduces learned policies, and may include realistic stochastic or stress-test extensions. Step-level ReT variants emitted by `supply_chain/env_experimental_shifts.py` are diagnostics or reward candidates. They are not thesis replication metrics.

The downstream dispatch quantity ambiguity is frozen before training. Use `figure_6_2` for thesis replication, reward selection, PPO smoke runs, DQN smoke runs, and paper-facing Track A claims. Use `table_6_20` only as a named robustness lane. A reward is eligible for training only after the Figure 6.2 lane is selected as the primary modeling source; Table 6.20 can falsify robustness, but it does not redefine the training default.

`learning_extension_v1` is frozen before held-out evaluation as a coupled realism extension: a persistent campaign-phase disruption regime plus stochastic ration demand. The persistent regime supplies the dose-response parameter `rho`; stochastic demand is justified by operational tempo rather than by test-set performance. These are moderators of learning value, not post hoc routes to a positive result.

The anti-fishing rule is simple: specify the regime parameters, pilot only on training/calibration tapes, freeze the contract, then run the held-out retained-vs-reset contrast once. The stationary thesis reference remains a separate fidelity lane; `current`, `increased`, and `severe` risk levels are reward-selection and stress-test gates, not automatic evidence for the final learning claim.

## Closed-Loop Learning Representation

For disruption cycle `k` and decision epoch `t`:

```text
a_{k,t} = pi_{theta_{k-1}}(s_{k,t})
s_{k,t+1} = F(s_{k,t}, a_{k,t}, d_{k,t}, xi_{k,t})
tau_k = {(s_{k,t}, a_{k,t}, r_{k,t})}_{t=1}^{T_k}
theta_k = U(theta_{k-1}, tau_k)
R_k = G(tau_k)
L_{k-1} = theta_{k-1}
```

Learning is therefore retained policy state, not ordinary DES memory. Inventories, backorders, and unattended orders are system state; only policy updates from prior disruption outcomes count as endogenous learning.

## Observation and Action Contract

The first Track A action surface is thesis-derived and factorized:

```text
MultiDiscrete([6, 3])
inventory level: 0, I168, I336, I504, I672, I1344
shift level: S1, S2, S3
```

Inventory level applies jointly to Op3, Op5, and Op9 using the thesis buffer table. The new `Discrete(18)` wrapper is a DQN-style view over the same 6 x 3 surface. The primary retained/reset evaluator uses `decision_cadence=block`: one configuration is selected at the start of a disruption block and held through the block. Weekly reselection is a sensitivity lane. The existing continuous 6D Track A lane remains available for PPO, SAC, and recurrent PPO comparisons, but it is not the thesis-factorized reward-surface audit.

Observable state for later learning runs should include only controller-observable quantities: inventories, WIP/proxy flow state, backorders, lost/unattended orders, fill rate, utilization/capacity, active disruption flags, downtime/severity observed so far, recent demand, current shift level, previous action, time within block, and physical resource-budget state if a budgeted extension is active.

## Reward Candidates and Gate

No reward is frozen for long training yet. The audit evaluates all current reward modes over the 18 static thesis policies without requiring DKANA or torch.

Raw `reward_total` is comparable only within a reward mode. Cross-mode selection uses:

- reward spread over the 18 policies;
- best policy by reward;
- shift collapse to S1 or S3;
- Spearman correlation with order-level ReT and fill rate;
- Spearman correlation against negative service-loss area, pending backlog, and cost;
- external outcomes: order-level ReT, fill rate, service-loss area, pending backlog, total cost proxy, shift mix, and 95th-percentile service-loss area.

Reward selection gate:

- reject or hold for audit rewards that trivially select S1 or S3 unless the selected policy also improves external ReT, service, loss, or cost metrics;
- shortlist rewards with positive rank correlation to order-level ReT/fill and positive correlation to negative backlog/service-loss;
- run only shortlisted rewards in small PPO/DQN learning smokes before any long experiment;
- keep `ReT_thesis` as an audit and negative-control reward, not the expected final training objective;
- do not add `control_v2` until all existing reward modes fail this gate.

## Policy Conditions

The confirmatory experiment must include:

- fixed heuristic policy;
- optimized static policy selected on training scenarios only;
- frozen RL policy with the same neural architecture but no evaluation-time updates;
- adaptive RL policy with retained policy updates;
- reset-learning ablation that learns within a disruption cycle but resets parameters or retained memory after the cycle.

The key causal comparison is adaptive RL versus frozen RL. The path-dependency comparison is persistent adaptive RL versus reset learning.

## Hypotheses

H1: The retained online learner produces higher held-out order-level Garrido ReT than the otherwise identical reset-learning ablation:

```text
E[ReT_retained - ReT_reset] > 0
```

H2: The retained-minus-reset effect increases monotonically with the disruption-persistence parameter of the frozen learning regime:

```text
d/d_rho E[ReT_retained - ReT_reset] > 0
```

Secondary checks compare retained learning against the robust static Garrido-aligned policy, the threshold heuristic, and the frozen neural policy. If retained beats static but not reset, the result is adaptive control rather than accumulated learning.

## Train/Evaluation Split

Training uses repeated disruption-centered episodes with domain randomization over disruption category, affected operations, severity, duration, demand level, initial inventory, and initial backlog.

Evaluation uses held-out random seeds and common random disruption streams for every policy. Report two regimes separately:

- probability-weighted evaluation using original disruption distributions;
- stress-test evaluation forcing rare events across varied timing and severity.

Stress tests reveal robustness; they do not estimate real-world event frequency.

## Immediate Test Plan

Re-run the thesis gates:

```text
Table 6.10 reproduction
decision tables
risk tables
operations table
design matrix
ReT schema
```

Run the reward-surface audit:

```text
uv run --no-project \
  --with simpy==4.1.1 --with numpy==2.1.3 --with pandas==2.3.1 \
  --with gymnasium==1.2.1 --with scipy==1.15.3 \
  python scripts/audit_thesis_reward_surface.py \
  --downstream-q-source figure_6_2 \
  --risk-levels current increased severe \
  --replications 3 \
  --max-steps 260
```

Run the downstream-Q robustness audit separately:

```text
uv run --no-project \
  --with simpy==4.1.1 --with numpy==2.1.3 --with pandas==2.3.1 \
  --with gymnasium==1.2.1 --with scipy==1.15.3 \
  python scripts/audit_thesis_reward_surface.py \
  --downstream-q-source table_6_20 \
  --risk-levels current increased severe \
  --replications 3 \
  --max-steps 260
```

Compare the two lanes with `scripts/compare_downstream_q_reward_surface.py`. Robustness failures must be reported, but they do not change the frozen Track A training source unless the protocol is explicitly revised.

Then run short learning smokes for shortlisted rewards only, with the same seeds, horizon, and static baselines. Acceptance metrics are order-level ReT, fill rate, service-loss area, pending backlog, total cost proxy, shift mix, and downside tail metrics.
