# Section 3.3 — Reinforcement Learning Formulation

*This section replaces the placeholder text in the v0 under "3.3 Hybrid Simulation Neural Model."*
*Written 2026-03-24 — ready for insertion into the DOCX.*

---

## 3.3 Hybrid Simulation–Neural Model

The second layer of the proposed framework converts the DES environment into a sequential decision problem amenable to reinforcement learning. Rather than executing a fixed shift and inventory policy for the full simulation horizon, the system is controlled by an agent that observes the supply chain state at weekly intervals and selects operational actions intended to minimize service loss while managing shift costs. The formulation follows the standard Markov Decision Process (MDP) interface (Sutton & Barto, 2018), implemented through the Gymnasium API (Towers et al., 2024), which wraps the SimPy-based DES described in Section 3.2.

### 3.3.1 State Space

The agent receives a 15-dimensional observation vector at each decision epoch. Table 5 summarizes the components and their normalization.

**Table 5.** Observation vector (v1) for the RL agent.

| Index | Variable | Description | Normalization |
|-------|----------|-------------|---------------|
| 0 | `raw_material_wdc` | Raw material at Warehouse (Op3) | /max_capacity |
| 1 | `raw_material_al` | Raw material at Assembly Line (Op5) | /max_capacity |
| 2 | `rations_al` | Rations buffer at QC (Op7) | /max_capacity |
| 3 | `rations_sb` | Rations at Supply Battalion (Op9) | /max_capacity |
| 4 | `rations_cssu` | Rations at CSSUs (Op11) | /max_capacity |
| 5 | `rations_theatre` | Rations at Theatre (Op13) | /max_capacity |
| 6 | `fill_rate` | Cumulative fill rate | [0, 1] |
| 7 | `backorder_rate` | Cumulative backorder rate | [0, 1] |
| 8 | `assembly_line_down` | Binary: assembly line disrupted | {0, 1} |
| 9 | `any_loc_down` | Binary: any LOC disrupted | {0, 1} |
| 10 | `op9_down` | Binary: Supply Battalion disrupted | {0, 1} |
| 11 | `op11_down` | Binary: CSSUs disrupted | {0, 1} |
| 12 | `time_fraction` | Simulation progress | [0, 1] |
| 13 | `pending_batch_norm` | Pending batch / batch size | [0, 1] |
| 14 | `contingent_demand_norm` | Pending contingent demand / 2600 | [0, 1] |

The observation captures both inventory levels and disruption indicators, allowing the agent to react to ongoing operational and known-unknown risks. Time fraction provides temporal context for long-horizon planning. No explicit memory of past observations is included at this stage; the implications of this design choice for partial observability are discussed in Section 5.

### 3.3.2 Action Space

The action space is five-dimensional and continuous, with each dimension bounded to [−1, 1]. Table 6 describes each action dimension and its mapping to the DES control parameters.

**Table 6.** Action space for the RL agent.

| Dim | Action | Mapping | DES parameter |
|-----|--------|---------|---------------|
| 0 | `op3_q` | multiplier = 1.25 + 0.75 × signal | Dispatch quantity at Op3 |
| 1 | `op9_q_max` | multiplier = 1.25 + 0.75 × signal | Upper dispatch quantity at Op9 |
| 2 | `op3_rop` | multiplier = 1.25 + 0.75 × signal | Reorder point at Op3 |
| 3 | `op9_rop` | multiplier = 1.25 + 0.75 × signal | Reorder point at Op9 |
| 4 | `shifts` | < −0.33 → S1; [−0.33, 0.33) → S2; ≥ 0.33 → S3 | Active assembly shifts |

The first four dimensions adjust inventory-control parameters as multiplicative perturbations around the thesis baseline values, providing the agent with fine-grained control over buffer sizing. The fifth dimension controls the number of active shifts in the assembly line (S1 = single shift, S2 = double shift, S3 = triple shift), which directly determines production capacity and operating cost. This action structure extends the two static strategies examined in Garrido-Rios (2017)—inventory buffering (Strategy I) and shift augmentation (Strategy II)—into a unified, dynamic control policy.

### 3.3.3 Decision Epoch and Horizon

The agent makes decisions at weekly intervals (168 simulated hours), which matches the natural operational cycle of the MFSC: procurement contracts operate on weekly cycles (R12), demand surges arrive at weekly granularity (R24, 672-hour intervals), and shift adjustments require coordination time that makes sub-weekly switching impractical. Each episode spans 260 decision steps, corresponding to the full 20-year simulation horizon of 161,280 hours after the warm-up period.

### 3.3.4 Reward Function Design

The choice of reward function is critical for RL-based control. Two reward formulations were evaluated during development:

**Resilience metric (ReT_thesis).** The thesis-aligned resilience metric defined by Garrido-Rios (2017, Equations 5.1–5.5) aggregates autotomy, recovery, non-recovery, and fill rate into a single scalar. While this metric is appropriate for *evaluating* resilience outcomes, it proved unsuitable as a *training* objective: preliminary experiments showed that the agent learned to minimize assembly shifts to S1 across all conditions, achieving a higher ReT score by reducing cost at the expense of severe service degradation (fill rate dropped from 0.99 to 0.84). The metric's structure rewards cost avoidance more than service maintenance, creating a misaligned incentive for the learning agent.

**Operational control reward (control_v1).** To address this misalignment, we designed a reward function that directly penalizes the two operational quantities that the shift-control agent can influence:

$$r_t = -\left( w_{bo} \cdot \frac{B_t}{D_t} + w_{cost} \cdot (S_t - 1) \right)$$

where:
- $B_t$ is the number of new backorders at step $t$,
- $D_t$ is the total demand at step $t$,
- $S_t \in \{1, 2, 3\}$ is the active shift count,
- $w_{bo} = 4.0$ weights the service-loss penalty,
- $w_{cost} = 0.02$ weights the shift-cost penalty.

The ratio $w_{bo} / w_{cost} = 200$ reflects the operational priority that maintaining service to forward-deployed units dominates shift operating cost. This design ensures that the agent is penalized for backorders (the primary resilience failure mode) while incurring a proportional cost for activating additional shifts, creating a meaningful service–cost tradeoff. The ReT metric is retained as a *reporting-only* evaluation metric for thesis-aligned comparison.

**Justification.** The control_v1 reward was selected over ReT_thesis based on three empirical criteria:

1. *Behavioral alignment:* Under control_v1, the trained agent uses a mix of shifts (12% S1, 25% S2, 63% S3 under increased risk), demonstrating genuine adaptive behavior. Under ReT_thesis, the agent collapsed to 99.99% S1.
2. *Service preservation:* control_v1-trained agents maintain fill rates comparable to the best static baselines (0.84 under increased risk, 0.63 under severe risk). ReT_thesis-trained agents degraded fill rate to 0.84 even under current (low) risk.
3. *Interpretability:* Each component of control_v1 maps directly to an observable operational quantity, making the reward transparent to supply chain practitioners.

### 3.3.5 Learning Algorithm

The agent is trained using Proximal Policy Optimization (PPO; Schulman et al., 2017), a policy-gradient algorithm with clipped surrogate objectives that provides stable learning in continuous action spaces. PPO was selected for its robustness to hyperparameter choices and its established track record in operations research applications (De Moor et al., 2022; Kemmer et al., 2018). The implementation uses Stable-Baselines3 (Raffin et al., 2021) with the hyperparameters listed in Table 7.

**Table 7.** PPO hyperparameters.

| Parameter | Value |
|-----------|-------|
| Learning rate | 3 × 10⁻⁴ |
| Rollout steps (n_steps) | 1,024 |
| Mini-batch size | 64 |
| Update epochs | 10 |
| Discount factor (γ) | 0.99 |
| GAE parameter (λ) | 0.95 |
| Clip range | 0.2 |
| Training timesteps | 500,000 |
| Policy architecture | MLP (64, 64) |

### 3.3.6 Evaluation Protocol

Each experimental condition is evaluated across 5 training seeds. For each seed, the trained policy is evaluated over 10 episodes with distinct simulation seeds. Three static baselines (S1-only, S2-only, S3-only) and a random policy are evaluated under identical conditions for comparison. Metrics are reported as seed-level means with bootstrap 95% confidence intervals. Two stress scenarios are examined:

- **Increased risk:** The baseline risk configuration from Garrido-Rios (2017), representing recurring operational and known-unknown disruptions.
- **Severe risk:** An escalated configuration in which disruption frequencies are increased 4–17× and recovery times are extended 2–3×, representing a high-stress operational theatre.

Stochastic processing times are enabled in both scenarios to capture the full variability of the DES environment.
