# Literature Positioning Matrix (2026-03-30)

## Purpose

This note replaces any broad "firstness" claim with a narrower and defensible positioning strategy.

As of **March 30, 2026**, the literature already contains:

- reinforcement learning for disrupted supply chains,
- multi-agent RL for resilience-oriented reconfiguration,
- PPO for disruption-aware multi-echelon inventory,
- reviews showing that DES + ML/RL in supply chains is now an active research line.

Therefore, the repo should **not** claim:

- "first recursive improvement analysis for supply chain resilience",
- "first self-learning SCRES study",
- "first learning-based resilience approach".

## Recommended Contribution Framing

The paper should instead position itself as:

1. A **Garrido-grounded reconstruction** of a 13-operation military food supply chain discrete-event simulation.
2. A **reproducible benchmark for operational resilience control under disruptions** with thesis-aligned static baselines.
3. An **auditable RL framework** that makes reward alignment, partial observability, and stress-regime-dependent performance explicit.

Short claim candidate:

> Prior work has applied RL and adaptive decision-making to disrupted supply chains and resilience-related problems. Our contribution is not a generic claim of first use of learning for SCRES, but a reproducible DES-based benchmark for operational resilience control grounded in Garrido-Rios' military food supply chain model, with exact thesis-aligned baselines, explicit disruption regimes, and auditable reward-resilience alignment.

## Positioning Matrix

| Paper | System / setting | Method | DES-based | Resilience explicit | Main decision problem | What it contributes | What remains distinct in this repo |
|---|---|---|---|---|---|---|---|
| Kim, Chen, and Linderman (2015), *Supply network disruption and resilience: A network structural perspective* | Supply **network** structures | Graph-theoretic / structural analysis | No | Yes | Structural vulnerability and resilience at network level | Establishes resilience as a network-structural property and separates node/arc disruptions from network-level resilience | Our repo is **not** about network redesign. It studies **operational control in a fixed military supply-chain structure** reconstructed from Garrido. |
| Gijsbrechts et al. (2022), *Can Deep Reinforcement Learning Improve Inventory Management?* | Inventory control; lost sales, dual sourcing, multi-echelon | DRL (A3C) | No | Not the primary construct | Inventory control under stationary settings | Shows DRL can match strong heuristics on classic inventory problems | Our repo is disruption-centric, DES-based, military, and explicitly tied to operational resilience rather than generic inventory cost optimization. |
| Kim et al. (2024), *A multi-agent reinforcement learning model for inventory transshipments under supply chain disruption* | Retail transshipment under disruption | MARL | Not DES-centered in our sense | Yes, disruption/resilience context | Inventory transshipment coordination | Demonstrates MARL for disruption-aware collaborative inventory decisions | Our repo focuses on a **single military MFSC process model**, not multi-retailer transshipment. It also includes exact Garrido thesis baselines. |
| Aboutorab et al. (2024), *Adaptive identification of supply chain disruptions through reinforcement learning* | Supply-chain risk information streams | RL + NLP | No | Indirectly | Disruption identification / early warning | Uses RL to identify disruption risks proactively | Our repo tackles **control** of operations inside a DES, not disruption classification or text-driven identification. |
| Ivanov-style recovery-policy work (2024), *Optimisation of recovery policies in the era of supply chain disruptions: a system dynamics and reinforcement learning approach* | Recovery after disruptions | RL + system dynamics | No, uses SD not DES | Yes | Recovery policy optimization | Shows RL can improve disruption recovery in a dynamic simulation setting | Our repo is a **detailed discrete-event military operations model**, not a system-dynamics abstraction, and includes explicit baseline replication against Garrido. |
| Lu et al. (2025), *Dynamic Optimization of Multi-Echelon Supply Chain Inventory Policies Under Disruptive Scenarios: A Deep Reinforcement Learning Approach* | Multi-echelon inventory under disruption regimes | PPO | No | Yes, but through cost/service under disruptions | Replenishment / inventory control | PPO beats classical policies as disruption severity rises; emphasizes disruption-aware inventory adaptation | Our repo does not study retailer replenishment rules; it studies **operational control in a 13-operation military food chain DES**, with service/cost/resilience reward auditing and thesis-derived static comparators. |
| Ding et al. (2026), *Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain system-of-systems under disruption risks* | SC system-of-systems (suppliers, manufacturers, distributors, consumers) | MARL / MAPPO in a POMDP | Simulation-based, but not Garrido-style DES replication | Yes, explicitly | Resilience-oriented **reconfiguration** | Strong prior art on resilience, disruption, POMDP, and learning-based reconfiguration with resilience/cost reward balancing | This is the closest neighboring paper. Our distinction must be **not architecture or firstness**, but a **Garrido-grounded operational-control benchmark** in a fixed MFSC DES with exact thesis-aligned baselines and reward-audit transparency. |
| Huber and Fikar (2025), *A literature review of supply chain analyses integrating discrete simulation modelling and machine learning* | Review of simulation + ML in supply chains | Review | Review | Mixed | Field overview | Confirms DES/ABS/SD + ML/RL is already an active line of work | This kills any claim that DES + ML/RL in supply chains is novel by itself. Our novelty must be the **specific benchmark, baseline fidelity, and operational resilience framing**. |

## What This Means for the Paper

### Claims to avoid

- Any absolute claim of being the first learning-based SC resilience study.
- Any claim that KAN, PPO, or DKANA alone makes the work novel.
- Any wording that implies no prior RL-based disruption control, recovery, or resilience work exists.

### Claims that remain defensible

- A **thesis-grounded DES reconstruction** of the Garrido-Rios MFSC.
- A **benchmark contribution** with exact Garrido-aligned static baselines.
- A **reward-design and auditing contribution** around operational resilience control.
- A **partial-observability and stress-regime analysis** for adaptive policies.

### Recommended positioning sentence

> Existing work already studies RL- and MARL-based supply-chain adaptation under disruption, including inventory optimization, recovery, transshipment, disruption identification, and resilience-oriented reconfiguration. This study contributes a different artifact: a reproducible, Garrido-grounded DES benchmark for operational resilience control in a fixed military food supply chain, with exact thesis-aligned baselines and explicit reward-resilience auditing.

## Practical Use in the Manuscript

Use this matrix to structure the literature review into four buckets:

1. **Structural / network resilience**
   - Kim et al. (2015)
2. **Inventory and transshipment RL**
   - Gijsbrechts et al. (2022)
   - Kim et al. (2024)
   - Lu et al. (2025)
3. **Disruption identification and recovery**
   - Aboutorab et al. (2024)
   - system dynamics + RL recovery paper (2024)
4. **Closest neighboring resilience-control work**
   - Ding et al. (2026)

Then position this repo as:

- fixed-structure,
- DES-grounded,
- military logistics,
- operational control,
- exact thesis baseline replication,
- reward-resilience auditability.

## Notes on Evidence Quality

- The rows above are based on publisher pages, abstracts, and public metadata available on **March 30, 2026**.
- For paywalled papers, the matrix should be updated after full-text reading before final submission.
- The 2026 IJPE paper is the most important comparison and should be read in full by a coauthor before freezing the final introduction.

## Source Links

- Kim, Chen, and Linderman (2015): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0272696314000746)
- Gijsbrechts et al. (2022): [Northwestern Scholars](https://www.scholars.northwestern.edu/en/publications/can-deep-reinforcement-learning-improve-inventory-management-perf)
- Kim et al. (2024): [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/24725854.2023.2217248)
- Aboutorab et al. (2024): [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0957417424003427)
- Recovery policies with SD + RL (2024): [Taylor & Francis](https://www.tandfonline.com/doi/abs/10.1080/00207543.2024.2383293)
- Lu et al. (2025): [MDPI](https://www.mdpi.com/2073-8994/17/12/2078)
- Ding et al. (2026): [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0925527326000861)
- Huber and Fikar (2025): [Taylor & Francis](https://www.tandfonline.com/doi/full/10.1080/17477778.2025.2500393)
