# SCRES-IA Must-Cite Bibliography (2026-06-28)

Union of references recommended across four advisor reports + the deep-research report. Trim to 15–20 for the
final manuscript per the target journal (IJPR / EJOR / M&SOM). Links are arXiv/publisher as supplied by advisors;
**verify each primary source before citing** (some links were ChatGPT-surfaced and need confirmation).

## Garrido core (anchor — verify against the two PDFs in repo)
- Garrido-Ríos, D. A. (2017). *A Mixed-Method Study on the Effectiveness of a Buffering Strategy in the Relationship
  between Risks and Resilience.* PhD thesis, University of Warwick. → MFSC, ReT, R1/R2/R3, buffers/shifts, 90 configs, 20yr.
- Garrido, Pongutá & Adarme (2024). *Enhancing the Operationalization of SCRES-Based Simulation Models with AI Algorithms:
  A Preliminary Exploratory Analysis.* → "Alzheimer effect" / open-loop DES; proposes NN/KAN/RL. (Univ. del Rosario)
- Garrido, Pongutá & García-Reyes (2024). *Zero-inventory plans, constant workforce, or hybrid approach? …factory resilience
  with demand variability.* IJPR. → Monte Carlo APP, Cobb-Douglas resilience index, S12 dominance.

## RL + inventory / supply chain
- Gijsbrechts, Boute, Van Mieghem & Zhang (2022). *Can Deep RL Improve Inventory Management?* M&SOM. (A3C; tuning-heavy)
- Madeka, Torkkola, Eisenach, Luo, Foster & Kakade (2022). *Deep Inventory Management.* arXiv:2210.03137
- Stranieri, Kouki, van Jaarsveld & Stella (2025). *Classical and Deep RL Inventory Control … Pharmaceutical … Perishability
  and Non-Stationarity.* arXiv:2501.10895 (PPO helps but doesn't universally dominate — key framing cite)
- Stranieri & Stella (2022). *Comparing DRL Algorithms in Two-Echelon Supply Chains.* arXiv:2204.09603
- Oroojlooyjadid, Nazari, Snyder & Takáč (2017). *A Deep Q-Network for the Beer Game.* arXiv:1708.05924
- Sultana et al. (2020). *RL for Multi-Product Multi-Node Inventory Management.* arXiv:2006.04037
- Leluc et al. (2023). *MARLIM: Multi-Agent RL for Inventory Management.* arXiv:2308.01649
- Meisheri et al. (2022). *Learning Framework for Uncertain Lead Times in Multi-Product Inventory.* arXiv:2203.00885
- Maggiar et al. (2025). *Structure-Informed Deep RL for Inventory Management.* arXiv:2507.22040
- Siems, Schambach, Schulze & Otterbach (2023). *Interpretable RL via Neural Additive Models for Inventory.* arXiv:2303.10382
- Sabal Bermúdez, del Rio Chanona & Tsay (2023). *Distributional Constrained RL for Supply Chain Optimization.* arXiv:2302.01727
- Hasturk et al. (2025). *Constrained RL for Dynamic Inventory Routing under Stochastic Supply/Demand.* arXiv:2503.05276
- Kotecha & del Rio Chanona (2024). *GNN + MARL for Inventory Control in Supply Chains.* arXiv:2410.18631
- Mousa et al. (2023). *Multi-Agent PPO for Decentralized Inventory Control.* arXiv:2307.11432
- Kotecha & del Rio Chanona (2025). *MORSE: Multi-Objective RL via Strategy Evolution for Supply Chain.* arXiv:2509.06490
  (Pareto fronts + CVaR — aligned with our framing)
- Wang, Wang & Sobey (2023). *Agent-based modelling for continuously varying supply chains.* arXiv:2312.15502 (PPO vs RPPO)

## DES / simulation + resilience
- Camur et al. (2023). *Integrated System Dynamics + DES for SCRES with Non-Stationary Pandemic Demand.* arXiv:2305.00086
- Che, Dong & Namkoong (2024). *Differentiable DES for Queuing Network Control.* arXiv:2409.03740
- Ramzy et al. (2022). *MARE: Semantic Disruption Management & Resilience Evaluation.* arXiv:2205.06499
- *Assessing supply chain Triple-R (responsiveness, resilience, robustness) by computer simulation: a systematic review.*
  IJPR 62(4), 2024.
- von Rueden et al. *Combining ML and Simulation: a Hybrid Modelling Approach.* (Springer) — sim+ML taxonomy.
- Zhang et al. (2024). *Coupling simulation and ML for predictive analytics in SCM.* IJPR 62(23).
- Chan et al. (2022). *Generation of synthetic manufacturing datasets (DES → ML training).* Prod. & Manuf. Research.
- Bruckler et al. *Resilience metrics taxonomy / resilience-curve capacities.*

## Architecture / learning mechanisms
- Parisotto et al. (2019). *Stabilizing Transformers for RL (GTrXL).* arXiv:1910.06764
- Hafner, Pasukonis, Ba & Lillicrap (2023). *Mastering Diverse Domains through World Models (DreamerV3).* arXiv:2301.04104
- Chen et al. (2021). *Decision Transformer: RL via Sequence Modeling.* arXiv:2106.01345
- Finn, Abbeel & Levine (2017). *MAML: Model-Agnostic Meta-Learning.* arXiv:1703.03400
- Duan et al. (2016). *RL²: Fast RL via Slow RL.* arXiv:1611.02779 (directly relevant to L_{t-1})
- Pan et al. (2025). *A Survey of Continual Reinforcement Learning.* arXiv:2506.21872
- Wang, Zhang, Su & Zhu (2023). *A Comprehensive Survey of Continual Learning.* arXiv:2302.00487
- Abbas et al. (2023). *Loss of Plasticity in Continual Deep RL.* arXiv:2303.07507 (methodological red flag for retained arm)
- Teece, Pisano & Shuen (1997). *Dynamic Capabilities and Strategic Management.* (framing — use with care)

## Adjacent (deep-research file; for closed-loop / predictive-accuracy framing)
- Knowledge-Assisted Deep RL (arXiv:2009.08346) — auxiliary structural signals improve convergence.
- Causal ML for supply-chain risk (intervention vs prediction); uncertainty-aware digital-twin RL.
