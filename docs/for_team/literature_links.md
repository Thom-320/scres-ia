# Literature Links for the Paper

Compiled references relevant to our SCRES+RL paper. All links verified as of April 2026.

---

## The "Dangerous Competitor" (read this first)

**Ding et al. (2026)** — Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain system-of-systems under disruption risks
*International Journal of Production Economics*, 297, 109995
https://www.sciencedirect.com/science/article/abs/pii/S0925527326000861

Key difference from our work: they do **network reconfiguration** (topology changes, filling/repairing/recruiting strategies) on an abstract SCSoS. We do **operational control** (shifts, inventory, downstream dispatch) on a detailed thesis-grounded DES. Different control granularity, different system representation, different mechanism.

---

## RL in Supply Chain Management — Reviews

1. **Yan et al. (2022)** — Reinforcement learning for logistics and supply chain management: Methodologies, state of the art, and future opportunities
   *Transportation Research Part E*, 162, 102712
   https://www.sciencedirect.com/science/article/pii/S136655452200103X

2. **Rolf et al. (2023)** — A review on reinforcement learning algorithms and applications in supply chain management
   *IJPR*, 61(20), 7151–7179
   https://www.tandfonline.com/doi/abs/10.1080/00207543.2022.2140221

3. **Boute et al. (2022)** — Deep reinforcement learning for inventory control: A roadmap
   *EJOR*, 298(2), 401–412
   https://www.sciencedirect.com/science/article/pii/S0377221721006111

---

## DES + ML / DES + RL

4. **Rolf et al. (2025)** — A literature review of supply chain analyses integrating discrete simulation modelling and machine learning
   *Journal of Simulation*
   https://www.tandfonline.com/doi/full/10.1080/17477778.2025.2500393
   (72 papers reviewed; 82% published in last 5 years; RL is the dominant ML paradigm)

5. **Perez-Pena et al. (2025)** — Application of simulation and machine learning in supply chain management
   *Computers & Operations Research*
   https://www.sciencedirect.com/science/article/pii/S036083522400771X

---

## Inventory / Operational Control with RL (closest methodological neighbors)

6. **Gijsbrechts et al. (2022)** — Can Deep Reinforcement Learning Improve Inventory Management?
   *M&SOM*, 24(3), 1349–1368
   https://pubsonline.informs.org/doi/10.1287/msom.2021.1064
   (A3C, 3-6% optimality gap, fails to consistently beat base-stock)

7. **De Moor et al. (2022)** — Reward shaping to improve the performance of DRL in perishable inventory management
   *EJOR*, 301(2), 535–545
   https://www.sciencedirect.com/science/article/pii/S0377221721008948

8. **Dehaybe et al. (2024)** — Deep RL for inventory optimization with non-stationary uncertain demand
   *EJOR*, 314(2), 433–445
   https://www.sciencedirect.com/science/article/pii/S0377221723007646

9. **Geevers et al. (2024)** — Multi-echelon inventory optimization using deep reinforcement learning
   *CEJOR*, 32(3), 653–683
   https://link.springer.com/article/10.1007/s10100-023-00872-2
   (PPO via SB3, 6.6–16.4% cost reduction — shows PPO+MLP works fine)

10. **Temizoz et al. (2025)** — Deep Controlled Learning for Inventory Control
    *EJOR* (in press)
    https://www.sciencedirect.com/science/article/pii/S0377221725000463
    (New SOTA: ≤0.2% optimality gaps, outperforms all DRL and heuristics)

11. **Vanvuchelen et al. (2025)** — Continuous action representations to scale DRL for inventory control
    *IMA J. Management Mathematics*, 36(1), 51–66
    https://academic.oup.com/imaman/article/36/1/51/7900570

---

## SCRES / Resilience / MARL

12. **Rolf et al. (2024)** — Optimisation of recovery policies in the era of supply chain disruptions
    *IJPR*, 63(5), 1649–1673

13. **Kim (2024)** — Cooperative MARL model for inventory transshipment under supply chain disruption
    *IISE Transactions*, 56(7)

14. **Liu et al. (2025)** — Multi-Agent DRL for Multi-Echelon Inventory Management
    *Production & Operations Management*, 34(7), 1836–1856
    https://journals.sagepub.com/doi/10.1177/10591478241305863

15. **Kotecha & del Rio Chanona (2025)** — Leveraging GNN and MARL for inventory control in supply chains
    *Computers & Chemical Engineering*, 199, 109111
    https://www.sciencedirect.com/science/article/pii/S0098135425001152

16. **Rachman et al. (2026)** — RL for Multi-Objective Multi-Echelon SC Optimisation
    *EJOR* (in press)
    https://www.sciencedirect.com/science/article/pii/S0377221726001177

---

## Cobb-Douglas for Resilience

17. **Garrido et al. (2024)** — Zero-inventory plans, constant workforce, or hybrid approach?
    *IJPR*, 63(10), 3589–3607
    https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2425771
    (C-D factory resilience index — the basis for our ReT_seq_v1 reward)

18. **Haleh et al. (2018)** — Resilient Structure Assessment using Cobb-Douglas
    *International Journal of Technology*, 9(5)

---

## Target Venue

**IJPR Special Issue: "The Agentic Supply Chain: Entering a New Era in AI in SCM"**
Deadline: **June 30, 2026**
(Direct match — AI agent for military SCRES with DES)

---

## Key Takeaway for Positioning

We are NOT the first to use RL in supply chains (refs 1-3 review dozens of papers).
We are NOT the first to combine DES+RL (ref 4 reviews 72 papers).
Ding et al. (2026) already exists for SCRES+MARL.

What IS novel:
- First thesis-grounded DES+RL benchmark for military SCRES with a validated operational model
- First empirical demonstration that action-space alignment with the active bottleneck determines RL effectiveness
- First Cobb-Douglas resilience function used as RL reward
- Honest dual-track result: RL fails under one contract, succeeds under another
