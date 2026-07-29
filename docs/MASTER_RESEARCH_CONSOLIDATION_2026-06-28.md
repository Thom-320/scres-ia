# Master Research Consolidation (2026-06-28)

**Purpose:** Preserve ALL research proposals from 4 independent analyses (ChatGPT Pro deep research + 3 detailed strategy reviews) for the SCRES + neural learning paper. Nothing is discarded. This document is the single source of truth for the paper's experimental roadmap, architecture decisions, and literature.

---

## Sources

- **S1:** ChatGPT Pro Deep Research — "How to Make the SCRES Learning Claim Defensible" (structured literature review + architecture recommendation + experimental design for H1-H4).
- **S2:** Detailed strategy analysis #1 — "Blind Spot / Brutal Truth" (architecture ranking, reward design, two-stage policy, holding-cost calibration, H4 design, 10-week plan, 15+ must-cite papers).
- **S3:** Detailed strategy analysis #2 — "18 papers, reward tracks, reformulated hypotheses" (dense static frontier, auxiliary risk head, per-risk hazard, constrained RL, 13-week plan).
- **S4:** ChatGPT Pro consolidated research — architecture comparison (DES + recurrent RL + optional KAN), L_{t-1} formalization, closed-loop vs open-loop framing, hypothesis refinement.

---

## 1. Current State of the Project

### Confirmed result (the WIN)
- **Lane:** continuous_its × risk_obs+hazard × ReT_excel_delta × war φ4/ψ1.5 × h104.
- **Seeds:** 10 mixed (1-5, 8501-8505), 60k timesteps, 8 eval episodes.
- **Result:** Excel ReT Pareto win=TRUE, CVaR95 Pareto win=TRUE, primary win=TRUE.
- **Numbers:** Excel ReT=0.002142, CVaR95=5.22e9, resource_composite=0.241, frac_std=0.257.
- **Survived:** 40k→60k timesteps, 5→10 seeds, Kaggle rerun.
- **Commit:** bbe5e32 on codex/garrido-replication-experiments.

### Mechanism (why it wins — MIXED, not purely preventive)
1. **Risk-specific reactive:** active_R22→buffer UP (+0.16 mean corr), active_R13→UP (+0.12), active_R24→UP (+0.28). Different risks get different responses.
2. **Resource-efficient:** holds ~48% of max buffer, varies (frac_std=0.257), uses less resource than statics at same service.
3. **Mildly preventive:** weeks_since_last_R3→buffer UP (+0.20 in some seeds). "Overdue for black-swan → prepare." Not dominant.

### What is NOT yet confirmed
- **H4 (L_{t-1}):** retained−reset never tested on the continuous_its winning lane. Only tested on Discrete(18) where it was NULL (memory Δ ≈ 0).
- **Raw ReT dominance:** current win is Pareto (resource-charged), not raw ReT vs free statics.
- **Generalization:** only tested on φ4/ψ1.5 + h104. h260 did NOT confirm the signal.
- **Lead-lag anticipation:** buffer before shock evidence is weak. "Preventive" is a stretch.

---

## 2. Convergent Proposals (UNANIMOUS across all 4 sources)

These 9 points are agreed by every analysis. They form the backbone of the paper.

| # | Proposal | Consensus |
|---|---|---|
| C1 | **PPO MLP [64,64] + hazard features as primary architecture.** Not exotic architectures first. The win already came from this. | 4/4 |
| C2 | **H4 (retained−reset) on continuous_its is the highest-value untested experiment.** Must run on the winning lane, not the dead Discrete(18) lane. | 4/4 |
| C3 | **Hazard ablation is mandatory:** full > risk_obs only > base > shuffled hazard. Without this, "preventive" claim is unsupported. | 4/4 |
| C4 | **Lead-lag event study:** corr(buffer_t, risk_{t+k}), k=1..8 weeks. Proves prevention vs reaction. | 3/4 |
| C5 | **Dense static frontier:** 21 fractions (0.00..1.00 step 0.05) × 3 shifts. The 18-discrete frontier is insufficient for continuous-action comparison. | 3/4 |
| C6 | **Reward: do NOT destroy ReT with heavy holding cost.** Use constrained/budgeted approach or zero-cost raw ReT lane. | 4/4 |
| C7 | **Paper positioning: NOT "first RL for SCRES."** Position as "frontier-dependent learning value" or "first thesis-grounded test of Garrido's AI-SCRES proposal." | 4/4 |
| C8 | **Garrido et al. (2024) "Alzheimer effect" paper is the gap-defining citation.** They explicitly call for AI integration into DES-SCRES. | 4/4 |
| C9 | **Continual RL survey (Pan et al. 2025) for L_{t-1} formalization.** Provides the theoretical scaffolding for retained knowledge. | 4/4 |

---

## 3. All Unique Proposals (NONE discarded)

### 3.1 Architecture proposals

| # | Proposal | Source | Priority |
|---|---|---|---|
| U1 | **Two-stage policy:** π_init (pre-warmup buffer/shift selector) × π_weekly (PPO adaptive). Fixes the credit-assignment problem of learning the initial decision with 104-week-delayed reward. | S2, S3 | Sprint 2 |
| U2 | **Auxiliary risk-prediction head:** shared encoder predicts P(R_i in next 1/4/8 weeks). Loss = L_PPO + λ_1·BCE(next_R) + λ_2·MSE(next_disruption_hours). Forces encoder to learn predictive risk representation. | S2, S3 | Sprint 4 |
| U3 | **Recurrent PPO (LSTM/GRU) as ablation:** if LSTM without hazard ≥ MLP+hazard, memory was learned endogenously. If not, features are the mechanism. Reference: Wang, Wang & Sobey (2023) on PPO vs RPPO in continuously varying supply chains. | S1, S4, S2 | Sprint 4 ablation |
| U4 | **GTrXL/Transformer-over-history:** Parisotto et al. (2019) stabilized transformers for RL. Use only if MLP+hazard wins AND LSTM doesn't explain the mechanism. For 104 weekly steps with structured hazard features, may be overkill. | S2, S3, S1 | Future work |
| U5 | **DreamerV3/world model:** Hafner et al. (2023). Learns latent dynamics including risk process, plans by imagination. Conceptually perfect for anticipation, computationally heavy for SimPy DES. | all sources | Future work |
| U6 | **MAML/meta-learning:** Finn et al. (2017). Train to adapt fast to new disruption campaigns. Directly relevant for H4 (path dependency across campaigns). | S2, S3, S1 | After H4 confirms |
| U7 | **Decision Transformer:** Chen et al. (2021). Offline sequence modeling from generated DES trajectories (static/heuristic/PPO/oracle). Good for policy distillation, not for online H4. | S2, S3 | Second paper |
| U8 | **KAN as interpretable surrogate:** Kolmogorov-Arnold Networks for interpretability of the learned resilience function. Not as primary control engine. | S1, S4 | Optional |

### 3.2 Reward proposals

| # | Proposal | Source | Priority |
|---|---|---|---|
| U9 | **ReT_excel_delta_bootstrap:** ΔReT from completed orders + proxy for pending orders' future ReT value. Fixes the problem of prepositioning buffer 20 weeks before reward arrives. | S2 | Sprint 2 |
| U10 | **Potential-based reward shaping (PBRS):** r'_t = r_t + γ·Φ(s_{t+1}) − Φ(s_t), where Φ(s) = −α·BO_pending − β·Lost + η·InventoryCoverage. Policy-invariant (preserves optimal policy), improves learning efficiency. | S2 | Sprint 2 |
| U11 | **Budgeted/constrained reward:** max E[ReT] s.t. E[resource] ≤ B. Implemented as r_t = ΔReT_t − λ·max(0, resource_t − B)². More elegant than holding cost for "ReT at equal resource." References: Sabal Bermúdez et al. (2023), Hasturk et al. (2025). | S3 | Sprint 2 |
| U12 | **Empirical holding-cost calibration:** λ_b = c · median(|ΔReT_step|) / median(buffer_frac), with c ∈ {0, 0.05, 0.10, 0.20, 0.40}. Select on calibration seeds, not test. | S2 | Sprint 2 |
| U13 | **Two reward tracks:** (A) raw ReT maximization (holding_cost=0, max ReT, resources reported not optimized), (B) Pareto constrained ReT (budgeted/constrained). Don't mix them. | S2, S3 | Sprint 2 |

### 3.3 L_{t-1} formalization

| # | Proposal | Source |
|---|---|---|
| U14 | **L_{t-1} = θ_e (policy weights):** the simplest interpretation. θ_{e+1} = U(θ_e, τ_e). Static baseline: θ_{e+1} = θ_e. Learning model: θ_{e+1} ≠ θ_e. | S1, S4 |
| U15 | **L_{t-1} = b_t (belief memory):** [weeks_since_last_Rxx, ewma_rate_Rxx, recent overlaps]. Explicit empirical hazard from realized history. | S2, S3 |
| U16 | **Two-level formalization:** L_{t-1} = (b_t, θ_{k-1}) — belief memory within episode + retained policy across campaigns. The policy is: a_{k,t} ~ π_{θ_{k-1}}(s_{k,t}, b_{k,t}). | S2 |
| U17 | **Cumulative learning stock:** L_e = Σ w_j · ℓ(τ_j), regression of later resilience on L_e while controlling for severity and shock type. | S1, S4 |
| U18 | **Formal update rule:** S_{t+1} = G_DES(S_t, A_t, D_t, ε_t); L_t = U_θ(L_{t-1}, S_t, D_t, A_t, Y_t); A_t ~ π_θ(A_t | S_t, D_t, L_{t-1}); R_t = F(S_t, D_t, L_{t-1}). | S4 |

### 3.4 H4 experimental design

| # | Proposal | Source |
|---|---|---|
| U19 | **Campaign tapes:** R1-heavy, R2-heavy, R3-overdue, mixed. Each with ρ ∈ {0.5, 0.7, 0.9}. Fixed tape per campaign (risk_id, start, duration, magnitude, affected_ops, demand realization). All arms receive same tape. | S2, S3 |
| U20 | **5 arms:** static charged frontier, hazard heuristic, frozen PPO (θ₀), reset PPO (θ₀ + prev campaign only), retained PPO (θ₀ + all prior campaigns). | S2, S3 |
| U21 | **History-match vs history-mismatch:** policy trained on R2-heavy campaigns tested on new R2-heavy vs new R1-heavy. If R2-trained > R1-trained on R2 tape → specific path dependency. | S2 |
| U22 | **Shuffle campaign order** (negative control): if effect survives shuffle → not temporal path dependency. | S2, S3 |
| U23 | **No-update control:** with update budget=0, retained−reset must be 0. | S2 |
| U24 | **Same-seed CRN:** all arms receive identical shocks. | S2, S3 |
| U25 | **Retained-wrong-history:** trained on different regime family than test. | S2 |
| U26 | **Hierarchical statistical model:** ΔR_{i,j,k} ~ β_0 + β_1·k + β_2·hazard + β_3·1[history match] + u_i + v_j + ε_{ijk}. H4: β_0 > 0. H2: β_1 > 0. H4-causal: β_3 > 0. | S2, S3 |
| U27 | **Bootstrap:** hierarchical (seed × tape), not episodes as independent. | S2, S3 |

### 3.5 Metrics and evidence

| # | Proposal | Source |
|---|---|---|
| U28 | **TTR (time-to-recovery)** mean/p95 as SCRES metric alongside CVaR95. | S2, S3 |
| U29 | **Action entropy / frac_std** as adaptivity measure. | S3 |
| U30 | **Probability of buffer increase before risk onset** (lead-lag). | S2 |
| U31 | **Generalization:** train φ4/ψ1.5, test φ3/ψ1.5, φ5/ψ1.5, φ4/ψ1.25, different regime seeds, h52/h104/h260. | S2, S3 |
| U32 | **Event windows:** mean buffer 4 weeks before/during/4 weeks after each R event type. | S3 |
| U33 | **Dense static frontier:** 21 fractions (0.00, 0.05, ..., 1.00) × 3 shifts × free + resource-charged. | S2, S3 |
| U34 | **Hazard ablation levels:** base obs → risk_obs only → risk_obs+hazard → shuffled hazard → time-only. | S2, S3 |

### 3.6 Paper framing and positioning

| # | Proposal | Source |
|---|---|---|
| U35 | **Novelty claim:** "This study provides the first thesis-grounded empirical test of Garrido et al.'s AI-enabled SCRES simulation proposal by operationalizing learned supply-chain experience as an endogenous state variable (L_{t-1})." | S2, S3 |
| U36 | **Core insight:** "AI+DES improves SCRES only if the action space exposes a controllable frontier." SCRES learning value = g(frontier, observability, resource pricing, shock recurrence). | S2, S3 |
| U37 | **Closed-loop vs open-loop framing:** traditional DES = open-loop (memoryless across runs). Our framework = closed-loop (retained learning updates future decisions). This is Garrido's "Alzheimer effect" fix. | S4 |
| U38 | **Reformulated RQ:** "Under what operational conditions do neural policies embedded in a Garrido-grounded DES improve SCRES relative to static and reset-learning baselines under recurring disruptions?" | S2 |
| U39 | **Reformulated objective:** "To develop and empirically evaluate a Garrido-grounded hybrid DES–neural control benchmark that distinguishes static buffering, reactive adaptation, retained learning, and preventive resource allocation under recurring heterogeneous disruptions." | S2 |
| U40 | **Paper sentence:** "Our findings show that learning-enabled SCRES does not emerge simply by embedding a neural network into a simulation. It emerges when the simulated supply chain exposes a decision frontier in which historical disruption signals can be transformed into preventive resource-allocation routines." | S3 |

### 3.7 Reformulated hypotheses

| # | Old version | New version (tighter, testable) |
|---|---|---|
| H1 | "Learning beats statics" | **Dynamic neural policy achieves non-inferior Excel ReT and lower CVaR95 than charged static frontier under war-stress** (already confirmed). |
| H2 | "Improves over time" | **Dynamic policy uses significantly lower resource than statics achieving comparable resilience.** (confirmed). OR: performance improves with more disruption exposure (learning curve). |
| H3 | "Reduces volatility" | **Removing/shuffling hazard features reduces the dynamic policy's ReT/CVaR advantage** (ablation). |
| H4 | "L_{t-1} matters" | **Retaining policy state across campaigns improves cold-start performance on new campaigns vs reset policy** (retained−reset > 0). |

---

## 4. Must-Cite Papers (consolidated, 18+)

### Garrido core
1. **Garrido-Rios, D. A. (2017).** PhD thesis, University of Warwick. MFSC, ReT, R1/R2/R3, 90 configs, buffers, shifts.
2. **Garrido, Pongutá & Adarme (2024).** "Enhancing the Operationalization of SCRES-Based Simulation Models with AI Algorithms." Gap: DES Alzheimer effect. Proposes NN/KAN/RL.
3. **Garrido, Pongutá & García-Reyes (2024).** "Zero-inventory plans, constant workforce, or hybrid approach?" IJPR. Cobb-Douglas resilience, APP strategies, Monte Carlo.

### RL + inventory/supply chain
4. **Gijsbrechts, Boute, Van Mieghem & Zhang (2022).** "Can Deep Reinforcement Learning Improve Inventory Management?" MSOM. A3C in lost sales, dual sourcing, multi-echelon.
5. **Stranieri et al. (2025).** "Classical and Deep RL Inventory Control Policies for Pharmaceutical SCs." PPO helps in complex but doesn't always dominate classical. [arXiv:2501.10895]
6. **Madeka et al. (2022).** "Deep Inventory Management." Lead times, lost sales, model-based RL. [arXiv:2211.13292]
7. **Oroojlooyjadid et al. (2017).** "A Deep Q-Network for the Beer Game." DRL in partially observable SC. [arXiv:1708.05924]
8. **Stranieri & Stella (2022).** "Comparing DRL Algorithms in Two-Echelon SCs." DRL vs (s,Q). [arXiv:2206.10368]
9. **Kotecha & del Rio Chanona (2024/2025).** "MORSE: Multi-Objective RL via Strategy Evolution for SC Optimization." Pareto + CVaR. [arXiv:2509.06490]
10. **Sabal Bermúdez, del Rio Chanona & Tsay (2023).** "Distributional Constrained RL for SC Optimization." Constrained RL, variance reduction.
11. **Meishari et al. (2022).** "A Learning Based Framework for Handling Uncertain Lead Times." [arXiv:2208.10432]
12. **Siems et al. (2023).** "Interpretable RL via Neural Additive Models for Inventory." [arXiv:2310.01840]

### SCRES / DES / simulation
13. **Camur et al. (2023).** "Integrated System Dynamics and DES for SCRES with Non-Stationary Pandemic Demand." [arXiv:2305.00086]
14. **Aboutorab et al. (2024).** "RL-SCRI for Proactive Disruption Identification." NLP + RL for disruption detection.
15. **Che, Dong & Namkoong (2024).** "Differentiable DES for Queuing Network Control." [arXiv:2409.03740]

### Architectures / learning mechanisms
16. **Parisotto et al. (2019).** "Stabilizing Transformers for RL" (GTrXL). [arXiv:1910.06764]
17. **Hafner et al. (2023).** "DreamerV3: Mastering Diverse Domains through World Models." [arXiv:2301.04104]
18. **Chen et al. (2021).** "Decision Transformer: RL via Sequence Modeling." [arXiv:2106.01345]
19. **Finn, Abbeel & Levine (2017).** "MAML." [arXiv:1703.03400]
20. **Duan et al. (2016).** "RL²: Fast RL via Slow RL." RNN retains experience across episodes. [arXiv:1611.02779]

### Continual learning / L_{t-1}
21. **Pan et al. (2025).** "A Survey of Continual RL." [arXiv:2506.21872]
22. **Wang et al. (2023).** "A Comprehensive Survey of Continual Learning." Stability-plasticity tradeoff.
23. **Abbas et al. (2023).** "Loss of Plasticity in Continual Deep RL." Red flag: retained learner can degrade. [Nature]

### Theory
24. **Teece, Pisano & Shuen (1997).** "Dynamic Capabilities and Strategic Management." Resilience as dynamic capability.

---

## 5. Red Flags (Reviewer #2 attack list)

1. **Pareto win ≠ raw ReT dominance.** Current win is resource-charged. Report both free and charged frontiers.
2. **War-stress φ4/ψ1.5 is an extension**, not thesis-faithful. Defend as stress scenario, not as Garrido baseline.
3. **Excel ReT has quirks:** ReT>1 (38 values), non-monotone in buffer. Explain, preserve for continuity, report CVaR/service alongside.
4. **Mechanism is more reactive than preventive.** Need lead-lag + hazard ablation before saying "anticipatory."
5. **H4 not yet proven on the winning lane.** Don't claim L_{t-1} in the abstract until retained−reset confirms.
6. **Static frontier must be dense** (21 fractions, not 18 discrete). Otherwise "dynamic wins because static grid is coarse."
7. **Holding cost in reward ≠ resource in eval.** Report both and explain the relationship.
8. **Don't do outcome fishing.** Pre-register calibration/validation/held-out. Run confirmatory ONCE.
9. **Hazard features are manual engineering.** Can't claim "the network discovered temporal patterns" with explicit weeks_since_last. Use LSTM ablation or auxiliary head to show learned prediction.
10. **h260 non-confirmation.** Report as temporal boundary. Don't extend claims beyond confirmed horizon.

---

## 6. Unified Execution Plan (6 Sprints)

### Sprint 0 — Guardar + commitear (done in this commit)
This document.

### Sprint 1 — Cerrar mecanismo de la señal actual (1-2 días)
**Goal:** Does hazard memory matter? Is it preventive or reactive?

| Experiment | Config | Seeds | Output |
|---|---|---|---|
| A0: base obs only | continuous_its, φ4/ψ1.5, h104, ReT_excel_delta | 5 | Pareto + service |
| A1: risk_obs only | + active/recent risks | 5 | Pareto + service |
| A2: risk_obs+hazard | + weeks_since_last, EWMA | 5 | Pareto + service |
| A3: shuffled hazard | hazard with randomized labels | 5 | Pareto + service |
| Dense static frontier | 21 fractions × 3 shifts, free + charged | — | Frontier reference |
| Lead-lag event study | corr(buffer_t, risk_{t+k}), k=1..8 | from A2 | Prevention evidence |

**Success:** A2 > A1 > A0; A2 > A3 (shuffled collapses). Lead-lag shows pre-shock buffer increase.

### Sprint 2 — Mejorar ReT vía reward + two-stage (1-2 días)
**Goal:** Can we win raw ReT? Does two-stage init stabilize?

| Experiment | Config | Seeds |
|---|---|---|
| B1: raw ReT (no cost) | holding_cost=0, max ReT | 5 |
| B2: budgeted ReT | max ReT s.t. resource ≤ B (Lagrangian) | 5 |
| B3: two-stage (static init + dynamic weekly) | best static a₀ + PPO weekly | 5 |
| B4: ReT_delta_bootstrap | completed + pending proxy | 5 |
| B5: holding_cost sweep | c ∈ {0, 0.05, 0.10, 0.20} × empirical_scale | 5 each |

**Success:** B1 or B2 beats dense free static on raw ReT with CI>0. If not, Pareto claim stands.

### Sprint 3 — H4 retained−reset en continuous_its (2-3 días)
**Goal:** Does L_{t-1} have empirical value? (the paper's central contribution)

| Experiment | Config |
|---|---|
| Campaign tapes | R1-heavy, R2-heavy, R3-overdue, mixed (4 types × 10 tapes) |
| Arms | static frontier, frozen PPO θ₀, reset PPO, retained PPO |
| Eval | cold-before-update on new campaign |
| Negative controls | shuffle campaigns, no-hazard ablation, no-update |
| Statistical model | ΔR ~ β_0 + β_1·k + β_2·hazard + β_3·1[history match] + (1|seed) + (1|tape) |
| Seeds | 10 learner × 10 campaign tapes × 20-40 campaigns |

**Success:** retained − reset > 0 (β_0 > 0, CI95 > 0). Late > early (β_1 > 0). Shuffle destroys effect (β_3 matters).

### Sprint 4 — Auxiliary risk head + per-risk hazard + LSTM ablation (1 día)
**Goal:** Make the agent more preventive. Test if memory is learned or feature-engineered.

| Experiment | Config |
|---|---|
| C1: PPO + auxiliary hazard prediction head | shared encoder + BCE(next_R_family, 1/4/8w) |
| C2: per-risk hazard (not per-family) | weeks_since_last_R11, R12, ..., R24 |
| C3: LSTM ablation (no explicit hazard) | if LSTM-no-hazard ≥ MLP+hazard → learned memory |

### Sprint 5 — Generalization (1 día)
| Experiment | Config |
|---|---|
| D1: cross-regime eval | train φ4/ψ1.5, test φ3/ψ1.5, φ5/ψ1.5, φ4/ψ1.25 |
| D2: cross-horizon | h52, h104, h260 |

### Sprint 6 — Paper writeup
- Reframed RQ + hypotheses (H1-H4).
- Results table (Sprints 1-5).
- Limitations (h260, war-stress, reactive mechanism, H4 boundary).
- Future work (GTrXL, DreamerV3, MAML, KAN, Decision Transformer).
- Dense static frontier figures.
- Lead-lag event study figures.
- Retained−reset learning curves.

---

## 7. Architecture Decision Summary

| Rank | Architecture | L_{t-1} as | When to use | Status |
|---|---|---|---|---|
| 1 | **PPO MLP [64,64] + hazard features** | Explicit belief features | Primary (already wins) | ✅ Confirmed |
| 2 | **PPO + auxiliary risk-prediction head** | Learned predictive representation | After Sprint 1 confirms | Planned |
| 3 | **PPO LSTM/GRU** (ablation) | Hidden state | If testing endogenous memory | Ablation |
| 4 | **Two-stage: static-init + PPO-weekly** | θ_init + θ_weekly | If credit assignment is the bottleneck | Sprint 2 |
| 5 | **GTrXL/Transformer** | Sequence attention | If MLP+hazard wins and LSTM doesn't explain | Future |
| 6 | **Decision Transformer** | Offline sequence | If generating oracle/static/PPO trajectory dataset | Second paper |
| 7 | **DreamerV3/world model** | Latent dynamics + planning | If DES can be made differentiable/surrogate | Future |
| 8 | **MAML/meta-RL** | Meta-learned θ₀ | If H4 confirms and we want fast adaptation | After H4 |

---

## 8. Reward Design Decision Summary

| Track | Reward | Holding cost | Objective | When |
|---|---|---|---|---|
| A: Raw ReT | ReT_excel_delta | 0 | Max ReT, beat free static | Sprint 2 |
| B: Budgeted ReT | ReT_excel_delta − λ·max(0, resource−B)² | Implicit (via B) | Max ReT s.t. resource ≤ B | Sprint 2 |
| C: Pareto (current winner) | ReT_excel_delta | 0 | Pareto dominance (emergent) | Confirmed |
| D: Holding-cost lane | ReT_excel_delta − λ_b·buffer − λ_s·shift | Calibrated | Force timing → preventive | Sprint 2 |
| E: Bootstrap | ΔReT_completed + proxy_pending | Configurable | Better credit assignment | Sprint 2 |
| F: PBRS | r_t + γ·Φ(s_{t+1}) − Φ(s_t) | Via Φ | Accelerate learning, preserve optimum | Sprint 2 |

---

## 9. Warmup/Priming Reference

- **WARMUP["estimated_deterministic_hrs"] = 838.8** (the "~900h" recalled by the user). Waits for the first batch of Q=5000 rations at Op9 (trigger_op=9, trigger_quantity=5000).
- **Priming:** priming_shifts=2, priming_step_hours=168, max_priming_hours=2016 (12 weeks at 2 assembly shifts to reach operational state). This IS buffer-related preparation.
- **clear_backlog_after_priming = False** (default). The inherited priming backlog is NOT cleared (Fix 3A concern).
- **init_frac parameter:** the continuous_its wrapper supports a fixed pre-warmup prepositioning (fraction of I1344 buffer applied before the agent acts). The winning run used init_frac=1.0 (max buffer pre-positioned).

---

## 10. Document Provenance

- Generated: 2026-06-28
- Sources: ChatGPT Pro deep research + 3 independent strategy analyses
- Branch: codex/garrido-replication-experiments
- Current HEAD: bbe5e32
- Status: living document — update after each sprint
