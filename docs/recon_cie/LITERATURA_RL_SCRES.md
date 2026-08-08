# Literature map: AI-for-supply-chain-resilience, for "Retained Search State Before Neural Architecture"

## 0. Method note and honest limits on this search

I used WebSearch plus Crossref/OpenAlex metadata APIs. **ScienceDirect, Taylor & Francis, SSRN and ResearchGate all returned 403 to automated fetches**, so for paywalled items I have verified *bibliographic* data against Crossref/OpenAlex but have **read abstracts/snippets, not full texts**, for: Ding et al. 2026, Preil & Krapp 2022, Rachman et al. 2026, Stranieri et al. 2024, Lang et al. 2026. Every "no one has done X" statement in §4 is therefore a **title/abstract-level absence claim**, not a full-text systematic screen. I give you the exact screening protocol you would need before writing the word "first" in print.

Existing repo bibliography: `<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/references.bib` (37 entries). Existing related work: `<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/sections/02_related_work.tex` — this is written for the **old** Track A/Track B action-space-alignment paper and does not survive the pivot to outer-loop search state. It must be rewritten, not patched: its entire gap argument ("does the policy's action space reach the bottleneck?") is a within-episode argument, i.e. exactly the layer the new paper's vocabulary rule forbids conflating.

**Bibliography defect found:** `kim2024` is wrong on both author and title. Actual record (Crossref-verified): Byeongmok Kim, Jong Gwang Kim, Seokcheon Lee, "A multi-agent reinforcement learning model for inventory transshipments under supply chain disruption", *IISE Transactions* 56(7):715–728, 2024, DOI 10.1080/24725854.2023.2217248. The bib currently says "Kim, Sungwook" and "Cooperative MARL model for…". Fix before submission.

---

## 1. The 22 works this manuscript MUST cite

Verification key: **[C]** = Crossref-verified this session; **[O]** = OpenAlex-verified; **[S]** = from search snippet, spot-check before submission; **[M]** = from prior knowledge, verify page numbers.

### A. The questions being answered (3)

1. Garrido, A., Pongutá, E., Adarme, W. (2024). *Enhancing the Operationalization of SCRES-Based Simulation Models with AI Algorithms: A Preliminary Exploratory Analysis.* ICCL 2024, LNCS 15168, pp. 80–94. Springer. **[M — already in bib]**
2. Garrido, A., Pongutá, E., García-Reyes, H. (2024). *Zero-inventory plans, constant workforce, or hybrid approach? Analysing pure production strategies for enhancing factory resilience with demand variability.* *International Journal of Production Research*. **[M — already in bib]**
3. Garrido-Ríos, A. (2017). *A Mixed-Method Study on the Effectiveness of a Buffering Strategy in the Relationship between Risks and Resilience.* PhD thesis. **[M — already in bib]**

### B. Direct competitors — RL/MARL for SC resilience and reconfiguration (6)

4. **Ding, W., Ming, Z., Wang, G., Yan, Y., Zhang, D. (2026).** *Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain system-of-systems under disruption risks.* *International Journal of Production Economics* **297**, 109995. DOI 10.1016/j.ijpe.2026.109995. **[C+O verified]** (Preprint: SSRN 5609791, DOI 10.2139/ssrn.5609791.)
5. **Kim, B., Kim, J.G., Lee, S. (2024).** *A multi-agent reinforcement learning model for inventory transshipments under supply chain disruption.* *IISE Transactions* **56**(7):715–728. DOI 10.1080/24725854.2023.2217248. **[C verified — fixes repo defect]**
6. **Bussieweke, F., Mula, J., Campuzano-Bolarín, F. (2025).** *Optimisation of recovery policies in the era of supply chain disruptions: a system dynamics and reinforcement learning approach.* *International Journal of Production Research*. DOI 10.1080/00207543.2024.2383293. **[S — verify volume/issue]**
7. **Li, Y., Krivtsov, V., Pan, Y., Nassehi, A., Gao, R.X., Ivanov, D. (2025).** *End-to-end supply chain resilience management using deep learning, survival analysis, and explainable artificial intelligence.* *International Journal of Production Research* **63**(3):1174–1202. DOI 10.1080/00207543.2024.2367685. **[S]** — cite as the *predictive* (non-control) neural SCRES strand.
8. **Preil, D., Krapp, M. (2022).** *Bandit-based inventory optimisation: Reinforcement learning in multi-echelon supply chains.* *International Journal of Production Economics* **252**, 108578. DOI 10.1016/j.ijpe.2022.108578. **[C verified]** — **the single most important addition to your bib.** See §4.
9. Kotecha, N., del Rio Chanona, A. (2025). *Leveraging graph neural networks and multi-agent reinforcement learning for inventory control in supply chains.* *Computers & Chemical Engineering*. **[M — already in bib]**

### C. Reviews that set reviewer expectations (5)

10. Yan, Y., Chow, A.H.F., Ho, C.P., Kuo, Y.-H., Wu, Q., Ying, C. (2022). *Reinforcement learning for logistics and supply chain management: Methodologies, state of the art, and future opportunities.* *Transportation Research Part E* **162**, 102712. **[M]**
11. Rolf, B., Jackson, I., Müller, M., Lang, S., Reggelin, T., Ivanov, D. (2023). *A review on reinforcement learning algorithms and applications in supply chain management.* *International Journal of Production Research* **61**(20):7151–7179. **[M]**
12. Boute, R.N., Gijsbrechts, J., van Jaarsveld, W., Vanvuchelen, N. (2022). *Deep reinforcement learning for inventory control: A roadmap.* *European Journal of Operational Research* **298**(2):401–412. **[M — note: repo bib lists the wrong author set (Boute/Gijsbrechts/Van Mieghem/Zhang); verify]**
13. Badakhshan, E., Mustafee, N., Bahadori, R. (2024). *Application of simulation and machine learning in supply chain management: A synthesis of the literature using the Sim-ML literature classification framework.* *Computers & Industrial Engineering*. **[M — already in bib; C&IE self-citation, keep it]**
14. Kogler, C., Maxera, P. (2026). *A literature review of supply chain analyses integrating discrete simulation modelling and machine learning.* *Journal of Simulation*. DOI 10.1080/17477778.2025.2500393. **[S]**

### D. The evidence base that strong non-neural comparators are hard to beat (4)

15. Gijsbrechts, J., Boute, R.N., Van Mieghem, J.A., Zhang, D.J. (2022). *Can deep reinforcement learning improve inventory management? Performance on lost sales, dual sourcing, and multi-echelon problems.* *M&SOM* **24**(3):1349–1368. **[M]**
16. Temizöz, T., Imdahl, C., Dijkman, R., Lamghari-Idrissi, D., van Jaarsveld, W. (2025). *Deep Controlled Learning for Inventory Control.* *European Journal of Operational Research*. **[M]** — carries the strongest single sentence in the field for your purposes: no existing DRL approach consistently surpasses the capped base-stock policy in lost-sales control.
17. Stranieri, F., Stella, F., Kouki, C. (2024). *Performance of deep reinforcement learning algorithms in two-echelon inventory control systems.* *International Journal of Production Research*. **[M]**
18. Powell, W.B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions.* Wiley. **[M]** — the canonical citation for "neural value-function approximation is one policy class among four."

### E. Measurement, and the C&IE anchor (2)

19. **Bruckler, M., Wietschel, L., Meßmann, L., Thorenz, A., Tuma, A. (2024).** *Review of metrics to assess resilience capacities and actions for supply chain resilience.* *Computers & Industrial Engineering* **192**, 110176. DOI 10.1016/j.cie.2024.110176. **[O verified; 78 citations]**
20. **Lang, S.M., Knak, L., Zanker, C., Glöser-Chahoud, S. (2026).** *Simulation-driven optimization for resilient and sustainable supply chain operations.* *Computers & Industrial Engineering* **217**, 112063. DOI 10.1016/j.cie.2026.112063. **[C verified]** — a 2026 C&IE paper that uses a genetic algorithm for resilience optimization over a simulation and explicitly names RL and Bayesian optimization as *future* work. It is your venue-adjacent proof that the outer-loop-search framing is live at C&IE and still open.

### F. The methodological controls we import from outside OR (2, both mandatory)

21. **Bischl, B., Kerschke, P., Kotthoff, L., Lindauer, M., Malitsky, Y., Fréchette, A., Hoos, H., Hutter, F., Leyton-Brown, K., Tierney, K., Vanschoren, J. (2016).** *ASlib: A benchmark library for algorithm selection.* *Artificial Intelligence* **237**:41–58. DOI 10.1016/j.artint.2016.04.003. **[C verified]** — this is where the **single-best-solver (SBS)** baseline comes from. Your "state-blind replay of its own search marginals" is the SBS/marginal analogue, and citing ASlib is what converts it from an ad-hoc control into a recognised standard.
22. **Yang, A., Esperança, P.M., Carlucci, F.M. (2020).** *NAS evaluation is frustratingly hard.* ICLR 2020. arXiv:1912.12522. **[S/dblp verified]** — the canonical demonstration that a search method must be scored as *relative improvement over the randomly sampled average from its own search space*, or the search space does the work and the method takes the credit. This is precisely RQ2's logic, already established in another field.

### Strongly recommended (add if space allows — 6 more)

23. Kerschke, P., Hoos, H.H., Neumann, F., Trautmann, H. (2019). *Automated algorithm selection: Survey and perspectives.* *Evolutionary Computation* **27**(1):3–45. **[M]**
24. Feurer, M., Springenberg, J.T., Hutter, F. (2015). *Initializing Bayesian hyperparameter optimization via meta-learning.* AAAI-15. **[S verified]** — retained search state across tasks, and the warm-start-vs-cold-start contrast, in its original form.
25. Agarwal, R., Schwarzer, M., Castro, P.S., Courville, A., Bellemare, M.G. (2021). *Deep reinforcement learning at the edge of the statistical precipice.* NeurIPS 2021. **[S verified]**
26. Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D. (2018). *Deep reinforcement learning that matters.* AAAI-18. arXiv:1709.06560. **[S verified]**
27. Mania, H., Guy, A., Recht, B. (2018). *Simple random search of static linear policies is competitive for reinforcement learning.* NeurIPS 31. **[S verified]** — and Rajeswaran, A., Lowrey, K., Todorov, E., Kakade, S. (2017), *Towards generalization and simplicity in continuous control*, NeurIPS 30. Together these are the ML-side precedent for RQ3's negative: linear/random carriers matching neural ones once the comparison is matched.
28. Hosseini, S., Ivanov, D., Dolgui, A. (2019). *Review of quantitative methods for supply chain resilience analysis.* *Transportation Research Part E* **125**:285–307. **[M — already in bib]**

---

## 2. Bruckler et al. 2024, C&IE 192:110176 — what it contributes and what it leaves

**What it actually does** (OpenAlex-reconstructed abstract, verified): it builds on the **resilience curve** framework; it **harmonises the terminology** for resilience capacities (absorptive / adaptive / restorative) and for the metrics that quantify them; it **identifies 17 metrics spanning all characteristics of the resilience curve** (depth of impact, recovery rate, time-to-recovery, performance loss integral, etc.); it then shows how those metrics are used to **evaluate the effectiveness of resilience actions**; and it **catalogues proposed actions and classifies them against conventional SCM functions**. It supplies **mathematical formulations** ("derived formulations") so that practitioners can operationalise each capacity, and it explicitly invites researchers to **embed these metrics into optimization models** and study trade-offs among economic, environmental and resilience objectives. Secondary finding worth quoting: only 52 of 220 screened studies present metrics grounded in economic, quality or availability criteria, and most economic metrics reduce to disruption/recovery cost or lost profit.

**The gap it leaves that we fill — stated precisely.** Bruckler et al. standardise *what to measure on a single realised trajectory*. Every one of the 17 metrics is a functional of one performance curve under one configuration. The review is silent on the **evaluation of the procedure that chooses the configuration**. There is no metric in the taxonomy for the cost of *finding* a resilient configuration, no notion of **development regret** across a sequence of design evaluations, and no control for whether an adaptive search procedure's apparent advantage is attributable to retained state or to the marginal distribution of the design space it happened to be given. Our contribution sits exactly there: we take their measurement layer as given and add the missing layer above it — **outer-loop search efficiency, measured with a state-blind marginal-replay control** — and we do so in the journal that published their harmonisation.

**What it does NOT leave — do not overclaim these.** (a) It does not leave the resilience *metric* unresolved: if a reviewer asks "why this resilience index?", Bruckler is the answer, and we should map our reported index onto one of their 17 rather than argue for a new one. (b) It does not leave the *terminology* open: use their absorptive/adaptive/restorative vocabulary, and do not invent parallel terms. (c) It does not leave a "no one measures resilience properly" gap — 220 studies were screened. (d) It says essentially **nothing about AI, ML or RL**, which cuts both ways: we cannot cite it as evidence that AI-SCRES metrics are unsettled, only as evidence that the metric layer is settled *and orthogonal* to the search-procedure layer we address.

---

## 3. Ding et al. 2026 IJPE — fair positioning

**Where it is genuinely strong.**
- It is the correct formulation for its question: reconfiguration under disruption as a POMDP, three coordinated strategy classes (filling, repairing, recruiting), reward jointly weighting resilience and cost. That is a harder and more realistic control problem than most SC-RL papers attempt.
- It benchmarks against **two credible contemporary MARL algorithms** (QMIX, MADDPG), not against a straw rule.
- It runs **three disruption scenarios plus a generalization analysis** over network attributes — more external-validity effort than the median paper in this literature.
- It is in IJPE, funded by NSFC, and it is *current* (vol. 297, 2026). Any reviewer working in this area will know it. Treating it dismissively would be a mistake.

**Where our evidence is stronger.**
- **Hypothesis-class closure.** Every one of its baselines is itself a neural multi-agent learner. Its design can establish *which neural learner is best*; it structurally cannot establish *that a neural carrier is needed*. Our RQ3 asks the question its design forecloses, and answers it negatively.
- **The marginal-replay control.** Nothing in its reported structure separates "the learner transferred something" from "the learner's search space and visit marginals were favourable". Our RQ2 makes that separation explicit and prospective.
- **Statistical custody.** Convergence curves without confidence intervals, no seed custody, no preregistered falsifiers. We have virgin disjoint seeds, preregistered falsifiers that state why they can fail, and interval estimates. Under the Agarwal et al. (2021) standard this is a material difference, and it is the kind of difference C&IE reviewers increasingly ask about.
- **Prospective expansion.** Our 288 → 4,608 expansion is a genuine out-of-sample enlargement of the design space; their "generalization analysis" is an attribute sweep within the same construction.

**Where their evidence is stronger — say so.**
- **Scale and realism of the decision object.** A multi-agent system-of-systems with three intervention classes is a bigger, more managerially recognisable artefact than a single-facility DES with six thesis-derived panels.
- **Multi-scenario disruption coverage.** Three distinct disruption regimes vs. our inherited demand process, which is a real external-validity limit on our side and should be conceded in Limitations.
- **Positive result.** They report a working method. We report a refinement plus a negative on the neural question. Positive results travel further; we compensate with control quality, not with claims.

**Draft sentences for Related Work** (use verbatim or nearly so; the vocabulary rules are respected):

> The closest contemporary comparator is Ding et al. (2026), who cast supply-chain system-of-systems reconfiguration as a POMDP over filling, repairing and recruiting actions and solve it with MAPPO, benchmarked against QMIX and MADDPG across three disruption scenarios and an attribute-level generalization analysis. Their design settles which neural multi-agent learner performs best on that control problem, but because every comparator in the study is itself a neural learner, it cannot address whether a neural carrier is required at all, and it reports no control distinguishing genuine transfer from a favourable search space. The present study is complementary rather than competing: it operates in the outer loop between simulation runs rather than within an episode, and it supplies the two comparisons that closed hypothesis classes cannot provide — a matched non-neural stateful search carrier, and a state-blind replay of the carrier's own search marginals under a prospective sixteen-fold expansion of the design space.

If you want a third sentence for the Table 1 analogue: their Table 1 grid ends with a "This Paper" row; we should build the same object but with **columns they cannot fill** — *non-neural comparator present*, *marginal-replay control*, *seed custody*, *interval estimates on the headline*, *prospective expansion*. That is the single most efficient rhetorical move available against a strong competitor, and it is honest because those columns are real design features, not spin.

---

## 4. THE KEY POSITIONING QUESTION

**Short answer: in supply-chain resilience, no — with two qualifications that force us to narrow the claim, and one precedent that we must cite rather than reinvent.**

### 4a. Nothing found in SC resilience that does both things

I searched title/abstract level across nine query formulations for: matched non-neural stateful comparators, marginal/visit-frequency replay controls, state-blind or observation-blinded placebo arms, and warm-start-vs-cold-start contrasts. **No SC-resilience paper does both (matched non-neural stateful comparator) and (control for replay of own marginals).** Ding et al. 2026 does neither. Bussieweke et al. 2025 (SD + RL recovery policies) does neither. Kim et al. 2024 does neither. Li et al. 2025 is predictive, not a search study.

### 4b. Two near-misses that force us to narrow

**(i) Preil & Krapp (2022), IJPE 252:108578 — "Bandit-based inventory optimisation".** A UCB-family bandit (PQ-UCB) applied to multi-echelon inventory, in our exact target-adjacent venue. **This means our carrier is not novel as an algorithm.** A reviewer who knows this paper will say "bandits for SC decisions have been in IJPE since 2022." We must cite it, concede it, and be precise that our novelty is (a) the bandit operating in the **outer loop across runs and contexts** rather than as an in-problem optimiser, and (b) the **control design**, not the estimator. Do not let this paper be discovered by a reviewer rather than by us.

**(ii) Rachman, R., Tingey, J., Allmendinger, R., Shukla, P., Pan, W. (2026), EJOR — "Reinforcement Learning for Multi-Objective Multi-Echelon Supply Chain Optimisation" (arXiv:2507.19788).** This benchmarks multi-objective RL against a **multi-objective evolutionary algorithm**. An MOEA *is* a stateful non-neural search carrier (the population is the retained state). This is the closest genuine instance of neural-vs-non-neural-stateful in the SC literature. **It is not resilience**, it does not run a marginal-replay control, and it compares final solution quality rather than retained search state across an expanded design space — but it is close enough that "first to compare a neural learner against a non-neural stateful comparator in supply chains" would be **false**, and a reviewer could produce it. Narrow accordingly. Add: Lang et al. (2026) C&IE uses a GA for resilience optimization over a simulation and names RL as future work — so the GA-for-SCRES-search side is already occupied too.

### 4c. The marginal-replay control exists — outside OR — and citing it makes us stronger, not weaker

The control we call "state-blind replay of its own search marginals" is a recognised standard in three adjacent fields:
- **Algorithm selection**: the *single-best-solver* baseline, and the VBS–SBS gap as the measure of how much per-instance selection can possibly buy (Bischl et al. 2016; Kerschke et al. 2019). Our RQ2 is structurally the SBS test.
- **Neural architecture search**: relative improvement over the randomly sampled average architecture from the same search space, precisely because engineered search spaces make every method look good (Yang, Esperança & Carlucci, ICLR 2020).
- **Meta-learned warm-starting of Bayesian optimization**: the cold-start-vs-transferred-initialisation contrast (Feurer, Springenberg & Hutter, AAAI 2015).

**This is good news, not bad.** A methodological control that is standard elsewhere is far more defensible than one we invented for this paper. The claim to make is: *the control is standard in algorithm selection and NAS; it has not been applied in supply-chain resilience; when we apply it, the neural advantage does not survive it.* That is unattackable in a way that "we invented a new control" is not.

### 4d. The claim we can defend, in one sentence

> To our knowledge, no prior study in supply-chain resilience evaluates an outer-loop search carrier against both a matched non-neural stateful comparator and a state-blind replay of the carrier's own search marginals; the marginal-replay control is standard in algorithm selection and neural architecture search but has not, to our knowledge, been transferred to this domain.

**Before that sentence goes to print**, run this screen and report it: Scopus/WoS, 2020–2026, `TITLE-ABS-KEY(("supply chain" OR logistic*) AND ("resilien*" OR disrupt* OR reconfigur*) AND ("reinforcement learning" OR "deep learning" OR "neural network" OR bandit OR "Bayesian optimi*") AND (baseline OR ablation OR control OR comparator OR "random search" OR heuristic))`, restricted to C&IE, IJPE, IJPR, EJOR, Omega, M&SOM, TRE, IISE Trans, JoS. Inclusion criterion: does the paper contain at least one comparator that is (a) non-neural **and** (b) carries state across evaluations? Report the count. If the count is zero outside Rachman et al., the "to our knowledge" is earned; if not, cite what you find and narrow further. Reviewers at C&IE routinely ask for exactly this and it costs one paragraph.

---

## 5. Standard reviewer critiques of RL-for-SC papers, with sources

These are the nine you will face. Six of them we are already immune to; note which.

| # | Critique | Source to cite | Our exposure |
|---|---|---|---|
| 1 | **Weak or stale baselines** — the comparator is a naive rule, not the best-known structured heuristic | Gijsbrechts et al. 2022 (DRL matches but rarely decisively beats SOTA heuristics); Temizöz et al. 2025 (no DRL method consistently beats capped base-stock in lost sales); Boute et al. 2022 | Immune — this is our RQ3 design |
| 2 | **Asymmetric tuning budget** — only the proposed method is tuned | Boute et al. 2022 roadmap; Henderson et al. 2018 | Immune if budget parity is stated explicitly per arm; **state it** |
| 3 | **Point estimates, no uncertainty, too few seeds** | Agarwal et al. 2021 (NeurIPS outstanding paper); Henderson et al. 2018 | Immune — but *show* the intervals on the headline figure, which Ding et al. do not |
| 4 | **Seed and implementation variance / non-reproducibility** | Henderson et al. 2018; Stranieri et al. 2024 (open-sourcing as the field's response) | Immune — seed custody is a differentiator, use it |
| 5 | **Reward–metric mismatch** — the training reward is reported as if it were the business outcome | Boute et al. 2022; Ng, Harada & Russell 1999 for the invariance theory | Immune, and we have a stronger version: our own measurement shows the naive index rewards abandonment, so we do not train on it |
| 6 | **Degradation off the training distribution / non-stationarity** | Dehaybe, Catanzaro & Chevalier 2024 (EJOR, non-stationary demand); Rolf et al. 2023 | **Exposed** — "the inherited demand process" is a real scope limit. Concede in Limitations; do not let a reviewer find it first |
| 7 | **No standardised benchmark; results not comparable across papers** | Yan et al. 2022; Rolf et al. 2023 | Partially exposed — mitigate by reporting on a Bruckler-mapped metric |
| 8 | **Simulator fidelity not established** | Hosseini, Ivanov & Dolgui 2019; Badakhshan et al. 2024; Kogler & Maxera 2026 | **Exposed and already handled correctly** — the "prospective reproduction of six thesis-derived comparative panels, not validation" framing plus the disclosed sumBt gap (>1.09% of 47,780 rows) is the honest move. Keep the number in the text; it converts a weakness into evidence of custody |
| 9 | **Attribution** — the gain is reported but never decomposed; no test of *what* carried it | Yang et al. 2020 (search space vs. search strategy); Bischl et al. 2016 (SBS gap); Mania et al. 2018 / Rajeswaran et al. 2017 (linear carriers match neural ones under matched comparison) | Immune — this *is* the paper |

**Critiques a reviewer will make of *us* specifically, which we should pre-empt:** (a) the negative on RQ3 is *development-stage* evidence, and the central claim already says so — do not let the abstract read stronger than the claim; (b) 4,608 configurations is a design-space expansion, not a new supply chain, so external validity is bounded by one DES; (c) our "regret" is development regret over an evaluation sequence, not managerial cost — define it against Bruckler's vocabulary in §2 so it cannot be confused with a resilience metric.

---

## 6. Positioning paragraph (draft, for Introduction or end of Related Work)

> Garrido, Pongutá and Adarme (2024) posed two questions to the resilience-simulation community: which family of AI algorithms best mimics the supply-chain-learning attribute, and how such an algorithm should be integrated into a discrete-event model for resilience assessment. They named backpropagation, Kolmogorov–Arnold networks and reinforcement learning as candidates, and described the absence of memory between simulation runs as an Alzheimer effect. The literature that has grown around that agenda has answered the second question far more thoroughly than the first. Reviews of reinforcement learning in supply chains (Yan et al., 2022; Rolf et al., 2023; Boute et al., 2022) and of hybrid simulation–machine-learning workflows (Badakhshan et al., 2024; Kogler and Maxera, 2026) document a large and growing body of integrations, while the resilience-measurement literature has converged on a harmonised set of seventeen curve-based metrics and their derived formulations (Bruckler et al., 2024). What remains unresolved is the first question, because the studies that could answer it are constructed so that they cannot. Ding et al. (2026) benchmark MAPPO against QMIX and MADDPG for system-of-systems reconfiguration; Kim et al. (2024) and Kotecha and del Rio Chanona (2025) compare multi-agent learners with one another; Li et al. (2025) apply deep learning to disruption prediction. In each case the comparison is within a single hypothesis class, so the studies establish which neural method is better without establishing that a neural carrier is what produced the gain. The inventory-control literature has repeatedly shown how consequential that omission is, finding that strong structured policies remain difficult for deep learners to beat (Gijsbrechts et al., 2022; Temizöz et al., 2025), and adjacent fields have formalised the corresponding controls: the single-best-solver baseline in algorithm selection (Bischl et al., 2016) and relative improvement over the randomly sampled average in neural architecture search (Yang et al., 2020), both of which exist to prevent a favourable search space from being credited to a search strategy. This paper transfers those controls into supply-chain resilience. We evaluate outer-loop search carriers on a Garrido-grounded discrete-event model, first across contexts at 288 configurations and then prospectively across a sixteen-fold expansion to 4,608, against two comparators that closed-hypothesis-class designs omit: a matched non-neural stateful carrier, and a state-blind replay of each carrier's own search marginals. The result refines Garrido et al.'s hypothesis rather than confirming it. The effective carrier of cross-run memory is persistent search state, and under a matched comparison it is not specifically neural.

---

## 7. What is and is not novel — honest statement

**Novel, and defensible:**
1. **The marginal-replay control applied in supply-chain resilience.** Standard in algorithm selection and NAS; to our knowledge not previously applied in this domain. The novelty is the *transfer plus the outcome*, not the invention of the control.
2. **Separating outer-loop search state from within-episode policy state as the object of study in SCRES.** The entire SC-RL literature we surveyed operates at the within-episode layer. This is a genuinely different question, and it is the question Garrido's Alzheimer-effect framing actually posed.
3. **A negative on the neural question obtained under a matched comparison** — with seed custody, preregistered falsifiers, and interval estimates. Negatives of this construction are rare in this literature and are the main reason the paper is publishable at C&IE rather than merely defensible.
4. **The prospective 288 → 4,608 expansion as the confirmation design.** Prospective enlargement of the design space, with the fit never reading the test block, is stronger than the retrospective sweeps that dominate the field.

**NOT novel — concede early and explicitly:**
1. **The bandit/UCB carrier itself.** Preil & Krapp (2022) put a UCB carrier in IJPE four years ago. Ours differs in loop position and factorisation, not in kind.
2. **Neural-vs-non-neural comparison in supply chains.** Rachman et al. (2026, EJOR) compare multi-objective RL against an MOEA; Lang et al. (2026, C&IE) use a GA for simulation-based resilience optimization. We are not first; we are first *with the marginal control, in resilience, under prospective expansion*.
3. **The finding that simple carriers match neural ones.** Established in continuous control (Mania et al., 2018; Rajeswaran et al., 2017) and in inventory control (Gijsbrechts et al., 2022; Temizöz et al., 2025). Our contribution is that it holds *in SCRES outer-loop search*, which nobody had checked.
4. **The resilience metric.** Bruckler et al. (2024) settled the metric layer. We adopt, we do not extend.
5. **The DES.** It reproduces a 2017 thesis, the original Simulink model is unavailable, and one ledger column is unreconstructed in >1.09% of 47,780 rows. Framed as prospective reproduction of six comparative panels, this is a strength. Framed as validation, it is a fatal overclaim.

**The one-line summary of the contribution:** *not a new learner, and not a new metric — a control that the field had not imported, applied prospectively to a sixteen-fold larger design space, which finds retained search state to be the carrier and neural architecture to be incidental.*

---

## 8. Concrete repo actions

- Fix `kim2024` in `<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/references.bib` (wrong author, wrong title — see §1 item 5).
- Verify `boute2022` author list; the repo currently lists Boute/Gijsbrechts/Van Mieghem/Zhang for the EJOR roadmap, which I believe should be Boute/Gijsbrechts/van Jaarsveld/Vanvuchelen.
- Add entries 4, 8, 19, 20, 21, 22 (Ding-verified DOI, Preil & Krapp, Bruckler, Lang, ASlib, NAS-eval) — these six carry the new paper's positioning.
- `<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/sections/02_related_work.tex` needs full replacement, not editing: its gap argument and its Table 1 columns ("Bottleneck action test", "Priv.-obs. defense") belong to the abandoned within-episode framing and would actively contradict the new vocabulary rules.

**Sources:**
- [Bruckler et al. 2024, C&IE 192:110176 (OpenAlex)](https://api.openalex.org/works/doi:10.1016/j.cie.2024.110176)
- [Bruckler et al. 2024 (Augsburg OPUS record)](https://opus.bibliothek.uni-augsburg.de/opus4/frontdoor/index/index/docId/112770)
- [Ding et al. 2026, IJPE 297:109995 (OpenAlex)](https://api.openalex.org/works/doi:10.1016/j.ijpe.2026.109995)
- [Ding et al. — SSRN preprint 5609791](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5609791)
- [Preil & Krapp 2022, IJPE 252:108578](https://www.sciencedirect.com/science/article/abs/pii/S0925527322001670)
- [Kim, Kim & Lee 2024, IISE Transactions 56(7)](https://www.tandfonline.com/doi/full/10.1080/24725854.2023.2217248)
- [Bussieweke, Mula & Campuzano-Bolarín 2025, IJPR](https://www.tandfonline.com/doi/full/10.1080/00207543.2024.2383293)
- [Li et al. 2025, IJPR 63(3):1174–1202](https://www.tandfonline.com/doi/abs/10.1080/00207543.2024.2367685)
- [Lang et al. 2026, C&IE 217:112063](https://www.sciencedirect.com/science/article/pii/S0360835226002640)
- [Temizöz et al. 2025, EJOR — Deep Controlled Learning](https://www.sciencedirect.com/science/article/pii/S0377221725000463)
- [Boute et al. 2022, EJOR — DRL for inventory control: a roadmap](https://www.sciencedirect.com/science/article/pii/S0377221721006111)
- [Rolf et al. 2023, IJPR — review of RL in SCM](https://www.tandfonline.com/doi/full/10.1080/00207543.2022.2140221)
- [Yan et al. 2022, TRE — RL for logistics and SCM](https://www.sciencedirect.com/science/article/abs/pii/S136655452200103X)
- [Kogler & Maxera 2026, Journal of Simulation](https://www.tandfonline.com/doi/full/10.1080/17477778.2025.2500393)
- [Rachman et al. 2026, EJOR — RL for multi-objective multi-echelon SC optimisation](https://arxiv.org/abs/2507.19788)
- [Bischl et al. 2016, Artificial Intelligence 237:41–58 — ASlib (Crossref)](https://api.crossref.org/works?query.bibliographic=ASlib+A+benchmark+library+for+algorithm+selection)
- [Yang, Esperança & Carlucci 2020, ICLR — NAS evaluation is frustratingly hard](https://arxiv.org/abs/1912.12522)
- [Feurer, Springenberg & Hutter 2015, AAAI — Initializing Bayesian HPO via meta-learning](https://ojs.aaai.org/index.php/AAAI/article/view/9354)
- [Mania, Guy & Recht 2018, NeurIPS — Simple random search of static linear policies](https://proceedings.neurips.cc/paper/2018/hash/7634ea65a4e6d9041cfd3f7de18e334a-Abstract.html)
- [Rajeswaran et al. 2017, NeurIPS — Towards generalization and simplicity in continuous control](https://arxiv.org/abs/1703.02660)
- [Agarwal et al. 2021, NeurIPS — Deep RL at the edge of the statistical precipice](https://dl.acm.org/doi/10.5555/3540261.3542505)
- [Henderson et al. 2018, AAAI — Deep Reinforcement Learning that Matters](https://arxiv.org/pdf/1709.06560)
- [Gijsbrechts et al. 2022, M&SOM — Can DRL improve inventory management?](https://www.researchgate.net/publication/329718922_Can_Deep_Reinforcement_Learning_Improve_Inventory_Management_Performance_on_Dual_Sourcing_Lost_Sales_and_Multi-Echelon_Problems)