## 0. BOTTOM LINE FIRST

Of your three novelty candidates, **one is genuinely defensible as new, one is rigor not novelty, and one is already published and you should cite it rather than claim it.** Details in §2. The real contribution is not on your candidate list — it is the *layer reassignment* (§2.4).

Marker convention below: **[V]** = verified this session against a primary/indexing source; **[M]** = from model knowledge, high confidence but **verify page numbers/volume before submission**.

---

## 1. THE SIX LITERATURES — CITATIONS

### 1.1 Simulation optimization & ranking-and-selection
This is your home methods literature and the current bib (`docs/manuscript_current/submission/elsevier/references.bib`, 37 entries, all supply-chain/RL) contains **none of it**. That is the single biggest citation gap.

- **[V]** Amaran, S., Sahinidis, N.V., Sharda, B., Bury, S.J. (2016). Simulation optimization: a review of algorithms and applications. *Annals of Operations Research* 240(1), 351–380.
- **[V]** Hong, L.J., Fan, W., Luo, J. (2021). Review on ranking and selection: A new perspective. *Frontiers of Engineering Management* 8(3), 321–343.
- **[V]** Eckman, D.J., Henderson, S.G., Shashaani, S. (2023). SimOpt: A testbed for simulation-optimization experiments. *INFORMS Journal on Computing* 35(2), 495–508.
- **[V]** Eckman, D.J., Henderson, S.G., Shashaani, S. (2023). Diagnostic tools for evaluating and comparing simulation-optimization algorithms. *INFORMS Journal on Computing* 35(2), 350–367. — **this is your reporting-conventions anchor; see §4.**
- **[M]** Fu, M.C. (ed.) (2015). *Handbook of Simulation Optimization*. Springer, ISOR vol. 216.
- **[M]** Chen, C.-H., Lee, L.H. (2011). *Stochastic Simulation Optimization: An Optimal Computing Budget Allocation*. World Scientific.
- **[M]** Chen, C.-H., Lin, J., Yücesan, E., Chick, S.E. (2000). Simulation budget allocation for further enhancing the efficiency of ordinal optimization. *Discrete Event Dynamic Systems* 10(3), 251–270.
- **[M]** Kim, S.-H., Nelson, B.L. (2006). Selecting the best system. In: *Handbooks in OR & MS, Vol. 13: Simulation*, Elsevier, ch. 17.
- **[M]** Nelson, B.L., Matejcik, F.J. (1995). Using common random numbers for indifference-zone selection and multiple comparisons in simulation. *Management Science* 41(12), 1935–1945.
- **[M]** Hsu, J.C. (1996). *Multiple Comparisons: Theory and Methods*. Chapman & Hall. — the MCB procedure; R&S reviewers expect MCB or an equivalent when you rank arms.
- **[M]** Xu, J., Nelson, B.L., Hong, L.J. (2010). Industrial Strength COMPASS. *ACM TOMACS* 20(1), art. 3.
- **[M]** Barton, R.R., Meckesheimer, M. (2006). Metamodel-based simulation optimization. In: *Handbooks in OR & MS, Vol. 13*, Elsevier.
- **[M]** Glynn, P., Juneja, S. (2004). A large deviations perspective on ordinal optimization. *Proc. Winter Simulation Conference*, 577–585. — the bridge from R&S to best-arm identification; cite it where you justify a bandit formulation of a design search.

**Why this matters for you:** your outer loop over 288→4,608 configurations *is* a ranking-and-selection problem with a shared-information structure. A methods reviewer will ask why you did not frame it as R&S. Answer it in one paragraph, citing Hong/Fan/Luo and Glynn/Juneja: R&S assumes independent (or GP-correlated) alternatives and allocates budget within one context; you are transferring allocation structure *across* contexts, which R&S does not address.

### 1.2 Bandits: UCB1, Thompson sampling, factorized/combinatorial

- **[M]** Lai, T.L., Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics* 6(1), 4–22.
- **[M]** Auer, P., Cesa-Bianchi, N., Fischer, P. (2002). Finite-time analysis of the multiarmed bandit problem. *Machine Learning* 47(2–3), 235–256. — **UCB1; your carrier's ancestor, must be cited.**
- **[M]** Thompson, W.R. (1933). On the likelihood that one unknown probability exceeds another in view of the evidence of two samples. *Biometrika* 25(3/4), 285–294.
- **[M]** Russo, D., Van Roy, B., Kazerouni, A., Osband, I., Wen, Z. (2018). A tutorial on Thompson sampling. *Foundations and Trends in Machine Learning* 11(1), 1–96.
- **[M]** Chapelle, O., Li, L. (2011). An empirical evaluation of Thompson sampling. *NIPS 24*, 2249–2257.
- **[M]** Bubeck, S., Cesa-Bianchi, N. (2012). Regret analysis of stochastic and nonstochastic multi-armed bandit problems. *Foundations and Trends in Machine Learning* 5(1), 1–122.
- **[M]** Lattimore, T., Szepesvári, C. (2020). *Bandit Algorithms*. Cambridge University Press.

**Factorized / combinatorial — this is the sub-literature your carrier sits in and you must own it:**
- **[V]** Zimmert, J., Seldin, Y. (2018). Factored Bandits. *NeurIPS 31*. arXiv:1807.01488. — **NOTE: NeurIPS, not ICML. A wrong venue here is exactly the kind of error a methods reviewer notices.**
- **[M]** Chen, W., Wang, Y., Yuan, Y. (2013). Combinatorial multi-armed bandit: General framework and applications. *ICML 2013*, PMLR 28(1), 151–159.
- **[M]** Kveton, B., Wen, Z., Ashkan, A., Szepesvári, C. (2015). Tight regret bounds for stochastic combinatorial semi-bandits. *AISTATS 2015*, PMLR 38, 535–543. — CombUCB1.
- **[M]** Cesa-Bianchi, N., Lugosi, G. (2012). Combinatorial bandits. *Journal of Computer and System Sciences* 78(5), 1404–1422.

**Adaptivity as a measurable quantity — directly relevant to your marginal-replay control:**
- **[M]** Perchet, V., Rigollet, P., Chassang, S., Snowberg, E. (2016). Batched bandit problems. *Annals of Statistics* 44(2), 660–681.
- **[M]** Agarwal, A., Agarwal, N., Assadi, S., Khanna, S. (2017). Learning with limited rounds of adaptivity. *COLT 2017*, PMLR 65, 39–75.
- **[V]** The "limited adaptivity / adaptivity barrier" line is active through 2025 (e.g. arXiv:2511.03708, batched nonparametric bandits). **This literature asks your exact question — "what does sequential adaptation buy over a fixed allocation?" — but answers it with regret bounds, not with an empirical control.** That gap is your opening in §3.

**Justifying the *factorization* specifically:**
- **[V]** Hutter, F., Hoos, H., Leyton-Brown, K. (2014). An efficient approach for assessing hyperparameter importance. *ICML 2014*, PMLR 32(1), 754–762. — functional ANOVA; the empirical case that response surfaces over configuration spaces are dominated by low-order (mostly main) effects. **This is the citation that makes a factorized UCB a principled choice rather than a convenience, and it also predicts why your neural carrier gains nothing.**
- **[V]** Bergstra, J., Bengio, Y. (2012). Random search for hyper-parameter optimization. *JMLR* 13, 281–305. — low effective dimensionality; also the canonical null.

### 1.3 Bayesian optimization: transfer, warm-starting, meta-learning

Foundations:
- **[M]** Jones, D.R., Schonlau, M., Welch, W.J. (1998). Efficient global optimization of expensive black-box functions. *Journal of Global Optimization* 13(4), 455–492.
- **[M]** Snoek, J., Larochelle, H., Adams, R.P. (2012). Practical Bayesian optimization of machine learning algorithms. *NIPS 25*, 2951–2959.
- **[M]** Shahriari, B., Swersky, K., Wang, Z., Adams, R.P., de Freitas, N. (2016). Taking the human out of the loop: A review of Bayesian optimization. *Proceedings of the IEEE* 104(1), 148–175.
- **[M]** Frazier, P.I. (2018). A tutorial on Bayesian optimization. arXiv:1807.02811.

Transfer / warm-start — **the section that decides whether your paper survives the "this is decades old" objection:**
- **[V]** Bai, T., Li, Y., Shen, Y., Zhang, X., Zhang, W., Cui, B. (2023). Transfer learning for Bayesian optimization: A survey. arXiv:2302.05927. — **Read this before writing Related Work.** It taxonomizes transfer-BO into four channels: *initial-points design, search-space design, surrogate model, acquisition function*. Your "state-blind marginal replay" is a **search-space/initial-design transfer** and your "retained search state" is a **surrogate/acquisition transfer**. Naming your arms inside their taxonomy is the single cheapest way to look literate.
- **[V]** Feurer, M., Springenberg, J.T., Hutter, F. (2015). Initializing Bayesian hyperparameter optimization via meta-learning. *AAAI 29(1)*, 1128–1135. — MI-SMBO.
- **[V]** Perrone, V., Shen, H., Seeger, M., Archambeau, C., Jenatton, R. (2019). Learning search spaces for Bayesian optimization: Another view of hyperparameter transfer learning. *NeurIPS 32*. arXiv:1909.12552. — **the closest published relative of your marginal-replay arm; see §2.1.**
- **[V]** Hvarfner, C., Stoll, D., Souza, A., Lindauer, M., Hutter, F., Nardi, L. (2022). πBO: Augmenting acquisition functions with user beliefs for Bayesian optimization. *ICLR 2022*. arXiv:2204.11051. — a prior over the optimum's *location*, decayed over time. Also a state-blind marginal, deployed as a method.
- **[V]** Volpp, M., Fröhlich, L.P., Fischer, K., Doerr, A., Falkner, S., Hutter, F., Daniel, C. (2020). Meta-learning acquisition functions for transfer learning in Bayesian optimization. *ICLR 2020*. arXiv:1904.02642. — MetaBO; an RL-learned acquisition function. **This is the "neural outer-loop search strategy" comparator your RQ3 needs; if you do not cite it a reviewer will supply it.**
- **[V]** Golovin, D., Solnik, B., Moitra, S., Kochanski, G., Karro, J., Sculley, D. (2017). Google Vizier: A service for black-box optimization. *KDD 2017*, 1487–1495. — transfer learning via a stack of GPs, deployed at scale since 2017. Cite this the moment you claim cross-run memory is under-explored; it is the counter-example a reviewer reaches for.
- **[M]** Swersky, K., Snoek, J., Adams, R.P. (2013). Multi-task Bayesian optimization. *NIPS 26*, 2004–2012.
- **[M]** Bardenet, R., Brendel, M., Kégl, B., Sebag, M. (2013). Collaborative hyperparameter tuning. *ICML 2013*, PMLR 28(2), 199–207.
- **[M]** Wistuba, M., Schilling, N., Schmidt-Thieme, L. (2016). Two-stage transfer surrogate model for automatic hyperparameter optimization. *ECML PKDD 2016*, 199–214.
- **[M]** Perrone, V., Jenatton, R., Seeger, M., Archambeau, C. (2018). Scalable hyperparameter transfer learning. *NeurIPS 31*, 6845–6855.
- **[M]** Feurer, M., Letham, B., Bakshy, E. (2018). Practical transfer learning for Bayesian optimization. arXiv:1802.02219. — RGPE. *(Title changed across versions; verify.)*
- **[M]** Salinas, D., Shen, H., Perrone, V. (2020). A quantile-based approach for hyperparameter transfer learning. *ICML 2020*, PMLR 119, 8438–8448. arXiv:1909.13595.
- **[M]** Wistuba, M., Grabocka, J. (2021). Few-shot Bayesian optimization with deep kernel surrogates. *ICLR 2021*.

### 1.4 Meta-learning and learning-to-optimize
- **[M]** Thrun, S., Pratt, L. (eds.) (1998). *Learning to Learn*. Kluwer.
- **[V]** Vanschoren, J. (2018). Meta-learning: A survey. arXiv:1810.03548.
- **[M]** Hospedales, T., Antoniou, A., Micaelli, P., Storkey, A. (2022). Meta-learning in neural networks: A survey. *IEEE TPAMI* 44(9), 5149–5169.
- **[M]** Andrychowicz, M., Denil, M., Gómez, S., Hoffman, M.W., Pfau, D., Schaul, T., Shillingford, B., de Freitas, N. (2016). Learning to learn by gradient descent by gradient descent. *NIPS 29*, 3981–3989.
- **[M]** Chen, Y., Hoffman, M.W., Colmenarejo, S.G., Denil, M., Lillicrap, T.P., Botvinick, M., de Freitas, N. (2017). Learning to learn without gradient descent by gradient descent. *ICML 2017*, PMLR 70, 748–756. — **an RNN that carries search state across black-box optimization runs. This is the most direct neural instantiation of "retained search state" and your RQ3's true antagonist.**
- **[M]** Finn, C., Abbeel, P., Levine, S. (2017). Model-agnostic meta-learning for fast adaptation of deep networks. *ICML 2017*, PMLR 70, 1126–1135.
- **[V]** Chen, T., Chen, X., Chen, W., Heaton, H., Liu, J., Wang, Z., Yin, W. (2022). Learning to optimize: A primer and a benchmark. *JMLR* 23(189), 1–59.

### 1.5 Hyperparameter transfer / algorithm configuration (the SMAC line)
- **[M]** Hutter, F., Hoos, H.H., Leyton-Brown, K. (2011). Sequential model-based optimization for general algorithm configuration. *LION 5*, LNCS 6683, 507–523.
- **[V]** Lindauer, M., Hutter, F. (2018). Warmstarting of model-based algorithm configuration. *AAAI 2018*, 1355–1362. arXiv:1709.04636. Reports **up to 165× speedups** from warm-starting on new benchmark families. — **This is the paper that most directly pre-empts "retaining search state helps." Cite it in your first Related Work paragraph, not your last.**
- **[M]** Lindauer, M., Eggensperger, K., Feurer, M., Biedenkapp, A., Deng, D., Benjamins, C., Ruhkopf, T., Sass, R., Hutter, F. (2022). SMAC3: A versatile Bayesian optimization package for hyperparameter optimization. *JMLR* 23(54), 1–9.
- **[M]** Bergstra, J., Bardenet, R., Bengio, Y., Kégl, B. (2011). Algorithms for hyper-parameter optimization. *NIPS 24*, 2546–2554. — TPE.
- **[M]** Li, L., Jamieson, K., DeSalvo, G., Rostamizadeh, A., Talwalkar, A. (2018). Hyperband. *JMLR* 18(185), 1–52.
- **[M]** Falkner, S., Klein, A., Hutter, F. (2018). BOHB: Robust and efficient hyperparameter optimization at scale. *ICML 2018*, PMLR 80, 1437–1446.
- **[M]** Feurer, M., Hutter, F. (2019). Hyperparameter optimization. In: *AutoML: Methods, Systems, Challenges*, Springer, ch. 1, 3–33.
- **[M]** Bischl, B., Binder, M., Lang, M., Pielok, T., Richter, J., Coors, S., Thomas, J., Ullmann, T., Becker, M., Boulesteix, A.-L., Deng, D., Lindauer, M. (2023). Hyperparameter optimization: Foundations, algorithms, best practices, and open challenges. *WIREs Data Mining and Knowledge Discovery* 13(2), e1484.

### 1.6 Kolmogorov–Arnold Networks and its critiques
Garrido named KAN, so you need both the primary and the adversarial literature.
- **[V]** Liu, Z., Wang, Y., Vaidya, S., Ruehle, F., Halverson, J., Soljačić, M., Hou, T.Y., Tegmark, M. (2024). KAN: Kolmogorov–Arnold Networks. arXiv:2404.19756.
- **[V, verify venue]** A follow-up "Kolmogorov–Arnold Networks Meet Science" appears in *Physical Review X* (link.aps.org/doi/10.1103/4t7t-v19l). Verify authors/volume before citing — this is the KAN-2.0/science line.
- **[V]** Yu, R., Yu, W., Wang, X. (2024). KAN or MLP: A fairer comparison. arXiv:2407.16674. — under matched parameters/FLOPs, **MLP generally outperforms KAN except on symbolic-formula representation, and that advantage traces to the B-spline activation, not the architecture.**
- **[V]** Poeta, E., Giobergia, F., Pastor, E., Cerquitelli, T., Baralis, E. (2024). A benchmarking study of Kolmogorov–Arnold Networks on tabular data. arXiv:2406.14529.
- **[V]** Girosi, F., Poggio, T. (1989). Representation properties of networks: Kolmogorov's theorem is irrelevant. *Neural Computation* 1(4), 465–469. — **the 37-year-old rebuttal to the theoretical motivation for KAN.** The inner functions of the Kolmogorov representation are non-smooth and not learnable in any useful sense; the representation theorem does not license the interpretability claim. This single citation does more positioning work for you than any 2024 benchmark, because it shows the "which neural family?" question has a prior answer that the 2024 wave did not engage.
- **[V, verify authors]** "Efficiency bottlenecks of convolutional Kolmogorov–Arnold Networks" (arXiv:2501.15757) — compute-cost critique on ImageNet/tabular.
- **[V, verify]** A survey exists: "A Survey on Kolmogorov–Arnold Network" (arXiv:2411.06078).

---

## 2. PRIOR-ART CHECK — BLUNT ASSESSMENT

**The objection you must survive:** *"Warm-starting an optimizer on a related task helps. Feurer 2015, Golovin 2017, Lindauer & Hutter 2018 (165× speedup), Perrone 2019, and a whole survey (Bai 2023) say so. What is new?"*

Your central claim as frozen — "all tested stateful variants achieved lower development regret than their memoryless counterparts" — **is a replication of established knowledge, and you should say so in the paper before a reviewer says it for you.** RQ1's answer is not the contribution. RQ2's and RQ3's are.

### 2.1 Candidate (a): the state-blind marginal-replay falsifier — **PARTIALLY NEW; DO NOT OVERCLAIM**

**Against it:**
- **Yu, Sciuto, Jaggi, Musat, Salzmann (2020), *Evaluating the search phase of neural architecture search*, ICLR 2020, arXiv:1902.08142** does the same *logical move*: hold the search space fixed, replace the search strategy with a control that has no learned state, and show state-of-the-art NAS is indistinguishable from it. Their control is uniform random within the space. **[V]**
- **Li, L., Talwalkar, A. (2019). Random search and reproducibility for neural architecture search. *UAI 2019*.** **[M]** — same finding, same year.
- **Bergstra & Bengio (2012)** established random-in-the-space as the canonical null.
- Worse for you: the *object* your control replays — a transferred marginal over configurations — is a **published method, not a null**. Perrone et al. (2019) transfer a learned search space; πBO (Hvarfner et al. 2022) transfers a prior over the optimum's location; Feurer et al. (2015) transfer an initial design. Bai et al. (2023) give these their own taxonomy slots. So a reviewer can correctly say: *"your falsifier is Perrone-2019 with the sequential update switched off."*

**For it:**
- Your control is **strictly stronger than the NAS null.** Uniform random matches only the support; **frequency-matched replay of the method's own visit marginals matches the first-order allocation** and therefore isolates *exactly* the sequential/conditional component. Yu et al. cannot separate "the search space was already good" from "the search strategy was useless"; your design can. That distinction is real and I found no paper that runs it as a control.
- The bandit theory literature poses this question formally — the value of adaptivity (Perchet et al. 2016; Agarwal et al. 2017; the ongoing "adaptivity barrier" line) — **but answers it with regret bounds on synthetic classes, never as an empirical ablation on a real optimizer.** You are supplying the empirical instrument for a question the theorists own.

**Verdict.** Not a new *technique*. It is a **stronger and, as far as I can determine, unnamed empirical control** that separates two transfer channels the survey literature already distinguishes conceptually. Claim it at exactly that strength:

> *"We separate the two transfer channels that Bai et al. (2023) taxonomize — search-space/initial-design transfer versus surrogate/acquisition transfer — by running the former as a frequency-matched, state-blind control against the latter. This is a stronger null than the uniform-random control used to audit neural architecture search (Yu et al., 2020), because it matches the first-order visit distribution and therefore attributes any residual advantage to sequential conditioning alone."*

Cite Yu 2020 + Perrone 2019 + Bai 2023 **in the same sentence as your control**. That sentence is your armour. Without it, this is the finding that gets you rejected.

### 2.2 Candidate (b): prospective 16× expansion on a reserved seed block — **RIGOR, NOT NOVELTY**

- Every ingredient is standard: reserved test instances (Bartz-Beielstein et al. 2020 explicitly recommend a tuning/test split of benchmark sets **[V]**); selection bias from optimizing a noisy criterion (Cawley & Talbot 2010 **[V]**); the optimizer's curse (Smith & Winkler 2006 **[V]**); NAS best practices demanding held-out evaluation (**[M]** Lindauer & Hutter 2020, *Best practices for scientific research on NAS*, JMLR 21(243)); and preregistration has two NeurIPS workshop proceedings (**[V]** PMLR v148, 2020; PMLR v181, 2021) with the exact exploratory/confirmatory split you use.
- What I could **not** find a precedent for: holding out **design-space cardinality** rather than instances or seeds. Held-out protocols hold out data; you hold out *the size of the problem*. That is unusual.
- But it is an experimental-design choice, not a method. Reviewers will read it as *"good, I believe your result"* — which is worth a great deal and is precisely why it should not be sold as a contribution. **Put it in Methods, not in the contribution list.**

**One live confound to pre-empt.** A reviewer will say: *"288→4,608 changes the difficulty; a null at 4,608 is confounded with the space being harder."* Your defence already exists in the design — cold start and marginal replay are measured **in the same expanded space**, so the contrast is within-space. State this explicitly in the paragraph that introduces RQ2, and put the three arms' absolute regret levels in the same panel so the difficulty shift is visible rather than argued.

### 2.3 Candidate (c): predictive fit and search quality dissociate — **ALREADY PUBLISHED. CITE IT, DO NOT CLAIM IT.**

This is your weakest novelty claim and asserting it as new would be the clearest signal to a methods reviewer that the literature was not read.

- **[V]** Tom, G., Lo, S., Corapi, S., Aspuru-Guzik, A., Sanchez-Lengeling, B. (2025). Ranking over regression for Bayesian optimization and molecule selection. *APL Machine Learning* 3(3), 036113. arXiv:2410.09290. — *"minimizing RMSE does not always lead to correctly ranked candidates"*; surrogate **ranking** ability correlates with BO performance, regression accuracy does not.
- **[V]** arXiv:2503.00844 — *Impact of surrogate model accuracy on performance and model management strategy in surrogate-assisted evolutionary algorithms*: Kendall's τ between surrogate accuracy and search performance can be **negative**.
- **[M]** Eggensperger, K., Hutter, F., Hoos, H., Leyton-Brown, K. (2015). Efficient benchmarking of hyperparameter optimizers via surrogates. *AAAI 2015*, 1114–1120. — rank correlation, not RMSE, is the criterion for a useful surrogate.
- **[M]** Jones, D.R. (2001). A taxonomy of global optimization methods based on response surfaces. *Journal of Global Optimization* 21(4), 345–383. — EGO works through calibrated *uncertainty*; a better point-fit with worse uncertainty is a worse optimizer.
- Hutter et al. (2014) fANOVA **[V]** makes the same point structurally: what governs search is the low-order effect structure, not global fit quality.

**What you can legitimately claim.** Not the phenomenon — the **instance and its consequence**. Two fields (Garrido et al. ICCL 2024; Ding et al. IJPE 2026) are choosing an AI family for supply-chain resilience on the assumption that the better function approximator is the better carrier. You show, in that domain, that it is not — and you have the mechanism measured (`docs/` memory: curvature 0.076 vs noise 0.317; MLP worse than linear). Frame as: *"a known methodological dissociation, demonstrated in a domain currently making the opposite assumption."* For C&IE that is a legitimate and publishable contribution. For a methods venue it would not be.

### 2.4 WHAT IS ACTUALLY NEW (not on your list)

**The layer reassignment.** Garrido et al. (2024) asked *which family of AI algorithms* carries cross-run memory (the "Alzheimer effect"), and named backpropagation/KAN/RL — all **inner-loop function approximators**. Your result is that the effective carrier lives at a **different layer entirely**: the outer-loop search state, where the appropriate object is a bandit allocation, not a network. That is a **reframing of an open published question, supported by a null on the family they proposed and a positive on a family they did not consider.** No citation I found makes this argument.

Second genuinely new thing: **a negative result with a control credible enough to believe.** Negatives are publishable exactly in proportion to the strength of the control. Yours is stronger than the NAS audits' — that is why (a) and this are the same contribution stated twice, and you should merge them.

Third: **the measurement instrument** — development regret across contexts under seed custody in a DES resilience setting, with a metric you have separately shown is not curvature-invariant and not step-cadence-invariant. Domain contribution, methodologically load-bearing.

**Suggested Related Work subsection, and I mean this literally: title one "What is already established."** List Feurer 2015, Golovin 2017, Lindauer & Hutter 2018, Perrone 2019, Bai 2023 (warm-starting works), Chen 2013/Kveton 2015/Zimmert & Seldin 2018 (factorized bandits), Tom 2025/Eggensperger 2015 (fit ≠ search quality), Yu 2020/Li & Talwalkar 2019 (search-phase audits). Then one paragraph: *"None of these is our contribution."* A reviewer who sees this cannot use any of it against you.

---

## 3. THE MARGINAL-REPLAY CONTROL: IS THERE A NAME?

**Short answer: no single accepted name. Searched BO, bandit, RL, NAS, and causal-inference vocabularies. Nothing standard.** The nearest named things, and what each buys you:

| Field | Nearest named concept | Citation | Fit |
|---|---|---|---|
| Causal inference / epi | **negative control**, **placebo test** | **[M]** Lipsitch, Tchetgen Tchetgen, Cohen (2010), *Epidemiology* 21(3):383–388; **[V, verify venue]** Eggers, Tuñón, Dafoe, *Placebo Tests for Causal Inference* | Good for *rhetoric*. "Negative control" = an analysis that must show no effect if the design is sound. Your control is not quite that — it *may* legitimately show an effect. |
| Causal inference | **randomization / permutation inference**, Fisher sharp null | **[V]** standard; synthetic-control placebo permutation | Structurally close: permute away the sequential ordering, keep the marginals. This is the honest analogy. |
| Bandits | **oblivious / non-adaptive allocation**; **limited adaptivity**; **batched bandits** | Perchet et al. 2016; Agarwal et al. 2017 | **Best technical fit.** Your control is an *oblivious* allocation matched to the adaptive one's marginals. |
| NAS | **random-search control / search-phase audit** | Yu et al. ICLR 2020; Li & Talwalkar UAI 2019 | Same purpose, weaker null. |
| RL | **state-marginal matching** | **[V]** Lee, Eysenbach, Parisotto, Xing, Salakhutdinov, Levine (2019), arXiv:1906.05274 | Same *mathematical object* (match a visitation marginal, discard the policy), but used as a training objective, not a control. Cite as terminological precedent for "marginal". |
| Transfer BO | **search-space transfer / initial-design transfer** | Perrone et al. 2019; Feurer et al. 2015; Bai et al. 2023 taxonomy | What your control *is*, as a method. |

**Recommendation — coin it, but anchor it.** Use:

> **"frequency-matched oblivious replay"** (technical), or **"state-blind marginal replay"** (your existing term, more readable)

and define it in one sentence that name-checks three ancestors: *oblivious allocation* (bandits), *search-space transfer* (Perrone et al. 2019), *search-phase audit* (Yu et al. 2020). Coining is fine — reviewers punish unacknowledged reinvention, not naming. **Do not call it a "placebo"** despite your internal vocabulary: in this literature "placebo" implies it must be null, and yours legitimately might not be. That would misdescribe your own falsifier. (This also matters given your standing rule that a falsifier must be able to PASS.)

---

## 4. HOW SEARCH-EFFICIENCY COMPARISONS ARE REPORTED

Four conventions coexist. Reviewers of a C&IE simulation-optimization paper will expect the **first**, and a methods reviewer will expect the second or third.

**(1) Simulation-optimization native — USE THIS AS PRIMARY.**
Eckman, Henderson & Shashaani (2023, *IJoC* 35(2):350–367) **[V]** is the field's standard reference for exactly this. Their apparatus: **progress curves** (normalized optimality gap vs. normalized budget), **area under the progress curve**, **solvability profiles** (fraction of problem–seed pairs reaching a relative-optimality target within budget), explicit **common random numbers** guidance, and **bootstrap** error estimation on all of it. This is your home-venue convention and it is recent enough that a reviewer will expect it.

**(2) Derivative-free / black-box optimization.**
- Dolan & Moré (2002), *Benchmarking optimization software with performance profiles*, *Mathematical Programming* 91(2):201–213 **[M]**. Caveat to note: **[M]** Gould & Scott (2016), *A note on performance profiles for benchmarking software*, *ACM TOMS* 43(2), art. 15 — ratio-based profiles distort under small comparate sets. With 3 arms, performance profiles are a poor choice.
- Moré & Wild (2009), *Benchmarking derivative-free optimization algorithms*, *SIAM J. Optimization* 20(1):172–191 **[V]** — **data profiles**, designed for *expensive* evaluations under a budget. Better fit than performance profiles for a DES.
- **[V]** Hansen, Auger, Brockhoff, Tušar, Tušar, *COCO: Performance Assessment*, arXiv:1605.03560; **[M]** Hansen et al. (2021), *COCO: A platform for comparing continuous optimizers in a black-box setting*, *Optimization Methods and Software* 36(1):114–144 — ECDF of runtimes-to-target, i.e. anytime performance.

**(3) AutoML/HPO.**
Normalized regret / **average distance to the minimum (ADTM)** vs. budget, plus **mean-rank-over-budget** curves, aggregated and per-search-space. **[V]** Arango, S.P., Jomaa, H.S., Wistuba, M., Grabocka, J. (2021). HPO-B: A large-scale reproducible benchmark for black-box HPO based on OpenML. *NeurIPS Datasets & Benchmarks*. arXiv:2106.06257. Also **[M]** Eggensperger et al. (2021), HPOBench, *NeurIPS D&B*; **[M]** Pfisterer et al. (2022), YAHPO Gym, *AutoML Conf*, PMLR 188; **[M]** Turner et al. (2021), *Bayesian optimization is superior to random search…*, PMLR 133:3–26.

**(4) Cross-context aggregation with significance.**
- **[M]** Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *JMLR* 7, 1–30. — Friedman + Nemenyi, critical-difference diagram.
- **[V]** Benavoli, Corani, Demšar, Zaffalon (2017). Time for a change: a tutorial for comparing multiple classifiers through Bayesian analysis. *JMLR* 18(77), 1–36; and Corani et al. (2017), *Machine Learning* 106:1817–1837, arXiv:1609.08905. — the standard critique: rank-only tests ignore effect magnitude and reject on n alone.
- **[V]** Bartz-Beielstein, T., Doerr, C., van den Berg, D., Bossek, J., et al. (2020). Benchmarking in optimization: Best practice and open issues. arXiv:2007.03488.

### What you should actually plot

**Primary metric: normalized *simple* regret, not cumulative regret.** Your outer loop recommends one configuration for deployment; only the final recommendation is realized. Cumulative regret prices every evaluation as if it were deployed, which is the *inner-loop* accounting. **Say this in one sentence — it does double duty as the clearest possible statement of the outer-loop/inner-loop distinction the whole paper rests on.**

1. **Progress curves, one panel per context** (six for RQ1, then the expanded space for RQ2). Normalized simple regret vs. evaluations. Three arms: cold start / state-blind marginal replay / retained search state. **Bootstrap 95% bands over seeds.** (Eckman et al. 2023 give the two-level-simulation theory and the CRN guidance.)
2. **Normalized regret AUC (nAUC) per (arm, context)**, reported as **paired within-context contrasts with CIs** — this already matches the RQ1 framing in your repo ("six paired within-family contrasts"). One number per contrast, CI on the difference, not on the levels.
3. **A solvability / data profile** — fraction of (context, seed) pairs within ε of best-known by budget b. **This is the figure that reports a null honestly**, because overlapping curves are visible rather than averaged away. For RQ3 that is the point.
4. **Mean-rank-over-budget** across contexts. Add a CD diagram only if you carry ≥5 contexts *and* state that with 6 contexts Nemenyi is underpowered, citing Benavoli et al. Prefer the paired per-context intervals as the inferential backbone.
5. **Do NOT plot episode-reward convergence curves** in the Ding et al. (2026) style. Those are inner-loop training curves; plotting them would silently re-conflate the two layers. **Make this an explicit sentence in the paper** — it converts a formatting decision into a positioning argument, and it is the cheapest paragraph in the manuscript.

### The Table-1 grid against Ding et al.

Their Table 1 is `Topic | Literature | Method | Main Advantage-Limitation` ending in "This Paper". Mirror the form, change the columns to the ones you win on:

`Study | Carrier of cross-run memory | Non-neural comparator | State-blind control | Prospective / reserved evaluation | Interval estimates on headline`

Ding et al. (2026) — MAPPO vs QMIX vs MADDPG — is **neural in every row**, with no non-neural comparator, no state-blind control, no reserved block, no CIs, by your own account. Garrido et al. (2024) proposes carriers without an empirical comparison at all. Populate the grid and the last row writes itself. This is fair, checkable, and far more damaging than any adjective.

---

## 5. THREE THINGS TO FIX BEFORE DRAFTING

1. **`docs/manuscript_current/submission/elsevier/references.bib` has 37 entries and zero methods citations.** Every work in §1 is missing. A methods reviewer opening the bibliography currently sees a supply-chain RL paper, not a simulation-optimization paper.
2. **Zimmert & Seldin is NeurIPS 2018, not ICML 2018.** Getting the venue wrong on the paper your carrier is named after is the kind of error that colours a whole review.
3. **Verify before submission:** the Physical Review X KAN follow-up (authors/volume); Eggers/Tuñón/Dafoe placebo-tests final venue; Feurer/Letham/Bakshy title (changed across arXiv versions); every **[M]**-marked page range.

Sources:
- [Warmstarting of Model-Based Algorithm Configuration (AAAI 2018)](https://ojs.aaai.org/index.php/AAAI/article/view/11532)
- [Factored Bandits (NeurIPS 2018)](https://proceedings.neurips.cc/paper_files/paper/2018/file/226d1f15ecd35f784d2a20c3ecf56d7f-Paper.pdf)
- [Learning search spaces for Bayesian optimization (NeurIPS 2019)](https://arxiv.org/pdf/1909.12552)
- [Evaluating the Search Phase of Neural Architecture Search (ICLR 2020)](https://arxiv.org/pdf/1902.08142)
- [KAN or MLP: A Fairer Comparison](https://arxiv.org/html/2407.16674v2)
- [Representation Properties of Networks: Kolmogorov's Theorem Is Irrelevant](https://direct.mit.edu/neco/article-abstract/1/4/465/5509/Representation-Properties-of-Networks-Kolmogorov-s)
- [A Benchmarking Study of Kolmogorov-Arnold Networks on Tabular Data](https://arxiv.org/abs/2406.14529)
- [Transfer Learning for Bayesian Optimization: A Survey](https://arxiv.org/abs/2302.05927)
- [Initializing Bayesian Hyperparameter Optimization via Meta-Learning (AAAI 2015)](https://ojs.aaai.org/index.php/AAAI/article/view/9354)
- [πBO (ICLR 2022)](https://arxiv.org/abs/2204.11051)
- [Meta-Learning Acquisition Functions for Transfer Learning in BO (ICLR 2020)](https://arxiv.org/abs/1904.02642)
- [Google Vizier (KDD 2017)](https://dl.acm.org/doi/10.1145/3097983.3098043)
- [An Efficient Approach for Assessing Hyperparameter Importance (ICML 2014)](https://proceedings.mlr.press/v32/hutter14.html)
- [Simulation optimization: a review of algorithms and applications](https://link.springer.com/article/10.1007/s10479-015-2019-x)
- [SimOpt: A Testbed for Simulation-Optimization Experiments (IJoC 2023)](https://pubsonline.informs.org/doi/10.1287/ijoc.2023.1273)
- [Diagnostic Tools for Evaluating and Comparing Simulation-Optimization Algorithms](https://par.nsf.gov/biblio/10398953-diagnostic-tools-evaluating-comparing-simulation-optimization-algorithms)
- [Benchmarking Derivative-Free Optimization Algorithms (SIAM J. Opt. 2009)](https://epubs.siam.org/doi/10.1137/080724083)
- [COCO: Performance Assessment](https://arxiv.org/pdf/1605.03560)
- [Ranking over regression for Bayesian optimization and molecule selection](https://pubs.aip.org/aip/aml/article/3/3/036113/3359544/Ranking-over-regression-for-Bayesian-optimization)
- [HPO-B benchmark](https://arxiv.org/pdf/2106.06257)
- [The Optimizer's Curse (Management Science 2006)](https://pubsonline.informs.org/doi/10.1287/mnsc.1050.0451)
- [On Over-fitting in Model Selection (JMLR 2010)](https://www.jmlr.org/papers/v11/cawley10a.html)
- [Benchmarking in Optimization: Best Practice and Open Issues](https://arxiv.org/abs/2007.03488)
- [Random Search for Hyper-Parameter Optimization (JMLR 2012)](https://jmlr.org/papers/v13/bergstra12a.html)
- [Learning to Optimize: A Primer and A Benchmark (JMLR 2022)](https://jmlr.org/papers/v23/21-0308.html)
- [Meta-Learning: A Survey](https://arxiv.org/abs/1810.03548)
- [Efficient Exploration via State Marginal Matching](https://github.com/RLAgent/state-marginal-matching)
- [Time for a change: Bayesian comparison of classifiers (JMLR 2017)](https://dl.acm.org/doi/10.5555/3122009.3176821)
- [NeurIPS Pre-registration workshop proceedings (PMLR v148)](https://proceedings.mlr.press/v148/)