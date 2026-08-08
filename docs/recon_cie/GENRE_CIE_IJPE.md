# GENRE ANALYSIS — C&IE/IJPE methods-and-application paper

Source read: `<HOME>/Library/Mobile Documents/com~apple~CloudDocs/1-s2.0-S0925527326000861-main.pdf` (23 pp). Text extracted to `/private/tmp/claude-501/-Users-thom-Projects-research-scres-ia/2c6b3a01-62a0-499d-957d-694424855dff/scratchpad/ding2026.txt` (1,698 lines, page breaks recoverable from running heads).

---

## 1. DING ET AL. 2026 (IJPE 297:109995) — FULL ANATOMY

### 1.1 Section structure with measured page budget (23 pp total)

| § | Title | Pages | Budget |
|---|---|---|---|
| — | Title, abstract (215 w), keywords (5), **Nomenclature & Abbreviations table** (unnumbered, 2-col, 14 entries) | p1 (top half) | 0.5 |
| 1 | Introduction | p1 (col 2) – p3 (top) | **1.9** |
| 2 | Background and related work | p3 – p5 (mid) | **2.2** |
| 2.1 | SC optimization design (+ Table 1) | p3–p4 | 1.0 |
| 2.2 | SoSs resilience in SC | p4 | 0.4 |
| 2.3 | Reinforcement learning for SC | p4–p5 | 0.4 |
| 2.4 | **Research gaps and our contributions** | p4 (bot) – p5 | 0.4 |
| 3 | Dynamic reconfiguration approach of SCSoSs under disruption risks | p5 – p14 | **9.0** |
| 3.1 | Problem description (3.1.1 definitions, 3.1.2 disruption risk model) — Definitions 1–6, Eqs 1–20 | p5–p8 | 3.3 |
| 3.2 | Resilience reconfiguration strategies (3.2.1 three-phase resilience, 3.2.2 directed network, 3.2.3 quantitative resilience model, 3.2.4 cost calculation) — Defs 7–10, Eqs 21–35 | p8–p11 | 3.0 |
| 3.3 | MARL for SCR (3.3.1 POMDP, 3.3.2 state/action/reward, 3.3.3 MAPPO training) — Defs 11–13, Eqs 36–53, Algorithm 1 | p11–p14 | 2.7 |
| 4 | **Case study: SCSoSs reconfiguration** | p14 – p21 | **7.0** |
| 4.1 | Experimental setup (4.1.1 subject+datasets, 4.1.2 algorithm parameters, 4.1.3 evaluation indicators) | p14–p16 | 2.2 |
| 4.2 | Experimental simulation analysis (4.2.1/4.2.2/4.2.3 = three disruption scenarios) | p16–p19 | 2.8 |
| 4.3 | **Generalization effect analysis** (4.3.1/4.3.2/4.3.3 = supplier / manufacturer / distributor attribute sweeps) | p19–p21 | 1.4 |
| 4.4 | **Managerial insights** (4 numbered recommendations) | p21 | 0.4 |
| 4.5 | Further discussion (*Potential applications* / *Possible drawbacks*) | p21 | 0.2 |
| 5 | **Closing remarks** | p21 – p22 | **1.2** |
| — | CRediT statement, Acknowledgements, Data availability | p22 | 0.3 |
| — | References (~64 entries) | p22 – p23 | **1.5** |

**Related-work share: 2.2 / 23 = 9.6% of the paper, ≈10% of body text excluding references.** That is the number to hold ourselves to. Not 20%.

Two structural facts worth copying: (a) there is **no separate "Discussion" section** — discussion is dissolved into §4.4 managerial insights + §4.5 further discussion, and §5 is one page of "Closing remarks"; (b) **method:results = 9:7**, i.e. the method section is *larger* than the results section. C&IE/IJPE readers pay for the model, not the plots.

### 1.2 Complete figure list (16 numbered figures, ~35 panels)

| # | p | What it is | What it *does* |
|---|---|---|---|
| 1 | 7 | Two-panel icon schematic of a 4-tier chain, (a) hard disruption (b) soft disruption, red X's on broken links | **Defines the phenomenon.** No method, no results. |
| 2 | 8 | Ring/cloud layout of SoSs with three arrows labelled Recruiting / Filling / Repairing | **Defines the action set** — one figure whose only job is the decision alphabet |
| 3 | 9 | The resilience-triangle curve r(t), labelled r₀, r_b, r_m, r_s and t₀…t_s | **Defines the measurand.** Single line plot, hand-drawn quality, zero data |
| 4 | 11 | Generic agent↔environment loop, n agents over "Supply chain environment" bar | Textbook RL loop |
| 5 | 12 | POMDP rollout graph s₀→a₀→s₁…, with o_t, r_t, tp_t | Formalism illustration |
| 6 | 12 | CTDE box: "Centralized Training" over 3 "Agent" boxes | Architecture at 30,000 ft |
| 7 | 13 | MAPPO actor-critic training framework | Architecture at 10,000 ft |
| 8 | 13 | Policy-network structure + action output module | Architecture at 1,000 ft |
| 9 | 15 | Map: distribution of the case SC in North China | **The instance.** Grounds the case study in a real geography |
| 10 | 17 | **6-panel grid**, hard disruption: (a)(b)(c) per-agent convergence curves ×3 algorithms ×2 scales, (d) total cumulative reward, (e) SoSs resilience vs iterations, (f) reconfiguration cost vs iterations | Main result, scenario 1 |
| 11 | 18 | Same 6-panel grid, soft disruption | Main result, scenario 2 |
| 12 | 19 | Same 6-panel grid, complex disruption | Main result, scenario 3 |
| 13 | 18/19 | Grouped bar chart: SoSs resilience pre-disruption / post-disruption / post-reconfiguration × 3 algorithms × 2 scales | **The money figure** — the before/after that a manager reads |
| 14 | 20 | 3-panel bar chart, supplier attributes at levels {0.5, 0.75, 1, 1.25, 1.5}, y = reconfiguration performance-cost ratio | Sensitivity / generalization |
| 15 | 20 | Same, manufacturer attributes (3 panels) | Sensitivity |
| 16 | 21 | Same, distributor attributes (2 panels) | Sensitivity |

Shape of it: **3 concept figures (1–3) + 5 architecture figures (4–8) + 1 instance figure (9) + 4 result figures (10–13) + 3 sensitivity figures (14–16)**. Note the ratio: 9 figures before a single number is plotted.

### 1.3 Table list (only 3 numbered tables + 1 unnumbered + 1 algorithm)

- **Nomenclature and Abbreviations** (unnumbered, p1) — 14 acronyms, two-column.
- **Table 1** (p3, ~half a page): "Summary and comparison of SC optimization design". Columns `Topic | Literature | Method | Main Advantage/Limitation`. Five topic blocks (Rule-based, SC Network Design, SC Network Redesign/Reconfiguration, Inventory Management Optimization, Risk Management), 19 literature rows, **final row `This Paper | / | Reinforcement Learning: MAPPO | Uncertainty model; Quantifiable resilience; Integrated configuration costs; Dynamic response/Data dependency`**. Every competitor row states advantage *and* limitation with a slash; the own row states four advantages and one self-declared limitation.
- **Table 2** (p15): hyper-parameters, `Type | Parameters | Value`, split Shared vs Specific (MADDPG / MAPPO).
- **Table 3** (p18): average response time, `Scenario × Scale × {QMIX, MADDPG, MAPPO}`, wall-clock seconds to 3 decimals.
- **Algorithm 1** (p14): 22-line MAPPO pseudo-code box with Input/Output, plus a prose walkthrough keyed to line ranges ("Lines 1-2 outline… Lines 4 to 15 describe… Lines 16 to 21 delineate…").

**56 numbered equations. 13 numbered Definitions** in bracket style (【Supplier】, 【Action】, 【Reward】). The definitional apparatus *is* the methodological contribution in this genre.

### 1.4 How contributions are worded — stated three times, escalating

1. **End of §1** — "The primary contributions of this work can be summarized as follows. (1)…(4)". Four numbered items. Item (1) = the problem formulation, (2) = the model, (3) = the algorithmic embedding, (4) = the experimental verification plus baselines by name.
2. **End of §2.4**, immediately after the gaps — "To address these gaps, this paper proposes… The key contributions are summarized as follows. ● … ● …". Two bullets, tighter, each one paired 1:1 to a numbered gap.
3. **§5 Closing remarks** — "The novelties of this paper include." Three bullets.

Each contribution is a *construct + what it enables*, never a result: "This model provides a structured basis for evaluating and guiding reconfiguration actions under disruptions." Only the last bullet cashes out in performance.

**Research gaps (§2.4)** are two numbered paragraphs with bolded lead-ins and a recurring formula: `[named gap]: While existing studies have explored X… there remains a significant gap in Y… Most prevailing approaches rely on Z, which are ill-suited for W… remains insufficiently explored.` Then: `Crucially, there is a notable absence of frameworks that…`.

### 1.5 How limitations and future work are worded

Split across two places, both short and both immediately neutralized.

- **§4.5 "Further discussion"** — a *Potential applications* paragraph (edge deployment, cloud platforms) followed by a *Possible drawbacks* paragraph naming exactly two: (i) the model assumes perfect and timely information sharing for centralized training, difficult under data silos/privacy/communication delay; (ii) the cost models "may oversimplify real-world complexities such as dynamic pricing, negotiation overheads, and the full logistical costs".
- **§5** — one sentence of concession followed immediately by a generality claim: *"We recognize that this study is conducted within a simulated environment with a specific set of disruption models and resilience strategies. However, due to the generality of the MARL framework and resilience quantification model, the approach can be extended to…"*
- **Future work** — one sentence, three items: real-world SC data, heterogeneous/large-scale agents (">10 agents"), coordination-mechanism × real-time-response interplay.

Total limitations budget: **~15 lines.** No limitation is allowed to threaten the headline.

---

## 2. OTHER PDFs IN THE REPO

`find . -name "*.pdf"` returns **no external C&IE/IJPE/IJPR papers** — everything is our own output or the collaborator's draft. Two useful comparators anyway:

**(a) Our current manuscript** — `<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/main.pdf`, "When Apparent Reinforcement-Learning Gains Depend on the Static Frontier". **53 pages, 16 figures, 12 tables**, single-column elsarticle preprint (≈26–30 double-column print pages). Results section alone runs §4.1–§4.10 — **ten** results subsections. §5 Discussion has 7 subsections. This is the lab-notebook failure mode in a directly comparable artifact: Ding gets three result subsections and one page of closing remarks; we currently have ten and seven. **Table 5 is literally titled "Experimental evidence map used in the manuscript"** — that is an internal audit object in the results section.

**(b) Garrido's own draft** — `<HOME>/Projects/research/scres-ia/tmp/docx_garrido_draft/v.0_neuralNet-scres.pdf`, 21 pp. Structure: `1. Introduction / 2. Background and related works / 3. Research Methodology (3.1 Core Hypotheses, 3.2 Discrete Event Simulation, 3.3 Python Hybrid Simulation-Learning Model) / 4. Analysis of the Results (4.1 DES, 4.2 Hybrid Neural, 4.3 Findings) / 5. Conclusions, limitations, theoretical contribution and future research (5.2 Limitations, 5.3 Theoretical Contribution) / 6. Bibliography / 7. Annexes`. 5 tables, Fig. 1 = "Initial configuration of the supply chain under analysis (SCUA)". Two things to inherit: **§3.1 "Core Hypotheses" as an explicit numbered subsection before any model** (this is exactly where our three RQs and the frozen central claim belong), and **an explicit "Theoretical Contribution" subsection in §5** — a C&IE/IJPE expectation that a pure-methods write-up usually forgets.

Submission mechanics already audited in `<HOME>/Projects/research/scres-ia/papers/cie_chassis/CIE_GUIDE_AUDIT_2026-07-29.md`: double-anonymized review, separate title page, abstract ≤250 words, 1–7 keywords, **3–5 highlights each ≤85 characters**, separate generative-AI declaration before references, CRediT, data-availability statement. C&IE states no hard page/figure cap; the cap is genre, not policy.

---

## 3. THE TEMPLATE — 22-page C&IE methods-and-application paper

### Counts

| Element | Target | Hard ceiling |
|---|---|---|
| Numbered figures | **10** | 12 |
| Numbered tables | **5** | 6 |
| Numbered equations | 25–40 | 56 |
| Numbered definitions (bracket style) | 6–10 | — |
| Algorithm boxes | **1** (the outer-loop search procedure) | 2 |
| References | 60–80 | — |
| Results subsections | **3–4** | 5 |
| Discussion subsections | **4** | 5 |
| Appendices | 2 (falsifier register; custody + seed schedule) | 3 |

Current manuscript is 16 figures / 12 tables. **Cut 6 figures and 7 tables.** Everything cut becomes an appendix or supplementary material — not deleted, *relocated*. The evidence map (`Table 5`) and the ablation/screen tables (`Table 9`, `Table 11`) go to appendix; the frozen-contract cell listings go to supplementary.

### Per-section page budget

| § | Section | Pages | Contents |
|---|---|---|---|
| — | Title page (separate), abstract ≤250 w, 5 keywords, 5 highlights ≤85 ch | — | — |
| — | Nomenclature table (unnumbered) | **0.4** | Copy Ding. Defines `outer-loop search state`, `search carrier`, `development regret`, `state-blind replay`, `cold start`, `factorized UCB`, `configuration`, `panel`. **This table is where the policy/carrier vocabulary rule is enforced once, up front, so no reviewer re-imports "policy".** |
| 1 | Introduction | **2.0** | Para 1–2 problem; para 3 Garrido/Ponguta/Adarme ICCL 2024 two questions + the Alzheimer effect *in their words*; para 4 the layer confusion (between-run vs within-episode) as the reason the field has not answered them; para 5 what we do; **numbered contributions (1)–(4)**; roadmap paragraph. |
| 2 | Background and related work | **2.2** | 2.1 SCRES measurement and DES-based assessment · 2.2 Learning and search in simulation optimization (incl. bandits/UCB — this is where the non-neural comparator gets its literature) · 2.3 RL and neural methods for supply-chain resilience (Ding et al. 2026 cited here, respectfully, as the strongest current instance) · **2.4 Research gaps and our contributions** with 2–3 numbered gaps + 2–3 bullet contributions mapped 1:1. **Table 1 lives here.** |
| 3 | Methodology | **8.0** | 3.1 Core hypotheses and research questions (RQ1/RQ2/RQ3 + the frozen central claim as a displayed statement) — 0.7 pp · 3.2 The DES environment: **prospective reproduction of six thesis-derived comparative panels** (never "validation") — 2.3 pp · 3.3 The outer-loop search problem: configuration space, the 288 → 4,608 expansion, development regret as the objective — 2.0 pp · 3.4 Search carriers compared: cold start / state-blind marginal replay / factorized UCB / stateful and memoryless neural variants — 2.0 pp, **Algorithm 1 here** · 3.5 Experimental protocol and inference — 1.0 pp. |
| 4 | Case study / results | **6.0** | 4.1 Experimental setup (instance, panels, budgets, hyper-parameters) — 1.5 pp · 4.2 **RQ1** development regret, stateful vs memoryless, six paired within-family contrasts — 1.5 pp · 4.3 **RQ2** prospective confirmation at 4,608 configurations vs cold start and state-blind replay — 1.5 pp · 4.4 **RQ3** is the carrier neural? (development, mixed) — 0.8 pp · 4.5 **Managerial insights**, 4 numbered items — 0.5 pp · 4.6 Further discussion: applications / drawbacks — 0.2 pp. |
| 5 | Discussion | **2.0** | 5.1 What this refines in Garrido et al. (2024) — the carrier is persistent search state, not architecture · 5.2 **Theoretical contribution** (Garrido's own §5.3 convention) · 5.3 Limitations — ≤0.5 pp · 5.4 Future work — ≤0.15 pp. |
| 6 | Conclusion | **0.8** | Restate the frozen claim verbatim + 3 novelty bullets. |
| — | CRediT, gen-AI declaration, data availability, acknowledgements | **0.3** | Per `CIE_GUIDE_AUDIT_2026-07-29.md`. |
| — | References (60–80) | **1.5** | |
| — | Appendix A: preregistered falsifier register · Appendix B: seed schedule and custody | **1.5** | Everything that reads like a protocol goes here. |
| | **Total** | **~22.5** | |

### The 10 figures (in order)

| # | Figure | Job | Genre precedent |
|---|---|---|---|
| **1** | **Two-layer loop: inner DES episode loop (frozen) vs outer configuration-search loop (where memory lives)**, with Garrido Fig. 2 nodes ③ and ⑧ marked as the open ends and the retained search state drawn as the arc that closes ⑧→③ | Defines the phenomenon *and* enforces the vocabulary in a single image | Ding Fig. 1 |
| 2 | The MFSC topology / decision points under study | Defines the instance and the configuration space | Ding Fig. 9, Garrido Fig. 1 |
| 3 | The measurand: what "development regret" is, drawn as a best-so-far curve over the run budget with the regret area shaded | Defines the objective with zero data | Ding Fig. 3 |
| 4 | The carrier ladder: cold start / state-blind marginal replay / factorized UCB / memoryless neural / stateful neural, drawn as *what each one retains between runs* | Defines the comparator alphabet — **this is the figure Ding structurally cannot draw** | Ding Fig. 2 |
| 5 | Factorized UCB search-state architecture (one panel, not three) | Method at one level of zoom, not three | Ding Figs 6–8, compressed |
| 6 | **RQ1**: six paired within-family contrasts, stateful − memoryless, with bootstrap intervals — one forest/dot-and-interval panel | The paired design made visible | (no precedent — an upgrade) |
| 7 | **RQ1**: regret trajectories over the run budget, one panel per family, with intervals as shaded bands | Convergence, but with dispersion | Ding Figs 10–12, done right and at 1/3 the panel count |
| 8 | **RQ2**: 288 vs 4,608 — the three carriers (UCB / cold start / state-blind replay) at both scales | The confirmation, and the only figure a skimming reviewer will remember | Ding Fig. 13 (before/after bar) |
| 9 | **RQ3**: curvature-vs-noise — the measured surface curvature against the replication noise floor, with the linear/MLP comparison | Why the answer is "not neural" and why that is a *measurement*, not a failure to train | (no precedent) |
| 10 | Sensitivity: transfer under panel/regime variation | Generalization, genre-required | Ding Figs 14–16, compressed to one |

### The 5 tables

1. **Table 1 — Positioning grid.** `Topic | Literature | Method | Carrier of cross-run memory | Non-neural comparator? | Uncertainty reported | Main advantage/limitation`, last row "This Paper". Ding's four columns plus three of ours. **The two columns "Non-neural comparator?" and "Uncertainty reported" do the entire competitive positioning without a word of criticism** — the column reads `—` down the neural rows including Ding's and the reviewer draws the conclusion.
2. **Table 2 — The configuration space**: factors, levels, 288 cells → 4,608 cells, with what changed in the expansion.
3. **Table 3 — Search-carrier specification**: carrier | what is retained between runs | what is reset | budget.
4. **Table 4 — RQ1/RQ2 headline results**: contrast | n pairs | point estimate | bootstrap percentile CI95 | preregistered threshold | verdict.
5. **Table 5 — Prospective reproduction of the six thesis-derived panels**: panel | thesis direction | reproduced direction | magnitude | **declared fidelity price** (including the sumBt column unreconstructed in >1.09% of 47,780 rows).

---

## 4. WHAT FIGURE 1 DOES, AND WHAT OURS MUST BE

**The convention.** In this genre Figure 1 is never a result and almost never an architecture. It is one of two things: **the phenomenon** (Ding Fig. 1 — two panels of a broken supply network, icons, red X's, no equations) or **the system under analysis** (Garrido's draft Fig. 1 — "Initial configuration of the supply chain under analysis"). It appears late — Ding's is on **page 7**, after the full problem formalization. It carries no data, and its caption is a noun phrase, not a claim.

**Ours must be the two-layer loop.** The single largest risk to this paper is a reviewer reading "search state" as "RL policy state" and then asking why we did not just train a recurrent policy — which would collapse RQ1, RQ2 and RQ3 into a misunderstanding. That risk is closed in the first figure or it is not closed.

Concretely: an outer rectangle labelled **outer-loop configuration search** containing an inner rectangle labelled **DES episode (frozen physics, no learning within)**. The inner loop is drawn once and marked *no cross-run memory*. The outer loop runs ③ data gathering → configure → simulate → ⑧ verification & validation, with Garrido's node numerals ③ and ⑧ printed on it, and a dashed arc from ⑧ back to ③ labelled **retained search state (the carrier)** — the arc that is absent in Garrido Fig. 2 and that they call the Alzheimer effect. Three annotations, no more: the arc is what varies across our arms; the inner box is identical in every arm; the word *policy* appears nowhere.

That figure simultaneously (i) defines the object, (ii) names the gap in the target audience's own diagram and vocabulary, (iii) makes the vocabulary rule visually self-enforcing, and (iv) shows a reviewer in five seconds why "just use a recurrent policy" is a different layer. Caption as a noun phrase: *"The two loops of a DES-based resilience study: the within-episode loop and the between-run configuration search."*

---

## 5. UNCERTAINTY — THEIRS VS OURS

### What Ding et al. actually report

- *"Each scenario is repeated 10 times for each scale, with the results meticulously recorded"* and *"the average derived from ten experiments."* That is the entire uncertainty statement in a 23-page paper.
- **No confidence intervals. No standard deviations. No error bars on any of the 35 panels. No significance test. No seed disclosure. No mention of common random numbers or pairing.**
- Convergence curves (Figs 10–12) are single traces, evidently smoothed, with no dispersion band.
- Headline superiority is declared on differences with no dispersion attached: *"MAPPO and MADDPG converge to a reward value of approximately 50, while QMIX reaches only 46"*; *"MAPPO and MADDPG converge to about 160, while DQN reaches approximately 155."* (Note also that the text says "DQN" where the experiment used QMIX — an uncorrected slip in a published IJPE paper.)
- Table 3 gives wall-clock times to three decimals (0.934 s, 2.746 s) with no dispersion, and orders the algorithms from those point estimates.
- Table 2 hyper-parameter choices are justified *narratively* ("standard starting points", "a good trade-off between bias and variance") rather than by a tuning protocol; the paper claims "a fair and effective comparison" on that basis.

That is the field norm, not an outlier. Across C&IE/IJPE simulation-optimization papers you typically get mean over N replications, sometimes ± SD or a gap-% against a best-known solution, occasionally a Wilcoxon or paired t-test. Bootstrap intervals are rare. Preregistration is essentially absent. "Seed custody" is not a term of art in this literature.

### Where our style will look unfamiliar — and the fix in each case

| Our practice | Why it reads foreign | Fix |
|---|---|---|
| Bootstrap percentile intervals | Reviewer expects ± SD or a t-test and will ask "why not a t-test?" | One sentence in §3.5: paired differences over a common configuration set are not normally distributed and the estimand is a difference of order statistics; percentile bootstrap avoids the assumption. Then never defend it again. |
| **Preregistered falsifiers** with named pass/fail thresholds | Reads as clinical-trial apparatus; C&IE reviewers have no schema for it and may read it as defensiveness | Call them **"prespecified decision rules"** in the body; give the full register as **Appendix A**. In §3.5 give two sentences and a forward reference. Do *not* narrate a falsifier failing inside the results text. |
| Seed custody, virgin/disjoint blocks, "sealed tapes" | Sounds like it came from another discipline | Translate: *"a held-out replication block, opened once, after the analysis plan was fixed."* Ban "tape", "sealed", "burned" from the manuscript; keep them in the repo. |
| RQ3 reported as a negative | The genre publishes negatives only when they are framed as design rules | Frame as an engineering decision rule, in the managerial-insights section: *when measured surface curvature is below the replication noise floor, the return is in retaining search state, not in architecture.* Never as "we failed to find a neural advantage." |
| Declaring a fidelity price (sumBt unreconstructed in >1.09% of 47,780 rows) | More self-incriminating disclosure than the genre expects | Keep it — but put it in **Table 5's own column**, so it reads as *measured and bounded*, not as a confession. And keep the section title "prospective reproduction of six thesis-derived comparative panels", which is itself the honest framing. |
| 53 pp / 16 figs / 12 tables | Ding does it in 23 / 16 / 3; our table count is 4× the genre | The template above. This is the single largest formatting gap. |

### Where our style will look clearly better — and how to cash it in

1. **A non-neural comparator.** Every one of Ding's baselines (MAPPO, QMIX, MADDPG) is neural. That design *structurally cannot* separate "neural" from "this particular neural" — the question Garrido et al. actually asked. Our factorized UCB is the comparator that makes the question answerable. **This is the strongest single positioning sentence in the paper**, and it belongs in §2.4 as a gap, not in the discussion as a boast.
2. **The state-blind replay control.** A comparator that reuses only the *marginal* statistics of the same search with the joint state destroyed. It is the placebo the competing literature has no analogue for, and it is what converts "retention helped" into "retention of *joint* state helped." Name it plainly in the abstract.
3. **Any interval at all is an upgrade.** Ding declares superiority on 50-vs-46 at n=10 with no dispersion. A paired design with bootstrap CIs is not merely more rigorous, it is *categorically* different evidence. Make the intervals visible in Figures 6, 7, 8 — the shaded band is the argument.
4. **Prospective vs post-hoc.** Ding's §4.3 "generalization effect analysis" sweeps attributes at {0.5, 0.75, 1, 1.25, 1.5} around a fiducial value **on the same instance**. Ours expands the design space **16×, prospectively, with the analysis plan fixed first**. Draw that contrast once, in Related Work, as a methodological category — "post-hoc sensitivity sweep" vs "prospective expansion" — without naming any paper as deficient.
5. **The paired within-family design.** Six paired contrasts of stateful vs memoryless *within* each family isolates the retention factor from the architecture factor. Ding's comparison confounds algorithm identity with everything else.

**The one-line positioning to put in §2.4:** *the current state of the art compares neural carriers against neural carriers and reports point estimates over ten repetitions; the question of whether the carrier must be neural has therefore not been posed.* That sentence does all the work, cites Ding et al. respectfully as the leading instance, and requires no criticism of anyone.

Sources: [C&IE guide for authors](https://www.sciencedirect.com/journal/computers-and-industrial-engineering/publish/guide-for-authors) (returns 403 to automated fetch; requirements taken from the repo's own audit at `<HOME>/Projects/research/scres-ia/papers/cie_chassis/CIE_GUIDE_AUDIT_2026-07-29.md`).