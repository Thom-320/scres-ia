# C&IE SUBMISSION BRIEF — "Retained Search State Before Neural Architecture"

Prepared 2026-08-07. Everything below is either read from the repo or read from the live journal pages today. Where I could not verify something I say so.

---

## PART 1 — WHAT THE CHASSIS ALREADY ENCODES

Files: `/Users/thom/Projects/research/scres-ia/papers/cie_chassis/`

### 1.1 `CIE_GUIDE_AUDIT_2026-07-29.md` — 17 requirements

| # | Requirement | Chassis status |
|---|---|---|
| 1 | Double-anonymized review | anonymous `main.tex` + separate `TITLE_PAGE.tex` |
| 2 | Separate title page (authors, affiliations, acknowledgements, declaration, corresponding contact) | template, all human fields `PENDING` |
| 3 | Editable LaTeX sources (no PDF as source) | present |
| 4 | Abstract ≤ 250 words | 215 words (old paper) |
| 5 | 1–7 English keywords | six |
| 6 | 3–5 highlights, ≤ 85 chars each | five, in `HIGHLIGHTS.txt` |
| 7 | Graphical abstract | encouraged, deliberately omitted |
| 8 | Separate generative-AI declaration before references | drafted |
| 9 | Competing-interest declarations-tool output as `.doc/.docx` | **human-blocked** |
| 10 | Funding sources + sponsor role | **human-blocked** |
| 11 | Research data deposited and cited, or non-sharing explanation | RC1 GitHub release exists; **archival DOI + anonymous routing pending** |
| 12 | Data-availability statement | present; DOI pending |
| 13 | Bidirectional reference check | must be redone on final source |
| 14 | Separate artwork with captions | 4 PNGs (old paper) |
| 15 | One corresponding author, email + postal + phone | **human-blocked** |
| 16 | Independent language review | **pending** |
| 17 | Journal snapshot | CiteScore 13.5 · IF 7.3 · 4 d to first decision · 90 d to decision after review · 208 d to acceptance · 7 d to online |

Plus a **double-anonymization boundary rule**: the chassis directory itself contains author names, email message-IDs, branch names and a public GitHub URL and must never be uploaded as anonymous supplementary material.

### 1.2 `RELEASE_AND_SUBMISSION_CHECKLIST.md` — 8 closed, 11 open

Open items, all of which block submission: all-author approval of order/CRediT/affiliations/corresponding author; **Garrido's written return of the bounded face-validation request**; funding/acknowledgements/conflicts/AI disclosure finalized; declarations-tool `.doc/.docx` generated; permissions & security review of public military-model wording and assets; final language and format review; clean-room build from the tagged release; immutable evidence bundle archived and DOI inserted; DOI routing that preserves anonymity; cover-letter placeholders removed; Q1 status rechecked on the submission date.

### 1.3 `HIGHLIGHTS.txt` — **format survives, content is prohibited**

Current five lines belong to the retracted RecurrentPPO paper ("RecurrentPPO robustly exceeds the complete open-loop frontier", "The protocol separates feedback value from **neural premium**"). Two of these violate the binding vocabulary rules directly, and per `README.md` the whole claim family is prohibited by `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §7. **Rewrite all five.**

### 1.4 `TITLE_PAGE.tex` — `elsarticle[preprint,12pt]`

Carries the **old title** ("When Feedback Beats an Exhaustive Open-Loop Frontier…"). Sections stubbed: corresponding-author contact, acknowledgements (with the explicit rule *do not include acknowledgements in the anonymized manuscript*), competing interest, funding, CRediT. All `PENDING`.

### 1.5 `GENERATIVE_AI_DISCLOSURE_DRAFT.md`

Names OpenAI Codex and Anthropic Claude for manuscript organization/drafting/clarity; states authors reviewed and take responsibility; states **no generative-AI-produced scientific result, table or figure is included**; and correctly separates *writing* disclosure from *scientific computation*, which is governed by methods/code/frozen artifacts. It is anonymous-safe. Keep the last paragraph — it is protective — but the methods section must actually deliver on it.

### 1.6 The missing piece (actionable, blocking)

`README.md` says the anonymous-bundle builder with the identity guard is **not in the branch**. Confirmed: `scripts/` on `codex/expanded-contract-comparators-v2` contains only `audit_benchmark_bundles.py`, `build_cie_outer_loop_figures.py`, `screen_program_h_visible_v1_observable_policies.py`, `watch_local_scientific_run.py`.

It exists in history — commit **`ba96890` "Prepare Program Q double-anonymous CIE package"**. Recover it, then (a) parameterize `PAPER_ROOT` (hard-wired to `papers/submission_a_program_q`), (b) add identity tokens for every new author, and (c) **add the repo name, the GitHub org/URL, and distinctive program names** ("Program Q", "SCRES-IA", branch names) to the token list — without that the guard does not guard.

---

## PART 2 — THE LIVE C&IE AUTHOR GUIDE (read in full today)

Source: <https://www.sciencedirect.com/journal/computers-and-industrial-engineering/publish/guide-for-authors> (bot-gated to plain fetch; read via browser).

### 2.1 Aims/scope and the application-paper test — verbatim

> "Papers reporting on applications of industrial engineering techniques to real life problems are welcome, as long as they satisfy the criteria of **originality in the choice of the problem and the tools utilized to solve it, generality of the approach for applicability to other problems, and significance of the results produced**."

Also: the journal publishes "original contributions on the **development of new computerized methodologies** for solving industrial engineering problems, as well as the **applications of those methodologies**." Two doors. **Our paper should walk through the methodology door, not the application door** — see Part 4.

<https://www.sciencedirect.com/journal/computers-and-industrial-engineering/about/aims-and-scope>

### 2.2 Article types and length — the honest answer

- **The live guide encodes no word limit, no page limit, and no per-article-type length norm.** I read the complete guide text; the only numeric limits anywhere are: abstract ≤ 250 words, 1–7 keywords, 3–5 highlights ≤ 85 characters, graphical abstract 531 × 1328 px, video ≤ 150 MB/file and 1 GB total.
- The guide **does not enumerate article types either**. Published items on the journal front page are labelled "Research article" and "Review article". A scispace listing (Full Length Article / Review / Short Communication / Case Report / Data / Micro-article / Original Software Publication / Practical Guideline / Protocol / Replication Study / Short Survey / Video) is a generic Elsevier list and I could **not** verify it against C&IE — treat as unverified. The definitive list lives inside Editorial Manager, login-gated: <https://submit.elsevier.com/CAIE>.
- One article type is verified because a special issue names it: **`VSI: Simulation-Driven Industrial Transformation`**.
- Practical read: length is governed by reviewers, not by rule. Budget a full-length research article and push apparatus to appendices.

### 2.3 Abstract — **not structured**

≤ 250 words, "concise and factual", must stand alone, avoid references (if essential, cite author + year in full), avoid non-standard abbreviations. **No structured-abstract requirement.** Note: your frozen central claim is ~60 words on its own — the abstract needs to be engineered around it.

### 2.4 Highlights

Separate **editable** file with the word "highlights" in the filename. **3–5 bullets, each ≤ 85 characters including spaces.** Required at submission.

### 2.5 Declarations required

| Declaration | Mechanism |
|---|---|
| Competing interests | Elsevier declarations tool; resulting **Word `.doc`/`.docx`** uploaded at "attach/upload files". No signatures. "I have nothing to declare" option exists. |
| Funding | Named-source format; sponsor's role in design/collection/analysis/writing/decision-to-submit; if none, the recommended sentence "This research did not receive any specific grant from funding agencies in the public, commercial, or not-for-profit sectors." |
| CRediT | **Corresponding author is required** to assign the 14 roles. |
| Generative AI | **New section immediately before the reference list**, titled "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process". Template: "During the preparation of this work the author(s) used [TOOL] in order to [REASON]. After using this tool/service, the author(s) reviewed and edited the content as needed and take(s) full responsibility for the content of the published article." Does **not** apply to basic grammar/spelling/reference tools. AI figure use must additionally be disclosed **in each image caption**. |
| Authorship | All authors listed at submission. Changes only before acceptance, only via the Authorship Change Request form with written confirmation from everyone. **No authorship changes after acceptance, at all.** |

### 2.6 Figures and tables — two rules that bite us

**Figures:** separate files, logical names (`Figure_1`, …), all cited in text, every figure captioned (brief title not shown on the figure + description). Formats: vector → EPS/PDF with embedded fonts; halftone → TIFF/JPG/PNG ≥ **300 dpi** (single column min 1063 px, full page 2244 px); bitmapped line drawings → ≥ **1000 dpi** (3543 / 7480 px); line+halftone → ≥ 500 dpi (1772 / 3740 px).

> **"Please do not submit … different images or graphs combined into one, as this affects accessibility."**

That is a direct instruction *against* the competitor's signature 6-panel convergence-curve grid. Do not imitate it. Ship single-purpose figures, or one genuinely coherent figure per number.

**Tables:** editable text, never images; captions provided; notes below the table body; **avoid vertical rules and shading**; use sparingly and do not duplicate data reported elsewhere. (Our comparison grid and per-family transfer table must be booktabs-style, no vrules.)

### 2.7 Structure, math, appendices

Numbered sections 1, 1.1, 1.1.1; cross-reference by number, never "the text"; abstract excluded from numbering. Footnotes sparingly. Appendices lettered A, B; equations `Eq. (A.1)`; tables/figures `Table A.1`, `Fig. A.1`. Math as editable text, variables italic, `exp` for powers of e, solidus for small fractions, displayed equations numbered consecutively.

### 2.8 References — **APA 7th**

In-text APA style; reference list **alphabetical then chronological**; same-author-same-year gets a/b/c. DOIs encouraged. `[dataset]` tag immediately before data references. Preprints must be marked with the word "preprint" or the server name + preprint DOI, and replaced by the formal publication where one exists. Web references need full URL + access date.

### 2.9 Data and code — **Option C, mandatory**

> "For this journal, **Option C** instructions from our research data guidelines apply. This means that you are **required to**: deposit your research data in a relevant data repository; cite and link to this dataset in your article. If this is not possible, make a statement explaining why research data cannot be shared."

Plus a **data statement is required at submission**. "Research data" is defined to include **software, code, models, algorithms, protocols and methods**. Optional co-submission to **MethodsX** or **Data in Brief**, linked on ScienceDirect if both are accepted.

### 2.10 Double-anonymous — **not an option, it is the process**

> "This journal follows a double anonymized review process, meaning author identities are concealed from reviewers and vice versa."

Title page and anonymized manuscript as **separate files**. The title page additionally carries acknowledgements, the competing-interest declaration (if no separate file), and the corresponding author's **full postal address and email**. The anonymized manuscript contains the main body **including references and tables**, and **"any supplementary materials"** must also be free of identifying information.

### 2.11 Process facts worth planning around

- Minimum two reviewers; editors decide; **one appeal only**, final.
- **4 days submission → first decision.** That is the desk-reject clock. Everything in Part 5 has to be right on day zero.
- LaTeX: use the Elsevier template; double-column layout is permitted **only** for LaTeX; PDF is not an acceptable source file.
- Free **SSRN preprint posting** offered at submission, publicly available once it passes desk review, with no effect on the editorial outcome. Requires all co-authors' agreement. Given the "clock is real" pressure in `CLAUDE.md`, this is worth a decision.
- Open access APC **USD 3,740** excl. tax; subscription route is free to authors.
- Editor-in-Chief: Yasser Dessouky (San José State University).
- Proof corrections due in **two days**.

---

## PART 3 — SPECIAL ISSUES (verified today on the live call-for-papers page)

<https://www.sciencedirect.com/journal/computers-and-industrial-engineering/about/call-for-papers>

**★ THE MATCH — deadline 30 September 2026 (7.5 weeks from today)**

> **"Simulation-Driven Industrial Transformation: Towards Resilient, Sustainable, and Human-Centric Operations"**
> Posted 4 Nov 2025. Guest editors: **Masood Fathi** (Univ. Skövde), **Mehdi Toloo** (Univ. Surrey), **Ming-Lang Tseng** (Asia Univ.).
> Article type to select in the editorial system: **`VSI: Simulation-Driven Industrial Transformation`**.
> <https://www.sciencedirect.com/special-issue/327465/simulation-driven-industrial-transformation-towards-resilient-sustainable-and-human-centric-operations>

Listed topics that our paper hits **verbatim**:
- *Simulation for Assessing Industrial System Resilience*
- *Supply Chain Simulation for Resilience and Sustainability*
- *AI-Powered Simulations for Decision Support and Optimization*
- *Simulation-Driven Decision Support Systems*
- *Complex Industrial Systems Simulation*
- *Multi-Objective Optimization in Simulation-Based Industrial Processes*

Honest caveat: two of the SI's three adjectives (sustainable, human-centric) do not describe our paper. Lead the cover letter with "Simulation for Assessing Industrial System Resilience" and "AI-Powered Simulations for Decision Support and Optimization", quoted from their own topic list, and do not manufacture a sustainability angle.

**Other open calls (verified, weaker fit):**
- *Bridging Data-Driven Innovation and Sustainable Practices* — Arbaoui, Hadj-Hamou, Masmoudi — deadline **30 Nov 2026**.
- *Manufacturing and Service Digital Ecosystems* — Dou, Matsuno, Zhou — deadline **31 Oct 2026**.

**Closed:** *Operations Research and AI in Logistics: methodology, case studies and applications* — deadline was **01 Aug 2026**, six days ago. Do not plan against it.

SI review is the same standard as regular: the guest editor may route reviews and recommend, but the journal editor oversees and makes the final decision.

---

## PART 4 — THE GENERALITY PROBLEM: WHAT WE MUST ADD

C&IE's application test names generality explicitly. One DES of one military food supply chain will not clear it **if the paper's object is the supply chain**. It clears easily if the paper's object is the **outer-loop search contract** and the DES is the instrument. Nine concrete moves, each with a verified in-journal precedent.

### Move A — Re-declare the unit of contribution: a protocol, not a case ★ highest leverage

Reframe the paper as a *methodology* contribution under the first scope door ("development of new computerized methodologies"). The contribution is the **evaluation contract**: retained search state across contexts, the **state-blind marginal replay** falsifier, prospective 16× design-space expansion on virgin seeds. The DES supplies the response surface.

**Precedent — C&IE publishes exactly this shape:**
Kordestani, Ahmadi & Chiong (2021), *"Comparison of sampling methods for algorithm configuration problem: A case for tuning differential ant-stigmergy (DASA) algorithm parameters"*, **C&IE 156:107277**, DOI `10.1016/j.cie.2021.107277`. Its generality does **not** come from more industrial cases — it comes from evaluating the configuration methods on the **Sphere function and the first seven noiseless BBOB-2010 benchmark functions**. That is the cheapest generality proof available to us.

**What to add:** port the four carriers (factorized UCB1, the Fig.-5 neuron, GP-EI, OFAT) onto a **standard synthetic benchmark family** under the identical 288 → 4,608 expansion contract and identical falsifier. If the factorized bandit still beats its own state-blind marginal replay there, the result is no longer about rations.

### Move B — A second, structurally different testbed

Two environments turn "one case" into "replicates across environment families". It does not need to be expensive — a published (s,S) inventory testbed or a small job-shop DES, wrapped in the same outer loop, is enough. The point is that the **design space** differs in factor count, interaction structure and noise, not that the domain is glamorous.

### Move C — Two instance families: standard + "more realistic"

**Precedent:** Chaurasia & Sun (2019), *"Order Acceptance and Scheduling with Sequence-dependent Setup Times: A New Memetic Algorithm and Benchmark of the State of the Art"*, **C&IE 134:106102**, DOI `10.1016/j.cie.2019.106102`. Verified abstract: it evaluates on "a set of standard benchmark instances" **and** "a set of new benchmark instances with more realistic properties", then reports **where each algorithm wins and loses** ("Sparrow is distinguished by its ability to solve difficult instances… HSSGA performs well on large instances").

**What to add:** a per-regime win/loss table, not a single headline. "UCB1 transfers when X; the neuron does not transfer when Y" is a stronger C&IE result than "UCB1 wins."

### Move D — Ship a pre-run screening rule, not just a verdict ★ converts the negative into a tool

The strongest transferable object we own is the **decision rule for when to bother**. We have measured: curvature 0.076 against noise 0.317 (`neural-premium-needs-curvature-above-noise`); `H_regime` = 0 under Cobb-Douglas on the 288 grid; `H_regime` × 7.4 with buffers added (R6). Package this as a **screening procedure a reader can run on their own design space before spending on a learner**: compute curvature-to-noise, compute `H_regime`, then choose carrier class.

This is literally "the tools utilized to solve it" plus "applicability to other problems" in the aims-and-scope sentence. It is the single addition that most changes the desk editor's read.

### Move E — Characterize the testbed by dimensionless features, not by its story

Report results as a function of quantities a reader can compute for their own problem: number of factors, levels per factor, expansion ratio (16×), effect-size-to-noise ratio, interaction strength, evaluation budget per context. Then the reader can locate their own problem on our axes. Put "military food supply chain" in one paragraph of §3 and never again as an explanatory variable.

### Move F — Managerial implications, explicitly, at two levels

**Precedent:** Ivanov-style single-case resilience simulation is publishable in C&IE. Verified abstract, **C&IE 160:107593**, DOI `10.1016/j.cie.2021.107593` ("Simulation-based assessment of supply chain resilience with consideration of recovery strategies in the COVID-19 pandemic context"): one multi-stage SC, and its generality rests on three things — it proposes a **reusable measurement method** ("for the first time, we propose a method to deduce quantitative resilience assessment from simulation"), it discusses **managerial implications at the descriptive and predictive levels**, and it states that decision-makers and scholars can **reuse the model and method**. Copy that architecture.

### Move G — Anchor in the C&IE conversation the gap already lives in

Per `docs/ESTRATEGIA_CIE_2026-08-06.md`, **Garrido 2024 cites seven C&IE papers**: Bruckler et al. 2024 (C&IE 192:110176 — verified, DOI `10.1016/j.cie.2024.110176`, *"Review of metrics to assess resilience capacities and actions for supply chain resilience"*), Carvalho et al. 2012 (62(1)), Habibi et al. 2023 (183), Ivanov 2019 (127), Moosavi & Hosseini 2021 (160), Pires Ribeiro & Barbosa-Póvoa 2018 (115), Rahman et al. 2022 (170).

**Say this in the cover letter.** The gap we are closing was posed by citing this journal seven times. And our `ret_excel`-rewards-abandonment finding is a direct, testbed-independent contribution to Bruckler et al.'s metrics review — a second contribution that generalizes without a second case.

Also cite the direct in-journal ancestor of RQ1: **C&IE 40:133–148 (2001), DOI `10.1016/S0360-8352(01)00013-4`, "Empirical comparison of search algorithms for discrete event simulation."** C&IE has published outer-loop search comparison for a quarter century. And speak the journal's simulation-optimization dialect: **C&IE 167:108007 (2022), DOI `10.1016/j.cie.2022.108007`, "Bayesian-based indifference-zone multi-objective ranking and selection procedures"** — position the factorized UCB1 carrier in **ranking-and-selection / simulation-optimization** vocabulary, not RL vocabulary.

Domain precedents proving the testbed is not disqualifying: **C&IE 161:107752 (2021), DOI `10.1016/j.cie.2021.107752`** (military workforce, asset and fleet planning via risk-averse multi-objective simulation-based optimization) and **C&IE 190:110145 (2024), DOI `10.1016/j.cie.2024.110145`** (digital twin + ML + optimization for resilient production–distribution under disruptions). Most recent and closest: **C&IE (2026), DOI `10.1016/j.cie.2026.112011`, "Operationalizing resilience optionality via hybrid simulation: a context-aware MTO execution framework."**

### Move H — Table 1 as a comparison grid, with the differentiator row

Mirror the competitor's Table 1 (Topic | Literature | Method | Advantage–Limitation | "This Paper") because C&IE reviewers read that shape. Our last row's differentiators, none of which Ding et al. have: **a non-neural comparator**, **a state-blind replay of the carrier's own marginals**, **seed custody with virgin confirmation blocks**, **preregistered falsifiers**, **confidence intervals on every headline**.

The positioning sentence writes itself and is factually checkable: *every baseline in the MAPPO/QMIX/MADDPG comparison is itself neural, so that design cannot identify whether the transferring ingredient is neural.* We can. That is the gap.

Competitor citation, verified: Ding, W., Ming, Z., Wang, G., Yan, Y., & Zhang, D. (2026). *Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain system-of-systems under disruption risks.* **International Journal of Production Economics, 297**, Article 109995. DOI `10.1016/j.ijpe.2026.109995`.

### Move I — The released artifact *is* generality evidence

Option C forces a deposit anyway. Deposit the **4,608-cell response surface, the outer-loop runner, the falsifier definitions and the seed ledger** as a citable dataset with a DOI. A reusable design-space artifact plus a runnable protocol lets any reader apply the contract to their own testbed — which is the operational meaning of "applicability to other problems". Consider a **MethodsX** co-submission carrying the protocol, linked to the main article; that also relieves length pressure (Move / risk R11).

### Minimum viable set, given 7.5 weeks

If only three can be done: **A** (synthetic benchmark port), **D** (screening rule), **G** (in-journal anchoring + the metrics contribution). A and D are new compute; G is writing. **C** and **H** are writing-only and should be done regardless.

---

## PART 5 — DESK-REJECT RISKS, RANKED, WITH PRE-EMPTIONS

The journal reports **4 days to first decision**. Assume a single editor read.

**R1 — "This is a negative result."** *Highest risk.* The frozen claim ends "no neural-specific transfer advantage was found," and one of the three RQs is labelled *mixed, development*.
→ Lead every surface (title, highlights, abstract's first result sentence, cover letter's first paragraph) with the **prospective confirmation that passed**: on virgin block `8200001–060`, expanding 288 → 4,608, the factorized UCB1 carrier beat cold start (+0.05744) **and** beat the state-blind replay of its own marginals (+0.03073, LCB +0.01990), while the neuron beat cold start (+0.05439) but **lost** to its own marginal replay (−0.01178 [−0.01849, −0.00484]). That is a positive, prospective, falsifier-surviving result. The neural finding is then a *boundary*, not the headline. Never let the paper's first 100 words read as "nothing worked."

**R2 — "Single case, no generality."** The scope statement names generality explicitly.
→ Part 4, Moves A/C/D/E/G/I. Add an explicit subsection titled with the word **"Generality"** or **"Transferability of the protocol"** so the editor can find it in 30 seconds.

**R3 — "Out of scope: this is an ML paper."**
→ Frame in **simulation-optimization, ranking-and-selection and design-of-experiments** vocabulary. Cite `10.1016/S0360-8352(01)00013-4`, `10.1016/j.cie.2022.108007`, `10.1016/j.cie.2021.107277`. State in the intro that Garrido's own third candidate is "simulation-optimization approach as a form of reinforcement learning" — that is a C&IE-native object, not an ML-conference object.

**R4 — Two-loop conflation.** If a reviewer reads "search strategy" as "policy", they will ask why we did not run MAPPO and will compare us to Ding et al.
→ A dedicated early figure and a two-column table separating **inner-loop control (within episode)** from **outer-loop search (between runs)**, mapped onto Garrido's Fig. 2 node numbers — ③ `Decision variables, ρ (experiment design)` and ⑧ `Metric of SCRES` — with the accumulation `L = {0+ℓ₁}, {0+ℓ₂}, …` running **between runs and configurations**. Then a one-sentence, cited scope exclusion for intra-episode RL. Enforce the vocabulary ban mechanically: grep the final source for `policy`, `organizational learning`, `the chain learns`, `neural premium` before build. (`HIGHLIGHTS.txt` currently fails this grep.)

**R5 — The DES section overclaims, or underclaims dishonestly.** ★ **flagging a real discrepancy**
The brief's phrasing — *"one ledger column (sumBt) is unreconstructed in >1.09% of 47,780 rows"* — is inverted relative to the repository record. `docs/REGISTRO_DE_HUECOS_2026-08-07.md` §A4 and `docs/PREGUNTAS_GARRIDO_2026-08-07.md` §1 both say: **no convention we tried reconstructs the column in more than 1.09% of the 47,780 rows** — i.e. reconstruction *succeeds* on at most 1.09%, so the column is unreconstructed on **≥ 98.91%** of rows. "More than 1.09%" is technically true and materially misleading; a reviewer who checks the artifact will find the understatement, and understating your own limitation is worse than the limitation.
→ Publish the exact repo sentence: *"The algebraic reproduction of the ReT formula is proven; the order-by-order behavioural reproduction of the DES is not — no convention we tested reconstructs `sumBt` in more than 1.09% of the 47,780 delivered ledger rows."* Keep the section titled **"prospective reproduction of six thesis-derived comparative panels"**, and per the repo's own contingency, if Garrido does not answer, fix a convention, declare it as **our** decision, and **remove every occurrence of "behavioural reproduction" from the manuscript**.

**R6 — Data availability fails Option C.** The chassis records "final archival DOI pending" and the anonymity conflict.
→ Deposit to Zenodo/Mendeley Data **now** with a reserved DOI and a review-only anonymous access link; insert the DOI in the manuscript and the metadata; state the routing decision in the cover letter (editors-only vs. post-review visibility). Blocking item; needs a human decision, not a code change.

**R7 — Anonymity leak.** The chassis warns; the guard script is missing from the branch; the repo is public and searchable on our distinctive terms.
→ Recover `ba96890`, parameterize `PAPER_ROOT`, add every new author's tokens **plus** repo name, GitHub org/URL, "SCRES-IA", "Program Q/O/S", and branch names. Run the guard over **supplementary material too** — the guide requires supplementary to be anonymous as well. Check figure file metadata and PDF producer strings.

**R8 — Stale prohibited content ships in the package.** `HIGHLIGHTS.txt` and `TITLE_PAGE.tex` carry the retracted RecurrentPPO claim family.
→ Rewrite both before any build. Enforce the repo rule: every number in the manuscript must exist in `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` or its amendment.

**R9 — Figure non-compliance.** The guide explicitly forbids combining different graphs into one image, and sets 300/500/1000 dpi floors.
→ Do not build a competitor-style 6-panel grid. One idea per figure, individual captions, resolution checked at build time by `scripts/build_cie_outer_loop_figures.py`. Colour must be accessible to impaired colour vision.

**R10 — Reference-style failure.** C&IE is **APA 7th, alphabetical then chronological** — a real trap for a group whose other targets use numeric styles.
→ Set the `.bst`/CSL correctly, re-run the bidirectional citation check on the **final tagged source** (chassis row 13), tag `[dataset]` on the deposit, and cite Ding et al. 2026 as a formal publication (`10.1016/j.ijpe.2026.109995`), not a preprint.

**R11 — Sprawl.** Custody apparatus, falsifiers, seed ledgers and the exhaustion record are large.
→ Appendices A/B with `Eq. (A.1)` / `Table A.1` numbering; the seed ledger and falsifier definitions go to the deposited artifact; consider a **MethodsX** co-submission for the protocol itself.

**R12 — Ethics/permissions on the military model.** An open checklist item.
→ Complete the security review of public wording and assets before submission; note the guide's neutral-jurisdiction and full-institutional-name rules for the title page.

**R13 — Authorship not settled.** C&IE will not consider authorship changes after acceptance at all, and generally not after submission.
→ Garrido's role and the CRediT matrix must be closed **before** the file is uploaded. This is currently open (checklist: "Garrido returns the bounded face-validation request in writing").

**R14 — GenAI disclosure mismatch.** Our draft is good but must sit in a **new section immediately before the references** and use the template's shape. Keep the sentence separating writing assistance from scientific computation — then make sure the methods/reproducibility section actually documents the agentic experiment execution, or the sentence becomes a liability.

**R15 — Title legibility.** "Retained Search State Before Neural Architecture" is precise but cryptic for indexing and for a 4-day editor read; "Prospective Transfer in Supply-Chain Resilience Simulation Optimization" is the part that will be searched.
→ Consider surfacing the IE object and the confirmed positive, e.g. a form that contains *simulation optimization*, *supply-chain resilience*, and *search state* in the first eight words. Avoid abbreviations per the guide.

---

## APPENDIX — VERIFIED CITATION LIST FOR THE MANUSCRIPT

| Purpose | Reference | DOI |
|---|---|---|
| The gap we answer | Garrido, A., Pongutá, F., & Adarme, W. (2024). *Enhancing the Operationalization of SCRES-Based Simulation Models with AI Algorithms*. LNCS, 80–94 | `10.1007/978-3-031-71993-6_6` |
| Contemporary competitor | Ding, Ming, Wang, Yan & Zhang (2026). IJPE 297, 109995 | `10.1016/j.ijpe.2026.109995` |
| Metrics contribution target (cited by Garrido) | Bruckler et al. (2024). C&IE 192, 110176 | `10.1016/j.cie.2024.110176` |
| In-journal ancestor of RQ1 | *Empirical comparison of search algorithms for discrete event simulation* (2001). C&IE 40, 133–148 | `10.1016/S0360-8352(01)00013-4` |
| Generality-via-benchmark precedent (Move A) | *Comparison of sampling methods for algorithm configuration problem* (2021). C&IE 156, 107277 | `10.1016/j.cie.2021.107277` |
| Two-instance-family precedent (Move C) | *Order Acceptance and Scheduling… Benchmark of the State of the Art* (2019). C&IE 134, 106102 | `10.1016/j.cie.2019.106102` |
| Single-case resilience precedent (Move F) | *Simulation-based assessment of supply chain resilience…* (2021). C&IE 160, 107593 | `10.1016/j.cie.2021.107593` |
| Simulation-optimization dialect | *Bayesian-based indifference-zone multi-objective ranking and selection* (2022). C&IE 167, 108007 | `10.1016/j.cie.2022.108007` |
| Military-domain precedent | *…risk-averse multi-objective simulation-based optimization for a military workforce planning, asset and fleet management problem* (2021). C&IE 161, 107752 | `10.1016/j.cie.2021.107752` |
| Nearest recent neighbour | *Operationalizing resilience optionality via hybrid simulation: a context-aware MTO execution framework* (2026). C&IE | `10.1016/j.cie.2026.112011` |
| Resilience + DT/ML precedent | *Digital twin model with machine learning and optimization for resilient production–distribution systems under disruptions* (2024). C&IE | `10.1016/j.cie.2024.110145` |

**Key URLs**
- Guide for authors: <https://www.sciencedirect.com/journal/computers-and-industrial-engineering/publish/guide-for-authors>
- Aims and scope: <https://www.sciencedirect.com/journal/computers-and-industrial-engineering/about/aims-and-scope>
- Calls for papers: <https://www.sciencedirect.com/journal/computers-and-industrial-engineering/about/call-for-papers>
- Target SI: <https://www.sciencedirect.com/special-issue/327465/simulation-driven-industrial-transformation-towards-resilient-sustainable-and-human-centric-operations>
- Submission portal: <https://submit.elsevier.com/CAIE>

**Repo paths referenced:** `/Users/thom/Projects/research/scres-ia/papers/cie_chassis/` (all five chassis files + `README.md`), `/Users/thom/Projects/research/scres-ia/docs/ESTRATEGIA_CIE_2026-08-06.md`, `/Users/thom/Projects/research/scres-ia/docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, `/Users/thom/Projects/research/scres-ia/docs/REGISTRO_DE_HUECOS_2026-08-07.md`, `/Users/thom/Projects/research/scres-ia/docs/PREGUNTAS_GARRIDO_2026-08-07.md`, `/Users/thom/Projects/research/scres-ia/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md`. Missing script to recover from commit `ba96890`: `scripts/build_submission_a_cie_review_bundle.py`.