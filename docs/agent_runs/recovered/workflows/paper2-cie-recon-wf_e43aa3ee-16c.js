export const meta = {
  name: 'paper2-cie-recon',
  description: 'Reconnaissance for the Paper 2 C&IE manuscript: journal conventions, literature, figure set, outline',
  phases: [
    { title: 'Reconocimiento', detail: 'C&IE conventions, two literature sweeps, evidence inventory, genre analysis' },
    { title: 'Diseño', detail: 'figure set and manuscript outline built from the sweeps' },
    { title: 'Adversarial', detail: 'reviewer-2 attack and completeness critic' },
  ],
}

const REPO = '/Users/thom/Projects/research/scres-ia'


const PHASE1_FINDINGS = `
WHAT PHASE 1 ESTABLISHED — bind these, they change the paper:

1. RQ1 IS A REPLICATION, NOT A CONTRIBUTION. "Warm-starting an optimizer on a related task helps" is
   Feurer 2015, Golovin 2017, Lindauer & Hutter 2018 (165x speedups), Perrone 2019, and the Bai 2023
   survey. Related Work carries a subsection literally titled "What is already established", listing
   them, then one sentence: none of these is our contribution.

2. THE ACTUAL CONTRIBUTION IS THE LAYER REASSIGNMENT. Garrido et al. asked which FAMILY of AI carries
   cross-run memory and named three inner-loop function approximators. The effective carrier lives at
   a different layer: outer-loop search state, where the right object is a bandit allocation. No
   citation found makes this argument.

3. TWO PAPERS NARROW THE CLAIM AND WE FOUND THEM, NOT A REVIEWER.
   - Preil & Krapp (2022), IJPE 252:108578: a UCB-family bandit in multi-echelon inventory. OUR
     CARRIER IS NOT NOVEL AS AN ALGORITHM. Cite, concede; novelty is the outer loop across runs and
     contexts, plus the control design.
   - Rachman et al. (2026), EJOR: RL benchmarked against a multi-objective evolutionary algorithm,
     whose population IS retained non-neural search state. "First to compare neural against
     non-neural stateful in supply chains" would be FALSE.

4. THE MARGINAL-REPLAY CONTROL, at its exact strength: a STRONGER null than the uniform-random
   control that audited neural architecture search (Yu et al., ICLR 2020), because it matches the
   first-order visit distribution and therefore isolates sequential conditioning alone. Introduce it
   in the SAME SENTENCE as Yu 2020, Perrone 2019 and Bai 2023. DO NOT call it a placebo: in this
   literature that implies it must come out null, and ours legitimately might not.

5. FIT-VERSUS-SEARCH DISSOCIATION IS ALREADY PUBLISHED (Tom et al. 2025; Eggensperger 2015; Jones
   2001). Cite it; claim only the instance and its consequence in a domain making the opposite
   assumption.

6. TEN EVIDENCE DEFECTS the figure/outline design must not inherit:
   - "excludes zero" for the ofat contrast is FORBIDDEN (opposite-signed bounds on byte-identical
     replicates; positive in 26 of 40 resampling seeds). Quote BOTH sealed bounds, say indistinguishable.
   - "the KAN fits better yet searches worse" SPLICES TWO ARTIFACTS: surrogate_architecture_bakeoff
     stores NO fit metric at all. A one-artifact fit-vs-search scatter DOES NOT EXIST.
   - H_regime = 0.00380 has no path, digest or normalisation stated, and its artifact has a falsifier
     in red which IS the finding.
   - retention_contrasts is NOT preregistered and must never be described as such; it is a post-hoc
     re-analysis whose citability rests on f3 reproducing the source's sealed contrast bit-for-bit.
   - garrido_h2_h3_confirmation_v1 has NO machine-readable grade: cite result AND sibling receipt.
   - Per-context breakdowns do not exist for RQ1 or RQ2; contexts are averaged before storage.
   - repro_probe/A and /B are the SAME architecture and SAME seed (9492) — a determinism probe, not a
     cross-architecture reproduction.
   - demand_seasonal_engine is ENGINE_PARTIAL with g4 and g5 failing; it supports no seasonal claim.
   - The formal claim uses the RUN index k, never physical time t:
     Y_k = DES(x_k, c_k, w_k); L_k = U(L_{k-1}, x_k, Y_k); x_{k+1} = pi(L_k, c_{k+1}).
     Physical state resets between runs; only L_k persists.
   - We cannot claim "first integration of AI, simulation and SCRES": the MAPPO work already studies
     dynamic reconfiguration under risk.

7. THE MANUSCRIPT IS INCOHERENT TODAY: papers/paper2/03_results.md tells the new story, but the
   introduction, methods, discussion and master table still tell the old Program O story. Results is
   the base; the other three are replaced wholesale.
`

const SHARED = `
CONTEXT. You are helping design a manuscript for Computers & Industrial Engineering (Elsevier).
Repo: ${REPO}. The machine is BUSY running a long verification job -- do NOT launch heavy compute,
do NOT run pytest, do NOT run anything that takes more than ~30s of CPU. Read files, grep, and use
web search. Read-only.

THE PAPER. Working title: "Retained Search State Before Neural Architecture: Prospective Transfer
in Supply-Chain Resilience Simulation Optimization".

Central claim (do not weaken or strengthen it, it is frozen):
"Within the evaluated outer-loop contract and the inherited demand process, all tested stateful
variants achieved lower development regret than their memoryless counterparts. In a prospective
expansion from 288 to 4,608 configurations, only a factorized UCB SEARCH STRATEGY outperformed both
cold start and a state-blind replay of its own search marginals; no neural-specific transfer
advantage was found."

Three research questions:
  RQ1 does retaining decision-search state improve search across contexts?   [REPLAY evidence]
  RQ2 does the transfer survive a state-blind replay of its own marginals
      when the design space grows 16x?                                        [CONFIRMATION]
  RQ3 is the surviving carrier specifically neural?                           [mixed, development]

KEY VOCABULARY RULES, binding:
 - "outer-loop search strategy" or "search carrier", NEVER "policy" -- the loop is BETWEEN runs, not
   within an episode. The whole paper depends on not conflating those two layers.
 - never "organizational learning", never "the chain learns", never "neural premium".
 - the DES section is "prospective reproduction of six thesis-derived comparative panels", NOT
   "validation of the DES": the original Simulink model is unavailable and one ledger column
   (sumBt) is unreconstructed in >1.09% of 47,780 rows.

THE INTELLECTUAL SETTING. Garrido, Ponguta & Adarme (ICCL 2024) asked two questions: which family of
AI algorithms best mimics the supply-chain-learning attribute, and how to integrate it into a DES
for resilience assessment. They named backpropagation, KAN and RL as candidates and called the
absence of cross-run memory "the Alzheimer effect". Our answer refines their hypothesis rather than
confirming it: the effective carrier is persistent search state, and it is not neural.

A CONTEMPORARY COMPETITOR the user just supplied: Ding, Ming, Wang, Yan & Zhang (2026),
"Multi-agent reinforcement learning-based resilience reconfiguration approach of supply chain
system-of-systems under disruption risks", Int. J. Production Economics 297:109995. It uses MAPPO
against QMIX and MADDPG baselines on a POMDP formulation of supply-chain reconfiguration, with
resilience and cost in the reward. Its structure: Intro / Background+related work with an explicit
research-gap subsection / method / case study with three disruption scenarios / generalization
analysis / conclusion. Its Table 1 is a literature comparison grid (Topic | Literature | Method |
Main Advantage-Limitation) whose last row is "This Paper". Its figures are framework diagrams,
convergence curves (6-panel grids), a before/after resilience bar chart, and attribute-sweep bar
charts. NOTE FOR POSITIONING: every one of its baselines is also neural. It has no non-neural
comparator, no state-blind replay control, no seed custody, no falsifiers, and no confidence
intervals on the headline curves.
`

phase('Reconocimiento')

const recon = await parallel([
  () => agent(`${SHARED}

YOUR TASK: Computers & Industrial Engineering as a venue. Produce a concrete, actionable brief.

1. Read ${REPO}/papers/cie_chassis/CIE_GUIDE_AUDIT_2026-07-29.md and
   ${REPO}/papers/cie_chassis/RELEASE_AND_SUBMISSION_CHECKLIST.md and
   ${REPO}/papers/cie_chassis/HIGHLIGHTS.txt and TITLE_PAGE.tex and
   GENERATIVE_AI_DISCLOSURE_DRAFT.md. Summarise every requirement they encode.
2. Web-search the current C&IE author guide and aims/scope. Report: article types and their length
   norms, what C&IE says it requires of an APPLICATION paper (originality, generality,
   significance), structured abstract or not, highlights format, declarations required, figure and
   table formatting rules, reference style, data/code availability policy, double-anonymous option.
3. Search for any current C&IE special issue relevant to simulation-guided industrial
   transformation / resilient operations, with deadlines. Report only what you can verify.
4. The hardest question: C&IE demands that an application generalise beyond its case. Our paper is
   one DES of one military food supply chain. Enumerate the specific moves that make a
   single-testbed methods paper acceptable at C&IE, with examples of published C&IE papers that
   did it. Be concrete about what we must add.
5. Report the desk-reject risks for this specific paper and how to pre-empt each.

Return a structured brief with citations/URLs. Be specific, not generic advice.`,
    { label: 'cie-venue', phase: 'Reconocimiento' }),

  () => agent(`${SHARED}

YOUR TASK: the literature this paper must engage, on the AI-for-supply-chain-resilience side.

Use web search plus the repo's own bibliography if one exists (grep for .bib files).

1. Map the current literature on RL / MARL / deep learning for supply chain resilience and
   reconfiguration, 2022-2026. Identify the 15-25 works this manuscript MUST cite, with full
   citations. Prioritise recent, high-impact, and venue-adjacent (C&IE, IJPE, IJPR, EJOR, Omega).
2. Bruckler et al. 2024, C&IE 192:110176, "Review of metrics to assess resilience capacities and
   actions for supply chain resilience" -- read what it actually contributes (harmonised
   terminology, 17 metrics organised on the resilience curve, derived formulations). Say precisely
   what gap it leaves that we fill, and what it does NOT leave.
3. Ding et al. 2026 IJPE (described above). Position us against it precisely and fairly. Where is it
   strong? Where is our evidence stronger, and where is theirs? Draft two or three sentences we
   could actually put in Related Work.
4. THE KEY POSITIONING QUESTION: is there ANY published work in SC resilience that compares a neural
   learner against a matched non-neural stateful comparator, or that controls for "the method is
   just replaying its own visit marginals"? Search hard. If the answer is no, that is our
   contribution and we must be able to say so defensibly. If someone has done it, we must cite them
   and narrow our claim.
5. Report the standard critiques reviewers make of RL-for-SC papers, with sources.

Return full citations, a positioning paragraph, and an honest statement of what is and is not novel.`,
    { label: 'lit-rl-scres', phase: 'Reconocimiento' }),

  () => agent(`${SHARED}

YOUR TASK: the METHODOLOGICAL literature -- the neighbours of "retained search state", which is
where a methods reviewer will look for prior art.

1. Our object is simulation optimization with an outer loop that carries state across related design
   contexts. Map the relevant literatures and give full citations for the works we must cite:
   - simulation optimization and ranking-and-selection
   - multi-armed bandits, UCB1, Thompson sampling, and FACTORIZED / combinatorial bandits
   - Bayesian optimization: transfer / warm-starting / meta-learning across related tasks
   - meta-learning and learning-to-optimize
   - hyperparameter transfer (the SMAC / warm-start literature)
   - Kolmogorov-Arnold Networks (the 2024 Liu et al. paper and its critiques)
2. CRITICAL PRIOR-ART CHECK: "warm-starting an optimizer on a related task helps" is decades old.
   State plainly what is genuinely new in our contribution given that. Our candidates are (a) the
   state-blind marginal-replay falsifier as a control, (b) the prospective 16x design-space
   expansion on a reserved seed block, (c) the finding that predictive fit and search quality
   dissociate. Assess each honestly -- is it new, or has someone done it?
3. The marginal-replay control: is there a named technique for this in the BO / bandit /
   causal-inference literature? If it has a name we should use it. Search for "visit frequency
   replay", "state-blind replay", "marginal policy control", ablation conventions.
4. What is the standard way to report search-efficiency comparisons in this literature? Normalised
   regret AUC? Simple regret? Performance profiles (Dolan-More)? Report what reviewers expect to
   see and what we should therefore plot.

Return full citations plus a blunt assessment of novelty.`,
    { label: 'lit-methods', phase: 'Reconocimiento' }),

  () => agent(`${SHARED}

YOUR TASK: inventory EXACTLY what sealed evidence exists for each manuscript section, with numbers.
This becomes the figure and table data source. Read the artifacts; do not re-run anything.

For each of these, report claim_status, scope, evidence grade, seed block, endpoint, n, the key
numbers with intervals, and which falsifiers passed/failed:
  results/grid_transfer_confirmation_v2/       RQ2, the confirmation
  results/retention_contrasts/                 RQ1, six paired within-family contrasts (NEW today)
  results/search_ladder_v5/                    the 15-method ladder
  results/surrogate_architecture_bakeoff/      RQ3b KAN vs matched MLP as a search surrogate
  results/dmlpa_kan_latent/                    RQ3c KAN in a PPO latent -- DIFFERENT contract
  results/garrido_fig5_surrogate/              RQ3a the Fig-5 identity
  results/garrido_wrap_q1/
  results/garrido_h2_h3_confirmation_v1/       the six DES panels
  results/twin_surface_v2/                     the non-anticipative normaliser control
  results/garrido_normaliser_audit_v3/         prefix vs oracle
  results/frozen_path_equivalence/             provenance (a v2 is running now)
  results/demand_seasonal_engine/              the seasonal demand engine, ENGINE_PARTIAL
  results/manuscript/h1_h3_originales_v3/      H1/H3, appendix material
  results/manuscript/h2_learning_curve/
  results/garrido_h3_merge_adjudication/

Also read ${REPO}/papers/paper2/03_results.md and every other file under ${REPO}/papers/paper2/,
and any claim_lock.json that exists, and report what prose already exists so we do not rewrite it.

Then: for EACH of the three RQs plus the two support sections, say which artifact is the primary
evidence and which numbers would go in a table. Flag any number a manuscript would want that does
NOT exist in a sealed artifact.

Return a dense evidence table. Exact values, exact paths.`,
    { label: 'evidence-inventory', phase: 'Reconocimiento' }),

  () => agent(`${SHARED}

YOUR TASK: genre analysis -- what a C&IE/IJPE paper of this type LOOKS like, so ours is not
formatted like a lab notebook.

1. The user supplied Ding et al. 2026 IJPE at
   "/Users/thom/Library/Mobile Documents/com~apple~CloudDocs/1-s2.0-S0925527326000861-main.pdf"
   (23 pages). Read it with the Read tool using the pages parameter, max 20 pages per call; read
   pages 7-12 and 21-23 which have not been read yet. Report its full section structure with
   approximate page budget per section, its complete figure list with what each one does, its table
   list, how it words its contributions, how it words its limitations and future work, and how long
   the related-work section is relative to the whole.
2. Look in the repo for any other C&IE/IJPE/IJPR paper PDFs (find . -name "*.pdf" in docs/, papers/,
   deliverables/) and do the same lighter-weight analysis on one or two.
3. Derive a TEMPLATE: for a 20-25 page C&IE methods-and-application paper, how many figures, how
   many tables, what each typically is, and the page budget per section.
4. Specifically: what does the FIRST figure of such a paper usually do, and what should ours be?
5. What conventions do these papers use for reporting uncertainty, and how does that compare with
   what we have (bootstrap percentile intervals, paired designs, preregistered falsifiers)? Where
   will our style look unfamiliar, and where will it look better?

Return a concrete template with counts and a per-section page budget.`,
    { label: 'genre', phase: 'Reconocimiento' }),
])

const [venue, litRL, litMethods, evidence, genre] = recon.map(r => r || 'AGENT FAILED')

phase('Diseño')

const design = await parallel([
  () => agent(`${SHARED}
${PHASE1_FINDINGS}

YOUR TASK: design the complete FIGURE AND TABLE SET for this manuscript.

Inputs from the reconnaissance phase:

=== EVIDENCE INVENTORY ===
${evidence}

=== GENRE TEMPLATE ===
${genre}

=== METHODS LITERATURE (what reviewers expect to see plotted) ===
${litMethods}

Design every figure and every table. For each one give:
  - number and caption
  - exactly what is plotted or tabulated, with the axes and units
  - WHICH SEALED ARTIFACT and which field supplies every number
  - why a reviewer needs it (what claim would be unsupported without it)
  - whether it is main text or supplement

Hard constraints:
  - Every figure must be generated from a sealed result.json by a builder with an admission guard.
    No hand-drawn numbers. Name the builder script for each.
  - Orientation traps to avoid: RQ1/RQ2 use auc_regret_norm where LOWER is better; the KAN-latent
    result uses ret_mean_track_b_v1 where HIGHER is better. A figure mixing them without saying so
    is a defect.
  - The headline figure must carry the CONFIRMATION (RQ2), not the development ladder.
  - RQ1 wants a forest plot of the six paired contrasts.
  - The 5.83-runs figure is BANNED from the main text (censoring differs by arm: 0.056/0.153/0.222/
    0.611); AUC under the prefix normaliser is the endpoint.
  - Style is already fixed in the repo: Okabe-Ito colorblind-safe palette, STIX serif, vector PDF
    plus 300-dpi PNG. See scripts/build_manuscript_figures.py and
    scripts/build_paper1_evidence_v1.py (whose lines 83-96 hold the admission guard to copy).

Also design the literature comparison table in the style of Ding et al.'s Table 1, with our row
last, and say exactly what our row should claim.

Return a numbered figure list, a numbered table list, and the builder plan.`,
    { label: 'figures', phase: 'Diseño' }),

  () => agent(`${SHARED}
${PHASE1_FINDINGS}

YOUR TASK: write the complete manuscript OUTLINE, section by section, ready to be drafted against.

Inputs from reconnaissance:

=== C&IE VENUE BRIEF ===
${venue}

=== RL/SCRES LITERATURE AND POSITIONING ===
${litRL}

=== METHODS LITERATURE AND NOVELTY ASSESSMENT ===
${litMethods}

=== EVIDENCE INVENTORY ===
${evidence}

=== GENRE TEMPLATE ===
${genre}

Produce a section-by-section outline with, for each section: its purpose in one line, its page
budget, the subsections, the specific claims it makes, the artifacts backing each claim, and the
figures/tables it carries.

Requirements:
 - The contributions list must be specific and checkable, in the style C&IE expects.
 - There must be an explicit research-gap subsection, as Ding et al. have, positioning against the
   literature the recon found.
 - The evidence-grade distinction must be visible to the reader without turning into internal
   governance: one confirmation, the rest development/replay. Do NOT put a count of confirmations
   in the manuscript.
 - Limitations must include, honestly: the inherited U(2400,2600) demand process; a single DES
   testbed; the seeds for RQ1 being a declared re-execution not virgin; sumBt unreconstructed; and
   the fact that the outer loop is between runs, not adaptive control within an episode.
 - There is an appendix reconciling the original H1-H4 hypotheses with the estimands finally
   identified, including the restricted_ttr chronology (the estimand was written 2026-08-06, BEFORE
   the preregistration that used it -- without that date on the page a reviewer reads endpoint
   shopping).
 - Say where the seasonal-demand robustness panel lands if it succeeds, and what the text says if
   it fails or is not finished.

Return the outline plus a draft of the contributions list and a draft abstract of <=250 words.`,
    { label: 'outline', phase: 'Diseño' }),
])

const [figures, outline] = design.map(r => r || 'AGENT FAILED')

phase('Adversarial')

const attacks = await parallel([
  () => agent(`${SHARED}

${PHASE1_FINDINGS}

You are Reviewer 2 for Computers & Industrial Engineering. You are technically excellent, you
dislike overclaiming, and you have read Ding et al. 2026 and Bruckler et al. 2024.

Here is the proposed manuscript outline and figure set:

=== OUTLINE ===
${outline}

=== FIGURES AND TABLES ===
${figures}

=== THE EVIDENCE THAT EXISTS ===
${evidence}

Write the review that would REJECT this paper. Be specific and technical. Cover at least:
 - the single-testbed generality problem C&IE cares about
 - whether "retained search state beats architecture" is actually supported or is a ranking artifact
 - whether the marginal-replay control is the right control, and what it does NOT rule out
 - the inherited low-variance demand process and whether the negative result is an artifact of it
 - the n=12 seeds for RQ1 and n=60 for RQ2 -- is that enough
 - whether a bandit beating a neural net on a 4,608-cell grid says anything about realistic problems
 - the fact that only one result is a prospective confirmation
 - anything that reads as endpoint shopping or post-hoc reformulation

Then, for each objection, say what the authors could do about it: fix with existing data, fix with
cheap new analysis, fix with words in the limitations, or cannot fix. Rank the objections by how
likely they are to actually sink the paper.

Be harsh. Do not be constructive until the second half.`,
    { label: 'reviewer2', phase: 'Adversarial' }),

  () => agent(`${SHARED}

${PHASE1_FINDINGS}

YOUR TASK: completeness critic. Find what the design MISSED.

=== OUTLINE ===
${outline}

=== FIGURES AND TABLES ===
${figures}

=== VENUE BRIEF ===
${venue}

Ask and answer: what claim in the outline has no artifact behind it? What figure has no data source?
What section of a C&IE paper is missing entirely? What required declaration or submission artifact
is unaccounted for? What number in the outline contradicts the evidence inventory? What has been
scheduled that the running compute will not deliver in time?

Also verify against the repo directly: check that every artifact path the outline and figure plan
cite actually EXISTS on disk (ls / test -f), and report any that do not. Check that the vocabulary
rules are not violated anywhere in the outline (grep the outline text for "policy" applied to the
carrier, "neural premium", "organizational learning", "excludes zero" near ofat).

Return a prioritised list of gaps, each with the concrete action that closes it.`,
    { label: 'completeness', phase: 'Adversarial' }),
])

return {
  venue, litRL, litMethods, evidence, genre, figures, outline,
  reviewer2: attacks[0] || 'AGENT FAILED',
  completeness: attacks[1] || 'AGENT FAILED',
}
