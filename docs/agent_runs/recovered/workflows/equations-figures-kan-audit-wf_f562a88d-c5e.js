export const meta = {
  name: 'equations-figures-kan-audit',
  description: 'Parallel deep-read of Garrido sources + manuscript + code to decide equations/figures/KAN additions',
  phases: [
    { title: 'Read', detail: 'six parallel readers over sources' },
  ],
}

phase('Read')

const SRC = {
  draft: '/Users/thom/Downloads/v.0_neuralNet-scres.pdf',
  ai2024: '/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Supernote/Document/20_RESEARCH/PhD-Papers/garrido2024 scres+AI.pdf',
  factory2024: '/Users/thom/Downloads/garrido et al 2024 factory resilience.pdf',
  thesis: '/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf',
  repo: '/Users/thom/Projects/research/scres-ia',
  ms: '/Users/thom/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier',
}

const results = await parallel([
  () => agent(`Read the PDF ${SRC.draft} (21 pages; use Read with pages parameter, e.g. "1-11" then "12-21").
This is v.0 of a draft that Garrido himself wrote for the neural-net/SCRES paper built on our project.
Report exhaustively:
1. EVERY equation in the draft: write each out in LaTeX as faithfully as you can, with the page number and what it defines (e.g. ReT metric, AP/RP/DP, Cobb-Douglas index, reward, NN formulation).
2. EVERY figure: number, caption, what it shows, page.
3. EVERY table: what it contains (risk distributions? parameters? results?), page.
4. How the draft describes the LEARNING MECHANISM: reward function, training loop, NN architecture — quote the key sentences.
5. Sections/content present in this draft that a results-focused LaTeX manuscript might lack (e.g. notation table, assumptions list, risk modelling detail).
Your final message is raw data for an orchestrator — dense, complete, no pleasantries.`, {label: 'read:draft-v0'}),

  () => agent(`Read the PDF "${SRC.ai2024}" (15 pages; use Read with pages parameter, "1-15" is fine or split).
This is Garrido et al. 2024 "Enhancing the Operationalization of SCRES-Based Simulation Models" (SCRES+AI, the paper recommending AI integration into DES).
Report exhaustively:
1. The EXACT argument for why Kolmogorov-Arnold Networks (KAN) are recommended: quote the relevant passages, page numbers. What specific properties (interpretability? learnable activations? symbolic extraction? accuracy on small data?) do they claim make KAN suitable for SCRES-DES integration?
2. What they say about backpropagation NNs and about reinforcement learning / simulation-optimization as the other two alternatives — quotes + pages. Do they rank the three? Do they state criteria?
3. Figure 5 (neural network integrated into a DES model for SCRES: drivers d_i, weights rho_i, transfer function Sigma, activation f, threshold theta, output SCRES): describe its exact role in their argument and any equations attached to it.
4. Their novel SCRES definition emphasizing learning — quote it verbatim with page.
5. The "Alzheimer's effect in SCs" framing — quote.
6. Any equations in the paper (write in LaTeX) and any concrete implementation guidance they give.
Dense final report, no pleasantries.`, {label: 'read:garrido2024-ai'}),

  () => agent(`Read the PDF "${SRC.thesis}" (Garrido-Rios 2017 Warwick thesis, 187 pages; Read with pages parameter, max 20 pages/request).
Strategy: first read pages 1-12 (contents/lists of figures/tables) to locate: (a) the resilience metric chapter defining ReT, AP_j, RP_j, DP_j (likely ch. 3 or 5), (b) the risk modelling tables (R11, R12, R13, R14, R21, R22, R23, R24, R3 with probability distributions), (c) the simulation model chapter with the 13-operation flowchart and parameters. Then read the specific page ranges.
Report:
1. The EXACT resilience metric equations as the thesis writes them: ReT definition, AP_j (autonomy period), RP_j (recovery period), DP_j (disruption period), the 0.5/RP branch or whatever branch/case logic exists, fill-rate FR_t, and any resilience index formula. Write each in LaTeX with thesis equation numbers and page numbers. This is the single most important output — be exact about subscripts, cases, and constants.
2. The risk table(s): for each risk ID (R11..R24, R3), distribution family, parameters, modelling assumptions — condensed but complete, with page numbers.
3. Candidate figures worth adapting for a journal paper: figure number, caption, page (especially the metric lineage/tree figure and the 13-operation model figure — note their thesis figure numbers).
4. The decision variables (buffer I_LS levels, shifts S) and experimental design (5 inventory x 3 shift etc.) with page refs.
Dense final report.`, {label: 'read:thesis', effort: 'high'}),

  () => agent(`Read the PDF "${SRC.factory2024}" (Garrido et al. 2024 factory resilience / IJPR, 20 pages; Read with pages "1-20").
Report:
1. Every equation, in LaTeX, with page numbers — especially the Cobb-Douglas factory resilience index (Re = f(...)), its variables, normalization, and any sigmoid/log transformation.
2. How an IJPR-published paper in this exact lineage presents math: how many numbered equations, where they sit (methodology?), notation-table presence, how metrics are introduced.
3. Their figures/tables inventory (what a published Garrido-lineage paper includes).
4. Anything they say about learning, AI, or future work pointing to the AI/KAN agenda.
Dense final report.`, {label: 'read:factory2024'}),

  () => agent(`Working dir: ${SRC.repo}. Read the current LaTeX manuscript: ${SRC.ms}/main.tex and all files in ${SRC.ms}/sections/.
Report:
1. EVERY piece of display math / equation currently in the manuscript (there are very few — list them all with file:line).
2. Every place the training reward 'control_v1' is described in prose — quote each (file:line). Note that no formula is given anywhere.
3. Every metric used in tables (Excel ReT, order-level ReT, CVaR05, CTj/RPj/DPj p99, service-loss AUC, flow fill, Cobb-Douglas sigmoid) and whether it is mathematically DEFINED anywhere in the manuscript or only named.
4. The current figure set (7 figs) and current tables (list labels), so an orchestrator can decide insertion points for: (a) a resilience-metric equations block, (b) a reward-function equation, (c) a risk-distribution table, (d) possibly a notation table.
5. Suggest precise insertion points (file + after which line/paragraph) for those four additions.
Dense final report with file:line references.`, {label: 'audit:manuscript'}),

  () => agent(`Working dir: ${SRC.repo}. Extract GROUND TRUTH formulas from the code so equations added to the paper match the implementation exactly. Use Grep/Read.
1. The 'control_v1' training reward: find its implementation (grep reward_mode / control_v1 in supply_chain/ and scripts/run_track_b_smoke.py or env files). Write the EXACT formula in LaTeX including all coefficients (backorder fraction term, shift-switching cost term, any scaling), with file:line evidence. Also note what the per-step observation-normalized quantities are.
2. The Garrido/Excel order-level ReT: find the branch/case formula (supply_chain/garrido_replication.py has excel_ret_value; also episode metrics order_ret_excel). Write the exact case logic in LaTeX (cases for delivered on time / late / never etc., the 0.5/72 or 0.5/RP branch, constants), file:line evidence.
3. CVaR05 as computed in the stats bundle (scripts/build_track_b_q1_stats.py build_cvar05_effect_row): exact definition in LaTeX.
4. The Cobb-Douglas sigmoid metric (cd_sigmoid / ret_garrido2024_sigmoid): exact formula, file:line.
5. AP_j / RP_j / DP_j and CTj as computed at order level (episode_metrics or order ledger code): exact definitions.
6. The 8D track_b_v1 action contract: the exact action fields and their bounds (found earlier in run_track_b_dense_crn_static.py action_for and env code) — list all 8 dims with ranges.
Dense final report, LaTeX-ready formulas, file:line for every claim.`, {label: 'extract:code-truth', effort: 'high'}),
])

return {
  draft: results[0],
  ai2024: results[1],
  thesis: results[2],
  factory2024: results[3],
  manuscript: results[4],
  code: results[5],
}