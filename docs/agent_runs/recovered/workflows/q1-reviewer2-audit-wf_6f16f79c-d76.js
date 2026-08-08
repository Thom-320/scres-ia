export const meta = {
  name: 'q1-reviewer2-audit',
  description: 'Audit SCRES-IA repo evidence, literature, and manuscript; run adversarial Reviewer #2 panel for Q1 readiness',
  phases: [
    { title: 'Evidence audit', detail: 'parallel auditors: experiment status, stats bundle, literature, manuscript, Track A/fidelity' },
    { title: 'Reviewer panel', detail: 'adversarial reviewers: stats/methods, OM/SCRES domain, RL/ML' },
    { title: 'Editor synthesis', detail: 'journal fit + minimal sufficient experiment set' },
  ],
}

const ROOT = '/Users/thom/Projects/research/scres-ia'

const AUDIT_SCHEMA = {
  type: 'object',
  required: ['summary', 'findings', 'gaps'],
  properties: {
    summary: { type: 'string', description: 'Executive summary, 5-12 sentences, concrete numbers and file paths' },
    findings: {
      type: 'array',
      items: {
        type: 'object',
        required: ['topic', 'verdict', 'evidence'],
        properties: {
          topic: { type: 'string' },
          verdict: { type: 'string', description: 'e.g. DONE / PENDING / STALE / CONTRADICTION / VERIFIED / UNVERIFIABLE' },
          evidence: { type: 'string', description: 'concrete file paths, numbers, dates' },
        },
      },
    },
    gaps: { type: 'array', items: { type: 'string' }, description: 'what is missing or must be done before Q1 submission, from this auditor\'s lens' },
  },
}

const REVIEW_SCHEMA = {
  type: 'object',
  required: ['recommendation', 'kill_points', 'major_comments', 'minor_comments'],
  properties: {
    recommendation: { type: 'string', description: 'reject / major revision / minor revision / accept, plus 1-paragraph rationale' },
    kill_points: {
      type: 'array',
      items: {
        type: 'object',
        required: ['attack', 'severity', 'evidence', 'fix'],
        properties: {
          attack: { type: 'string', description: 'the reviewer objection, stated as the reviewer would write it' },
          severity: { type: 'string', description: 'fatal / high / medium' },
          evidence: { type: 'string', description: 'what in the repo/manuscript grounds this attack (paths, numbers)' },
          fix: { type: 'string', description: 'the minimal concrete action that neutralizes the attack' },
        },
      },
    },
    major_comments: { type: 'array', items: { type: 'string' } },
    minor_comments: { type: 'array', items: { type: 'string' } },
  },
}

const COMMON_CONTEXT = `
CONTEXT (verified by the orchestrator, treat as ground truth unless you find contradicting files):
- Repo: ${ROOT} (SCRES-IA: Garrido-grounded DES of a military food supply chain + RL).
- Paper goal: Q1 journal submission. Candidate framing: "frontier-dependent adaptive recovery control" — Track A (Garrido's original buffer/shift decision family) shows no publishable dynamic frontier under dense static CRN evaluation; Track B (operational extension exposing downstream Op10/Op12 dispatch controls at the bottleneck, 8D action contract) shows PPO beating the best dense static comparator.
- Claims source of truth: ${ROOT}/docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md. Its "Primary result": Excel ReT PPO 0.005893 vs dense static S2_op10_2.00_op12_1.50 0.005466, delta +0.000426, CI95 [+0.000389, +0.000463], paired Cohen's d 2.87 (n_pairs=60 = 5 seeds x 12 eval episodes). NOTE: claim row C1 in the same file cites DIFFERENT numbers (0.005666 vs 0.005251) — an inconsistency to investigate.
- Canonical run: ${ROOT}/outputs/experiments/track_b_gain_2026-06-30/top_tier_confirm_v3_output/track_b_top_tier_confirm_5seed_60k_h104
- Stats bundle: ${ROOT}/outputs/audits/track_b_q1_stats_2026-07-01/ and ${ROOT}/docs/track_b_q1_stats_2026-07-01/ (README.md, effect_sizes.csv, mechanism_audit.json, pareto_summary.json, ablation_decision.md, ledger_tail_panel.csv, mechanism_metric_panel.csv, pareto figures).
- Fidelity gate: ${ROOT}/docs/FIDELITY_GAP_BUFFER_SATURATION_2026-06-28.md — header says RESOLVED / GATE PASSED (DES reproduces Garrido H2/H3; buffer saturation at I168 is faithful-consistent).
- Consolidated plan: ${ROOT}/docs/TRACK_B_FINAL_CONSOLIDATED_PLAN_2026-07-01.md
- Manuscript: ${ROOT}/docs/manuscript_current/submission/elsevier/ (main.tex, sections/01..06, references.bib).
- Registry checklist says still OPEN as of 2026-07-01: final canonical workbook, current-contract 8D ablations, mechanism lead/lag audit, generalization table, H4 retained-vs-reset, retired-claim scrub.
- New in git status (untracked, may postdate the registry): ${ROOT}/kaggle/track_b_adaptive_confirm_v9/ and ${ROOT}/scripts/watch_track_b_adaptive_confirm_v9.sh (v9 observation confirm run, possibly on Kaggle now).
Your final message is consumed by an orchestrator, not a human — return dense raw findings, no pleasantries. Cite file paths for every claim.`

// ---------- Phase 1: parallel evidence auditors ----------
phase('Evidence audit')

const audits = await parallel([
  // A1: experiment status
  () => agent(`${COMMON_CONTEXT}

You are the EXPERIMENT-STATUS AUDITOR. Determine what is actually DONE vs PENDING right now (2026-07-01), because review documents from earlier today may already be stale.

Tasks:
1. Scan ${ROOT}/outputs/experiments/ and ${ROOT}/outputs/audits/ (ls with dates; read summary.csv/README/json where present) for: (a) current-contract 8D ablation results (joint / downstream_only / shift_only / fixed-shift / no-risk), (b) mechanism lead-lag audit outputs, (c) generalization runs across risk levels (current/increased/severe/adaptive_benchmark_v2), families (R2-only/R24-only/mixed), horizons (h52/h104/h260), (d) any heuristic-baseline (threshold/hysteresis dispatch) results, (e) SAC/TD3 runs, (f) H4 retained-vs-reset runs.
2. Inspect ${ROOT}/kaggle/track_b_adaptive_confirm_v9/ and ${ROOT}/scripts/watch_track_b_adaptive_confirm_v9.sh — what exactly is the v9 confirm run (obs version, reward, seeds, timesteps, risk level, horizon)? Is it running/complete? Check ${ROOT}/kaggle/ for other track_b_* runs and their states.
3. Read ${ROOT}/docs/TRACK_B_FINAL_CONSOLIDATED_PLAN_2026-07-01.md and ${ROOT}/docs/TRACK_B_LEARNING_PROTOCOL_2026-06-30.md and reconcile: which planned items have artifacts on disk, which do not.
4. Check ${ROOT}/docs/track_b_q1_stats_2026-07-01/ablation_decision.md — what does it decide?
5. Check git log (last ~30 commits) for context on what was run most recently.
6. Confirm exact seed/episode structure of the canonical run (how many learner seeds, how many eval episodes each, CRN protocol) by reading the run dir's config/metadata.

Return for each item: DONE (with path + headline numbers) / RUNNING / PENDING / NOT FOUND.`, { label: 'audit:experiments', phase: 'Evidence audit', schema: AUDIT_SCHEMA }),

  // A2: stats bundle verification
  () => agent(`${COMMON_CONTEXT}

You are the STATISTICS AUDITOR. Verify the paper's headline inference with a hostile eye.

Tasks:
1. Read every file in ${ROOT}/docs/track_b_q1_stats_2026-07-01/ and ${ROOT}/outputs/audits/track_b_q1_stats_2026-07-01/ (both if both exist): README.md, effect_sizes.csv (full contents), pareto_summary.json, mechanism_audit.json, mechanism_metric_panel.csv, ledger_tail_panel.csv, ablation_decision.md.
2. Resolve the number inconsistency: registry C1 says PPO 0.005666 vs 0.005251; the registry primary-result section says 0.005893 vs 0.005466. Which run produced which? Are they different runs (e.g., earlier confirm vs top_tier_confirm_v3)? Which is canonical? Check the canonical run dir's own summary files.
3. Pseudo-replication: confirm whether n_pairs=60 treats 5 seeds x 12 episodes as 60 independent pairs. Is there any seed-clustered or hierarchical bootstrap anywhere in the repo (grep scripts/ for 'cluster', 'hierarchical', 'seed_boot')? If the CI is computed over 60 pooled pairs, estimate how fragile the conclusion is: from effect_sizes.csv or per-seed data, can you compute a per-seed (n=5) paired summary — mean delta per seed, min/max — to see if all 5 seeds are individually positive? If per-episode data exists in the run dir, actually compute a seed-level paired t/bootstrap with pandas (use the venv at ${ROOT}/.venv if present, else python3).
4. Comparator-selection bias / winner's curse: the best dense static is SELECTED by Excel ReT on the same CRN episodes used for the paired test. Check whether selection and testing use the same episodes. How many static policies are in the dense grid (count rows in pareto_points.csv or the dense frontier file)? Does anything correct for selecting the max of N statics before testing (e.g., split selection/eval seeds)? If not, quantify the risk: what is the delta between PPO and the 2nd/3rd/5th-best statics — does the win survive against the whole top-10?
5. Metric sanity: Excel ReT ~0.0059 scale — what is this metric's range/meaning? Check for the known quirk 'ReT > 1 in 38 rows, no clamp' (grep docs for it). Is CVaR reported as CVaR of what distribution (per-episode ReT?)?
6. Cost index: verify PPO 0.682 vs static 0.667, CI crossing zero, and exactly which cost definition.
Return concrete numbers with file paths; if you run python, include the code's key outputs.`, { label: 'audit:stats', phase: 'Evidence audit', schema: AUDIT_SCHEMA }),

  // A3: literature
  () => agent(`${COMMON_CONTEXT}

You are the LITERATURE AUDITOR. The user says "in the repo is an investigation with a lot of papers" — find it, inventory it, and gap-check it.

Tasks:
1. Read: ${ROOT}/docs/SCRES_BIBLIOGRAPHY_2026-06-28.md, ${ROOT}/docs/LITERATURE_POSITIONING_MATRIX_2026-03-30.md, ${ROOT}/docs/LITERATURE_RESEARCH_REQUEST_2026-07-01.md, ${ROOT}/docs/for_team/literature_links.md, ${ROOT}/docs/chatgpt_pro_review_package_2026-07-01/LITERATURE_SCOUTING_BRIEF.md, and ${ROOT}/docs/manuscript_current/submission/elsevier/references.bib. Inventory: how many entries, what streams are covered, what is annotated vs just listed.
2. The anchor paper Garrido/Ponguta/Adarme 2024 (LNCS 15168 pp.80-94, DOI 10.1007/978-3-031-71993-6_6) cites among others: Garrido 2017 Warwick thesis; Garrido/Ponguta/Garcia-Reyes IJPR 2024 (zero-inventory/constant-workforce, DOI 10.1080/00207543.2024.2425771); von Rueden et al. 2020 hybrid ML+simulation; Zhang et al. IJPR 2024 coupling simulation+ML; Greasley & Edwards 2021; Chan et al. 2022 synthetic DES data; Moosavi & Hosseini 2021; Ivanov 2019 disruption tails; Fattahi et al. 2020; Pires Ribeiro & Barbosa-Povoa 2018; Saisridhar et al. 2024 Triple-R; Bruckler et al. 2024; Taghizadeh et al. 2021; Levitt & March 1988 organizational learning; Eryarsoy et al. 2022. Check which of these are in references.bib / the repo bibliography. These are near-mandatory cites since the paper positions itself as the empirical answer to Garrido 2024.
3. Gap-check against the RL-for-SCM canon a Q1 reviewer expects: Gijsbrechts et al. 2022 M&SOM (DOI 10.1287/msom.2021.1064); Oroojlooyjadid et al. Beer Game DRL; Boute et al. 2022 EJOR "Deep RL for inventory control"; Harsha et al. M&SOM deep policy iteration; Geevers et al. 2024 CEJOR PPO multi-echelon; Stranieri et al. pharma DRL; Madeka et al. 2022 Deep Inventory Management; Schulman 2017 PPO; Haarnoja SAC; Fujimoto TD3; Tamar/Chow CVaR RL; Wieland & Durach 2021 JBL; Ponomarov & Holcomb 2009; Hosseini/Ivanov/Dolgui 2019 review; Ivanov & Dolgui digital twin / viability. Which are present, which missing?
4. VERIFY-OR-FLAG dubious citations that appeared in earlier external reviews (they may be hallucinated). Use WebSearch (load via ToolSearch first) to verify each exists with matching title/authors: (a) 'Ghasemloo et al. Accelerating RL Training Using Simulation Surrogate Models' arXiv 2605.27556; (b) 'ReflectiChain: Epistemic Grounding in LLM-Driven World Models for Supply Chain Resilience' arXiv 2606.10359; (c) 'Pan et al. A Survey of Continual Reinforcement Learning' arXiv 2506.21872; (d) 'Maggiar et al. Structure-Informed DRL for Inventory Management' arXiv 2507.22040; (e) 'Temizoz et al. Zero-shot Generalization in Inventory Management' arXiv 2411.00515; (f) 'Kotecha & del Rio Chanona GNN+MARL inventory' arXiv 2410.18631; (g) 'Stranieri et al. pharma inventory RL' arXiv 2501.10895; (h) 'Che, Dong, Namkoong Differentiable DES' arXiv 2409.03740; (i) 'MORSE multi-objective RL supply chain by Kotecha/del Rio Chanona 2025'; (j) Garrido IJPR DOI 10.1080/00207543.2024.2425771. Mark each VERIFIED / NOT FOUND / MISMATCH.
5. Produce: (a) must-cite list actually missing from references.bib, (b) citations in references.bib that look weak/wrong (broken DOIs, preprints where published versions exist), (c) the 3-5 closest competitor papers that could scoop or undercut novelty, with one line each on differentiation.`, { label: 'audit:literature', phase: 'Evidence audit', schema: AUDIT_SCHEMA }),

  // A4: manuscript
  () => agent(`${COMMON_CONTEXT}

You are the MANUSCRIPT AUDITOR. Read the actual draft and scrub it against the claims registry.

Tasks:
1. Read ALL of: ${ROOT}/docs/manuscript_current/submission/elsevier/main.tex and sections/01_introduction.tex, 02_related_work.tex, 03_methodology.tex, 04_results.tex, 05_discussion.tex, 06_conclusion.tex.
2. Retired-claim scan: find every occurrence (file + approximate line) of: 'thesis-faithful' applied to Track B; '7D'; 'perfect fill' / 'fill = 1.000' as headline; 'strictly Pareto-dominat'; 'anticipat*' (anticipates/anticipatory) applied to the policy; H4 / L_{t-1} / retained-learning stated as demonstrated; 'resource-efficient' / cost-win claims against the best dense static; 'impossible' applied to Track A. Also 'universal', 'proves', 'first ever' style overclaims.
3. Which numbers does the draft use for the primary result — 0.005893/0.005466 or 0.005666/0.005251 or something else? Do table values match ${ROOT}/docs/track_b_q1_stats_2026-07-01/effect_sizes.csv?
4. Structure check: does the draft have (a) formal MDP/POMDP definition, (b) CRN protocol description, (c) dense static frontier definition (grid size), (d) ablation section, (e) generalization section, (f) limitations paragraph matching registry constraints, (g) mechanism section with lead-lag or properly hedged language?
5. Journal fit: which journal template/format is it (Elsevier — which journal named in main.tex?), approximate length, figure/table inventory. Does the framing match 'frontier-dependent adaptive recovery control'?
6. What hypotheses does the draft state (H1..Hn) and do they match what evidence exists per the registry?
Return findings with file:line evidence.`, { label: 'audit:manuscript', phase: 'Evidence audit', schema: AUDIT_SCHEMA }),

  // A5: Track A / fidelity
  () => agent(`${COMMON_CONTEXT}

You are the FIDELITY & TRACK-A AUDITOR. Establish exactly what Track A boundary claim is defensible and how solid the fidelity gate is — this is the foundation of the paper's Track A vs Track B contrast.

Tasks:
1. Read ${ROOT}/docs/FIDELITY_GAP_BUFFER_SATURATION_2026-06-28.md fully, then the gate evidence: ${ROOT}/outputs/benchmarks/garrido_static_fidelity_stress/paired_h2_h3_full_cf1_30_thesis_1rep_2026_06_28/FIDELITY_GATE_ANALYSIS.md and policy_family_summary.csv. Summarize the gate design (episodes, configs, frequencies) and results (H2: 10/10,10/10,9/10; H3: 10/10,10/10,7/10 — verify). How strong is this as a validation section in a Q1 paper? What would a reviewer still attack (e.g., only lever-vs-none, not per-level response; horizon h104 vs thesis 20 years; R3 weakness)?
2. Read ${ROOT}/THESIS_FIDELITY_AUDIT.md (repo root) — what deviations from the 2017 thesis remain documented?
3. Read ${ROOT}/docs/SAME_VARIABLES_NO_FRONTIER_2026-06-28.md, ${ROOT}/docs/GARRIDO_TRACK_A_FRONTIER_FREEZE_2026-06-26.md, ${ROOT}/docs/TRACK_A_REPAIR_LOCAL_ANALYSIS_2026-06-30.md, ${ROOT}/docs/TRACK_A_HEADROOM_SEARCH_2026-06-29.md, ${ROOT}/docs/S3_NONMONOTONICITY_AUDIT_2026-06-28.md. What exactly was tried in Track A (action families, rewards, algorithms) and what is the precise defensible statement of the Track A null?
4. Read ${ROOT}/docs/WIN_CONFIRMED_2026-06-29.md and ${ROOT}/docs/PROMISING_LANES_REGISTRY.md — what is the lineage of the Track B win and are there retired lanes a reviewer could use to allege fishing (count how many lanes/rewards/observations were tried before the win)? Be precise: the fishing defense needs the full trial count and the pre-registration/gate structure.
5. Excel ReT metric: read ${ROOT}/docs/RET_GARRIDO2024_AUDIT_2026-06-18.md (and RET_GARRIDO2024_IMPLEMENTATION.md if useful) — what is the formula, the validation against Garrido's workbook (47,546 rows / 0 mismatches?), and known quirks (ReT>1 rows, non-monotonicity, no clamp)?
Return the precise defensible wording for: (a) DES fidelity, (b) Track A null, (c) Excel ReT validity, plus the list of residual attack surfaces.`, { label: 'audit:fidelity', phase: 'Evidence audit', schema: AUDIT_SCHEMA }),
])

const [expAudit, statsAudit, litAudit, msAudit, fidAudit] = audits
const auditDigest = JSON.stringify({
  experiments: expAudit, stats: statsAudit, literature: litAudit, manuscript: msAudit, fidelity: fidAudit,
}, null, 1)

log('Evidence audit complete; launching adversarial reviewer panel')

// ---------- Phase 2: adversarial reviewers (need ALL audit results -> barrier was genuine) ----------
phase('Reviewer panel')

const REVIEWER_COMMON = `${COMMON_CONTEXT}

You are simulating a HOSTILE but competent Reviewer #2 for a Q1 journal (IJPR / EJOR / Omega / M&SOM tier). The submission's framing: "frontier-dependent adaptive recovery control — RL adds SCRES value only when the action space reaches the binding downstream bottleneck; Track A (Garrido's original variables) is a boundary case, Track B (downstream dispatch extension) shows a CRN-paired PPO win over a dense static frontier."

Below is a machine-generated evidence audit of the actual repo state (5 independent auditors). Ground every attack in it — or in files you read yourself. Do NOT recycle generic RL-paper complaints unless they actually bite here; find the attacks that would genuinely kill THIS paper. You may read any file in ${ROOT} to sharpen an attack.

EVIDENCE AUDIT:
${'${auditDigest}'}
`

const reviews = await parallel([
  () => agent(REVIEWER_COMMON.replace('${auditDigest}', auditDigest) + `
Your lens: STATISTICS & EXPERIMENTAL METHODOLOGY. Attack: pseudo-replication (n_pairs=60 from 5 seeds), comparator selection on the same data used for testing (winner's curse over the dense static grid — max of N statics then paired test), CRN validity, effect-size plausibility (paired d=2.87 — is that credible or a symptom of pooled clustering?), reward/observation fishing across the historical lane count, multiple-comparison exposure across metrics panel, the two inconsistent headline number sets, power of 5 seeds, selection gates pre-registered or post-hoc. For each attack give: exact wording a reviewer would use, severity, repo evidence, and the cheapest fix that fully neutralizes it. Also state which attacks the CURRENT evidence already survives (be fair — false alarms waste the author's week).`, { label: 'reviewer:stats', phase: 'Reviewer panel', schema: REVIEW_SCHEMA, effort: 'high' }),

  () => agent(REVIEWER_COMMON.replace('${auditDigest}', auditDigest) + `
Your lens: OM / SCRES DOMAIN (IJPR/EJOR editorial culture). Attack: contribution positioning (is 'frontier-dependent learning' a real theoretical contribution or a benchmark report?), single topology/case, Track B not thesis-faithful, Excel ReT as primary metric (obscure, workbook-specific — why should the field adopt it?), practical relevance of Op10/Op12 dispatch multipliers to real SC managers, SCRES theory linkage (Wieland/Durach adaptation vs engineering resilience; does the paper actually engage resilience THEORY or just metrics?), what Garrido 2024 being an LNCS conference paper implies for anchoring a Q1 submission on it, generalizability language, and whether the Track A null is interesting to this audience or reads as 'we failed then moved the goalposts'. Also: which journal would actually want this paper and what would each demand? For each attack: wording, severity, evidence, cheapest fix.`, { label: 'reviewer:domain', phase: 'Reviewer panel', schema: REVIEW_SCHEMA, effort: 'high' }),

  () => agent(REVIEWER_COMMON.replace('${auditDigest}', auditDigest) + `
Your lens: RL / ML METHODS. Attack: PPO-only with no off-policy or bandit/heuristic baseline, 'more knobs' confound (8D vs Track A action space — ablation status per the audit), observation design v7/v8/v9 (what's in the 52 dims — does the agent see privileged info the statics can't use? Statics are open-loop constants, so ANY feedback controller has an unfair information advantage — is the right comparator class dynamic rules, not constants?), 60k timesteps (tiny — undertrained? or conversely, is the env so easy a PID/threshold rule solves it?), reward=control_v1 vs evaluation=Excel ReT mismatch (is the agent even optimizing the reported metric — and is that a strength or incoherence?), generalization/transfer status, seed count, and whether 'frontier-dependent' is just 'controllability matters', which control theory has known for 70 years. For each attack: wording, severity, evidence, cheapest fix. Also list which standard RL-reviewer attacks do NOT bite here and why (so the authors don't over-invest).`, { label: 'reviewer:rl', phase: 'Reviewer panel', schema: REVIEW_SCHEMA, effort: 'high' }),
])

// ---------- Phase 3: editor synthesis (needs all reviews -> genuine barrier) ----------
phase('Editor synthesis')

const [statsReview, domainReview, rlReview] = reviews
const editorVerdict = await agent(`${COMMON_CONTEXT}

You are the HANDLING EDITOR of a Q1 OM journal receiving this submission plus three referee reports. Your job: triage, not repetition.

EVIDENCE AUDIT:
${auditDigest}

REFEREE 1 (stats/methods): ${JSON.stringify(statsReview, null, 1)}
REFEREE 2 (OM/SCRES domain): ${JSON.stringify(domainReview, null, 1)}
REFEREE 3 (RL/ML): ${JSON.stringify(rlReview, null, 1)}

Deliver:
1. TRIAGE: merge the three reports' kill points into a single deduplicated list ranked by real-world lethality (what actually causes rejection at IJPR/EJOR/Omega, not what sounds scary). Mark each: BLOCKING (must fix before submission) / MAJOR (fix or pre-empt in text) / COSMETIC.
2. JOURNAL TARGETING: rank 4-6 concrete target journals (e.g., IJPR, EJOR, Omega, IJPE, Computers & IE, Annals of OR, Simulation Modelling Practice & Theory, M&SOM) by fit x acceptance probability for THIS paper with the blocking fixes done. Note each journal's specific expectations (case grounding, methodological novelty, length limits) and where the Garrido-anchor helps or hurts. State your single recommended target and a realistic fallback ladder.
3. MINIMAL SUFFICIENT EXPERIMENT SET: the smallest set of additional experiments/analyses that makes the paper defensible at the recommended target — with the specific decision each experiment de-risks. Distinguish 'needed for submission' from 'hold for response letter'.
4. FRAMING RULING: given all evidence, the exact one-paragraph contribution statement you would advise the authors to use, and the 3 sentences they must NOT write.
Return as structured markdown text in the summary field.`, { label: 'editor:synthesis', phase: 'Editor synthesis', schema: AUDIT_SCHEMA, effort: 'high' })

return {
  audits: { experiments: expAudit, stats: statsAudit, literature: litAudit, manuscript: msAudit, fidelity: fidAudit },
  reviews: { stats: statsReview, domain: domainReview, rl: rlReview },
  editor: editorVerdict,
}