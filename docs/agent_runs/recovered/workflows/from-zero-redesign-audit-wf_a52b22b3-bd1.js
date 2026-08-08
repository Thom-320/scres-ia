export const meta = {
  name: 'from-zero-redesign-audit',
  description: 'Parallel thesis deep-read + repo capability/failure audit for the from-zero contract redesign',
  phases: [
    { title: 'Thesis', detail: '4 readers over Garrido 2017 page ranges' },
    { title: 'Repo', detail: '4 auditors: env capabilities, failures, ideas, headroom evidence' },
  ],
}

const PDF = '/Users/thom/Library/CloudStorage/GoogleDrive-chisicathomas@gmail.com/My Drive/Archive/Misc_Unsorted/Unsorted/WRAP_Theses_Garrido_Rios_2017.pdf'
const REPO = '/Users/thom/Projects/research/scres-ia'

const FACTS = {
  type: 'object',
  properties: {
    facts: { type: 'array', items: { type: 'object', properties: {
      topic: { type: 'string' },
      fact: { type: 'string', description: 'precise fact with numbers/units and thesis table/section or file:line citation' },
      relevance: { type: 'string', description: 'why it matters for designing an RL contract/env that could beat a static policy' },
    }, required: ['topic', 'fact', 'relevance'] } },
    surprises: { type: 'array', items: { type: 'string' } },
  },
  required: ['facts', 'surprises'],
}

const thesisTasks = [
  { label: 'thesis:ret-metric', pages: '60-78', q: `Extract from Ch5 (Operationalization of Resilience): the exact disruption taxonomy (partial vs total), the DP_j/AP_j/RP_j/FT definitions and how ReT is derived step by step (weights, sub-indicators, Figure 5.6-5.11), what makes ReT high vs low mechanistically, and any temporal structure in the metric (what does the metric reward reacting QUICKLY to?). Also note anything about time granularity (hours/days) of the underlying quantities.` },
  { label: 'thesis:mfsc-risks', pages: '79-96', q: `Extract from Ch6 §6.2-6.4: the 13 MFSC operations (names, roles, which have inventory/capacity/transport), raw materials, assembly-line theoretical vs effective capacity (Tables 6.2/6.3), demand structure for combat rations (Table 6.4, DP/DA cycles, order cycle period), and the FULL risk roster: every R1r operational risk, R2r natural/intentional, R3 black swan from Tables 6.6a/6.6b/6.7a/6.7b/6.8a/6.8b — for each: which operation(s) it hits, probability distribution + parameters, duration/impact model, and whether its effect is a capacity loss, transport/LOC interruption, supply delay, or demand surge.` },
  { label: 'thesis:experiment-design', pages: '96-112', q: `Extract from Ch6 §6.5-6.9: ALL simulation assumptions (6.5.1-6.5.8, especially 6.5.2 proactive strategy, 6.5.4 backorders/scheduling rule, 6.5.5 LOC/warehouses, 6.5.6 suppliers/orders, 6.5.7 maintenance/strikes, 6.5.8 steady state), verification/validation criteria (Tables 6.9/6.10), and the experiment design: scenario I/II/III definitions, risk frequency levels (Table 6.11-6.15), buffer levels I_t,S at Op3/Op5/Op9 (Table 6.16-6.19 exact values), short-term capacity S levels (Table 6.20-6.23), run length/warm-up (6.8.1-6.8.2), and the 90 configurations structure (Table 6.25).` },
  { label: 'thesis:results-future', pages: '123-149', q: `Extract from Ch7-8: for each hypothesis H1a-H3c, WHICH risk category × buffer/capacity combinations showed significant moderation and which did not (Tables 7.4-7.9 directionality); the association rules found (Table 7.1-7.3, which risks drive ReT low); empirical findings 8.2; limitations 8.5 (especially 8.5.2 cost non-inclusion — quote what Garrido says cost would change); and future work 8.6.1-8.6.3 verbatim gist (synergy question, optimum SCRes search). These tell us where GARRIDO HIMSELF found buffers/capacity matter, i.e. where decision variables have authority.` },
]

const repoTasks = [
  { label: 'repo:env-capabilities', q: `In ${REPO}, inventory EVERY environment capability and flag: read supply_chain/config.py, supply_chain/env_experimental_shifts.py (init kwargs, ACTION_CONTRACT_OPTIONS, reward modes list, observation versions, risk_level options, surge_inertia, step_size_hours, raw_material_flow_mode, any cost/holding machinery), supply_chain/external_env_interface.py (make_track_b_env signature/kwargs incl. replenishment), supply_chain/track_bp_env.py (11D contract, fixed-buffer env). For EACH capability state: what it does, its options/ranges, and whether it was ever used in a canonical experiment (grep scripts/ and docs/ for usage) or is DEFINED-BUT-NEVER-USED. I especially need: all risk_level presets and what they change; risk-frequency/impact override kwargs; cost/lambda knobs; surge_inertia semantics; obs v1..v10 contents (dimensions and field groups); reward modes incl. control_v1_pbrs. Return precise file:line cites.` },
  { label: 'repo:failure-catalog', q: `In ${REPO}, build the complete failure/attempt catalog from docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md, docs/PROMISING_LANES_REGISTRY.md, and the verdict docs (docs/TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md, docs/TRACK_B_CLEAN_REPLICATION_PROTOCOL_2026-07-10.md RESULT section, docs/TRACK_B_CONTRACT_FACTORIAL_VERDICT_2026-07-10.md, docs/TRACK_BP_*.md, docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md, docs/E1_GO_NO_GO_VERDICT_2026-07-02.md). For every attempted approach (contract, reward, architecture, regime, prevention gate): what was tried, at what scale, the numeric outcome, and the ROOT CAUSE of failure/success. End with: which root causes are ENVIRONMENTAL (the world makes constants optimal) vs LEARNER (PPO could not find it) vs PROTOCOL (comparator/tape artifacts).` },
  { label: 'repo:ideas-inventory', q: `In ${REPO}, read docs/PROMISING_LANES_REGISTRY.md and docs/TRACK_A_EXTENSION_IDEA_BANK_2026-06-17.md fully, plus any other idea/proposal docs you find (grep docs/ for 'idea', 'propuesta', 'Contrato 1', 'Contrato 2', 'epoch', '24h'). List EVERY idea not yet executed or only partially executed, with: what it proposes, status (never-run / partial / superseded), and what running it would require. Include the LOC dispatch lane (docs mentioning op9/op10/op12 dispatch as decision variable under LOC caps) and the per-op buffer contract Box([op3,op5,op9,shift]) if present.` },
  { label: 'repo:headroom-evidence', q: `In ${REPO}, quantify WHERE adaptive or preventive headroom has ever been MEASURED (not narrated): read docs/E1_GO_NO_GO_VERDICT_2026-07-02.md (regime-conditioned lookup vs statics numbers), docs/TRACK_BP_GATES_0_1_VERDICT_2026-07-09.md and docs/TRACK_BP_GATE2_SCREEN_VERDICT_2026-07-09.md (R21 starvation oracle numbers, 11D-8D confirm, fixed-posture equivalence), the prevention headroom sweep outputs (grep outputs/experiments for track_bp_breadth and read summary.json files for the R21 freq x impact grid oracle values), and docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md Case C numbers. Return a table: environment/regime -> measured oracle/dynamic headroom -> what captured it (constant posture vs timing) -> CI when available. Also: what the holding-cost sensitivity showed (lambda crossover) with file cites.` },
]

phase('Thesis')
const all = await parallel([
  ...thesisTasks.map(t => () => agent(
    `You are auditing the Garrido-Rios 2017 thesis PDF at: ${PDF}\n` +
    `Read pages ${t.pages} with the Read tool (pages parameter, max 20 pages per call; the printed page number is close to the PDF page number but verify and adjust by reading a probe page first).\n\n${t.q}\n\n` +
    `Be exhaustive and numeric. Cite thesis tables/sections for every fact.`,
    { label: t.label, phase: 'Thesis', schema: FACTS },
  )),
  ...repoTasks.map(t => () => agent(
    `${t.q}\n\nWork read-only. Be exhaustive, numeric, and cite file:line or doc names for every fact.`,
    { label: t.label, phase: 'Repo', schema: FACTS },
  )),
])

const [ret, risks, design, results, envcap, failures, ideas, headroom] = all
return {
  thesis: { ret, risks, design, results },
  repo: { envcap, failures, ideas, headroom },
}