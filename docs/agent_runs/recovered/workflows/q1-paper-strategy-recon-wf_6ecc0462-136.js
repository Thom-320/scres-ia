export const meta = {
  name: 'q1-paper-strategy-recon',
  description: 'Parallel read-only recon: env prevention levers, results inventory, Ding rival paper, manuscript gaps',
  phases: [
    { title: 'Recon', detail: '4 parallel readers' },
  ],
}

phase('Recon')

const FINDINGS = {
  type: 'object',
  properties: { report: { type: 'string' } },
  required: ['report'],
}

const results = await parallel([
  () => agent(`You are exploring the repo /Users/thom/Projects/research/scres-ia (read-only). Report raw facts with file:line citations.

MISSION: map every EXISTING environment lever that could create PREVENTIVE headroom (value for acting BEFORE a risk event) with minimal code change, plus the full risk roster mechanics.

1. In supply_chain/env_experimental_shifts.py: find 'inventory_replenishment_lead_time' (~lines 400-540, 2460-2480). Explain exactly what it does mechanically (what gets delayed, units hours/weeks?), its default, and whether it's exposed through make_track_b_env / external_env_interface.py and scripts/run_track_b_smoke.py's build_env_kwargs (i.e., can a Track B training run set it TODAY without code changes?).
2. In supply_chain/config.py (or wherever RISKS are defined): list ALL 9 risks (R11,R12,R13,R14,R21,R22,R23,R24,R3x?) with: name/meaning, occurrence dist, recovery dist, affected_ops, detection, impact mechanism (what state does it damage: raw material? production rate? demand? transport?). For each, note whether the damage is mechanically MEDIABLE by these action levers: op3_q/op9_q/op3_rop/op9_rop/op5_q (upstream inventory/order-up-to), shift (assembly capacity S1-S3), op10_q/op12_q (downstream dispatch multipliers). I.e., could building buffer or capacity BEFORE the event reduce that risk's damage in principle?
3. Action contracts: what does each of the 8 dims of track_b_v1 control exactly (find the action application code)? Are there other ready action contracts (per-op buffer, continuous_its Box([frac,shift]))? Cite where.
4. Search for any phi/psi risk-modulation knobs or a 'binary resilience' sweep (grep for risk_frequency_multiplier usage, calibrate_headroom_env.py, sweep_env_b_headroom.py, run_risk_regime_headroom.py, docs/PREVENTIVE_PARETO_RESULTS_2026-06-28.md) — summarize what those earlier headroom searches found (was there ever an intermediate regime where dynamic beats static buffering?).
5. Does demand fulfillment allow holding/prepositioning inventory forward of Op10/Op12 at all? (Is there ANY state the agent can build up downstream before an event, or is fulfillment purely flow-through demand-driven?) Look at how op10/op12 multipliers act and whether orders can be completed early relative to demand arrival.

Return a dense factual report.`, { label: 'recon:env-levers', schema: FINDINGS, agentType: 'Explore', effort: 'high' }),

  () => agent(`You are exploring the repo /Users/thom/Projects/research/scres-ia (read-only). Produce a complete RESULTS INVENTORY ("what survived / what died") with exact numbers and artifact paths, from these docs (read them fully):
- docs/CLAIMS_REGISTRY_Q1_DEFENSE_2026-07-01.md (all claims C1-C25 with status)
- docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md
- docs/PREVENTION_GATE_AUTOPSY_AND_CLOSURE_2026-07-07.md
- docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md
- docs/PROMISING_LANES_REGISTRY.md (sections ⭐13-16 + 'Win bars' + open lanes/queue)
- docs/E2/E1 privileged observation + generalization docs if referenced (E3 cross-regime matrix numbers, severe-h52 boundary case)
- Track A status: is Track A closed as negative? cite the final Track A verdict docs.
- H4 / retained-learning probe status and numbers.
Structure: (a) SUPPORTED claims table w/ numbers+CIs+artifacts; (b) RETIRED/dead claims table w/ why; (c) OPEN/untested lanes explicitly listed as still-live options (e.g., per-op buffer contract, LOC dispatch decision lane, SAC/TD3 screen, upstream-risk prevention never ceiling-tested?, Real-KAN efficiency, retained-learning probe); (d) known boundary cases (severe h52) and how they're currently explained.`, { label: 'recon:results-inventory', schema: FINDINGS, agentType: 'Explore', effort: 'high' }),

  () => agent(`Read the PDF at "/Users/thom/Library/Mobile Documents/com~apple~CloudDocs/1-s2.0-S0925527326000861-main.pdf" using the Read tool with pages parameter (23 pages total; read pages 1-8, then 9-16, then 17-23). This is believed to be Ding et al., IJPE 2026, MAPPO-based supply chain resilience (topology reconfiguration). Extract precisely:
1. Full title, authors, journal, year; exact problem statement and decision variables.
2. Method (algorithm, agents, action/state spaces), and evaluation standard: what baselines? how many seeds/replications? statistical tests? any dense baseline frontier? any negative results? any causal/ablation tests? do they validate their simulator against anything?
3. Their claims (verbatim-ish) about resilience improvement, and any claims about anticipation/prevention vs reaction.
4. Weaknesses a rigorous reviewer would flag (baseline strength, confounds, generalization).
5. Overlap/differences with a paper about: single-agent PPO operational control inside a FIXED 13-op validated military DES, dense 147-cell static frontier, action-space/bottleneck ablation, privileged-observation defense, prevention-boundary result.
Return a dense factual report; quote exact numbers where available.`, { label: 'recon:ding-rival-pdf', schema: FINDINGS, agentType: 'Explore', effort: 'high' }),

  () => agent(`You are exploring the repo /Users/thom/Projects/research/scres-ia (read-only). Inventory the CURRENT manuscript state vs a strengthening checklist. Read docs/manuscript_current/submission/elsevier/main.tex and all sections/*.tex (01_introduction, 02_related_work, 03_methodology, 04_results, 05_discussion, 06_conclusion) plus the abstract.
Checklist — for each item say PRESENT (where, quote a line) or MISSING:
1. TOC (Theory of Constraints) bridge foregrounded in abstract+intro (Goldratt cited? action-space alignment as headline?)
2. Prevention-boundary result / prevention audit (any mention of the 2026-07-07 closure: splice-gate retraction, forced-prep null, clairvoyant null)?
3. Off-policy algorithm screen (SAC/TD3) or explicit limitation wording
4. Operational-significance translation paragraph (CTj p99 8113->1207 hours, what it means in ration-delivery terms)
5. Military logistics doctrine grounding for Op10/Op12 dispatch extension
6. Reproducibility statement (GitHub repo, branch/commit)
7. Severe-h52 boundary case mechanistic explanation
8. Which Track B package is the spine: adaptive_benchmark_v2 v7 (+0.000426/+8-9%) or the new no-forecast Case A/B/C package (+4.55% Case A)? Quote the headline numbers currently in the abstract/results.
9. Ruta B / efficiency-architecture result mentioned anywhere?
10. Related work: are Ding et al. 2026, Bussieweke 2025, Gijsbrechts 2022, Kogler&Maxera, Rolf 2023, Boute roadmap, Powell, Hosseini 2019, Wieland&Durach cited? List which of these are missing from references.bib.
Also report overall section structure and approximate length. Return a dense factual report.`, { label: 'recon:manuscript-gaps', schema: FINDINGS, agentType: 'Explore', effort: 'high' }),
])

return {
  envLevers: results[0]?.report ?? 'FAILED',
  resultsInventory: results[1]?.report ?? 'FAILED',
  dingRival: results[2]?.report ?? 'FAILED',
  manuscriptGaps: results[3]?.report ?? 'FAILED',
}