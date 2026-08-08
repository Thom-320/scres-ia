export const meta = {
  name: 'paper2-candidate-theory-screen',
  description: 'Blind independent proposal + adversarial audit of >=12 materially-distinct DES adaptive-headroom mechanism families for SCRES-IA Paper 2',
  phases: [
    { title: 'Generate', detail: '12 blind proposers, one distinct mechanism region each' },
    { title: 'Audit', detail: '3-lens adversarial panel per family (DES/OR, RL/identification, domain/resilience)' },
    { title: 'Synthesize', detail: 'rank, dedup, classify ACTIVE/BLOCKED_PENDING_PI/FALSIFIED, pick strongest candidate + Garrido questions' }
  ]
}

const REPO = '/Users/thom/Projects/research/scres-ia'
const GROUND = `You are one independent research stream in an exhaustive search for a NEW discrete-event-simulation (DES) decision contract that could support "Paper 2" in the SCRES-IA military food supply chain (MFSC) project. Before answering you MUST read these two files for ground truth (they contain thesis-native physics and the list of ALREADY-CLOSED lanes you may NOT reopen):
- ${REPO}/research/paper2_exhaustive_search/source_reconstruction.md  (Op1-Op13 semantics, risks R11/R14/R21/R22/R23/R24/R3, binding thesis assumptions, and the source-derived candidate boundary table)
- ${REPO}/research/paper2_exhaustive_search/phase0_failure_taxonomy.json  (13 closed decision families D1/DRA1/DRA2b/E/F/G/H/I/J/K/K2/K3/bottleneck, each with its exact failure arrow)
Optionally skim ${REPO}/docs/REPOSITORY_SOURCE_OF_TRUTH.md.

HARD RULES:
- Canonical endpoint is order-level ReT (ret_excel_visible_v1). Practical bar: LCB95(delta ReT) >= 0.01, with co-directional service improvement, equal resources, no increase in lost orders, no worst-CSSU starvation.
- FORBIDDEN: reopening any closed lane by only changing architecture/reward/horizon/metric; increasing disruption magnitude until a learner wins; creating inventory/capacity by action; resources without cost/conservation; future information unavailable to operators; two-week shelf life for the ~3-year non-perishable ration; selecting a metric after seeing rankings; removing hard orders from the denominator; supplier decisions in Op1-Op2.
- The target mechanism must have ALL of: a scarce shared resource; mutually incompatible actions; persistent/intertemporal consequences; delayed activation or commitment; frequent STATE-DEPENDENT action-ranking reversals; informative-but-imperfect observations available BEFORE the action with lead >= physical lead time; canonical-ReT consequences; and NO tape-independent open-loop schedule that matches feedback everywhere.
- A family is only materially new (not a reopening) if it introduces a genuinely new physical mechanism, information source, decision right, or causal estimand relative to the closed lanes.
- Be adversarial against your own idea. A large perfect-information gap (H_PI) is NOT evidence of deployable value. If the strongest constant/periodic-calendar policy plausibly dominates, say so.`

const REGIONS = [
  { id: 'R01_condition_based_repair_crew', seed: 'A FINITE shared repair crew dispatched at breakdown/degradation EVENTS across disaggregated Op5/Op6/Op7 stations (condition-based maintenance with observable wear and imperfect repair). Contrast explicitly with closed lane J (which was EQUAL-BUDGET weekly maintenance CALENDAR allocation); argue whether event-driven finite-crew dispatch is materially new or collapses to the same dominant calendar.' },
  { id: 'R02_dynamic_wip_buffer_allocation', seed: 'Dynamic reallocation of a FINITE TOTAL WIP/buffer space across Op3/Op5/Op7/Op9 under blocking-and-starvation coupling. Contrast with closed Track A strategic buffer SIZING; is dynamic reallocation under blocking materially new?' },
  { id: 'R03_finite_fleet_persistent_location', seed: 'A finite transport fleet with PERSISTENT vehicle location, route-specific R22 line-of-communication delays, alternate routes, and return cycles; observations include route status and vehicle ETA. Contrast with closed DRA2/DRA2b finite convoy hold/dispatch and Program E; is persistent-location + alternate-route choice materially new or already dominated?' },
  { id: 'R04_inspection_qc_vs_throughput', seed: 'State-dependent inspection/QC effort vs throughput at Op7 under R14 rework/non-conforming risk: more inspection lowers escapes but consumes capacity and delays release. Never tried in the closed set. Assess ranking reversal and the dominant-constant counterexample.' },
  { id: 'R05_order_release_backlog_timing', seed: 'Order-RELEASE timing and backlog re-sequencing under the cap-60 pending list beyond SPT/FIFO/age (closed D1 covered SPT/FIFO/age service rules but not release timing). Assess whether release-timing authority is materially new, and whether mission/due-date classes are required (if so, mark the exact Garrido fact needed).' },
  { id: 'R06_multiechelon_prepositioning', seed: 'Multi-echelon prepositioning between the supply battalion (Op9) and the two CSSUs (Op11) with REAL relocation lead time and finite relocatable stock, anticipating R23 CSSU outage/reactivation. Contrast with closed DRA1 (reallocatable CSSU capacity, negligible oracle headroom); is physical prepositioning with lead time materially new?' },
  { id: 'R07_product_mix_kit_completeness', seed: 'Multi-ration product mix / kit-completeness control (the thesis compresses a real 21-product assortment to one homogeneous ration). Requires product shares, bill-of-materials, substitution, setup facts. Assess the mechanism AND state precisely which Garrido domain facts are required (BLOCKED_PENDING_PI is a legitimate honest status).' },
  { id: 'R08_information_acquisition_censored', seed: 'Active information-acquisition / demand sensing with cost under CENSORED demand (current DES records orders; hidden true demand and active sensing are introduced). Assess whether this creates irreversible multi-stage authority, and which system-of-record Garrido facts are required.' },
  { id: 'R09_admission_priority_abandonment', seed: 'Admission/priority control with exogenous mission ABANDONMENT/expiry (current orders backorder rather than expire). Keep ALL demand in the denominator (no shed-to-win). Assess ranking reversal and the mission-deadline / rejection-authority Garrido facts required.' },
  { id: 'R10_integrated_prod_maint_routing', seed: 'JOINT control of a live production constraint AND a live downstream transport/maintenance constraint under ONE shared resource budget, where the interaction produces ranking reversals that neither subsystem produces alone. Contrast with closed post-K3 bottleneck migration (one equal-cost team M/R/T, which lost to constant M); argue what interaction structure, if any, is materially new.' },
  { id: 'R11_regime_forecast_anticipation', seed: 'Demand-regime anticipation using an operationally-named leading indicator (thesis Ch8 flags demand-pattern change and advance cross-functional forecasting as a real future direction). The signal must have a named source, lead time, sensitivity/specificity, false pos/neg, and cost; must beat block-shuffled/delayed/wrong-location placebos. Assess and state the signal-quality Garrido facts required.' },
  { id: 'R12_recovery_resource_sequencing', seed: 'Finite RECOVERY-resource sequencing across a persistent multi-operation outage: R21 natural disaster (~120h recovery on Op3/Op5-7/Op9) or R3 black-swan (672h multi-op), where a scarce recovery crew must be sequenced across simultaneously-down operations with persistent consequences. Assess whether recovery-sequencing under concurrent outage yields state-dependent reversals or collapses to a fixed priority order.' }
]

const FAMILY_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['family_id','one_line_hypothesis','binding_scarce_resource','decision_epoch','ranking_reversal_mechanism','constant_policy_counterexample','survives_counterexample','likely_failure_arrow','materially_new_vs_closed_lanes','garrido_fact_required','self_status'],
  properties: {
    family_id: { type: 'string' },
    one_line_hypothesis: { type: 'string' },
    physical_actuator: { type: 'string' },
    controlled_ops: { type: 'array', items: { type: 'string' } },
    binding_scarce_resource: { type: 'string' },
    decision_owner: { type: 'string' },
    decision_epoch: { type: 'string' },
    action_latency_dwell: { type: 'string' },
    observable_state: { type: 'array', items: { type: 'string' } },
    hidden_state: { type: 'array', items: { type: 'string' } },
    information_lead_vs_leadtime: { type: 'string' },
    disruption_families: { type: 'array', items: { type: 'string' } },
    ranking_reversal_mechanism: { type: 'string' },
    constant_policy_counterexample: { type: 'string' },
    survives_counterexample: { type: 'boolean' },
    likely_H_PI_sign_and_scale: { type: 'string' },
    likely_H_obs_sign_and_scale: { type: 'string' },
    likely_failure_arrow: { type: 'string', enum: ['physical_magnitude','perfect_info_only','observable_conversion','dominant_calendar','resource_purchase','metric_reversal','fairness_starvation','thesis_incompatible','blocked_pending_pi','none_survives_screen'] },
    materially_new_vs_closed_lanes: { type: 'string' },
    garrido_fact_required: { type: 'string' },
    self_status: { type: 'string', enum: ['ACTIVE','BLOCKED_PENDING_PI','LIKELY_FALSIFIED'] }
  }
}

const VERDICT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['lens','kill_attempt','verdict','rationale','confidence'],
  properties: {
    lens: { type: 'string' },
    kill_attempt: { type: 'string' },
    verdict: { type: 'string', enum: ['SURVIVES','FALSIFIED','BLOCKED_PENDING_PI'] },
    rationale: { type: 'string' },
    confidence: { type: 'number' }
  }
}

const LENSES = [
  { key: 'des_or', role: 'a discrete-event-simulation / operations-research reviewer. Attack the family for: a dominant robust constant or periodic calendar that matches feedback everywhere; resource-purchase (winning by spending more of the scarce resource); redistribution without throughput value; inventory/capacity creation; CRN/event-log identity failure; open-loop schedule confounding (the K3 lesson). Give the exact constant/calendar counterexample. Default to FALSIFIED under uncertainty.' },
  { key: 'rl_ident', role: 'a reinforcement-learning / causal-identification reviewer. Attack for: perfect-information-only headroom (H_PI>0 but H_obs~0), insufficient predictive information, signal lead shorter than physical lead time, latent-label leakage, action-label accuracy masquerading as policy value, learner collapse to open-loop. Judge whether a deployable non-anticipative observable policy could plausibly convert the headroom. Default to FALSIFIED/BLOCKED under uncertainty.' },
  { key: 'domain', role: 'a supply-chain-resilience / military-logistics / resilience-measurement reviewer. Attack for: thesis-incompatible physics (e.g. perishability, supplier decisions, un-validated demand/threat process); un-validated invented parameters; worst-CSSU fairness/starvation; metric-induced ranking reversal; whether the required domain facts exist in the thesis or need Garrido face-validation. If the mechanism needs facts not in the thesis, return BLOCKED_PENDING_PI and name the exact fact.' }
]

phase('Generate')
const audited = await pipeline(
  REGIONS,
  (region) => agent(
    `${GROUND}\n\nYOUR ASSIGNED MECHANISM REGION: ${region.seed}\n\nPropose exactly ONE materially-distinct mechanism family in this region. Fill every field. Be brutally honest: if the strongest constant/periodic policy dominates, set survives_counterexample=false and self_status=LIKELY_FALSIFIED. If the mechanism is defensible but needs domain facts the thesis does not supply, set self_status=BLOCKED_PENDING_PI and name the exact fact in garrido_fact_required. Only use self_status=ACTIVE if a state-dependent ranking reversal plausibly survives the best constant/calendar counterexample AND the required physics/signals are already thesis- or literature-defensible without new Garrido facts.`,
    { schema: FAMILY_SCHEMA, phase: 'Generate', label: `propose:${region.id}` }
  ),
  (fam, region) => {
    if (!fam) return null
    return parallel(LENSES.map(L => () =>
      agent(
        `${GROUND}\n\nYou are ${L.role}\n\nA colleague proposed this candidate family (JSON):\n${JSON.stringify(fam)}\n\nTry hard to KILL it or to prove it needs domain facts. Give your single strongest kill_attempt (a concrete counterexample, leakage, resource-purchase, or missing fact), then your verdict. SURVIVES only if you genuinely cannot kill it and it needs no new Garrido facts.`,
        { schema: VERDICT_SCHEMA, phase: 'Audit', label: `audit:${region.id}:${L.key}` }
      )
    )).then(verdicts => {
      const v = verdicts.filter(Boolean)
      const nFals = v.filter(x => x.verdict === 'FALSIFIED').length
      const nBlock = v.filter(x => x.verdict === 'BLOCKED_PENDING_PI').length
      const nSurv = v.filter(x => x.verdict === 'SURVIVES').length
      let panel = 'FALSIFIED'
      if (nSurv >= 2) panel = 'SURVIVES'
      else if (nFals >= 2) panel = 'FALSIFIED'
      else if (nBlock >= 1 && nFals < 2) panel = 'BLOCKED_PENDING_PI'
      return { region: region.id, family: fam, verdicts: v, panel_classification: panel, votes: { SURVIVES: nSurv, FALSIFIED: nFals, BLOCKED_PENDING_PI: nBlock } }
    })
  }
)

const clean = audited.filter(Boolean)
log(`Audited ${clean.length}/${REGIONS.length} families. Panel: ` +
  clean.map(c => `${c.region}=${c.panel_classification}`).join(', '))

phase('Synthesize')
const SYNTH_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ranked_families','strongest_active_candidate','boundary_or_positive','garrido_question_set','recommended_paper2_identity','recommended_paper3_identity','rationale'],
  properties: {
    ranked_families: { type: 'array', items: { type: 'object', additionalProperties: false,
      required: ['family_id','status','strongest_failure_arrow','publishability_note'],
      properties: { family_id: {type:'string'}, status: {type:'string', enum:['ACTIVE','BLOCKED_PENDING_PI','FALSIFIED']}, strongest_failure_arrow: {type:'string'}, publishability_note: {type:'string'} } } },
    strongest_active_candidate: { type: 'string' },
    boundary_or_positive: { type: 'string', enum: ['BOUNDARY_CERTIFIED','POSITIVE_CANDIDATE_FOUND'] },
    garrido_question_set: { type: 'array', items: { type: 'string' } },
    recommended_paper2_identity: { type: 'string' },
    recommended_paper3_identity: { type: 'string' },
    rationale: { type: 'string' }
  }
}
const synth = await agent(
  `${GROUND}\n\nYou are the synthesis lead. Here are ${clean.length} audited candidate families with their adversarial panel verdicts (JSON):\n${JSON.stringify(clean.map(c => ({ region: c.region, panel: c.panel_classification, votes: c.votes, family: c.family, kill_attempts: c.verdicts.map(v => ({lens:v.lens, verdict:v.verdict, kill:v.kill_attempt})) })))}\n\nProduce the final registry. A family is ACTIVE only if the panel says SURVIVES and it needs no new Garrido facts. BLOCKED_PENDING_PI means defensible-but-gated on a specific domain fact. FALSIFIED means a constant/calendar dominates or the arrow is fatal. Then: identify the single strongest_active_candidate (or "NONE"); decide boundary_or_positive (POSITIVE_CANDIDATE_FOUND only if >=1 ACTIVE family plausibly clears the 0.01 ReT bar with an observable policy); write a crisp Garrido question set (one falsifiable question per BLOCKED family that would reopen it); and recommend the strongest HONEST Paper 2 and Paper 3 identities given that Paper 1 is already the "when does RL work / when does it fail in resilience operational control" eligibility paper (retained learning reserved as Paper-1 future work). Paper 2 must NOT duplicate Paper 1's eligibility/comparator-contribution.`,
  { schema: SYNTH_SCHEMA, phase: 'Synthesize', label: 'synthesize', effort: 'high' }
)

return { audited: clean, synthesis: synth }
