export const meta = {
  name: 'verify-5-audits-qr1',
  description: 'Adversarially verify the 5 external audits\' claims against the audited commit efae6514',
  phases: [{ title: 'Verify' }],
}

const REPO = '/private/tmp/claude-501/-Users-thom-Projects-research-scres-ia/79a20ac6-ef07-49db-ab45-48bab5e28d63/scratchpad/qr1-audit-wt'

const SCHEMA = {
  type: 'object',
  required: ['claim_id', 'verdict', 'evidence', 'numbers', 'notes'],
  properties: {
    claim_id: { type: 'string' },
    verdict: { type: 'string', enum: ['CONFIRMED', 'REFUTED', 'PARTIAL'] },
    evidence: { type: 'string', description: 'file:line references + short quotes proving the verdict' },
    numbers: { type: 'object', description: 'any recomputed numbers, keyed by name' },
    notes: { type: 'string' },
  },
  additionalProperties: false,
}

const COMMON = `You are verifying a claim made by external auditors about a research codebase.
Repo checkout (READ-ONLY — never write, never git-commit, work only by reading files and running
read-only python/jq in a temp dir): ${REPO}
Be adversarial in BOTH directions: confirm only what the code/artifacts actually show; refute or
mark PARTIAL if the auditors overstated. Quote exact file:line evidence.`

const TASKS = [
  { id: 'V1_splice', effort: 'low', prompt: `${COMMON}
CLAIM V1: The "common continuation" is an off-policy calendar splice, not a common policy.
Check: (a) supply_chain/q_r1_retained_learning.py — function common_continuation_calendar (~line 188)
returns arm[:2] + reset_calendar[2:]; (b) scripts/run_q_r1_d0_cold_start_reanalysis.py (~L116-139)
and scripts/run_q_r1_d1_demand_memory.py (~L206-229) compute the FULL calendar only for the reset
arm's own trajectory and splice its weeks 3-8 into every other arm. Verdict on whether the executed
estimand is "retained prefix + fixed reset-trajectory suffix" rather than "same policy replanned on
each arm's reached state".` },
  { id: 'V2_metric_population', effort: 'medium', prompt: `${COMMON}
CLAIM V2: early_ret_2w excludes unresolved/lost early orders from its mean, contradicting its own
docstring; empty visible set defaults the mean to 1.0; early_omitted_rows counts ALL-campaign
non-visible orders (so a 12-order cohort can report 36 omitted).
Check: supply_chain/q_r1_retained_learning.py (early cohort function ~L224 and its docstring);
supply_chain/ret_thesis.py ~L477-598 (visible-ledger filter: not lost AND OATj not None AND id in
emit set; mean over visible only; the n_omitted_rows computation; any np.ones/default-1.0 behavior
when no visible rows). Then confirm from results/q_r1/cold_start_replication_v1/d0_retained_context.json
that rows exist with early_generated_orders=12, early_visible_rows=12, early_omitted_rows=36.` },
  { id: 'V3_d3_recompute', effort: 'high', prompt: `${COMMON}
CLAIM V3: D3's residual is contaminated: scripts/run_q_r1_d3_residual_bound.py (~L34-50) maximizes
per-episode over ALL arms including shuffled_posterior_mpc and wrong_posterior_mpc, tie-breaking by
arm name string. Auditors computed from the raw data: across 528 selected episodes wrong won 297 and
shuffled 124; and the residual bounds are: all-candidates pooled mean +0.0251202 (LCB95 +0.0202288);
excluding placebos +0.0144192 (LCB +0.0100910); retained+oracle-parameters only +0.0107465 (LCB
+0.0073755); deployable retained+reset only +0.0093674 (LCB +0.0057039).
Task: (1) confirm the selector code; (2) RECOMPUTE all four bounds and the winner counts from
results/q_r1/cold_start_replication_v1/d1_candidate_calendars.json using python3+numpy in a temp
dir. To match the frozen inference: read the bootstrap implementation in
scripts/run_q_r1_cold_start_replication.py (clustered by history root, seed 20260722, 10000 draws)
and replicate it exactly for the LCBs. Report your recomputed numbers vs the auditors'.` },
  { id: 'V4_pair_distribution', effort: 'high', prompt: `${COMMON}
CLAIM V4: In D0 at persistence 0.90 the 264 retained-vs-reset pair deltas on early_ret_2w split as
110 favorable / 145 exactly zero / 9 adverse (so median 0, p75 ~ +0.0597), meaning the mean +0.0226
comes from a minority of pairs where the action actually changed — and among NONZERO pairs the
favorable share is ~92%. Task: recompute from
results/q_r1/cold_start_replication_v1/d0_retained_context.json using python3 in a temp dir:
per-pair deltas (retained minus reset, same history root + campaign index, excluding campaign 0 if
that is what the adjudicator does — read scripts/run_q_r1_cold_start_replication.py to replicate
pairing exactly), then report counts favorable/zero/adverse, median, p75, favorable-among-nonzero,
and the same for persistence 0.75. Also verify the headline means +0.0226361 (0.90) and +0.0137836
(0.75) against results/q_r1/cold_start_replication_v1/adjudication.json.` },
  { id: 'V5_dual_bootstrap', effort: 'low', prompt: `${COMMON}
CLAIM V5: Two different bootstrap implementations produce two different "LCB95" for the same
contrast: scripts/run_q_r1_d0_cold_start_reanalysis.py uses seed 20260721 with 5000 resamples;
scripts/run_q_r1_cold_start_replication.py uses seed 20260722 with 10000 resamples; raw D0 reports
~0.0191247 while adjudication reports 0.0190475 for the same persistent-0.90 contrast. Confirm both
code sites (quote the seed/resample constants) and both numbers in the artifacts.` },
  { id: 'V6_mpc_weaknesses', effort: 'medium', prompt: `${COMMON}
CLAIM V6: The MPC comparator has these specific weaknesses: (a) supply_chain/program_t_joint_belief.py
(~L137-146) draws only count=particles samples WITH replacement from a 6-state exact posterior, RNG
seeded from a hash of the observation (so different arms use different integration draws); (b)
supply_chain/program_t_full_des_mpc.py (~L324-383): beyond the planning horizon the candidate
calendar repeats the LAST action; modes nominal/scenario/robust force feasibility True (only
constraint_aware enforces the worst-product floor); constraint_aware is fail-open (if no candidate
is feasible it still returns the best infeasible one); the planning objective is ret_visible; CVaR
tail uses ceil(0.1*n_scenarios) so with 4 particles the "CVaR10" is the single worst draw. Confirm
each sub-claim with line quotes.` },
  { id: 'V7_online_ms', effort: 'low', prompt: `${COMMON}
CLAIM V7: Matched-compute cannot be audited because prefix arms log online_ms=0: the controller
prefix function does not return an online_ms key, and the D1 runner does
detail.get("online_ms", 0.0). Check scripts/run_q_r1_d1_demand_memory.py and the controller
functions it calls in supply_chain/ (find controller_prefix / controller_calendar return dicts).
Confirm or refute with quotes.` },
  { id: 'V8_degenerate_guardrails', effort: 'medium', prompt: `${COMMON}
CLAIM V8: Two guardrails are structurally degenerate in this experiment: (a) ret_full equals exactly
0.0 in ALL raw rows of d0_retained_context.json and d1_candidate_calendars.json (auditors say 8640
rows total) while ret_visible ~0.85 — check with python3/jq over
results/q_r1/cold_start_replication_v1/; (b) lost_orders is structurally 0 because each campaign
generates only 48 requests against a backlog cap of 60 — find the cap-60 / lost-order logic in
supply_chain/program_o_full_des_transducer.py and the request count in the Q-R1 config, and check
lost fields in the raw rows. Report exact counts of rows checked and any exceptions found.` },
  { id: 'V9_repro_environment', effort: 'medium', prompt: `${COMMON}
CLAIM V9: Reproducibility gaps: (a) requirements.txt has only lower bounds but requirements-pinned.txt
EXISTS with exact pins (one audit falsely said no pinned env exists) — list 5 example pins; (b) no
Python version pin anywhere (check .python-version, setup.py/pyproject/toml, CI workflows); (c)
supply_chain/program_o_ret_freeze.py contains frozen source hashes and ~3 of them no longer match
the current files at this commit — compute the hashes it checks (read how it hashes, replicate in
python3 in a temp dir) and report which match/mismatch; (d) adjudication.json embeds an absolute
/private/tmp/... path. Verify each.` },
  { id: 'V10_placebo_semantics', effort: 'medium', prompt: `${COMMON}
CLAIM V10: Two placebo caveats: (a) the iid null is ALGEBRAIC, not empirical — with kappa=0.5 the
between-campaign transition b' = kappa*b + (1-kappa)*(1-b) maps every posterior to exactly 0.5, so
retained==reset by construction; find the transition code (supply_chain/q_r1_retained_learning.py or
the belief module) and confirm the formula; (b) wrong_posterior is a PRIVILEGED anti-oracle: it is
constructed from the TRUE initial regime of the current campaign (complement), not from a merely
mistaken history — find its construction and confirm. Quote lines for both.` },
]

phase('Verify')
log(`Verifying ${TASKS.length} audit-claim classes against ${REPO} (commit efae6514)`)
const results = await parallel(TASKS.map(t => () =>
  agent(t.prompt, { label: t.id, phase: 'Verify', schema: SCHEMA, effort: t.effort })
))
const out = results.filter(Boolean)
log(`${out.length}/${TASKS.length} verifiers returned`)
return { verified: out }