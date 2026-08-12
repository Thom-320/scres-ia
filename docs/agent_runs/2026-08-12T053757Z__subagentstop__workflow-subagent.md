# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `workflow-subagent`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a778bf0e0ca355583`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T05:37:57.681225+00:00`
- **commit**: `b68bcf19`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a778bf0e0ca355583.jsonl`

## Final response

Verification complete. The arithmetic survives; the classification does not.

**What held up.** The field exists and every number matches to the last digit. `inference.estimates["rho90_share90::H_OL"]` = point `0.1172400203711007`, lcb95 `0.10614476924833394`, ucb95 `0.12833527149386748` (plus an unreported `se` `0.003760011633529455`). The supporting figures are exact too: `favorable_tapes_fraction_vs_open_loop` `0.95703125` (= 245/256), `positive_learner_seeds_H_OL` `10`, `best_classical_config` `max_pressure__0`. The bar of 0.01 is real — `contracts/program_q_frozen_policy_replication_v1.json` → `.inference.adaptation_rule` — and I independently confirmed that file's sha256 equals the `contract_sha256` recorded in the result. The estimand wording is right: `.inference.H_OL = "learner - max(open_loop)"`. No supersession edge exists; the registry has zero `program_q` occurrences, and the posterior C1 diagnostic is a different program on burned roots 757xxxx marked `EXPLORATORY_NO_CLAIM`.

**What I refute.** Two things.

The provenance is misattributed. The cited `result.json` contains no `claim_status` field at all, and its own `terminal_verdict` reads `PENDING_DIRECT_FULL_DES_REPLAY_AND_ADJUDICATION` — not the STOP string. Both halves of the quoted claim_status live in a sibling file the report never names, `.../artifacts/confirmation/adjudication.json`.

The cause of death is wrong, and wrong in the direction that matters most to you. Every scientific cell gate passed in all three cells — H_OL, equivalence, favorable_tapes, learner_seeds. `neural_premium` failed, but the contract explicitly anticipates that and routes to the equivalence rule, which passed. The single failing gate anywhere in the artifact is `integrity_gates.worst_product_fill_noninferior = false`: a worst-product fairness guardrail at margin −0.02, breached in all three cells with point estimates that are only mildly negative (−0.0045 in rho90_share90) but lower bounds that cross. This lane won on the mean by 10.6× its bar and was killed by the lower tail of a fairness endpoint — the exact pattern you named as the objective of the audit, filed by the other agent under "not killed."

## Raw payload

```json
{
 "agent_id": "a778bf0e0ca355583",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a778bf0e0ca355583.jsonl",
 "agent_type": "workflow-subagent",
 "background_tasks": [
  {
   "description": "Find the lanes that came closest to a claim, verify their numbers against artifacts, and rank what is reopenable",
   "id": "wydnbmt6b",
   "name": "closest-losses-audit",
   "status": "running",
   "type": "workflow"
  }
 ],
 "cwd": "<HOME>/Projects/research/scres-ia",
 "effort": {
  "level": "xhigh"
 },
 "hook_event_name": "SubagentStop",
 "permission_mode": "auto",
 "prompt_id": "ee334d76-63aa-4489-9fa7-aac74d371f0b",
 "session_crons": [],
 "session_id": "fc8fefc1-10a6-4eea-bcbf-d92791120134",
 "stop_hook_active": false,
 "transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl"
}
```
