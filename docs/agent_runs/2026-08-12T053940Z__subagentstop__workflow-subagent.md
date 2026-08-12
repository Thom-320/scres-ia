# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `workflow-subagent`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `a07a2a371588b4e78`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T05:39:40.269489+00:00`
- **commit**: `b68bcf19`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a07a2a371588b4e78.jsonl`

## Final response

I have everything I need. Here is the adversarial verification.

## Verdict: the numbers are exact, the death certificate is wrong

### 1–2. File and fields exist; every number matches to the last digit

`<HOME>/Projects/research/scres-ia/results/program_g/g5/verdict.json`

| reported | actual field | match |
|---|---|---|
| 560.1958333333333 | `virgin.vs_static_ci95.tree[0]` | exact |
| [407.70020833333336, 736.8651041666666] | `virgin.vs_static_ci95.tree[1:3]` | exact |
| [-60.133, -130.393, +7.742] | `virgin.ppo_minus_cover_ci95` = [-60.13333333333333, -130.39302083333334, 7.742499999999995] | exact |
| n=240 | `n_per_split` | exact |
| ABAB | `frozen_best_calendar` = ["A","B","A","B"] | exact |
| -0.02274 [-0.02995, -0.01634] | `results/program_g/retexcel/verdict.json` → `virgin.cover_minus_static_full_ledger_ci95` | exact |
| -0.02317 | `results/program_g/terminal_metric_audit/verdict.json` → `policy_audits.cover.paired_delta_ci95.ret_order[0]` | exact |
| +3370 | same file → `...unfulfilled_rations_at_horizon` = 3370.2775 | exact |

Estimand direction confirmed arithmetically: `virgin.mean_service_loss.best_static` 26120.604166666668 − `.tree` 25560.408333333333 = 560.1958. Static minus policy, positive = policy better.

**I could not refute a single reported figure.** I also failed to refute my strongest hypothesis: I suspected `retexcel/verdict.json` was the defect-contaminated pre-fix artifact, since its doc header says "treat the tables below as the pre-fix version." `git show 5bc03e56` proves the **JSON was re-run post-fix** (virgin full-ledger moved -0.033136 → -0.022740). The doc tables are stale; the JSON the report cited is not.

### 3. `claim_status` is not a field of this artifact

No file under `results/program_g/` contains a `claim_status` key. The real status fields are `g5/verdict.json → interpretation` = `G5_LEARNER_BEATS_STATIC_OOS_NO_NEURAL_INCREMENT`, and the terminal `terminal_metric_audit/verdict.json → verdict` = `STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT`. `SUPERSEDED_PROXY_ONLY` is the auditor's paraphrase of doc prose — faithful, but invented, and it should not be quoted as an artifact field.

### 4. The registry does not carry this at all

`research/supersession_registry.json` has 23 edges and **zero** mention of `program_g` (grep count 0). The supersession lives only in Markdown prose. That is a real custody gap in the registry, not in the report.

### 5. The cause of death is misclassified — this is the finding

The repo publishes its own machine-readable death certificate: `results/paper2_search/failure_taxonomy.json` → `decision_families[8]` (`family_id: "G_spatial_commitment"`).

```
failure_class: ["metric_reversal", "fairness_concentration", "dominant_open_loop_calendar"]
```

**Three classes. The report gave one.** It missed the two that matter most to this audit:

- **`fairness_concentration`** → your `tail_or_fairness_guardrail`. `policy_audits.cover.paired_delta_ci95.worst_cssu_fill` = **-0.17335543032044837 [-0.20330147948914484, -0.1465781613815386]**; worst-served CSSU fill collapses 0.8465 → 0.6731 (`means.best_periodic_static.worst_cssu_fill` / `means.cover.worst_cssu_fill`). Plus the shed guardrail `attended_orders` = **-3.5075 [-4.1451, -2.9498]**.
- **`dominant_open_loop_calendar`** → your `comparator`. `strongest_comparator` = "full four-period periodic frontier; ABAB". A blind open-loop calendar dominates every closed-loop policy.

Worse, the report's stated *mechanism* is the one the repo explicitly retracted. Commit `5bc03e56`: "the order-vs-mass explanation is **REFUTED**; the reversal is continuity/fairness metrics vs a non-fairness-preserving aggregate index." The report reproduces exactly that refuted story ("the win is measured on a ration-mass service-loss proxy... the sign flips"). It is refuted because `ret_quantity` is *ration-weighted* and ABAB still wins it: **-0.00695449561166123 [-0.008688414592678132, -0.005484890065843363]**.

**This lane is in fact your target category, and the report buried it.** `results/program_g/triangulation/verdict.json` (never cited by the report, `kind: "exploratory_metric_sensitivity"`, `interpretation: METRIC_INDUCED_POLICY_REVERSAL_CONFIRMED`) shows cover **winning in mean with LCB95 > 0** on both Cobb-Douglas lenses — `cover_minus_ABAB_ci95.cd_sigmoid` = +0.00478144092132831 [+0.0036769730202516684, +0.005975055890041531], `cd_spatial` = +0.04803884094839869 [+0.03725037698014028, +0.05926272124701088] — while `worst_cssu_fill` = -0.173262241113646 [-0.2133743825817509, -0.13646179169672787]. Won on the mean of the aggregate index, died on equity.

### 6. Scope degradation the report never disclosed

- `terminal_metric_audit/verdict.json → scope` = `"stylized_program_g_order_adapter_not_full_des"`; `forbidden_claims` = `["full_des_confirmation","cobb_douglas_rescue","virgin_mfsc_confirmation"]`. Neither the +560 win nor the −0.023 loss is a full Op1–Op13 DES measurement.
- The report calls retexcel "virgin". The JSON key is `virgin`, but `docs/PROGRAM_G_RETEXCEL_CONFIRMATION_2026-07-12.md` states it is **"NOT virgin-confirmatory. Tapes 1010001+ were already opened by G5"** — a replay on burned tapes. `results/paper2_search/seed_burn_ledger.json` confirms 1010001+ in both. Only 1040001+/1050001+ are preregistered-fresh.
- The taxonomy's `exact_failure` states **"only the terminal metric audit is canonical"** — so citing retexcel as a co-equal killing number is citing a non-canonical artifact, even post-fix.
- The "neural question answered cleanly" rests entirely on the retracted proxy: the terminal preregistration says **"No PPO training or tuning is permitted"**, and `policy_audits` contains only cover/mpc/ret_tree_depth3/service_tree_depth3. PPO was never re-tested on the canonical endpoint.
- The three "learners" are degenerate: `virgin.mean_service_loss` gives tree = bandit = cover = 25560.408333333333 identically. The doc confirms "the two depth-3 trees collapsed to the same tested behavior as cover." There is no learning increment over a hand-written one-liner, let alone a neural one.

### Corrected framing, with a real bar

The report's `barra: 0 / distancia: -407.70` is doubly off: the sign is inverted (the lane *cleared* by +407.70), and the bar is proxy-local prose, not a JSON field. The project bar is `results/paper2_search/failure_taxonomy.json → practical_gate_ret_lcb95` = **0.01**, against which:

- `decision_families[8].h_pi` = **+0.01641384654689033 [+0.013953358880973493, +0.018888441969943784]** — clairvoyant headroom **clears** (LCB +0.01395 > 0.01).
- `decision_families[8].h_obs` = **-0.021186874058491956 [-0.02554859191998398, -0.017183620638142404]** (this is *mpc*, the least-bad policy, not cover). Distance to bar = **-0.027183620638142404**.

So Program G is a clean `physics_no_headroom`-adjacent boundary result at the observable layer, not a near-miss: −0.027 from a 0.01 bar.

### Bonus, directly on your stated priority

The pure "won in mean, died on equity" case is **Program I**, not G — `results/headroom_gsa/oos_guardrail_check.json`. H_obs positive and OOS-stable across three blocks (`blocks.FRESH_4200001.ci95` = [0.0087, 0.0141], `blocks.FRESH_4500001.ci95` = [0.0072, 0.0129]), killed by `worst_cssu_fill_delta` -0.1435/-0.1282/-0.1258 against a **-0.02** threshold. Its own `verdict` field names it "the Program-G concentration/fairness artifact" — the same mechanism, and it missed the 0.01 bar by 0.0013–0.0028 on the LCB. That is your near-miss-plus-fairness lane.

## Raw payload

```json
{
 "agent_id": "a07a2a371588b4e78",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-a07a2a371588b4e78.jsonl",
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
