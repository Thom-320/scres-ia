# Agent run — SubagentStop

> **`UNADJUDICATED_DO_NOT_CITE`** — raw agent transcript saved by the `SubagentStop` hook. It may contain WITHDRAWN claims; the canon lives in sealed artifacts and amendments, never here.

- **event**: `SubagentStop`
- **agent_type**: `workflow-subagent`
- **session_id**: `fc8fefc1-10a6-4eea-bcbf-d92791120134`
- **agent_id**: `aec37680a1d7fba46`
- **cwd**: `<HOME>/Projects/research/scres-ia`
- **transcript_path**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134.jsonl`
- **saved_at**: `2026-08-12T05:25:20.072175+00:00`
- **commit**: `b68bcf19`
- **answer read from**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-aec37680a1d7fba46.jsonl`

## Final response

All three tracks read. Everything below is reconstructible from a named field; where a CI is not a stored field I say so explicitly.

## Direct answers to the three asks

**1. Track B-P — the posture effect and how much a FIXED posture absorbs**

| estimand | point | CI95 (seed-clustered, n=5) | seeds + | tapes + |
|---|---|---|---|---|
| 11D − 8D (adding the 3 buffer dims) | **+0.028488** | [+0.015815, +0.041161] | 5/5 | 117/120 |
| **8D + frozen posture** − 8D | **+0.028928** | [+0.016285, +0.041572] | 5/5 | 117/120 |
| 11D − (8D + frozen posture) | **−0.000440** | [−0.001679, +0.000799] | 2/5 | 47/120 |

A **fixed** per-operation posture `(Op3=0.15311, Op5=0.247975, Op9=0.206824)` absorbs **101.5 %** of the contract effect. The dynamic residual is negative and straddles zero. Two independent controls agree: within-checkpoint clamp `self − clamp_perop = +0.000153 [−0.000899, +0.001206]`, and disjoint-calibration `self − calibration_frozen_global = −0.000156 [−0.000746, +0.000433]` (2/5 seeds positive). So the entire Track B-P win is a **learned constant reserve level**, not scheduling, anticipation, or state-contingency.

**2. Track C — the switcher's maximum against its material bar**

Pass rule is `run_track_c_gates.py:438` → `lo > 0 AND delta.mean() >= 0.05·ret_base`.

| iter | switcher − constant (J) | CI95 | bar (0.05·ReT_base) | fraction of bar | short by |
|---|---|---|---|---|---|
| 4 (**max**) | +6.4649e-05 | [+5.4447e-06, +1.3590e-04] | 1.5422e-04 | **0.419×** | 8.957e-05 |
| 5 (final, 24/24) | +2.7784e-05 | [+2.4269e-05, +3.1422e-05] | 2.0880e-04 | 0.133× | 1.8102e-04 |

The maximum is 2.39× below its bar, and it is the iteration with the *worst* consistency (12/24 tapes). The cleanest-CI iteration is 7.5× below. Five iterations, zero PPO seeds spent.

**3. Gate A2 — mlp against the best non-neural**

`mlp − linear_feedback = −0.5594, LCB95 −0.7476, UCB95 −0.3856, 7/48 tapes favourable`, against a bar of `LCB95 ≥ +0.01`. Missed by **0.7576**. The mlp still beats the *narrow* class it was allowed in Gate A (vs constant +0.5510 [+0.3598, +0.7291]; vs ewma +0.5422; vs threshold +0.4717) — the premium existed only against a class that could not do linear feedback. `linear_feedback` beats the constant on **48/48** tapes (+1.1103 [+1.0375, +1.1824]).

## The distinction you care about: nothing here died in a tail or a fairness constraint

That is the clean finding of this audit. Across Track B, B-P and C there is **no** lane that won in the mean and was killed by a tail/CVaR/worst-product guardrail:

- **Track B** lost in the mean *and* the tail simultaneously — canonical PPO − best full-contract static = `−1.8049e-05 [−2.8615e-05, −8.0873e-06]` on ret_excel, and CVaR05 agrees at `−1.0192e-04 [−1.6328e-04, −4.2639e-05]`. The tail did not kill a mean win; both were already negative. Pure **comparator**: the historical +4.55 % (`delta_vs_best_static = 2.5676e-04`) was measured against a 147-cell family that varied only shift and dispatch while fixing upstream controls. Optimise the static policy over the *same* 8-D contract and the gain reverses.
- **Track B-P** won in the mean and was killed by (a) a stronger comparator — the fixed posture above — and (b) a **priced endpoint**, not a tail: holding is unpriced during training; at `λ_h = 0.10` the increment is `+0.007887 [−0.007603, +0.023378]` and the crossover is `λ_h* ≈ 0.138`. This is your "headroom needs a PRICED resource" pattern again, not a tail guardrail.
- **Track C** never got a material mean win to lose.

## Two instrument problems I found that are not in any verdict doc

**(a) Track B-P's pre-registered primary metric is identically zero.** The prereg fixes the primary as `order_ret_excel_mean` (Excel convention). In all 16 Gate-1 breadth cells and all three clock policies, `always_minus_never__order_ret_excel_mean.mean = 0.0`, `ci95 = [0.0, 0.0]`, `positive = 0/24` — including the cell the whole lane is built on (`f8_i4`). The entire +0.0285 result is measured on the *episode-level* `ret_excel` fallback. The declared primary discriminates nothing anywhere in this regime family.

**(b) Track C iter3's C2 verdict contradicts its own C1.** `c2_verdict.json` says `"passed": true, "verdict": "PROMOTE_TO_C3_TRAINING"`, but the rule at `run_track_c_gates.py:510` is `capture >= 0.5 and lo > 0 and c1.get("passed", False)`, and that directory's `c1_verdict.json` has `"passed": false`. Both files were written at 11:05 on 10 Jul, so the C1 verdict was regenerated around the C2 stage. Non-decisive — C1 is the material gate and it failed in 5/5 iterations, and no C3 training was launched — but a promotion flag that can read `true` while its precondition reads `false` is a live defect in that runner.

**(c) Custody, both Program N gates.** `falsifiers.custody.passed = false`, `status = "COLLISION"`, `registry_status = "BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED"`, 77 seeds each. The collision is with the gate's *own* registry entry (self-registration after the fact, `n_sealed_artifact_overlap = 0`), so it is bookkeeping rather than reuse — but both artifacts carry `run_role: "DEVELOPMENT"` and `scope: "DEVELOPMENT_PAIRED_ON_A_FRESH_BLOCK_NO_CONFIRMATORY_GRADE"`. Gate A2 has no confirmatory grade.

## The lane that came closest, and it is not close

The narrowest miss in these three tracks is **Track B's same-contract challenge: 2.86e-05 of ReT away from a zero bar**. But that is small in absolute terms only because ret_excel itself is ~5.9e-03 here; in relative terms the static policy wins by 0.31 % and takes 58/60 tapes. The retained positive is the within-learner dispatch increment: `factorial joint − upstream/shift with dispatch fixed at 2.0x/1.5x = +2.1106e-05 [+7.0145e-06, +3.4963e-05]`, 4/5 seeds, 58/60 tapes, CVaR05 also positive `+1.5577e-04 [+7.0531e-05, +2.4104e-04]`, 5/5 seeds. Dynamic dispatch access buys something *inside* the learner class; it does not beat a well-optimised constant.

Files: `<HOME>/Projects/research/scres-ia/results/program_n/gate_a_track_b/result.json`, `<HOME>/Projects/research/scres-ia/results/program_n/gate_a2_track_b/result.json`, `<HOME>/Projects/research/scres-ia/results/track_b_nonneural/result.json`, `<HOME>/Projects/research/scres-ia/outputs/experiments/track_b_same_contract_challenge_2026-07-10/summary.json`, `<HOME>/Projects/research/scres-ia/outputs/experiments/track_bp_*_2026-07-09/summary.json`, `<HOME>/Projects/research/scres-ia/outputs/experiments/track_c_gates_iter{1..5}_2026-07-10/c{1,2}_verdict.json`. No file was modified.

## Raw payload

```json
{
 "agent_id": "aec37680a1d7fba46",
 "agent_transcript_path": "<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/fc8fefc1-10a6-4eea-bcbf-d92791120134/subagents/workflows/wf_0c792111-6d8/agent-aec37680a1d7fba46.jsonl",
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
