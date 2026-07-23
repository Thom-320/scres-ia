# Pre-registration — constraint-aware c256/c1024 convergence escalation

**Date:** 2026-07-22
**Branch:** `codex/q-r1-comparator-reconciliation`
**Written BEFORE any result exists.** No shard has been launched at the time this file is
committed. If a result predates this commit, this document is void.
**Claim status:** `BURNED_DEVELOPMENT_NO_CLAIM`. Burned roots only; no fresh seeds; no
learner.

---

## 1. Why

The freeze amendment (`docs/Q_R1_COMPARATOR_V2_FREEZE_AMENDMENT_V1_2026-07-22.md`,
Finding C) records that the constraint-aware / service family has **no convergent budget at
any tested level**, and that its c256/c1024 escalation was **never run**:

| config | vs | roots | N | agreement | pass |
|---|---|---|---|---|---|
| `constraint_aware_h4_c16_wf0.70` | c64 | 4 | 16 | 0.93750 | False |
| `constraint_aware_h4_c64_wf0.70` | c256 | 24 | 96 | 0.90625 | False |
| `scenario_h4_c64_wf0.00_tol0.0020_service` | c256 | 24 | 96 | 0.83333 | False |
| `constraint_aware_h4_c4/h3_c4_wf0.80` | c16 | 4 | 16 | 0.00000 | False (`inf`) |
| `constraint_aware_h4_*_wf0.70` | **c256 vs c1024** | — | — | **never run** | — |

Consequently `RETAINED_SAFE_EFFECT` cannot be frozen today **for lack of evidence, not on
performance**. This run produces that evidence, in whichever direction it falls.

This mirrors the budget escalation that commit `51957969` predeclared for the pure-ReT
family: *"This is a budget escalation, not a changed outcome gate."* It is the same move on
the other family, and it is being written down first for the same reason.

## 2. What will run

Four shards over the same burned roots and the same four blocks as the frozen receipt:

```
scripts/run_q_r1_comparator_v2_preflight.py \
  --output <out>/convergence_sNN/result.json \
  --seed-start {7570801,7570807,7570813,7570819} \
  --histories 6 --campaigns 12 --states 12 \
  --horizons 4 --service-floors 0.70 \
  --low-paths 256 --high-paths 1024 \
  --value-indifference-tolerance 0.0 --tie-breaker legacy \
  --convergence-only
```

This emits **two** config pairs per shard, because `family()` always appends the scenario
controller before the service floors:

1. `constraint_aware_h4_c256_wf0.70` vs its c1024 counterpart — **the point of the run**;
2. `scenario_h4_c256_wf0.00` vs its c1024 counterpart — an incidental **replication** of
   the already-frozen receipt.

Execution host: VPS `ovh-agent-lab` (6 cores, idle), so it does not contend with the burned
Pareto running locally.

## 3. Gates — unchanged, copied from the merger

`merge_q_r1_comparator_v2_shards.py::merge_convergence`, verbatim: abstentions `== 0`,
first-action agreement `>= 0.95`, mean abs planning value error `< 0.005`, q95 `< 0.01`.
No threshold is relaxed, tightened, or added.

## 4. Decision rule, fixed now

- **If the constraint-aware pair passes** → the family becomes *eligible* for a separate
  freeze answering `RETAINED_SAFE_EFFECT`, executed with the same two instruments used for
  the ReT-pure family: `scripts/freeze_q_r1_comparator_v2_amended.py` (adapted only in its
  expected config id) **and** `scripts/audit_q_r1_comparator_v2_freeze.py` re-deriving the
  gate from the raw rows. Eligibility is not a freeze; the freeze is a separate step.
- **If it fails** → constraint-aware convergence is `UNRESOLVED` at this budget ladder, and
  **no safe-effect freeze is possible from it**. No further escalation to c4096 is
  authorized by this document.
- **If the incidental scenario pair disagrees with the frozen receipt** (0.96875 /
  0.00010953 / 0.00024667) → that is a red flag about reproducibility and **must be
  reported**, not quietly dropped. It cannot be used to revise the freeze either way.

## 5. What this run may NOT do

- It may **not** reselect, replace, or revise the frozen pure-ReT c256 comparator. The two
  families answer **different gates** (`RETAINED_RET_EFFECT` vs `RETAINED_SAFE_EFFECT`) and
  are not competing candidates for one slot.
- Selection may **never** use retained−reset, learner return, or any realized outcome. The
  only criterion here is numerical convergence against a higher budget.
- No fresh roots. No learner. Nothing here upgrades `BURNED_DEVELOPMENT_NO_CLAIM`.

## 6. Stated prior — the likely outcome is that this family is vacuous

Recorded now so it cannot be presented later as a discovery:

In the burned c64 Pareto the `wf0.70` floor was **inactive** — the `constraint_aware` output
was identical to `scenario`. If that holds at c256, then `constraint_aware_h4_c256_wf0.70`
is numerically **the same controller** as the already-frozen `scenario_h4_c256_wf0.00`, its
convergence will match trivially, and the "safe" family adds nothing beyond the ReT-pure
freeze. **That would itself be the finding**: the service floor as specified has no bite,
so a separate deployment-safety comparator cannot be built by tightening it — a genuinely
binding constraint would have to come from somewhere else. The neighbouring floor `wf0.80`
is already known to be *over*-binding (`inf` value error, 0.0 agreement), i.e. infeasible.

A pass here is therefore not automatically good news, and a numerically-identical result is
the outcome to expect, not a surprise to explain away afterwards.
