# SCRES-IA session provenance gate

**Audit date:** 2026-07-13  
**Requested terminal outcomes:** (A) confirmed Paper 2 environment, or (B) machine-verified boundary certificate  
**Audit verdict:** `BLOCKED_PROVENANCE_NO_TERMINAL_SCIENTIFIC_VERDICT`

## Scope available in this execution environment

Available:

- GitHub connector access to the public remote repository.
- Uploaded thesis, Garrido papers, Ding et al. (2026), manuscript drafts, and the PI-supplied current-state statement.

Unavailable:

- `/Users/thom/Projects/research/scres-ia` in the active runtime.
- Any local Git object database containing corrective commit `ef6b53b7`.
- A VPS endpoint, SSH credentials, scheduler, or mounted VPS artifact store.
- A Git bundle or immutable remote ref representing the current local state.

The task itself says that the local K3 correction is newer than the remote and requires provenance reconciliation before relying on PR descriptions. That reconciliation cannot be completed without the local objects and raw artifacts.

## Remote provenance independently verified

| Object | Verified state | Scientific implication |
|---|---|---|
| `main` | Earlier state rooted at `c6e6d08b...` | Not the July scientific source of truth. |
| Draft PR #2 | Open draft; base `c6e6d08b...`; head `3bcf6e96...` | Its Track-B/Cobb-Douglas description is stale relative to later terminal evidence. |
| DRA-2b | `3bcf6e96...`; `STOP_DRA2B_PRE_TREE_GATE` | Restricted PI headroom existed, but service-magnitude and horizon gates failed; no learner or virgin test. |
| Program E | `59bfd218...`; `STOP_PROGRAM_E_VALIDATION` | Ten MaskablePPO seeds failed observable conversion; no virgin tapes. |
| Program F | `STOP_PROGRAM_F_SCREEN` | Liveness and PI headroom existed, but observable rollout conversion passed in 0/24 cells. |
| Program G | `006b41c1...`; terminal metric audit | ABAB beat cover, MPC, and trees after calendar and ReT-ledger corrections. |
| Program H | terminal belief audit | The Bayes filter was informative, but no belief policy met the joint gates. |
| Program I | remote verifier head `9b758d45...` | Full-DES branching found practically negligible adaptive value; the stylized positive region violated worst-CSSU fairness. |
| Bottleneck migration | 2026-07-13 terminal verdict | Signal-adaptive response-team allocation did not beat constant manufacturing allocation. |

## Newer PI-supplied local claims that remain unauditable here

The supplied current-state statement says:

- Program J PPO beat the static comparator in `0/6` seeds.
- Program K's short-shelf-life result conflicts with the approximately three-year ration premise.
- K2 had substantial perfect-information headroom but no reliable observable conversion.
- K3 PPO emitted the same eight-week action sequence on every test tape:
  `(1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.0, 0.0) * D0`.
- A full-horizon period-8 schedule reproduced PPO and beat the purportedly adaptive MPC.
- The corrective local verdict is `RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND` at local commit `ef6b53b7`.

These statements are binding cautions, but they cannot be upgraded to independently verified evidence without the commit, verdict JSON, trajectories, full static frontier, and resource ledgers.

## Terminal-outcome logic

### A. `PAPER_2_EXISTS`

**Not authorized.** No independently verified accessible result satisfies the complete canonical-metric, comparator, resource, guardrail, virgin-tape, and state-feedback gates. The supplied K3 correction specifically says the apparent positive failed the fixed-sequence replacement test.

### B. `BOUNDARY_CERTIFICATE`

**Not authorized.** The remote contains strong family-specific boundary results, but not a machine-verified exhaustion certificate for the entire permitted envelope. The latest local tree and raw J/K/K2/K3 artifacts are unavailable, fresh candidate families cannot be executed without the current repository and compute environment, and no theorem closes all permitted researcher extensions.

Returning B would manufacture exhaustion just as returning A would manufacture a win. Scientific theater remains theater even when formatted as JSON.

## Authorization state

```text
paper_2_confirmed = false
paper_2_boundary_certificate_complete = false
paper_3_authorized = false
virgin_tapes_opened_by_this_session = 0
learners_trained_by_this_session = 0
```

## Required provenance package

1. A Git bundle or immutable remote branch/tag containing `ef6b53b7` and all parents, tags, contracts, verdicts, and raw K3 artifacts.
2. `results/k3/open_loop_confound_audit.json`, complete per-tape trajectories, full-horizon static-frontier output, and resource ledgers.
3. Current Program J, K, and K2 contracts, preregistrations, verdict JSON files, and raw result tables.
4. Current dependency lock, environment snapshot, and burned/unopened seed registry.
5. An executable compute endpoint or mounted artifact channel for screens, training, and one-time virgin confirmation.

## Binding conclusion

`NO_TERMINAL_RESULT_AUTHORIZED_FROM_ACCESSIBLE_EVIDENCE`.

This is a provenance gate, not a scientific boundary certificate. No Paper 2 positive claim and no Paper 3 retained-learning claim should be made from the accessible evidence.
