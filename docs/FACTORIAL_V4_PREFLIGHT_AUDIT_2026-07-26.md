# Audit of the factorial v4 burned instrument preflight

Audited `6677da97` on `codex/q-r1-oracle-v2`:
`results/q_r1/matched_retention_factorial_v4_preflight/receipt.json`.

**Verdict: the preflight PASSES. The freeze has not happened yet, and one custody question
should be settled before it does.**

## What the preflight demonstrates

| Property | Evidence |
|---|---|
| No fresh root touched | `fresh_development_roots_opened: false`, `confirmation_roots_opened: false`; the preflight scope is the burned root 7570801, campaign index 0 |
| Execution attestable | `commit: a7136357`, `worktree_clean_at_opening: true` |
| Contract unchanged since the run | the receipt attests sha256 `591eed17…`; I recomputed it on the current DRAFT bytes and it matches exactly |
| Same checkpoint across all four arms | `same_checkpoint_hash_all_neural_arms: true` — the property is now verified empirically, not merely stamped |
| Service ledger complete | `missing_service_rows: 0` |
| Arms symmetric | P0_H0 / P1_H0 / P0_H1 / P1_H1 = 36 rows each, plus `best_static_frozen` 36 and both structured arms |
| Structured cache works | 6 structured rows from 1 cache entry, 5 reuses |
| The three fixes exercised, not just declared | `aggregate_budget_checked_before_next_uncached_calendar`, `checkpoint_rows_persisted_before_next_checkpoint`, `structured_rows_and_cache_persisted_incrementally`, `development_workers_forbid_static_bar_recomputation` all true |
| Artifacts kept out of the repo | `external_output_directory` under /private/tmp |
| Tie honestly disclosed | `selected_checkpoint_timesteps: 0` with the note that the two tiny checkpoints tied exactly and this is not scientific evidence |

My earlier non-blocking observation is also closed in the contract: the preflight scope now
declares `training_and_evaluation_roots_are_identical: true` and
`held_out_interpretation_forbidden: true`, so a later reader cannot mistake a preflight number
for a held-out one.

## Open item 1 — the freeze has not been performed

The contract is still `DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY` and no frozen contract or
freeze receipt exists. Running the preflight before freezing is defensible, arguably better
than the order I originally recommended, since it avoids freezing an instrument that fails its
own preflight. But the freeze is still outstanding.

## Open item 2 — the preflight consumed a declared development seed

The preflight ran with `optimizer_seed: 7672001`, which is the **first of the five declared
development optimizer seeds** and the first of the three declared screen seeds. The contract
contains no policy about which seed a preflight may use.

This is not a leakage argument: the preflight trained from scratch on burned roots disjoint
from the development block, and no selection decision carries over. It is a custody hygiene
question, and it is much cheaper to settle now than to argue about later, because once the
contract is frozen the seed set becomes authority.

Two acceptable resolutions, either of which closes it:

1. run preflights on a seed **outside** the declared set, leaving all five development seeds
   pristine; or
2. add an explicit clause permitting preflight reuse of a development seed, stating the reason
   (training restarts from scratch on disjoint burned data, no selection is carried) so the
   reuse is a declared decision rather than an accident.

I mildly prefer (1), because it costs nothing and removes the question entirely.

## Recommendation

Settle open item 2, then freeze, then run the static-bar step and the development workers on
fresh roots. Confirmation roots stay sealed. I will audit the freeze receipt when it lands:
that it is separate and immutable, that its hash covers the frozen contract bytes, and that it
asserts the roots closed at the moment of freezing.
