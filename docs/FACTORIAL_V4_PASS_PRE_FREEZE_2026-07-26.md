# External pre-freeze review — Q-R1 matched-retention factorial v4

**Verdict: `PASS_PRE_FREEZE`.** The two custody gaps I conceded are closed, the five negative
tests exist and genuinely exercise the failures, the recovery policy is frozen in the contract,
and the preflight seed no longer belongs to the development block. The contract may be frozen.

Reviewed `543292e8`, `1d7a306d`, `1ce6e624` on `codex/q-r1-oracle-v2`. Every item below was
checked in the code and by running the tests myself, not accepted from the implementation
receipt.

## Gap A — the shared bar is now bound cryptographically: CLOSED

`validate_shared_static_bar` verifies a complete chain rather than a scope:

* opening receipt mode and contract hash;
* completion receipt schema, mode and contract hash;
* `completion.opening_receipt_sha256 == sha256(opening_receipt)` — binds completion to opening;
* `completion.static_bar_sha256 == sha256(static_bar)` — binds the receipt to the artifact bytes;
* `completion.identities_sha256 == json_sha256(bar.identities)` — binds it to the content;
* root coverage and campaign count on both the bar and the receipt;
* **`completion.calendar == bar.calendar`** and **`completion.frontier_row == bar.frontier_row`**.

The last two are the ones that close the exact hole I had missed: two bars over the same 16
roots and 576 campaigns but with a different decision can no longer both pass.

## Gap B — the over-cap unit is now attested before the stop: CLOSED

In `supply_chain/q_r1_factorial_v4.py` the rejection callback fires **before** the `raise`,
carrying the campaign identity, arm, prior, skeleton hash, cache key, observed seconds, the
cap, cumulative seconds, the calendar and sha256 digests of calendar, diagnostics and metrics,
with `action_eligible: False`. It is handed `rows` and `cache` so prior completed work is
persisted in the same call. The offending unit is appended to neither `rows` nor the cache —
both assignments sit after the raise — so it cannot become eligible by any path.

## Recovery policy — frozen, not a note

`execution_custody.failed_worker_recovery` declares: no resume, a new output directory
required, same commit/contract/config/seed required, the failed attempt must be preserved and
may not be deleted or reused, and only one complete attempt per worker may be action-eligible.
That is stronger than the operational note I proposed, and it is the right form.

## Optimizer seed custody — settled

Development seeds are now `7672101-7672105`, screen seeds `7672101-7672103`. The instrument
preflight seed `7672001` is recorded as `BURNED_INSTRUMENT_ONLY_NOT_DEVELOPMENT_ELIGIBLE`.
This is resolution (1) of the two I offered, the one that removes the question rather than
documenting it.

## Negative tests — present, and they exercise real failures

| Test | What it does |
|---|---|
| `test_static_bar_chain_rejects_an_altered_bar` | mutates the bar's calendar, expects a raise |
| `test_static_bar_chain_rejects_a_wrong_completion_receipt` | corrupts `static_bar_sha256`, expects a raise |
| `test_two_workers_cannot_accept_different_static_bar_hashes` | asserts the valid bar passes **and** a divergent `frontier_row` raises |
| `test_over_cap_unit_is_rejected_with_receipt_after_prior_rows_persist` | asserts the receipt exists and prior rows survive |
| `test_runner_rejects_an_existing_output_directory` | asserts the custody guard |

I ran the four v4 test files independently: **24 passed**.

## Non-blocking observation

All three static-bar tests trip on the *first* guard, "artifact hash mismatch", because any
content mutation also changes the file hash. The deeper guards — identities digest, calendar
match, frontier-row match — are therefore never individually exercised. One extra test would
close this: mutate only the **completion receipt's** `calendar` field while leaving
`static_bar_sha256` consistent with the untouched bar, so the hash check passes and the
calendar guard is the one that fires. Not a precondition for the freeze.

## State at the time of this review

Contract still `DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY`, sha256 `bb92a2cb…`; the new burned
preflight attests that same contract at execution commit `1d7a306d` with a clean worktree,
`same_checkpoint_hash_all_neural_arms: true`, zero missing service rows, and the structured
cache doing 1 computation with 5 reuses. `fresh_development_roots_opened: false`,
`confirmation_roots_opened: false`. The earlier preflight receipt attesting `591eed17…` is the
superseded first run and should be read as such.

## Authorization

`PASS_PRE_FREEZE`. Freeze the contract, then run the static-bar step and the development
workers on the fresh roots. Confirmation roots stay sealed until a prospective power audit.
