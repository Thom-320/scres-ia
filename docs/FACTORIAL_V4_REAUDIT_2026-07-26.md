# Re-audit of the three required fixes — factorial v4

Audited commits `4141136c`, `324baaee`, `5655680a`, `a7136357` on `codex/q-r1-oracle-v2`,
against the three defects raised in `FACTORIAL_V4_ADVERSARIAL_AUDIT_2026-07-26.md`. Each was
checked in the code, including inside the helper module, because arguments that are accepted
but not enforced would look identical from the call site.

> **RETRACTED IN PART, 2026-07-26.** A counter-audit found two gaps this review missed, both
> verified in the code and both conceded: the shared bar is never compared against an
> authoritative hash (only its scope is checked), and the per-calendar cap raises after the
> offending unit is computed but before it is persisted, leaving that unit unattested. The
> "ready to freeze" recommendation below is withdrawn. See
> `FACTORIAL_V4_PREFLIGHT_AUDIT_2026-07-26.md` and the concession section appended at the end.

**Verdict (as originally issued): two fixes PASS, one is PARTIAL.**

## Fix 1 — shared static bar: PASS

A separate `static-bar` mode computes the bar once. Workers run as `development-worker` and
cannot build their own: they require both the shared artifact and its opening receipt, and
fail closed when either is missing. Before using it a worker verifies that

* the opening receipt's mode is `static-bar` and its contract hash matches the contract;
* the bar's identities cover **every** selection root (`static_roots != integer_range(expected)`
  raises);
* the bar was built over exactly `16 * 12 * 3 = 576` campaigns;

and it records `static_bar_reference.json` carrying the artifact's sha256. The failure mode the
clause exists to prevent — two processes grading against different references — is now closed
by construction rather than by convention, and the ~24 minutes of per-process recomputation is
gone.

## Fix 2 — compute budget: PASS, and enforced in the right order

The caps are not decorative parameters. Inside `structured_pair_rows`, before starting any
uncached campaign:

```
if cumulative_compute_seconds + per_calendar_hard_cap_seconds > aggregate_hard_cap_seconds:
    raise RuntimeError("STOP_COMPUTE_BUDGET_PREDECLARED")
```

The guard therefore refuses work it has **not yet done**, instead of discovering an overrun
after paying for it. A per-calendar cap is also checked after each planning call, and
`progress_callback(rows, cache)` fires after **every** campaign row, persisting partial rows,
the planner cache and a manifest with sha256s and `complete: false`. A raise at any point now
leaves every completed unit on disk. This is the correct inversion of the abort-not-continue
pattern that discarded 137 completed arms this morning.

## Fix 3 — durability of checkpoint rows: PARTIAL

**Done:** every checkpoint writes `checkpoint_rows_t{step}.json` immediately, with its sha256
and the model archive hash recorded in `checkpoint_progress.json`, which is rewritten after
each checkpoint. The original defect — a late death losing every earlier checkpoint including
the expensive structured evaluation — is fixed. No paid work is lost.

**Not done:** there is no resume. The runner contains no resume path (zero matches for resume /
already-complete / skip-existing), and `args.output_dir.mkdir(parents=True)` without
`exist_ok` means a relaunch into the same directory fails by design. So a worker that dies at
checkpoint 8 of 10 keeps its rows but must retrain from zero in a fresh directory.

**Assessment:** this does not block the freeze. The contract-level property is durability —
nothing paid is discarded, and everything is attested by hash — and that property now holds.
Resume is a recovery convenience whose absence costs one worker's training time, not
scientific validity.

**One operational note that should be written down before the run:** because there is no
resume and the output directory must not pre-exist, the correct recovery from a dead worker is
a **new** output directory. Nobody should "fix" a failed relaunch by deleting and reusing the
old one — that would defeat the `output_directory_must_not_exist` custody guard, which is there
to make silent overwrites impossible.

## Also observed

`scripts/aggregate_q_r1_matched_retention_factorial_v4.py` exists and is explicitly documented
as aggregating workers "without opening or simulating data". Keeping aggregation incapable of
touching the simulator is the right separation.

The four factorial arms of a checkpoint are evaluated from one live model, and the archive's
sha256 is computed once before the arm loop and stamped on all four arms' rows, so
same-checkpoint identity is attested rather than assumed.

## Recommendation

**Ready to freeze.** Optionally add resume before the long run, since it is cheap now that the
per-checkpoint artifacts and model archives already exist; but it is an efficiency improvement,
not a precondition. After freezing, the sequence is: instrument preflight on burned/synthetic
data, then the static-bar step, then the development workers on fresh roots. Confirmation
roots stay sealed.


## Concession — two gaps this audit missed

**Gap A: the bar is verified by scope, not by identity.** I wrote that the different-references
failure mode was "closed by construction". It is not. The worker computes
`sha256(static_bar_path)` and writes it into `static_bar_reference.json` — it *records* the
hash, it never *compares* it against an authoritative value. The opening receipt cannot carry
the bar hash because it is written before the bar exists. So two workers could be handed
different bars that both cover all 16 selection roots and exactly 576 campaigns with valid
identities, yet carry a different `calendar` and `frontier_row`, and both would pass every
current guard. My checks close the narrower-scope hole only.

Fix: a `static_bar_completion_receipt.json` written after the bar is built, carrying the bar's
sha256, the contract hash, a digest of the identities, the calendar, the frontier row and the
exact coverage; every worker must compare the received bar's hash against it and fail closed.

**Gap B: the over-cap unit is computed and then dropped unattested.** In
`supply_chain/q_r1_factorial_v4.py` the per-calendar check raises at line 159, after
`calendar_builder` has returned — so the work is paid — and before `rows.append(...)` and the
`progress_callback` at line 191. Earlier units survive because earlier callbacks persisted
them, but the unit that triggered the stop vanishes with no record of why.

Fix: write a rejection receipt for that unit — calendar key, observed seconds, the cap, the
configuration, the hashes, status `REJECTED_OVER_CAP` — and never make that row eligible.

Both findings are correct, both were missed here, and together they mean the freeze should wait
until the two receipts exist and the negative tests pass.
