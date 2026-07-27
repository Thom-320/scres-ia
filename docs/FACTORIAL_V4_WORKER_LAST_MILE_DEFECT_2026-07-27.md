# Development workers: last-mile defect, no result, work salvageable

**Status: `NO_RESULT_PRODUCED`. Nothing to interpret. The computation survived on disk.**

The 24-worker development screen launched its first batch of 9. Eight completed the entire
computation and then crashed while assembling the final result. No `result.json` exists for any
worker, so there is no development outcome to read. The remaining 15 workers never started.

## The defect

`scripts/run_q_r1_matched_retention_factorial_v4.py`, lines 828 and 1028:

```python
"static_bar_sha256": sha256(args.output_dir / "static_bar.json"),
...
"sha256": sha256(args.output_dir / "static_bar.json"),
```

Both hash a bar located **inside the worker's own output directory**. That file exists only in
`static-bar` mode. A `development-worker` receives the bar through `--static-bar-path` and
records `static_bar_reference.json` instead, exactly as the shared-artifact design requires. So
the final result write raises

```
FileNotFoundError: .../q_r1_factorial_v4_screen_v1/s01_7672101/static_bar.json
```

after 96,000 timesteps, five checkpoints, four factorial arms per checkpoint and the full
structured comparator have already been paid for.

This is a consequence of the fix I asked for, not a flaw in it: once workers were forbidden from
building their own bar, two leftover call sites still assumed one existed locally. It is the
kind of defect a smoke in worker mode would have caught, and the preflight could not, because
preflight runs in a mode where the local bar does exist.

## What survived, and why that matters

The durability requirement I made a precondition for the freeze is what turned a total loss into
a recoverable one. Every one of the eight workers holds a complete artifact set:

* `checkpoint_rows_t000000` … `t096000` — all five checkpoints;
* `checkpoint_progress.json` with per-checkpoint hashes;
* `structured_rows.json` with `structured_progress.complete: true` — 192 rows per worker,
  1,536 across the eight;
* the model archive for every checkpoint;
* `static_bar_reference.json` and `worker_opening_reference.json`.

I verified the integrity of what survived rather than assuming it:

| Check | Result |
|---|---|
| Four factorial arms at t=96000 | 576 campaigns each for P0_H0, P0_H1, P1_H0, P1_H1 |
| One identical checkpoint hash across the four arms | PASS |
| Structured comparator complete | 192 rows, `complete: true` |
| Rows missing the service ledger | 0 |
| Confirmation roots touched | none |

Had the per-checkpoint persistence not been required before the freeze, all of this would have
been discarded exactly as the 137 VPS arms were.

## Custody

No `result.json` means no result: nothing here may be interpreted as a development outcome, and
no checkpoint selection has been performed. The training and selection roots and seeds
7672101-7672103 are opened and burned for this attempt. Seeds 7672104-7672105 were never
launched. Confirmation 7670201-7670264 remains sealed.

## Recommended recovery

Re-running costs the entire computation a second time for work that is already on disk and
hash-attested. The recovery policy requires that a failed attempt be preserved and never reused
as an output directory — it does not forbid **reading** it as evidence. So the clean path is:

1. fix both call sites to use the shared bar path in worker mode;
2. add a salvage step that reads the preserved attempt's persisted rows and emits the results
   into a **new** directory, leaving the failed attempt untouched;
3. only if the salvage cannot reproduce every required field, re-run.

If the salvage path is taken, I will audit that it reads only persisted artifacts, that it
recomputes nothing that would differ from the paid computation, and that the resulting
attestation names the failed attempt it recovered from.
