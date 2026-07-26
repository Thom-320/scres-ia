# Adversarial pre-freeze audit — Q-R1 matched-retention factorial v4

**Verdict: `PASS_WITH_THREE_REQUIRED_FIXES`.** The design, the custody discipline and the
information-rights construction are sound, and every criterion I published as a rejection
condition is met. Three defects must be fixed before the contract is frozen. None of them is a
scientific error; all three are operational failure modes this programme has already paid for
today.

Audited: contract `contracts/q_r1_matched_retention_factorial_v4.DRAFT.json` and runner
`scripts/run_q_r1_matched_retention_factorial_v4.py` at `bfbae25c` on branch
`codex/q-r1-oracle-v2`. Every finding below was checked against the code, not against the
authoring summary. No branch was modified by this audit.

## What passes

| Rejection criterion | Evidence in the code |
|---|---|
| Three kappa cells including the iid null | `KAPPAS = (0.50, 0.75, 0.90)`, used in development mode |
| Retained MPC as an arm; premium computable from the first run | `evaluate_structured` on the frozen c256 comparator, `neural_premium = P1_H1 - structured_retained` computed in the same pass |
| Service ledger in the rows | `worst_product_fill`, `unresolved_orders`, `lost_orders`, `service_loss` emitted per row |
| Four arms share one checkpoint, attested | all four arms evaluated from the same live model at each checkpoint, with `checkpoint_sha256` of the saved archive stamped on every row |
| Checkpoint selection touches only the selection split | development evaluates on `checkpoint_selection_history_roots`; confirmation never built |
| Runtime frozen in the contract | screen 96k / full 240k timesteps, interval 24k, rollout 480, batch 96, net arch and the hyperparameter grid all read from the contract |
| Opening receipt written at opening | `write_json(opening_receipt)` executes **before** `build_histories`, so it lands before a single root is materialized |
| Freeze receipt separate and immutable | contract requires it; the runner only reads and hashes it |
| Fresh roots and seeds | 7670001-40 / 7670101-16 / 7670201-64, seeds 7672001-5; I re-ran the collision check and the single apparent hit is a false positive (`7672001` occurring inside the float `355.7672001760811` in a metrics CSV) |
| Output directory must not exist | `args.output_dir.mkdir(parents=True)` without `exist_ok` |
| Seed must be in the frozen set | `if args.optimizer_seed not in allowed: raise` |
| Preflight isolated from fresh roots | separate scope, `BURNED_INSTRUMENT_PREFLIGHT_NO_CLAIM`, its own receipt name |

The compute-budget concern I raised before the runner existed was answered properly: the
structured comparator carries a skeleton-keyed cache and a **predeclared** budget with a hard
cap, so scope cannot be trimmed after seeing results.

## Required fix 1 — the static bar is recomputed per process, and it is expensive

`build_static_bar` enumerates the full 65,536-calendar frontier for every unique skeleton in
the evaluation histories. In development that is 16 selection roots x 3 kappa cells x 12
campaigns = **576 campaigns**, and the kappa cell changes the regime chain, so the skeleton
cache does not collapse them.

At the measured ~2.5 s per campaign that is **~24 minutes per process**, and the runner
computes it inside every process. With 5 optimizer seeds across the configuration grid this is
hours of recomputation producing a byte-identical artifact.

The contract already forbids this: `static_bar_protocol.shared_immutable_artifact_across_optimizer_seeds: true`.
The runner does not implement that clause. Beyond the waste, the clause exists for a
correctness reason — if any process ever computes the bar over a different scope, the arms are
silently graded against different references.

**Fix:** compute the bar once in a separate step, write it as the shared artifact, and have the
runner load and hash-verify it, failing closed when the hash does not match the contract's
declared bar.

## Required fix 2 — the compute budget is checked after the work, then discards it

```
rows = structured_pair_rows(...)        # the whole structured evaluation runs
elapsed = time.perf_counter() - started
if elapsed > cap:
    raise PredeclaredComputeBudgetExceeded(...)   # rows are lost
```

Two problems. The cap cannot prevent an overrun, only report one after it has been paid for.
And raising discards the rows that were just computed — the identical abort-not-continue
pattern that discarded 137 completed arms on the VPS earlier today, at 44-47 of 48 arms per
shard.

**Fix:** check the budget incrementally between units and stop cleanly, and persist the rows
already computed before raising, flagged as truncated. A budget guard should cost work it has
not yet done, never work it already has.

## Required fix 3 — checkpoint rows are held in memory until the end

`checkpoint_rows` accumulates across the whole checkpoint schedule and is only serialized when
the run completes. A death at the last checkpoint loses every earlier one, including the
structured comparator evaluation, which is the most expensive component in the run.

**Fix:** append each checkpoint's rows to disk as they are produced, and resume from what is
already there.

## One observation, not a required fix

In preflight mode `evaluation_roots == training_roots`, so the learner is graded on the
histories it trained on. That is acceptable for an instrument check and is correctly labelled
`BURNED_INSTRUMENT_PREFLIGHT_NO_CLAIM`, but the contract should say so explicitly so a later
reader cannot mistake a preflight number for a held-out one.

## Recommendation

Fix 1, 2 and 3, then freeze. I will re-audit the three fixes specifically and, if they hold,
the contract is ready to open fresh roots. Nothing in this audit requires re-opening any
scientific question; the design is right.
