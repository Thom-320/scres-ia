# Program S implementation status — amended v1.1 — 2026-07-17

## Current verdict

`PASS_S1_PREOPEN_AUTHORIZED_POST_Q_TERMINAL`

S0 remains frozen under the parent v1 contract:

- `PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL`;
- `PASS_S0_HOT_PATH_RISKOFF_PARITY_V1_1` on 18 burned episodes;
- `PASS_S1_TRANSDUCER_PREFLIGHT_ALL_MASKS_ELIGIBLE`.

Program Q opened and terminally adjudicated `7490001–7490256`. Program S intervals
751, 752 and 753 remain unopened.

## Binding v1.1 changes before S1

- S-NATIVE is the only primary/promotable physical screen.
- S-WARTIME is separated and hypothesis-generating; it has no seed block and cannot be an automatic fallback.
- Anticipatory alarms are forbidden in S-NATIVE.
- Program S-P is a separate, unseeded annex with a risk-family-specific, binned and noisy alarm generator.
- R23 retains deterministic physical liveness but is fixed at multiplier 1.0 as a negative control and cannot select a cell.
- Native S1 stops before S2 when `max LCB95(H_PI_safe) < 0.01`.
- Program Q is terminal. Its VPS precedence is discharged; Program S may proceed only under the fresh post-Q audit.
- S terminal labels describe risk-aware adaptation rather than claiming Paper 2 by label alone.

## Compute and design

The amended Morris design contains:

- 3 native groups, 30 trajectories, 160 points;
- capacity fixed at 1.0 inside Morris;
- 0.9/1.1 reserved as discrete post-selection stability anchors;
- 3 product-regime cells and 12 tapes;
- 5,760 projected native shards.

The burned-tape benchmark measured 1.91–3.22 seconds per complete 65,536-calendar shard. With a frozen 1.25 safety multiplier and two workers, projected wall time is approximately 11,582 seconds (3.22 hours), below the seven-day cap:

`PASS_S1_COMPUTE_BENCHMARK_FEASIBLE`

## Instrument corrections retained

The first hot-path audit used the rounded literal `2.22e-16`, slightly below exact binary64 epsilon. It stopped technically. The identical burned inputs were repeated with `numpy.finfo(float).eps`; max difference was exactly one epsilon and risk-off rework remained zero in 18/18 episodes. Both artifacts are retained.

The first compute preflight exposed that capacity had accidentally been generated as a continuous Morris coordinate despite the discrete contract. No timing episode or scientific seed ran. The design was corrected prospectively before 751: capacity is now fixed at 1.0 inside Morris.

## Post-Q authorization

The post-Q numeric-range auditor excludes burned 749 from virginity claims and
audits only the still-reserved S namespaces. The fresh S1 preopening audit binds
the terminal Q adjudication and authorizes S1 without changing any S gate.

The post-Q execution harness enumerates exactly 5,760 unique
`group × trajectory × point × product-cell × tape` shards, fixes two recycled
workers, starts an independent watcher before the producer, records PID/PGID,
RAM, stderr and atomic progress, refuses overwrite, and permits only an explicit
fail-closed resume over missing identities.

## Post-opening audits

The integrated exact-transducer suite initially failed closed because four new
R24 incidence counters in the shared simulator hot path were not assigned a
scientific state role. They are output/accounting ledgers, are never read by the
frozen bottleneck transition, and key-v3 conservatively serializes them anyway.
They were explicitly classified and reaudited without changing the running S1
source or any historical result: the exact-transducer/full-frontier suite passes
`65 passed, 1 skipped`.

The original `--resume` implementation detected missing shards but could not
resume after a normal failure receipt, because the immutable
`producer_exit.json` blocked re-entry. A prospective recovery-only harness now
uses append-only `resume_attempts/attempt-NNN` custody directories, validates
every preserved NPZ before reuse, refuses completed runs, and verifies that all
scientific files are byte-identical to the original source commit. This does not
modify or restart the healthy attempt already running from `adc0056`.

## Live execution update — 2026-07-18

S1 opened `7510001–7510012` once at `2026-07-18T15:34:38Z` on
`ovh-agent-lab`, source commit `adc0056`, after the post-Q authorization. The
watcher is active, the run is partial, and no scientific verdict exists. The
authoritative live seed registry is now v1.2; the v1.1 preopening registry and
audit remain immutable historical evidence.

Paper 3 remains sealed until a risk-aware S4 PASS, and its formerly open-ended namespace is bounded to `7530001–7539999`.
