# Program L(e-1) Gate 2 verdict — terminal headroom stop

**Verdict:** `STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM`  
**PPO trained:** no powered/scientific PPO run

**Virgin tapes opened:** no

## Valid evidence package

- Gate 0: `results/headroom/l_program_gate0_crn_v3_2026-07-10.json` — PASS.
- Gate 1: `results/headroom/l_program_gate1_crn_v3_2026-07-10/` — 60
  materialized calibration tapes, 18 static configurations and six observable
  heuristics.
- Gate 2: `results/headroom/l_program_gate2_crn_v3_2026-07-10/` — 100 states
  per buffer, three actions per state, four-week branches, GroupKFold by tape
  and a depth-four observable tree.

The unsuffixed and `crn_v2` result directories contain explicit
`INVALIDATED.md` notices. They are provenance records, not evidence.

## Why v3 is the first valid contract

Two confounders were removed before the final gate:

1. Risks are generated once after a risk-free warm-up, expressed relative to
   the treatment boundary, hashed, and replayed with endogenous risk generators
   disabled.
2. Every policy begins from the same S1 warm-up state within a buffer. S2/S3 are
   requested only through the weekly action and become effective one week
   later. The treatment begins at endogenous physical warm-up completion, so no
   policy consumes a different prefix of the demand RNG.

Tests verify that S1 and S3 policies starting from the common S1 state receive
identical post-warm-up demand and risk sequences.

## Gate 1

Integrity:

- 1,440 rows = 60 tapes x (18 statics + 6 heuristics);
- 60 unique materialized tape hashes;
- every row is calibration-only and has `initial_shift=1`;
- panel SHA256:
  `b6dbb1956a52baacc1df52ef21bcb556bb459d14d073012e22897fe6f12b5a5b`.

The unconstrained selector returned `I1344/S1`, tied exactly with S1 at every
other positive buffer:

- mean Excel ReT: `0.1099880856`;
- mean service-loss AUC: `67,285,878.27` ration-hours;
- extra shift-hours: `0`.

For every positive buffer, S1/S2/S3 produced identical mean Excel ReT and
service-loss AUC; extra shift-hours were 0, 17,304 and 34,608, respectively.
The observable heuristic also produced the same ReT/service loss while using
about 19,858 extra shift-hours. Therefore it is dominated.

B0 is the only distinct boundary: S2 modestly improved both ReT and service loss
relative to S1/S3, but B0 is excluded from the required non-extreme-buffer
promotion count.

## Gate 2

Prefix observations were asserted exactly equal before all 1,800 branches.
The outcome by buffer was:

| Buffer | States with loss effect | States with ReT effect | Cross-fitted tree result |
|---:|---:|---:|---|
| 0 | 96/100 | 92/100 | service worsened `-0.0401%`; resources `-5.25%`; fail |
| 168 | 0/100 | 0/100 | exact null; fail |
| 336 | 0/100 | 0/100 | exact null; fail |
| 504 | 0/100 | 0/100 | exact null; fail |
| 672 | 0/100 | 0/100 | exact null; fail |
| 1344 | 0/100 | 0/100 | exact null; fail |

For B0, the optimal-action labels varied (S1 43%, S2 45%, S3 12%), but the
observable depth-four tree did not convert that clairvoyant heterogeneity into
out-of-tape value. Its service-loss reduction CI95 was entirely negative:
`[-0.0738%, -0.00915%]`.

For every positive buffer the three branches were exactly identical in
four-week late-backlog AUC and final Excel ReT. There is no action variation for
an observable learner to recover.

Hashes:

- branch rows:
  `3b16d9fcb6e66c74c91ae1af836bb4c1af8f9e571786435ff8be9f5deaafea48`;
- verdict:
  `2390ffd73bbf3d1643387eb2fe64ab02d76e8624bd6955a0e4e94979a7bfb116`.

## Scientific decision

Gate 2 is terminal under the preregistration. Do not run Gate 3, tune PPO,
lengthen the branch horizon, alter the buffer clock, or add an architecture to
rescue Program L v1. The permitted conclusion is:

> In the Garrido-grounded fixed-buffer/weekly-shift contract, no deployable
> short-horizon shift headroom is observed at any tested positive buffer, while
> the no-buffer case contains local action heterogeneity that an observable
> cross-fitted tree cannot exploit. Powered retained-learning experiments are
> therefore not scientifically authorized on this decision surface.

This boundary does not retract Track B, which controls a different downstream
action surface.
