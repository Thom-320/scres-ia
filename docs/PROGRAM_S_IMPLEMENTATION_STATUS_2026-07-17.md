# Program S implementation status — amended v1.1 — 2026-07-17

## Current verdict

`HOLD_S1_TECHNICALLY_READY_PROGRAM_Q_HAS_VPS_PRIORITY`

S0 remains frozen under the parent v1 contract:

- `PASS_S0_RISK_ADAPTER_LIVE_AND_RISKOFF_IDENTICAL`;
- `PASS_S0_HOT_PATH_RISKOFF_PARITY_V1_1` on 18 burned episodes;
- `PASS_S1_TRANSDUCER_PREFLIGHT_ALL_MASKS_ELIGIBLE`.

No Program Q/S reserved seed in intervals 749, 751, 752, or 753 has been opened.

## Binding v1.1 changes before S1

- S-NATIVE is the only primary/promotable physical screen.
- S-WARTIME is separated and hypothesis-generating; it has no seed block and cannot be an automatic fallback.
- Anticipatory alarms are forbidden in S-NATIVE.
- Program S-P is a separate, unseeded annex with a risk-family-specific, binned and noisy alarm generator.
- R23 retains deterministic physical liveness but is fixed at multiplier 1.0 as a negative control and cannot select a cell.
- Native S1 stops before S2 when `max LCB95(H_PI_safe) < 0.01`.
- Program Q is the current primary Paper 2 instrument. S cannot open scientific seeds while Q keeps VPS priority.
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

## Remaining hold

The unified Q/S numeric-range seed auditor passes. The amended S1 preopening audit still withholds authorization solely while Program Q remains `FROZEN_POWER_PASS_N_256_PENDING_SEED_AUTHORIZATION` and retains VPS priority.

Paper 3 remains sealed until a risk-aware S4 PASS, and its formerly open-ended namespace is bounded to `7530001–7539999`.
