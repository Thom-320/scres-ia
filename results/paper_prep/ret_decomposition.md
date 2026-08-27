# ReT decomposition (preregistered) — results

Generated: 2026-08-27T06:24:21.030649+00:00 · script: `scripts/paper_prep/ret_decomposition.py`
Preregistration: `contracts/paper_prep/ret_decomposition_preregistration_v1.json` (FROZEN_BEFORE_ANALYSIS)

## Mechanistic finding (declared)

- Program Q confirmation physics is **risk-off**: every skeleton carries `risk_events = []`, so **no visible order of any arm on any tape is risk-active**.
- All visible rows score the **excel_fill_rate** branch, so the branch share vectors are identical across arms. The composition component is then **computed** from those share vectors and **checked** against zero — falsifier **F3** is evaluated, not assumed.
- The entire Δ falls in the **intra-regimen** component. This is a mechanistic property of the environment, not an implementation error.

| cell | H_PI (safe) | H_OL | Δ_N | comp (vs OL) | intra (vs OL) | comp (vs CL) | intra (vs CL) |
|---|---|---|---|---|---|---|---|
| rho75_share90 | 0.15151 | 0.07952 | -0.0015852 | 0.0 | +0.0795238 | 0.0 | -0.0015852 |
| rho90_share75 | 0.15151 | 0.07255 | -0.0007246 | 0.0 | +0.0725474 | 0.0 | -0.0007246 |
| rho90_share90 | 0.15151 | 0.11724 | -0.0004104 | 0.0 | +0.1172400 | 0.0 | -0.0004104 |

## Verification

- Shard custody: **768/768 SHA-256 verified** against `shards/{cell}/shard_files.sha256`.
- Bit-exact replay: open-loop frontier, classical ten and learner calendars reproduce every shard scalar with diff 0.0 (F2).
  - rho75_share90: learner 0.00e+00, open-loop 0.00e+00, classical 0.00e+00
  - rho90_share75: learner 0.00e+00, open-loop 0.00e+00, classical 0.00e+00
  - rho90_share90: learner 0.00e+00, open-loop 0.00e+00, classical 0.00e+00
- Reference anchors: H_OL and Δ_N identities hold against `result.json::inference.estimates` to 1e-12; best comparators match `cell_summaries`.

## Branch composition (visible rows)

| cell | arm | fill_rate | autotomy | recovery | risk_no_recovery |
|---|---|---|---|---|---|
| rho75_share90 | learner_per_seed | 120399 | 0 | 0 | 0 |
| rho75_share90 | open_loop_all_calendars | 667591430 | 0 | 0 | 0 |
| rho75_share90 | classical_ten | 120509 | 0 | 0 | 0 |
| rho90_share75 | learner_per_seed | 121292 | 0 | 0 | 0 |
| rho90_share75 | open_loop_all_calendars | 677366631 | 0 | 0 | 0 |
| rho90_share75 | classical_ten | 121553 | 0 | 0 | 0 |
| rho90_share90 | learner_per_seed | 120855 | 0 | 0 | 0 |
| rho90_share90 | open_loop_all_calendars | 606903783 | 0 | 0 | 0 |
| rho90_share90 | classical_ten | 120985 | 0 | 0 | 0 |

## Bootstrap (descriptive, Delta scale)

- 10000 two-way resamples (learner seeds × tapes), comparators reselected inside every resample; studentized max-t across the six estimands.
- Simultaneous t_0.95 = 3.0648; RNG = SHA256('paper-prep-ret-decomposition-v1')[:8].

| estimand | estimate | SE | LCB95 | UCB95 |
|---|---|---|---|---|
| rho75_share90::learner_vs_openloop | +0.0795238 | 0.0045582 | +0.0655536 | +0.0934940 |
| rho75_share90::learner_vs_bestclassical | -0.0015852 | 0.0015728 | -0.0064056 | +0.0032351 |
| rho90_share75::learner_vs_openloop | +0.0725474 | 0.0034562 | +0.0619547 | +0.0831400 |
| rho90_share75::learner_vs_bestclassical | -0.0007246 | 0.0015940 | -0.0056099 | +0.0041607 |
| rho90_share90::learner_vs_openloop | +0.1172400 | 0.0038293 | +0.1055039 | +0.1289761 |
| rho90_share90::learner_vs_bestclassical | -0.0004104 | 0.0007624 | -0.0027470 | +0.0019263 |
