# Program Q terminal verdict — 2026-07-18

## Binding adjudication

`STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION`

This is the frozen adjudicator's compound label. It does **not** mean that the
ReT superiority endpoint failed. Program Q independently replicated the
learner's advantage over the complete 65,536-calendar open-loop frontier in all
three cells, and the bilateral equivalence test against the best classical
controller passed in all three cells. The compound claim failed because the
preregistered worst-product-fill non-inferiority guardrail did not pass against
the best classical controller.

Historical records remain unchanged:

- Program O: `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`.
- Program O-R: `STOP_CALIBRATION_NOT_ELIGIBLE` at scientific commit `821c8d8`.
- The old block `7480101-7480148` remains sealed forever.

## Primary estimates

The confirmation used 256 new tapes per cell, ten frozen historical
RecurrentPPO checkpoints, two-way learner-seed/tape studentized max-t inference,
10,000 resamples, complete comparator reselection, and common random numbers.

| Cell | H_OL point | simultaneous LCB95 | Delta_N point | bilateral simultaneous CI95 | favorable tapes | positive learner seeds |
|---|---:|---:|---:|---:|---:|---:|
| `rho75_share90` | 0.07952 | 0.06608 | -0.00159 | [-0.00627, 0.00310] | 84.77% | 10/10 |
| `rho90_share75` | 0.07255 | 0.06233 | -0.00072 | [-0.00552, 0.00408] | 89.84% | 10/10 |
| `rho90_share90` | 0.11724 | 0.10614 | -0.00041 | [-0.00268, 0.00186] | 95.70% | 10/10 |

Thus:

- replicated learned ReT value versus every full-horizon open-loop calendar: **yes**;
- neural premium of at least 0.01 over the best classical controller: **no**;
- practical equivalence within +/-0.01 to the best classical controller: **yes**;
- eligibility under the complete frozen Program Q contract: **no**.

## Binding guardrail failure

Against the best classical controller, worst-product-fill contrasts were:

| Cell | point | simultaneous LCB95 | frozen margin |
|---|---:|---:|---:|
| `rho75_share90` | -0.01036 | -0.02266 | -0.02 |
| `rho90_share75` | -0.01573 | -0.02566 | -0.02 |
| `rho90_share90` | -0.00451 | -0.02632 | -0.02 |

The lower bound crossed the frozen margin in every cell. The ReT, feedback,
replacement, scheduled-resource, demand, mass, partition, full-ledger ReT, and
quantity-ReT gates passed. `ret_full` and `quantity_ret_full` were exact
deterministic zero contrasts under this evaluator, not noisy estimated effects.

## Direct audit and custody

- 768/768 immutable shards are present and verified by their SHA-256 manifest.
- The full remote manifest verified 789/789 listed files.
- Independent full-DES audit: 21,696 unique replays, zero failures.
- Maximum ReT replay error: `7.771561172376096e-16`.
- Evaluation result SHA-256:
  `62f6fd390471624f7c301b8baa96d31871db99e22dd5a22d6bb8cf7bba8088b2`.
- Direct audit SHA-256:
  `3da52ca129707e883be0179f82be8058d29ddf454c27a4f578918c26c7ec82eb`.
- Adjudication SHA-256:
  `e13e17f001a1d24f86f00257e145c26f9c09def68ef7b2ee2f90fcb23148b0e9`.
- Full remote manifest SHA-256:
  `f5858915e068199ddce54104aa7584a158cc495b47a85ebfa502dd7f020d5e1e`.

The raw 829 MiB custody remains immutable on `ovh-agent-lab`. A slim, reviewable
copy of the result, adjudication, direct audit, manifests, watcher, logs, and
Fable audit is stored under
`results/program_q/confirmation_v1_20260718/`. No shard was regenerated.

## Publication boundary

Program Q supports reporting a preregistered decomposition:

> RecurrentPPO independently replicated genuine state-dependent ReT superiority
> over every full-horizon open-loop calendar and was practically equivalent to
> the best structured controller. The neural premium was absent, and the frozen
> worst-product service non-inferiority criterion was not met.

It does not authorize `PASS_Q_*`, deployment safety, neural superiority, or
Paper 3 under the frozen Program Q gate. Any subsequent fairness-constrained or
learning-augmented controller is a new prospective program with new training
and evaluation tapes; it cannot rewrite this terminal result.
