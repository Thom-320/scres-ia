# Program L(e-1) preregistration — implementation freeze

**Status:** design frozen before any powered Program-L PPO run. Pipeline smokes
using at most two campaigns and three fixed probes are software tests, not
scientific evidence.

**Terminal update:** Gate 2 v3 returned
`STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM`. By the sequential rule below, Gates 3-5
are not authorized for Program L v1.

The governing contract is `configs/garrido_learning_v1.json`; the identification
rules are in `docs/CLAIM_AND_IDENTIFICATION_2026-07-10.md`.

## Sequential gates

1. Gate 0: runtime, proxy hash, causal invariants, CRN, metric and observation
   checks.
2. Gate 1: 18 static policies plus the fixed observable heuristic on 60
   calibration tapes.
3. Gate 2: 100 replayed states per buffer, S1/S2/S3 four-week branches, tree
   depth 4, GroupKFold by tape.
4. Gate 3: PPO pilot only if Gate 2 promotes.
5. Gate 4: retained/reset/frozen/matched-compute confirmatory comparison.
6. Gate 5: same campaign multiset in three different orders.

No later gate may run when an earlier gate returns a stop verdict.

## Tape separation

| Universe | Seed base | Permitted use |
|---|---:|---|
| Calibration | 700000 | statics, normalizer, reward scales, heuristic, theta0 |
| Training | 710000 | PPO updates |
| Fixed probes | 720000 | checkpoint curves, no updates |
| Virgin confirmatory | 730000 | one final confirmatory opening |
| R3 stress | 740000 | OOD evaluation only |

Opening a later universe early invalidates that evidence level and requires a new
seed block plus a dated amendment.

## Algorithms and arms

PPO uses a categorical three-action policy, MLP actor/critic `2x64`, tanh,
calibration-fixed observation normalization and no reward normalization.

- `frozen`: calibrated theta0, no campaign updates;
- `persistent_weights`: actor/critic retained, optimizer moments cleared;
- `reset_local`: theta0 restored before every campaign;
- `persistent_full`: weights and optimizer retained, technical sensitivity;
- `scratch_matched_onpolicy`: theta0 plus identical campaign multiset and update
  count in a canonical balanced order; it generates new on-policy rollouts.

The last arm is not called cumulative-scratch and never reuses PPO rollout files as
offline training data.

## Primary inference

Primary contrast: `persistent_weights - reset_local` at the frozen final
checkpoint. Inference uses a seed-by-probe matrix, two-way bootstrap, paired seed
signs, and a mixed-effects sensitivity. Weeks are never treated as independent.

Retained learning is promoted only if:

- Excel ReT CI95 lower bound > 0;
- service-loss AUC reduction and its CI95 lower bound are at least 5%;
- shift-hours differ by no more than 2% or persistence is Pareto non-dominated;
- the effect does not require optimizer or normalization persistence.

H4 path dependence is separate and non-directional. It requires reproducible
differences among three orders of the same 24-campaign multiset.

## Frozen exclusions

No KAN, RNN, SAC, TD3, MARL, predictor, buffer switching, prevention,
anticipation, attribution retuning, dynamic normalization, or post-virgin
hyperparameter changes belong to v1.
