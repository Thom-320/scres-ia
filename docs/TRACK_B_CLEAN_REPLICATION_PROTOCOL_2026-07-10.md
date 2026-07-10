# Track B clean-joint replication — frozen protocol

Status: **frozen before inspecting any tape in the 500001–500060 range**.

## Scientific question

Does the 5-seed factorial `joint` PPO bundle — the internally consistent
arm (100% observation-version v7 52-dim, `norm_reward=True`,
`net_arch=[64,64]`, 60k steps, `control_v1`, full 8-D track_b_v1 contract)
that showed a post-hoc +9.6e-6 mean ret_excel advantage over the frozen
best full-contract static policy on tapes 400001–400060 — retain that
advantage on a held-out tape battery that neither the checkpoints nor the
human have ever inspected?

This is a replication test, not a new training experiment. The existing
frozen checkpoints are evaluated as-is on fresh tapes.

## Why this test

The same-contract challenge verdict (`TRACK_B_SAME_CONTRACT_CHALLENGE_VERDICT_2026-07-10.md`)
correctly retired the canonical 10-checkpoint bundle's adaptive advantage.
However, the fresh factorial `joint` arm (5 seeds, 52-dim v7 throughout)
cleared the static comparator by +9.6e-6 on 400001–400060 (4/5 seeds
positive, 58/60 tapes positive). That result was computed post-hoc after
inspecting the tapes, so it cannot be promoted without a confirmatory test.

Two competing explanations:
1. The gap is real — the canonical bundle's failure was caused by
   observation-dimension heterogeneity (seeds 1–5 trained on 48-dim v7,
   seeds 6–10 on 52-dim v7), not by absence of adaptive value.
2. The gap is a post-hoc artifact — overfitting to the 400001–400060 tape
   set by selective interpretation.

These hypotheses are distinguished by evaluating the same frozen
checkpoints on completely fresh tapes.

## Selection and test separation

- The 5 joint checkpoints were trained by
  `scripts/run_track_b_contract_factorial.py --arms joint --seeds 1 2 3 4 5`
  in `outputs/experiments/track_b_factorial_joint_2026-07-09/` (the same frozen
  checkpoints evaluated by the same-contract challenge).
  They were frozen before any tape in 400001–400060 or 500001–500060 was
  opened.
- Calibration / selection tapes for the static comparator: 300001–300024
  (already used, unchanged).
- Prior held-out tapes: 400001–400060 (already inspected; burned for
  confirmatory purposes).
- **Confirmatory test tapes: 500061–500120 only.** No member of this range
  has been generated, inspected, or used for any selection or hyperparameter
  choice in any prior experiment. (Tapes 500001–500060 were used for a
  2-tape smoke test of the evaluation harness before the confirmatory run
  and are therefore excluded; see Provenance note below.)
- Primary endpoint: per-episode Garrido/Excel ReT (`ret_excel`).
- Secondary endpoint: `ret_excel_cvar05`.

## Frozen checkpoint provenance

Source: `outputs/experiments/track_b_factorial_joint_2026-07-09/models/joint/seed{1,2,3,4,5}/ppo_model.zip` + `vec_normalize.pkl`.

Training config (identical across all 5 seeds):
- `reward_mode`: `control_v1`
- `observation_version`: `v7` (52-dim)
- `risk_level`: `adaptive_benchmark_v2`
- `step_size_hours`: 168.0, `max_steps`: 104
- `learning_rate`: 3e-4, `n_steps`: 1024, `batch_size`: 256, `n_epochs`: 10
- `gamma`: 0.99, `gae_lambda`: 0.95, `clip_range`: 0.2
- `net_arch`: [64, 64] (shared pi/vf), `norm_reward`: True
- `train_timesteps`: 60,000 per seed
- `norm_obs`: True, `clip_obs`: 10.0
- Device: CPU

## Frozen static comparator

Unchanged from the same-contract challenge:
`outputs/experiments/track_b_static_contract_search_2026-07-10/frozen_static_policy.json`.
Selected by calibration-only Sobol search on tapes 300001–300024, frozen
before any test tape was opened. Calibration ret_excel: 0.005944627.

## Confirmatory contrast and stop rule

Primary contrast: factorial `joint` PPO (5 frozen checkpoints) minus frozen
best full-contract static policy, on tapes 500061–500120.

Inference: two-way bootstrap independently resampling checkpoint/training-seed
and tape axes (10,000 iterations, seed=0).

**PASS** (gap replicates; reopen adaptive-advantage case) if and only if ALL:
1. Two-way CI95 lower bound for (joint − static) ret_excel mean delta is
   strictly above zero.
2. At least 4 of 5 seed-level means are positive.
3. At least 50 of 60 tape-level means are positive.

**FAIL** (gap does not replicate on fresh tapes) otherwise.

If PASS: the post-hoc +9.6e-6 was a real held-out signal. Promote to a
pre-registered finding and reopen the adaptive-advantage case with a
cleanly described mechanism (protocol consistency + decision-contract
alignment).

If FAIL: the +9.6e-6 was a post-hoc artifact. Proceed to Paso 1 (reward
alignment with fresh training) as specified in the broader plan. The
same-contract challenge verdict stands.

No optimization, model selection, or hyperparameter tuning follows this
gate on these tapes.

## Provenance note

Tapes 500001–500002 were used for a smoke test of the evaluation harness
(`scripts/evaluate_track_b_clean_replication.py`) before the confirmatory
run. Their results were visible to the operator, so tapes 500001–500060 are
excluded from the confirmatory battery. The confirmatory battery is
500061–500120, which has not been inspected in any form.
