# Track B same-contract static challenge — frozen protocol

Status: **frozen before inspecting any tape in the ranges below**.

## Scientific question

Does the canonical joint PPO retain an out-of-sample Garrido/Excel ReT
advantage over (i) a calibration-only optimized constant policy spanning the
full eight-dimensional Track B contract and (ii) an otherwise matched learned
upstream-and-shift controller anchored at the strongest known fixed dispatch
posture, Op10 `2.0x` and Op12 `1.5x`?

## Selection and test separation

- Calibration tapes: `300001–300024` only.
- Final test tapes: `400001–400060` only.
- Tapes `200001–200060` are historical evidence and may not be used for
  selection or final confirmation here.
- Primary endpoint: per-episode Garrido/Excel ReT (`ret_excel`).
- Secondary endpoint: `ret_excel_cvar05`; the operational panel remains
  descriptive and direction-checked.

## Finite static optimization

1. One scrambled 128-point Sobol global screen over seven continuous Track B
   signals and the three-level shift decision, plus the prespecified dense-
   frontier winner. Evaluate on tapes `300001–300012`.
2. One refinement around the eight best screen points: the incumbent, plus or
   minus `0.15` in each continuous dimension, and alternative shift levels.
   Deduplicate and evaluate on all tapes `300001–300024`.
3. Freeze the highest calibration-mean `ret_excel` candidate. No restarts,
   additional optimizers, test-set tuning, or post-test refinement.

## Learned anchored arm

Train five PPO seeds for 60,000 steps using the corrected factorial settings,
while fixing action dimension 6 to `1.0` (Op10 `2.0x`) and dimension 7 to
`1/3` (Op12 `1.5x`). Compare it with the already-frozen five factorial joint
checkpoints on identical final tapes.

## Confirmatory contrasts and stop rule

Use a two-way bootstrap that independently resamples checkpoint/training-seed
and tape axes.

1. Canonical joint PPO (10 frozen checkpoints) minus frozen best full-contract
   static policy.
2. Factorial joint PPO (5 frozen checkpoints) minus
   `upstream_shift_best_dispatch` (5 matched training seeds).

Paper 1 closes experimentally with the IJPR → C&IE → SMPT ladder only if both
95% two-way confidence intervals are wholly above zero and seed-level mean
directions are positive. If the static full-contract policy matches PPO or the
anchored dispatch increment is not positive, retire the bottleneck/adaptive-
advantage claim and pivot to a benchmark/action-contract-design contribution
for C&IE or SMPT. No further optimization follows this gate.
