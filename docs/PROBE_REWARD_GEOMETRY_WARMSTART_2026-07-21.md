# Probe spec — reward geometry + warm-start ablation (EXPLORATORY_NO_CLAIM)

**One-page discovery spec, burned/dev tapes only, no sealed seeds.** Separates the three
candidate causes of PPO's static non-convergence: (A) reward SCALE, (B) optimization drift,
(C) whether closed-loop feedback beats the best static at all. Runs in PARALLEL with the
CROSS_ECHELON densification (`PROBE_CROSS_ECHELON_DENSIFICATION_2026-07-21.md`).

## The finding that motivates the reward arm (verified in code)

`program_u_static_search.py:298` — the static PPO reward is `objective.evaluate(calendar)` =
**mean ReT over tapes**, deterministic per calendar. Two facts:

- It is the correct metric (PPO trains on exactly what it is ranked on — no objective mismatch).
- But the **scale is pathological**: reward ≈ 0.84 for the best calendar; the top-38 of 65,536
  span only 0.0025 → the learnable signal is **~0.3% of the reward level**. Delivered
  terminal-only, `gamma=1`, `gae_lambda=1`, `n_steps=8`, batch 8, **no reward normalization**.
  The policy-gradient advantage `reward − V(s)` requires V to nail 0.84 to expose the 0.0025;
  it can't, so the advantage is value-approximation NOISE → the frozen policy DRIFTS and more
  training makes it worse (Stage2→3: regret 0.00162→0.00253, entropy 0.044→0.131, rank 6→38).

## Arms (same 65,536 answer key as the gate, identical budgets, CEM as reference)

1. `baseline_ppo` — current raw reward (~0.84). The Stage-3 incumbent.
2. **`rescaled_ppo`** — affine training reward `(ReT − c)/s`, with `c,s` frozen from calibration
   (mean and std of calendar ReT), **return-order-preserving** (does not change the optimal
   calendar). Optionally add SB3 VecNormalize. This is the return-equivalent
   `standardized_terminal`.
3. `warmstart_ppo` — policy pretrained by behavior cloning to reproduce the best static
   calendar (IDENTICAL initial inventory/pipeline/demand — only logits/weights, never physical
   state, unlike Track A's `S1_I1344` prewarmup which created real stock), then RL fine-tune.
   Run under BOTH raw and rescaled reward.

## Readouts (per arm, per seed)

frozen_policy regret + exact rank; best_seen regret; **entropy trajectory** (rising = drift);
does it HOLD the optimum or DRIFT away; for warm-start, does it learn deviations that BEAT the
frozen best static.

## Diagnostic verdicts (routing, not claims)

- `REWARD_SCALE_BETRAYAL` — `rescaled_ppo` converges (low frozen rank, no drift) while
  `baseline_ppo` drifts → the reward was betraying PPO by scale. Carry the affine rescale into
  the dynamic Q training as a return-equivalent credit aid (new contract, never a re-read of Q).
- `OPTIMIZATION_DRIFT` — even `rescaled_ppo` drifts → the PPO update itself destabilizes on the
  flat top → QR-DQN (off-policy replay, distributional) is the justified next algorithm.
- `WARMSTART_HOLDS` — warm-start retains the optimum → the failure was credit/init, curable.
- **`FEEDBACK_ADDS_VALUE`** (the one that matters) — warm-started feedback RL learns deviations
  that beat the frozen best static → closed-loop value confirmed, the necessary precondition
  before any "beat the MPC" claim.
- `NO_FEEDBACK_VALUE` — never deviates advantageously → feedback adds nothing in the stationary
  env (consistent with the flat top; the win is nonstationary).

## Invariants

Primary `ret_excel` stays thesis-faithful (0/47,546) — evaluation and ranking always use raw
ReT; the affine rescale is a TRAINING-reward transform only, never the metric. Answer key
inaccessible during search (only after candidate freeze). Burned/dev tapes, no sealed seed
opened. Report `best_seen` and `frozen_policy` separately. Compute preflight (amendment 2)
before launch; save weights + seed×config matrix + JSON (do not repeat David's no-bundle run).

## Why this is the right first move

CEM already wins the static search — this ablation is NOT about beating CEM. It answers two
things cheaply: (1) is our reward silently sabotaging every RL learner (relevant to Q's
RecurrentPPO too, same ~0.84 scale + tape variance), and (2) does closed-loop feedback beat the
best static at all (`FEEDBACK_ADDS_VALUE`) — the precondition for the whole learner lane. The
win-hunt itself stays the parallel CROSS_ECHELON densification.
