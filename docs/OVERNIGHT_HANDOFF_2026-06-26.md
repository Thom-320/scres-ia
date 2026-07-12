# Overnight Autonomous Run — Handoff / Coordination (2026-06-26 night)

Claude (this session) is running the Track-A learning pipeline autonomously overnight under
`docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md`. This note records what is running, what is frozen, and
**what NOT to touch** so the parallel Codex session does not collide.

## Decision recap (user, 2026-06-26)
Pivot accepted: **DQN retained-vs-reset** is the primary lane (not PPO); **ReT is an OUTCOME, not a
training reward**; keep **two envs** (faithful + fine-tuned headroom); full **complete-win + two
partial-wins** definition (Contract v2 §7). Most-promising paper track = the **decision-frontier**
claim, with the **operational-inertia (Ed.2) moderator** as the strong result: *accumulated memory
improves resilience only where a decision frontier exists; operational inertia makes memory
measurably valuable.*

## Files THIS session owns tonight (do not edit in Codex)
- `docs/EXPERIMENT_CONTRACT_V2_2026-06-26.md`, `docs/OVERNIGHT_HANDOFF_2026-06-26.md`
- `docs/LEARNING_ENVIRONMENT_CONTRACT_v1.md` (to be written when the frontier env is frozen)
- `docs/TRACK_A_RESULTS_*` (morning)
- `scripts/calibrate_headroom_env.py`, `scripts/calibrate_decision_frontier.py`,
  `scripts/reward_alignment_static_surface.py`, `scripts/run_learning_confirmatory*.py` (new)
- `scripts/retention_transfer.py`, `scripts/evaluate_retained_reset_learning.py` (runner + helper)
- `outputs/experiments/headroom_calibration_2026-06-26/`, `outputs/experiments/frontier_*`,
  `outputs/benchmarks/retention_transfer/*`

## Files Codex owns (THIS session will NOT edit)
`supply_chain/{supply_chain.py, ret_thesis.py, env_experimental_shifts.py, config.py}` and the
Garrido fidelity audits (`docs/GARRIDO_*_2026-06-26.md`, `scripts/audit_garrido_*`). Because of
this, the reward gate uses **only already-wired reward modes** (`control_v1`, `control_v2`,
`ReT_cd`, `ReT_garrido2024`) — no new reward mode is added tonight (that would touch
`env_experimental_shifts.py`). `service_first_v2` from the review is deferred to a coordinated edit.

## Pipeline stages (gated; each writes JSON, next reads it)
- **A. Headroom φ/ψ screen (severe)** — `calibrate_headroom_env.py`, RUNNING. Quick screen only.
- **B. Decision-frontier calibration (multi-regime, static-only)** —
  `calibrate_decision_frontier.py`. 18 statics × {current,increased,severe} × seeds across finalist
  (φ,ψ) cells. Score = oracle−robust gap on **bounded `ret_continuous`** + argmax regime-diversity +
  off-saturation + corner-free. **Never uses retained−reset to pick the env** (anti outcome-shopping).
- **C. Freeze winner** → `LEARNING_ENVIRONMENT_CONTRACT_v1.md` + `supply_chain/data/*.json`.
- **D. Reward gate (static surface, no RL)** — `reward_alignment_static_surface.py`: Spearman(reward,
  exact ReT) ≥ ~0.6, reward's best static within ~0.02 ReT of ReT's best, not always I1344/S3, not
  always S1, Pareto-dominated actions not in top-3. Pick the most parsimonious passing reward.
- **E. DQN hyperparam calibration (5×5)** — select by **retained−frozen** (total learning value),
  reserving retained−reset for confirmatory.
- **F. Confirmatory (10×10 held-out)** — `retention_transfer.py`, faithful + frozen-headroom envs.
- **G. Ed.2 inertia moderator** — same DQN/reward/seeds/tapes with `--surge-inertia`; test
  Δmemory(v2) > Δmemory(v1).
- **H. Morning docs** — `TRACK_A_RESULTS_v1.md` (+ `_v2_INERTIA.md`).

## Metric note (important)
The review's absolute frontier thresholds (oracle ReT ∈ [0.55,0.95], gap ≥ 0.03) were calibrated to
an earlier env epoch. Under the current faithful env (delay=54, every order late) even
`ret_continuous` tops out ≈ 0.25, so absolute bands are not directly comparable. Tonight the frontier
is judged by **relative** criteria (gap and off-saturation vs the faithful φ=ψ=1 baseline) on
`ret_continuous`, with `ret_excel` reported alongside as the primary outcome bar. Flagged in results.
