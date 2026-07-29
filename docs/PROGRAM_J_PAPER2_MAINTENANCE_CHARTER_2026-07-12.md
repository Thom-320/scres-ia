# Paper 2 — Adaptive maintenance on Op5–Op7 (the pre-committed "second family")

**Status: EXECUTING (charter frozen at run time, not after).** This is the designated second
family named in `PROGRAM_G_VERIFICATION_GATES §E` (Op5–Op7 disaggregated + finite repair crew +
condition-based hazard). It is the one lane that introduces the structural condition every closed
lane lacked: a **persistent, observable, controllable degradation state** with **intertemporal
opportunity cost** through a **single shared crew**, which makes R11 **endogenous** (degradation
decides whether an exogenous threat becomes a failure).

## What is genuinely new vs the closed lanes (D, DRA, E, F, G, H, I)

Every closed lane had H_PI ≈ 0 or ~1e-5 (no physical authority) **or** H_PI > 0 that never converted
to observable/learnable advantage. This lane is the first with **real, non-trivial clairvoyant
headroom**: on the strongest cell the open-loop clairvoyant oracle beats the best static periodic
calendar by ~2% of service loss, CI excludes zero. That makes an RL convertibility test *meaningful*
here (in Program E/I, H_PI≈0 made RL pointless a priori).

## Physics (stylized, disclosed, thesis-anchored) — `supply_chain/maintenance.py`

- 3 serial stations Op5→Op6→Op7 with finite WIP buffers (block/starvation).
- Per-station degradation `d_i ∈ [0,1]`, persistent; grows with exogenous wear (heterogeneous
  across stations) + utilization; reduced by PM by `pm_efficacy`.
- Weekly action = give the single 24 h crew to PM5 / PM6 / PM7 (crew always used).
- **R11 endogenous**: an exogenous threat `threat[w,i]` (thesis window: current 168 h / increased
  42 h) becomes a **corrective failure iff `d_i > TAU=0.4`**; a corrective repair **preempts the
  crew** (that week's PM is skipped) — the intertemporal opportunity cost.
- CRN: `wear`, `threat`, `demand` are fixed per tape; only *realized* damage depends on the action.
  Env and oracle both call the single shared `week_step` → identical dynamics.
- Primary screen metric: service loss (unmet weekly demand). ret_excel port is a later confirmation.

## Comparators (all evaluated on the SAME sealed holdout tapes)

- **Static**: best periodic calendar (period ≤ 6 over {PM5,PM6,PM7}), chosen **in-sample on
  calibration** (conservative — biases H_obs *against* the learner).
- **Clairvoyant oracle**: exact min over 3^weeks PM sequences (upper bound on any policy) → H_PI.
- **Observable heuristics** (fair, non-privileged obs): worst-observed-condition; predictive
  forecast policy (noisy 1-wk-ahead threat forecast — the project's hypothesized missing signal).
- **Learner**: PPO (MlpPolicy 2×64), 6 seeds, 120k steps, trained on calibration seeds, deterministic
  eval on holdout. Observation = [noisy condition×3, noisy 1-wk threat forecast×3, wip×2, week/H] —
  **no privileged access to true degradation or future threats**.

## Frozen tape universes

- Calibration (training + static selection): seeds `5300001..5300160`.
- **Sealed holdout** (evaluation only, never used for any tuning): seeds `5400001..5400120`.

## Decision rule (anti-p-hacking, binding)

- **H_obs = static − learner** on holdout. Report the full CI and **every seed**.
- **Convertible** ⇔ at least one seed's H_obs LCB95 > 0 **and** it does not increase lost demand at
  any node. Only then does Paper 2 proceed to belief-MPC / RecurrentPPO and virgin tapes.
- If no seed converts, the result is the honest close of the strongest lane: real clairvoyant
  authority that adaptive control **cannot** capture from fair observations — the central finding's
  hardest test. No metric switch, no risk inflation, no seed cherry-picking to rescue it.
- Note already visible before RL: the clairvoyant ceiling (~2%) is **below** the preregistered 5%
  service-practicality gate, and both observable heuristics convert **negatively**. RL is run as the
  definitive convertibility test, not because a win is expected.

## Result artifact

`results/paper2/rl_convertibility.json` — single source of truth (H_PI, H_obs per heuristic, PPO
per-seed H_obs + CI, any_seed_beats_static). Charter frozen before reading the PPO seeds.

## Open Garrido / ChatGPT-Pro doubt (flag to PI)

The degradation efficacy / hazard `TAU` and wear ranges are **chosen physics**. They are calibrated
to the thesis R11 window but not yet signed off by Garrido. If Paper 2 were to convert, a Garrido
sign-off on these ranges would be the critical validity gate before any managerial claim. Given the
pre-RL evidence (negative observable conversion, sub-5% ceiling), this is currently a robustness
note, not a blocker.
