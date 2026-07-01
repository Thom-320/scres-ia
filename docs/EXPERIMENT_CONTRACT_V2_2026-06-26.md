# Experiment Contract v2 (2026-06-26) — SINGLE SOURCE OF TRUTH

This contract reconciles three drifting epochs into one authoritative experiment:

- `docs/PAPER_CONTRACT_2026-06-24.md` (declared frozen reward `ReT_cd`, block cadence) —
  **reward/algorithm clause SUPERSEDED here**; its fidelity gates and observation spec stand.
- `docs/PPO_REWARD_FREEZE_2026-06-26.md` + `docs/PPO_EXPERIMENT_PROTOCOL_2026-06-26.md` (PPO,
  two co-primary rewards) — **algorithm SUPERSEDED here** (PPO → DQN); the two-env design and the
  win/dominance definition are kept and made primary below.
- `scripts/retention_transfer.py` (runner: `control_v1`, DQN, cold transfer) — **this is the
  paper runner**; this contract is written to match it.

Decided by user 2026-06-26 after three independent reviews converged on: use DQN (not PPO) for the
`Discrete(18)` surface; keep ReT as an OUTCOME, not the training reward (avoid circularity); do not
manufacture headroom and call it learning; reframe the claim around a decision frontier. The user
additionally requires: keep BOTH environments (faithful + fine-tuned headroom) and the full
dominance win definition; tune the reward to the env and the claim.

---

## 1. Primary claim

> **Endogenous retained learning improves resilience over a reset/static policy ONLY where the
> environment exposes a non-trivial decision frontier.** We show how to identify, build, and
> evaluate that frontier on a thesis-anchored MFSC, and we report when learning pays and when it
> does not.

This is stronger and more honest than "RL beats Garrido": the faithful env is near-binary (buffer
absorbs → ReT ≈ saturated, or it saturates → collapse), so the contribution is *characterizing the
frontier*, with a genuine win where one exists.

## 2. Algorithm (SUPERSEDES the PPO clause)

- **DQN**, `MlpPolicy`, hidden `[64, 64]`, ReLU. Action space **`Discrete(18)` = [6 inventory levels
  × 3 shifts]** (Track A, thesis-faithful inventory×shift surface).
- Hyperparameters frozen for reproducibility: `learning_rate=1e-3`, `buffer_size=10000`,
  `learning_starts=50`, `batch_size=32`, `gamma=0.99`, `target_update_interval` = SB3 default unless
  pinned in the run log.
- **Retention = online Q-network + target network weights.** Replay buffer **cleared between blocks**
  in the primary line; `--retain-buffer` (weights+replay) is a SECONDARY ablation that localizes
  where `L_{k-1}` lives.
- Rationale: discrete 18-action surface is natural for Q-learning; the Markov sanity gate validated
  DQN as a retention detector; `retention_transfer.py` is already built around DQN. PPO retained as
  an optional robustness arm only.

## 3. Estimand (cold transfer — the corrected protocol)

Runner: **`scripts/retention_transfer.py`** (three arms, all evaluated COLD on each unseen block k,
before any block-k training):

| Arm | θ at block k | Isolates |
| --- | --- | --- |
| `frozen` | θ₀ (no learning) | non-learning floor |
| `reset` | θ₀ + ONLY block k−1 | single-block adaptation |
| `retained` | θ₀ + ALL blocks 0..k−1 | accumulated memory |

- **Primary contrast:** `retained − reset` → cross-block memory `L_{k-1}`.
- **Secondary:** `retained − frozen` → total value of online learning.
- **DEPRECATED:** `scripts/evaluate_retained_reset_learning.py::main()` uses the old
  *adapt-then-eval* estimand (reset catches up). It is kept ONLY as a helper library
  (`load_model`/`online_update`/`clean_eval`); it must NOT be used as the paper runner. A banner is
  added in that file.

## 4. Training reward (tuned to env + claim)

- **Primary training reward: `control_v1`** (operational, dense):
  `−( w_bo·service_loss_step + w_cost·shift_cost_step )`, with `w_bo=4.0`, `w_cost=0.02`
  (`config.py` BENCHMARK_*). ReT is **NOT** in the reward — it is the outcome, so we never train on
  the bar we report (no circularity).
- **Pre-registered reward sensitivity (decide on training tapes only, via a reward gate):**
  `control_v2_backlog` — adds explicit *pending-backlog pressure* on top of `control_v1`
  (`control_v1` only sees NEW service loss + shift cost, not standing backlog). Reward gate (training
  tapes): Spearman(reward, order-level ReT) > 0, Spearman(reward, −service-loss area) > 0,
  Spearman(reward, −pending backlog) > 0; the reward's best static must not be Pareto-dominated; the
  reward surface must not be flat. Pick the most parsimonious reward passing the gate.
- **Reward is tuned to the env:** the faithful env (every order late, delay 54 > LT 48) makes strict
  on-time fill degenerate, so the operational signal keys on service-loss + backlog, not on-time fill.
- Weights are tuned on **training-seed sensitivity only** (anti-fishing). `ReT_cd` /
  `ReT_garrido2024` are NOT training rewards in the primary line; they appear only as outcome bars.

## 5. Outcomes / win bars (report ALL; ReT is outcome only)

- **Primary resilience:** **Excel-exact ReT** (`ret_thesis.compute_ret_per_order_excel_formula`,
  no clamp — the Garrido raw-Excel formula, gate-verified 0 mismatch over 47,546 rows).
- **Secondary resilience:** Cobb-Douglas index `ReT_garrido2024` (cost-aware), `ret_thesis`
  (bounded), `ret_continuous`.
- **Metric-lane rule:** Excel ReT and Cobb-Douglas are separate resilience bars. If the training
  reward is operational (`control_v1`/`control_v2`), Excel ReT remains the primary Garrido continuity
  bar and C-D is secondary. If the training reward is Cobb-Douglas or a derivative of it, then
  Cobb-Douglas becomes the **same-bar** resilience metric: compute the Pareto frontier on C-D for
  dynamic and static policies, and report Excel ReT only as a continuity / non-inferiority check.
  Do **not** describe a C-D win as an Excel-ReT win unless the Excel-ReT comparison independently
  passes.
- **Full panel** (`supply_chain/episode_metrics.compute_episode_metrics`): `flow_fill_rate`,
  `lost_rate`, `service_loss_auc`, `ttr_mean/p95`, backlog age, CTj/RPj/DPj p50/p90/p99,
  `delivered_rations`, plus per-step service-loss mean/p95/**CVaR95**.
- **Resources:** shift-hours, surge-hours, strategic-buffer units, unit-cost-per-ration.

## 6. Two environments

- **Env-faithful** (`THESIS_FAITHFUL_ENV_FREEZE_2026-06-26`): the 1-to-1 fair fight. The
  **R1-dominant / forensic `excel_risk_tape` lane is the verified replication** (ReT MAE ≈ 0.0038;
  R1 DES 0.0044 vs Excel 0.0063). The **endogenous R2 / CF11–20 tail is an OPEN fidelity
  limitation** (`GARRIDO_ORIGINAL_RUNS_GATE`): chronic backlog tail unresolved → **do NOT make a
  fidelity claim on, or train the paper line on, the endogenous R2 lane.** Faithful experiments run
  on the R1-dominant generic regimes (`current/increased/severe`) where R14 is ubiquitous.
- **Env-headroom** (fine-tuned): faithful env + pre-registered realism knobs justified by the
  thesis's OWN future work/limitations — non-stationary/variable demand (§8.5.1), `stochastic_pt`,
  risk frequency φ + impact ψ up, optional `surge_inertia`/lead-time (§8.6). Calibrated with the
  Part-2 static panel to a **decision-frontier band** (best static off-saturation, regime-dependent
  optimum) and **FROZEN before any confirmatory run.** Purpose = test the frontier claim, NOT to
  manufacture a win; one frozen headroom regime only.

Machine-readable freeze:
`supply_chain/data/headroom_env_contract_v2_2026-06-26.json`.

For the overnight confirmatory line, Env-headroom is split into declared profiles rather than one
mutable tuning surface:

- `envb_frontier_v2`: `risk_frequency_multiplier=1.0`,
  `risk_impact_multiplier=1.5`, `stochastic_pt=false`. Purpose: **primary DQN retained-transfer
  benchmark**. This is the full-horizon static-grid recommendation
  (`outputs/experiments/headroom_calibration_2026-06-26/summary.json`): off-saturation,
  corner-free, argmax `S3_I672`, flow `0.685`. It supersedes the aggressive cell for the DQN
  memory paper line.
- `envb_aggr_g24_raw`: `risk_frequency_multiplier=2.0`,
  `risk_impact_multiplier=1.5`, `stochastic_pt=false`, reward
  `ReT_garrido2024_raw`. Purpose: **Cobb-Douglas same-bar sensitivity**: train and evaluate on the
  cost-aware C-D resilience frontier, while reporting Excel ReT as a continuity / non-inferiority
  metric. This is a Kaggle confirmatory sensitivity, not the primary DQN frontier profile, because
  the full-horizon static grid marks it as too collapsed.
- `envb_cons_control_v2`: `risk_frequency_multiplier=1.0`,
  `risk_impact_multiplier=1.25`, `stochastic_pt=false`, reward `control_v2` with
  service-heavy weights (`fill=1.2`, `service=6.0`, `lost=4.0`, `inventory=0.04`,
  `shift=0.06`, `switch=0.01`). Purpose: **Partial Win B**, Excel non-inferiority plus resource
  / Pareto efficiency.

This split is deliberate: it prevents post-hoc switching between "Excel win" and "Pareto win" while
letting the paper report the closest publishable result if the complete win does not appear.

## 7. Win definition (the complete win + two separable partial wins)

A dynamic (retained-DQN) policy is judged against **every** baseline incl. Garrido's thesis policy
(Cf0 `S1_I0`) and all 18 static configs, under CRN.

- **Complete win (best of all worlds):** the dynamic policy **Pareto-dominates all static policies**
  on resilience (Excel ReT primary; + as many resilience bars as possible: Cobb-Douglas, etc.) AND
  on the other panel metrics AND on resources — strictly better resource use at ≥ resilience.
- **Partial win A — resilience:** beat **all** statics (incl. thesis Cf0) on **Excel ReT by a
  significant margin**, regardless of Pareto. *Significant margin → paper.*
- **Partial win B — efficiency:** **tie** on Excel ReT but **Pareto-dominate on resources** (strictly
  fewer shift/surge-hours + buffer units) → better resource use.

If the complete win is unavailable, **report A and B separately** — either standing alone is a
result. Dominance flags reuse `compare_garrido_dynamic_vs_static.py::strict_service_resource_dominates`.

## 8. Baseline ladder

1. **Original** static `S1_I0` (thesis Cf0).
2. **Best static** per regime (Part-2 panel — the strong lean bar; S1_I168-class).
3. **Threshold heuristic** — state-dependent non-neural rule (separates reactive from learned).
4. **Frozen DQN** — θ₀ checkpoint, no eval-time updates (separates neural control from learning).
5. **Reset DQN** / **Retained DQN** — the estimand (§3).

## 9. Design & statistics

- Tapes split **calibration / validation / held-out confirmatory** (confirmatory used once).
- Regimes: `current/increased/severe` (env-faithful) + the one frozen headroom regime (env-headroom).
- CRN paired across all policies. Pilot ≥ 10 learner × 10 tape seeds; confirmatory 20 learner × 10
  tape (or 20×20 if compute allows). Statistical unit = (learner seed, tape seed), hierarchical.
- Inference: paired bootstrap CIs per metric (dynamic − best static, retained − reset); Holm across
  metrics; effect sizes + CIs, not p-values alone. Mixed model:
  `ΔR_{i,j,k} = β0 + β1·k + β2·regime + u_i(learner) + v_j(tape) + ε`. H1: β0>0 (retained−reset>0);
  H2: β1>0 (grows with blocks).

## 10. Stop-rule (anti-fishing)

- Env-faithful: if dynamic does not beat best static on resilience, the defensible result is the
  **efficiency win** (Partial win B). Report as such.
- Env-headroom: if no `retained > reset` and no dominance at the frozen frontier after the
  pre-registered budget → report the **honest null**; do NOT re-tune env/reward/regime against
  held-out results. One frozen headroom regime; one declared escalation axis only.

## 11. Pre-flight gate before any confirmatory compute

Env builds under the reward × both envs; `compute_episode_metrics` runs; full static panel runs;
DQN trains a few hundred steps and `retention_transfer.py` emits a block-0 sanity (retained−reset =
0 with no online update) + the dominance table. No compute committed to a broken setup.

---

### Open coordination note

`supply_chain/{supply_chain.py,ret_thesis.py,env_experimental_shifts.py,config.py}` are being edited
concurrently (Codex, Garrido fidelity lane). This contract owns docs + the DQN/retention runners; it
does not edit those shared env files. The endogenous-R2 fidelity work (`GARRIDO_*_2026-06-26` audits)
is the gate that keeps the paper line on the R1-dominant/forensic replication.
