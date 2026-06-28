# SCRES-IA Research Proposals Registry (2026-06-28)

Durable capture of **every** proposal from four expert advisor reports + one deep-research report
(2026-06-28). **Nothing is discarded.** Source of the prioritized execution program:
`~/.claude/plans/let-s-define-a-win-bright-flask.md`. Companion: `docs/SCRES_BIBLIOGRAPHY_2026-06-28.md`.

## Current state (what is / isn't proven)
- **CONFIRMED:** `continuous_its × risk_obs+hazard × ReT_excel_delta × war φ4/ψ1.5 × h104` Pareto-dominates the
  **resource-charged** static frontier on Excel ReT AND CVaR95 (10 mixed seeds, 60k, Kaggle `primary_win=true`,
  resource 0.241, frac_std 0.257). See `docs/PREVENTIVE_PARETO_RESULTS_2026-06-28.md`.
- **MECHANISM (mine + Codex agree):** efficient/reactive resource-aware allocation with **weak** hazard-conditioning
  (hazard strength 0.07–0.11 < reactive 0.13), **not** clean pre-shock anticipation.
- **OPEN:** `L_{t-1}` retained-learning (discrete18 Δmemory ~null; continuous Δmemory Kaggle probe running).

## Central theoretical reframing (all advisors converge)
> **SCRES learning value is frontier-dependent:** neural policies improve resilience only when the action space,
> observation history, and reward expose a *controllable resilience–resource frontier*.
> `R_t = f(decision frontier, observability, resource pricing, shock recurrence)` — NOT "RL = resilience magic."

## Three SEPARATE objectives (never mix)
1. **Raw ReT** — beat best static on Excel ReT, resources uncharged (may be physically impossible).
2. **Pareto ReT** — ≥ ReT at fewer charged resources. **CONFIRMED.**
3. **Learning / path-dependency** — retained > reset on new campaigns. **OPEN; highest value.**

---

## P1 — Reward design
**Raw-ReT track (NO holding cost):**
- `ReT_excel_delta_v1` = completed orders only (current).
- `ReT_excel_delta_bootstrap` = completed + proxy/terminal value of pending orders (fixes ~20-week-late credit).
- `ReT_excel_terminal_shaped` = sparse terminal ReT + **potential-based shaping** (policy-invariant):
  `Φ(s) = -α·BO_pending - β·Lost + η·InventoryCoverage`; `r'_t = r_t + γΦ(s_{t+1}) - Φ(s_t)`.
  (Only `control_v1_pbrs` exists today; ReT-PBRS is new.)

**Preventive/Pareto track:** `r_t = ΔReT - λ_I·I_t - λ_S·S_t (- λ_Δ·|ΔI_t|)`, small λ, **calibration-seed gated**.
- λ by empirical scale: `λ_b = c · median(|ΔReT_step|)/median(buffer_frac)`, `c ∈ {0,0.05,0.10,0.20,0.40}`.
- Sweep `holding_cost ∈ {0,0.0005,0.001,0.002,0.004}`, `shift_cost ∈ {0.0005,0.001,0.003}` (wrapper supports both).
- Stop rule: highest Excel ReT among policies whose `resource_composite` < best static needed to match it;
  select on calibration/training, NEVER held-out.

**Budgeted / Lagrangian constrained RL (advisor-preferred for raw-ReT-under-budget):**
`max E[ReT] s.t. E[resource] ≤ B`; `r_t = ΔReT - λ·max(0, resource_t - B)²`.

## P2 — Architecture ranking
1. **PPO MLP [64,64] + hazard features + two-stage init** — PRIMARY (already won). +n_envs 4/8, reward norm,
   action-smoothing/switch penalty, curriculum h52→h104, h260 only as generalization stress.
2. **PPO + auxiliary risk-prediction head** — shared encoder predicts `P(R_i in next 1/2/4/8 wk)` + disruption hours;
   `L = L_PPO + λ1·BCE + λ2·MSE` (λ1∈{0.01,0.05,0.10}, λ2∈{0.001,0.01}). Predictive rep without oracle.
3. **Recurrent PPO (LSTM/GRU)** — ABLATION only (RecurrentPPO lost before). MLP+hazard vs LSTM-no-hazard vs LSTM+hazard.
4. **GTrXL / Transformer-over-history** — only if ablations prove memory needed; else reviewer-bait.
5. **DreamerV3 / world-model RL** — future / 2nd paper (needs DES surrogate).
6. **Decision Transformer** — offline sequence modeling from DES rollouts; appendix.
7. **MAML / meta-RL** — H4 extension (task = campaign); after retained-reset.
8. **KAN** — interpretable surrogate/explainer only, not control engine.

## P3 — Two-stage policy (prepositioning credit assignment; KEY for raw ReT)
Per `EXCEL_REWARD_PREPOSITIONING_AUDIT_2026-06-27.md`, prepositioning is the real lever; `learn_initial_decision` was
unstable. `π = π_init(a_0|z_0)·∏π_weekly(a_t|s_t,h_t)`. Arms: static-init+dynamic-weekly / learned-init+dynamic-weekly /
static-only / weekly-only. Success: static-init+dynamic-weekly > static-only → weekly dynamics adds value.

## P4 — Hazard / observation improvements
Per-risk granularity `weeks_since_last_R11..R24,R3`; `empirical_rate_Ri_{8w,26w}`; `ewma_Ri`;
`time_since_last_Ri/expected_interval_Ri`. Auxiliary TRAINING-ONLY labels `y_Ri_{1w,4w,8w}` (feed P2.2 head).

## P5 — Mechanism ablation (Sprint 1)
continuous_its, φ4/ψ1.5, h104, ReT_excel_delta, PPO[64,64], 10 seeds, 60k.
`A0 base / A1 risk_obs only / A2 risk_obs+hazard / A3 shuffled hazard` (+ optional hazard-only/time-only/random).
Success: `A2>A1>A0` AND `A2>A3`. Else relabel "state-dependent reactive allocation."

## P6 — Anticipation lead/lag event study
`corr(buffer_t, risk_{t+k})` for k=1,2,4,8; event windows `mean buffer {4w before/during/4w after} R2/R3`.
Pre-event buffer > baseline → anticipatory; else reactive.

## P7 — Stronger baselines + dense frontier
Dense static frontier `buffer_frac ∈ {0.00,0.05,…,1.00} × {S1,S2,S3}` (21×3 vs current 5×3). Report best FREE static,
best CHARGED static, best static at ≤ dynamic resource. Strong heuristics: hazard-threshold, EWMA risk-rate, recent-risk
reactive, weeks-since-risk ramp.

## P8 — Raw-ReT headroom GATE (run FIRST)
Dense free static frontier h104/φ4/ψ1.5, same seeds. If best free static ReT > dynamic by > CI → raw-ReT impossible →
pivot objective 1 to budget-constrained ReT (P1 Lagrangian). ~minutes; saves a week of GPU.

## P9 — H4 retained-vs-reset on the WINNING continuous lane
Dedicated runner (extend `retention_transfer.py --track continuous`, already wired/validated, or fork).
- Campaigns: {R1-heavy, R2-heavy, R3-overdue, demand-surge, black-swan, mixed} × ρ∈{0.5,0.7,0.9}; fixed CRN tape per
  campaign (risk_id,start,duration,magnitude,affected_ops,demand) shared by all arms.
- Arms: static frontier, frozen θ0, reset (θ0+prev), retained (θ0+all prior), retained-shuffled, retained-wrong-history,
  retained-no-hazard.
- Contrasts: `Δ^H4_k = R_retained − R_reset`; history-match `R(π_θR2, tape_R2)` vs mismatch `R(π_θR1, tape_R2)`.
- Timing: cold-eval campaign k BEFORE training on it, then update (transfer estimand — implemented).
- Negative controls (mandatory): shuffle order; memoryless ρ≈1/3; hazard ablation; zero-update (Δ→0); same-seed CRN.
- Stats: `ΔR_{ijk} = β0 + β1·k + β2·ρ/hazard + β3·1[history-match] + u_i(seed) + v_j(tape) + ε`; H4: β3>0/β0>0; H2: β1>0;
  hierarchical bootstrap; episodes NOT independent; watch loss-of-plasticity (Abbas 2023).
- Scale: probe 10×10×24; confirmatory 20×10×30–40 (Kaggle).

## P10 — Generalization
Train φ4/ψ1.5; eval φ3/ψ1.25, φ5/ψ1.5, φ4/ψ2.0, different risk-family mix, h52 & h104 (NOT h260). Pareto non-inferiority.

## P11 — Metrics panel (always together)
Excel ReT (+ bounded ReT robustness), CVaR95, TTR mean/p95, CTj/RPj p95/p99, service-loss AUC, lost orders, flow fill,
resource_composite, action entropy/frac_std, P(buffer↑ before risk), lead-lag. (`compute_episode_metrics` covers most.)

## P12 — Paper reframing + hypotheses
- RQ1: under what operational conditions do neural policies improve SCRES vs static + reset baselines under recurring
  disruptions? RQ2: gains from retained learning, reactive risk-specific control, or preventive allocation?
- Two-level `L`: belief memory `b_t` (hazard features) + parametric memory `θ_k = U(θ_{k-1}, τ_k)`.
- Closed-loop framing: DES is open-loop/memoryless (Garrido "Alzheimer effect") → closed-loop DES-neural.
- Hypotheses: **H1** dynamic frontier (non-inferior/superior ReT + lower tail vs charged static); **H2** resource-efficient
  adaptation; **H3** hazard-memory mechanism (shuffle/remove collapses advantage); **H4** path-dependency (retained>reset
  cold); **H5** predictive accuracy (q(S,D,L) beats q(S,D) on TTR/service-loss/trajectory).
- Priority: primary=frontier shift; secondary=mechanism; open=H4; negative evidence (DQN null, h260 non-confirm)=boundary.
- Novelty: "first thesis-grounded empirical test of Garrido 2024's AI-SCRES proposal, operationalizing learned experience
  as endogenous `L_{t-1}`, audited against original order-level Excel; learning creates value only when the DES exposes a
  controllable ReT–resource frontier." (Do NOT claim "first RL for SCRES.")

## P13 — Reviewer #2 red-flag checklist
War-stress φ4/ψ1.5 = extension not Garrido-faithful · Excel ReT non-monotone/unclamped (38 values >1) → report full panel ·
"preventive" oversold → "adaptive resource-aware w/ limited hazard-conditioning" · charged frontier stated from abstract +
also report free static · holding-cost-in-reward vs resource-in-eval (emergent efficiency OR show hc>0 timing sensitivity) ·
H4 not yet on winning lane → "next confirmatory test" · R2 endogenous fidelity limited · frontier 5×3 too coarse → 21×3 ·
reward tuning = fishing → calibration/validation/held-out split, confirmatory ONCE.

## P14 — Must-cite library
See `docs/SCRES_BIBLIOGRAPHY_2026-06-28.md` (full 30+ union with links).

---

## Execution order (user-approved 2026-06-28)
Sprint 0 (gate + dense frontier + heuristics) → Sprint 1 (mechanism ablation A0–A3 + lead/lag) → Sprint 2 (reward timing +
per-risk hazard + aux head) → Sprint 3 (two-stage prepositioning) → **Sprint 4 H4 DEFERRED** until the running Δmemory
Kaggle probe returns (then decide ownership + scope) → Sprint 5 write-up (doesn't wait for H4).
