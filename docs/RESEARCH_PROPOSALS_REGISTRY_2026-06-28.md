# SCRES-IA Research Proposals Registry (2026-06-28)

Durable capture of **every** proposal from four expert advisor reports + one deep-research report
(2026-06-28). **Nothing is discarded.** Source of the prioritized execution program:
`~/.claude/plans/let-s-define-a-win-bright-flask.md`. Companion: `docs/SCRES_BIBLIOGRAPHY_2026-06-28.md`.

## Current state (what is / isn't proven)
- **SUPERSEDED:** the early `continuous_its × risk_obs+hazard × ReT_excel_delta × war φ4/ψ1.5 × h104`
  preventive/Pareto claim was later falsified by the dense same-CRN static frontier. See
  `docs/PREVENTIVE_PARETO_RESULTS_2026-06-28.md`; do not reuse the old "Pareto-dominates" wording.
- **CONFIRMED PAPER-1 LANE:** Track B exposes the controllable downstream bottleneck. Under the same evaluation
  protocol and Garrido Excel-ReT metric, PPO+MLP on the 8D `track_b_v1` contract improves over the dense static
  grid; Real-KAN is a confirmed architecture sidecar with slightly higher Excel ReT and materially higher resource use.
- **MECHANISM STATUS:** current evidence supports adaptive/resource-aware control. Clean pre-shock anticipation is
  not yet proven; the prevention audit is still being calibrated because the original fixed-reset counterfactual
  failed sanity checks against known preventive/reactive heuristics.
- **OPEN PREVENTION QUESTION:** convert adaptive/reactive behavior into true prevention by learning risk recurrence
  from prior common risks (not black-swan shocks), then prove value with Excel-ReT counterfactuals and risk-history
  ablations.
- **H4 / retained adaptation:** a Track B retained-vs-reset probe has landed with a small positive Excel-ReT effect
  (`docs/H4_RETAINED_VS_RESET_VERDICT_2026-07-02.md`). It supports a minor retained-adaptation channel, not a pivot
  to organizational-learning claims.

## Central theoretical reframing (all advisors converge)
> **SCRES learning value is frontier-dependent:** neural policies improve resilience only when the action space,
> observation history, and reward expose a *controllable resilience–resource frontier*.
> `R_t = f(decision frontier, observability, resource pricing, shock recurrence)` — NOT "RL = resilience magic."

## Three SEPARATE objectives (never mix)
1. **Raw ReT** — beat best static on Excel ReT, resources uncharged (may be physically impossible).
2. **Dense-frontier ReT** — beat the dense static grid on the same scenario, same CRN, same Excel-ReT metric.
   **CONFIRMED for Track B.**
3. **Preventive learning** — actions improve future Excel ReT before service loss/backlog materializes. **OPEN.**
4. **Learning / path-dependency** — retained > reset on sequential adaptation cycles. **SUPPORTED as a small
   Track B side result, not a headline.**

---

## 2026-07-03 prevention roadmap and status

The prevention goal is now explicit: keep Track B as the paper spine, but test whether the learned controller can move
from **adaptive/reactive** control toward **preventive** control for frequent, learnable risks.

**What prevention means here**
- A policy is **reactive** if it waits for realized congestion, backlog, low fill rate, or downstream pressure, then
  increases dispatch/shift and improves recovery.
- A policy is **preventive** if it raises protective actions before realized service loss, using credible evidence such
  as risk history, recurrence rate, or a forecast signal, and those pre-event actions improve final Excel ReT.
- A policy is **mixed** if it shows both: moderate pre-positioning before common risks and stronger recovery actions
  after damage appears.

**Implemented / partially implemented**
- `control_v1_pbrs` is implemented and tested in `supply_chain/env_experimental_shifts.py` and
  `tests/test_control_reward_benchmark.py`; it shapes control reward, not a full Excel-ReT future-value reward.
- `FutureCreditRewardWrapper` in `scripts/run_cf20_learning_repair.py` implements `ReT_excel_delta_bootstrap` and
  `ReT_excel_terminal_shaped`, but in the older CF20/continuous lane. It is **not yet ported to the Track B 8D
  contract**, so it cannot be used as Paper-1 evidence today.
- Forecast masking/retraining infrastructure exists for Track B (`v7_no_forecast`, `v7_no_regime_forecast`). The
  no-forecast confirmatory run has landed:
  `outputs/experiments/track_b_e2_no_forecast_confirm_2026-07-03/v7_no_forecast/summary.json`.
  **CORRECTED 2026-07-03 — a narration of this run mixed up `order_level_ret_mean_mean` (0.005668)
  with the real primary metric `order_ret_excel_mean` (verified directly against the JSON:
  `0.00589519292585662`).** On the correct field: PPO without the explicit 48h/168h forecast scores
  `order_ret_excel_mean=0.005895` vs best static `0.005428` (delta `+0.000467`), and vs the
  canonical 10-seed with-forecast headline `0.0058977` — a gap of only **0.0000025, statistically
  indistinguishable, not merely "below canonical."** Cost is lower (`assembly_cost_index≈0.645` vs
  canonical's ~0.665-0.68). **Forecast costs us essentially nothing to remove — a clean, strong
  case for making no-forecast the headline observation contract**, not just "useful but not
  necessary." Treat forecast purely as a prevention probe from here on.
  (Note: `v7_no_regime_forecast`, the stricter mask that also removes the 5 regime one-hot fields,
  already has a verified retrain result — `docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md`,
  order_ret_excel_mean=0.005832 vs canonical 0.005898, ~93% of the gain retained, at *lower* cost —
  consistent with, slightly more conservative than, the narrower `v7_no_forecast` result above.)
- Real-KAN is now more than a surrogate: it is a working PPO feature extractor sidecar. It should be framed as
  interpretable/novel and resource-expensive, not as automatic proof of prevention.

**Still needed for true prevention**
- Per-risk memory features for common risks: `weeks_since_last_Ri`, `empirical_rate_Ri_{8w,26w}`, `ewma_Ri`,
  last duration/magnitude, affected node family, and time since recovery.
- An auxiliary risk-prediction head trained to predict `P(R_i in next 1/2/4/8 weeks)` and expected disruption hours
  from state/history. This is the cleanest way to make the model learn frequent-risk recurrence without giving it an
  oracle forecast.
- A Track B future-Excel-ReT reward/shaping variant: keep final evaluation as Garrido Excel ReT, but use potential
  shaping or bootstrapped terminal value during training to reward actions that improve near-future resilience before
  the order-level ReT is fully realized.
- A validated audit using policy-specific calm baselines, not one global reset action. Do not call a policy preventive
  unless pre-risk action changes have positive `R_full - R_reset(pre)` under the Excel-ReT formula.
  **IMPLEMENTED AND TESTED 2026-07-03** (`scripts/audit_track_b_prevention_sanity_check.py`): the
  per-policy calm-baseline reset is a real, necessary fix (a global fixed reset like `s2_d1.50`
  confounds "remove the reactive/preventive action" with "change this policy's whole operating
  point"). Confirmed it resolves one of two known-reference sanity checks
  (`heur_forecast_threshold` correctly classifies preventive under the episode-level delta). **But
  it is not sufficient alone**: at 5 seeds x 12 episodes the causal deltas are order 1e-5 against a
  ReT scale of ~0.005-0.006, and flip sign with small window-width changes (a windowed-ReT variant
  and a wider post-window both destabilized results that had been correct). Full write-up:
  `docs/TRACK_B_PREVENTION_COUNTERFACTUAL_VALIDATION_2026-07-03.md`. **Do not cite any
  "PPO+MLP/Real-KAN are reactive" classification from the current counterfactual as validated** —
  it needs materially more scale (seeds x episodes) or a redesign (per-signal-type anchors, per
  `docs/FORECAST_AND_PREVENTION_AUDIT_DECISION_2026-07-03.md`) before it can support a claim.

**Frequent-risk emphasis**
- Black-swan prediction is not the target. The prevention lane should focus first on frequent/low-impact risks where
  there is enough empirical recurrence for learning: classify risks by frequency and impact, train/evaluate common-risk
  hazard prediction, and only use rare risks as stress tests.

## P1 — Reward design
**Raw-ReT track (NO holding cost):**
- `ReT_excel_delta_v1` = completed orders only. **Implemented in older Track A/continuous lanes; not the current
  Track B spine.**
- `ReT_excel_delta_bootstrap` = completed + proxy/terminal value of pending orders (fixes ~20-week-late credit).
  **Partially implemented in `FutureCreditRewardWrapper`; still needs a Track B 8D port before it can support the
  prevention claim.**
- `ReT_excel_terminal_shaped` = sparse terminal ReT + **potential-based shaping** (policy-invariant):
  `Φ(s) = -α·BO_pending - β·Lost + η·InventoryCoverage`; `r'_t = r_t + γΦ(s_{t+1}) - Φ(s_t)`.
  **Partially implemented only outside Track B; `control_v1_pbrs` exists today, ReT-PBRS for Track B remains an
  active proposal.**
- **Prevention candidate:** train with a future-Excel-ReT shaped reward, but evaluate only with the original Garrido
  order-level Excel ReT. The training reward may encourage pre-risk preparation; the paper metric remains unchanged.

**Preventive/Pareto track:** `r_t = ΔReT - λ_I·I_t - λ_S·S_t - λ_Δ·|ΔI_t|`, small λ,
**calibration-seed gated**.
- λ by empirical scale: `λ_b = c · median(|ΔReT_step|)/median(buffer_frac)`, `c ∈ {0,0.05,0.10,0.20,0.40}`.
- Sweep `holding_cost ∈ {0,0.0005,0.001,0.002,0.004}`, `shift_cost ∈ {0.0005,0.001,0.003}` (wrapper supports both).
- Add switch/smoothing penalty `λ_Δ ∈ {0,0.0001,0.0005}` only as a calibration arm, not as an assumed improvement.
- **Lexicographic selection rule:** first require Excel ReT to stay within `ε` of the λ=0 dynamic policy
  (`ε = min(0.00002, 1% of baseline dynamic ReT)`); then prefer higher action variability/timing (`frac_std`),
  lower `resource_composite`, better lead-correlation to future risk, and non-worse CVaR. Select on calibration/training,
  NEVER held-out.

**Budgeted / Lagrangian constrained RL (advisor-preferred for raw-ReT-under-budget):**
`max E[ReT] s.t. E[resource] ≤ B`; `r_t = ΔReT - λ·max(0, resource_t - B)²`.
This is cleaner than making every week expensive when the scientific target is pure Excel ReT: the policy can spend
resources when prevention matters, but must stay inside an episode-level resource budget. Primary sources supporting
this direction include distributional constrained RL for supply-chain constraints (Bermudez, del Rio Chanona & Tsay,
2023) and constrained/lookahead RL for dynamic inventory routing under stochastic supply/demand (Hasturk et al., 2025).

## P2 — Architecture ranking
1. **PPO MLP [64,64] + hazard features + two-stage init** — PRIMARY (already won). +n_envs 4/8, reward norm,
   action-smoothing/switch penalty, curriculum h52→h104, h260 only as generalization stress.
2. **PPO + auxiliary risk-prediction head** — shared encoder predicts `P(R_i in next 1/2/4/8 wk)` + disruption hours;
   `L = L_PPO + λ1·BCE + λ2·MSE` (λ1∈{0.01,0.05,0.10}, λ2∈{0.001,0.01}). **Best next architecture for true
   prevention:** predictive representation without oracle forecast. Implementation plan:
   `docs/TRACK_B_RISK_BELIEF_AUX_HEAD_IMPLEMENTATION_PLAN_2026-07-03.md`.
3. **Recurrent PPO (LSTM/GRU)** — ABLATION only. Track B LSTM sidecar beat static but did not beat PPO+MLP; history
   alone is not enough evidence for prevention.
4. **GTrXL / Transformer-over-history / DMLPA** — DMLPA/history sidecars did not beat PPO+MLP under current protocol.
   Keep as David's future-work lane unless a same-protocol rerun beats PPO+MLP.
5. **DreamerV3 / world-model RL** — future / 2nd paper (needs DES surrogate).
6. **Decision Transformer** — offline sequence modeling from DES rollouts; appendix.
7. **MAML / meta-RL** — H4 extension (task = campaign); after retained-reset.
8. **KAN** — updated status: Real-KAN is now a working PPO sidecar, not only a surrogate. It slightly improves Excel
   ReT over PPO+MLP but at materially higher resource use; useful for Garrido's novelty/interpretability concern, not
   yet a better operational spine.

## P3 — Two-stage policy (prepositioning credit assignment; KEY for raw ReT)
Per `EXCEL_REWARD_PREPOSITIONING_AUDIT_2026-06-27.md`, prepositioning is the real lever; `learn_initial_decision` was
unstable. `π = π_init(a_0|z_0)·∏π_weekly(a_t|s_t,h_t)`. Arms: static-init+dynamic-weekly / learned-init+dynamic-weekly /
static-only / weekly-only. Success: static-init+dynamic-weekly > static-only → weekly dynamics adds value.

## P4 — Hazard / observation improvements
Per-risk granularity `weeks_since_last_R11..R24,R3`; `empirical_rate_Ri_{8w,26w}`; `ewma_Ri`;
`time_since_last_Ri/expected_interval_Ri`. Auxiliary TRAINING-ONLY labels `y_Ri_{1w,4w,8w}` (feed P2.2 head).

**Status:** Track B v7 contains selected downstream/hazard signals and forecast fields, but not the full per-risk memory
panel above. This remains the main route to prevention from prior common risks: classify risks by recurrence and impact,
then test whether removing/shuffling that risk history reduces Excel ReT. See the 2026-07-03 risk-belief auxiliary-head
plan for the recommended low-risk sequence: risk-event logging → supervised predictor → pretrained encoder → PPO fine
tuning → same-protocol ReT Excel evaluation.

## P5 — Mechanism ablation (Sprint 1)
continuous_its, φ4/ψ1.5, h104, ReT_excel_delta, PPO[64,64], 10 seeds, 60k.
`A0 base / A1 risk_obs only / A2 risk_obs+hazard / A3 shuffled hazard` (+ optional hazard-only/time-only/random).
Success: `A2>A1>A0` AND `A2>A3`. Else relabel "state-dependent reactive allocation."

## P6 — Anticipation lead/lag event study
`corr(buffer_t, risk_{t+k})` for k=1,2,4,8; event windows `mean buffer {4w before/during/4w after} R2/R3`.
Pre-event buffer > baseline → possible anticipation; else reactive.

**2026-07-03 update:** simple event windows and fixed-reset counterfactuals are not sufficient. The first prevention
audit failed sanity checks because one global reset action (`s2_d1.50`) does not represent each policy's own calm
behavior. Future audits must use policy-specific calm baselines and separate anchors:
- prevention anchor = forecast/regime warning before realized damage;
- reaction anchor = realized backlog/fill-rate/downstream-pressure deterioration;
- value test = `R_full - R_reset(window)` using the Garrido Excel-ReT formula.

## P7 — Stronger baselines + dense frontier
Dense static frontier `buffer_frac ∈ {0.00,0.05,…,1.00} × {S1,S2,S3}` (21×3 vs current 5×3). Report best FREE static,
best CHARGED static, best static at ≤ dynamic resource. Strong heuristics: hazard-threshold, EWMA risk-rate, recent-risk
reactive, weeks-since-risk ramp.

**Status:** for Paper 1, the defendable comparator is the Track B dense static grid under the same CRN/evaluation
protocol. For prevention-specific audits, add three reference policies before making claims: static calm, forecast-threshold
preventive, and backlog/downstream-pressure reactive. If the audit cannot classify these references correctly, do not
classify PPO/KAN as preventive or reactive.

## P8 — Raw-ReT headroom GATE (run FIRST)
Dense free static frontier h104/φ4/ψ1.5, same seeds. If best free static ReT > dynamic by > CI → raw-ReT impossible →
pivot objective 1 to budget-constrained ReT (P1 Lagrangian). ~minutes; saves a week of GPU.

## P9 — H4 retained-vs-reset / path dependency
**Status update 2026-07-03:** the older "winning continuous lane" framing is superseded. The paper spine is Track B.
`scripts/retention_track_b.py` has already run a confirmatory Track B retained-vs-reset probe; see
`docs/H4_RETAINED_VS_RESET_VERDICT_2026-07-02.md`.

What is supported:
- Track B retained adaptation is positive but small: masked-arm retained-reset `+0.0000493`, CI95
  `[+0.0000167,+0.0000819]`, 9/10 seeds positive.
- It survives masking the same regime/forecast fields removed in E2.
- It can be mentioned as a minor retained-adaptation channel.

What is **not** supported:
- Do not pivot the paper around H4.
- Do not call it cross-campaign organizational learning.
- Do not revive `continuous_its` as the main H4 lane unless a new Track-B-level frontier win is established there.

If H4 is reactivated as a dedicated future study, use the causal order:
1. evaluate campaign `k` cold with `θ_{k-1}`;
2. train/update on campaign `k`;
3. move to campaign `k+1`.

Recommended future-study arms:
1. dense static frontier;
2. hazard/risk-memory heuristic;
3. frozen PPO `θ0`;
4. reset PPO (`θ0` plus only previous campaign);
5. retained PPO (`θ0` plus campaigns `1...k-1`);
6. retained-shuffled;
7. retained-no-hazard / active-risk-only controls.

Primary contrast: `Δ^H4_k = R_{k,retained} - R_{k,reset}`. Use hierarchical bootstrap; do not treat episodes as
independent.

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
"preventive" oversold → "adaptive resource-aware w/ prevention still open" · charged frontier stated from abstract +
also report free static · holding-cost-in-reward vs resource-in-eval (emergent efficiency OR show hc>0 timing sensitivity) ·
H4 is small side evidence, not the main mechanism · R2 endogenous fidelity limited · frontier 5×3 too coarse → 21×3 ·
reward tuning = fishing → calibration/validation/held-out split, confirmatory ONCE.

## P14 — Must-cite library
See `docs/SCRES_BIBLIOGRAPHY_2026-06-28.md` (full 30+ union with links).

---

## Execution order (user-approved 2026-06-28)
Sprint 0 (gate + dense frontier + heuristics) → Sprint 1 (mechanism ablation A0–A3 + lead/lag) → Sprint 2 (reward timing +
per-risk hazard + aux head) → Sprint 3 (two-stage prepositioning) → Sprint 4 H4 **side-result only, already landed small
positive on Track B** → Sprint 5 write-up (doesn't wait for stronger H4).
