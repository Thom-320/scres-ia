# Promising Lanes Registry (living tracker — update every session)

Standing instruction (user, 2026-06-27): **never lose a promising lane.** Search for, record, and
try every promising combination. This file + the memory note `scres-ia-promising-lanes` are the
durable trackers. ⭐ = most promising.

## Win bars (per user)
Resilience = **Excel ReT** (primary) and/or **Cobb-Douglas** (when CD is the same-bar reward).
A win = beat best static by significant margin on resilience, OR Pareto (≥ resilience at fewer
resources), OR win a separable lens (CVaR tail / flow / lost). Measure consistently (episode-mean
CD; real service from `compute_episode_metrics`).

## Horizons / years still to try
Most fast screens so far used `max_steps=52` (≈1 post-warmup year). Do not claim a lane exhausted
until the promising candidates have been checked at:

| horizon | meaning | status |
|---|---|---|
| `52` weekly steps | fast 1-year screen | heavily used |
| `104` weekly steps | 2-year preparation/recovery check | local 2-seed win; Kaggle confirmatory in progress |
| `260` weekly steps | 5-year medium-horizon check | tried with ReT_excel_delta; not confirmed |

## LANES (status + artifact)

| # | lane (action × obs × reward × env) | status | result |
|---|---|---|---|
| ⭐1 | **continuous_its × v6 × ReT_excel_delta × war φ4/ψ1.5, init_frac=1.0** | PROMISING BUT NOT CONFIRMED | Original 2-seed pilot beat coarse static: Excel 0.00191 > 0.00188, CVaR 1.55e9 < 1.59e9; seed1 adaptive. Rigorous fixed-init 3-seed/fine-static failed: Excel 0.00186 < 0.00193, CVaR 1.65e9 > 1.59e9; 3/3 adaptive. Artifacts: `outputs/experiments/continuous_its_ReT_excel_delta_v6_2026-06-27`, `outputs/experiments/continuous_its_confirm_fixedinit_excel_delta_v6_phi4_psi1p5_3seed_2026-06-27` |
| ⭐1b | **continuous_its × v6 × ReT_excel_delta × war φ4/ψ1.5, horizon 104** | PROMISING BUT 5-SEED NOT CONFIRMED | 2-seed/60k won Excel+CVaR, but 5-seed local confirm did **not**: learned Excel 0.00213 < best static 0.00218; learned CVaR 5.35e9 > best static 4.94e9. Still adaptive; not a claim. Artifacts: `outputs/experiments/h104_ret_excel_delta_2seed_60k_2026-06-27`, `outputs/experiments/h104_confirm_excel_delta_5seed_2026-06-27`. |
| ⭐1b-audit | **learning audit for h104 ReT_excel_delta** | LEARNS ADAPTIVE BUT NOT FORECAST-DRIVEN | Checkpoint audit found adaptive actions (`frac_std≈0.28`) but no win under alternate eval seeds: Excel 0.00208 < 0.00225, CVaR 5.40e9 > 4.94e9. Forecast fields were constant zero (`risk_forecast_48h/168h std=0`), so behavior is not risk-forecast anticipation; it is mostly temporal ramping (`corr(frac,time_fraction)` high). Artifact: `outputs/diagnostics/h104_ret_excel_delta_learning_audit_2seed_2026-06-27`. |
| ⭐1b-v8 | **continuous_its × v8 realized-risk obs × ReT_excel_delta × war φ4/ψ1.5, horizon 104** | NEW CVaR SIGNAL, NEEDS CONFIRMATION | v8 exposes realized risk IDs (`active/recent/duration` for R11/R12/R13/R14/R21/R22/R23/R24/R3). 1-seed/20k quick screen did not beat Excel (0.00220 < 0.00225) but beat CVaR (4.59e9 < 4.94e9) with slight adaptivity (`frac_std=0.054`). Artifact: `outputs/experiments/h104_v8_realized_risk_ret_excel_delta_1seed_20k_2026-06-27`. |
| ⭐1b-v8-cvar | **continuous_its × v8 realized-risk obs × ReT_excel_plus_cvar α=0.2 × war φ4/ψ1.5, horizon 104** | NEW TOP LANE, 2-SEED WIN | 1-seed/20k won both bars. 2-seed/40k also won: Excel 0.00226 > 0.00225, CVaR 4.57e9 < 4.94e9; both seeds adaptive (`frac_std=0.118,0.145`). Small Excel edge, strong tail edge. This is the cleanest test of "pass risks as obs + reward tail." Artifacts: `outputs/experiments/h104_v8_ret_excel_plus_cvar_alpha02_1seed_20k_2026-06-27`, `outputs/experiments/h104_v8_ret_excel_plus_cvar_alpha02_2seed_40k_2026-06-27`. |
| ⭐1b-preventive | **continuous_its × risk_obs+hazard × ReT_excel_plus_cvar α=0.2 × resource-aware Pareto × war φ4/ψ1.5, h104** | 3-SEED EXCEL PARETO WIN, CVaR NOT CONFIRMED | Preventive design with history-derived hazard features plus holding/shift cost. 1-seed smoke won both Excel and CVaR. 3-seed/40k confirm holds **Excel Pareto** but not CVaR: dynamic Excel 0.002220 > best static-at-<=resource 0.002156; CVaR 5.29e9 is worse than efficient S2 static 4.94e9; resource 0.550; `frac_std=0.163`. Action audit shows risk features matter in 2/3 seeds (`active/recent_R22/R23` top drivers), but one seed still keys on consequence/hazard mix. Artifacts: `outputs/experiments/preventive_pareto_v6riskobs_excelcvar_smoke_2026-06-27`, `outputs/experiments/preventive_pareto_excel_plus_cvar_alpha02_3seed_40k_2026-06-27`. |
| ⭐1b-preventive-excel-delta | **continuous_its × risk_obs+hazard × ReT_excel_delta × resource-aware Pareto × war φ4/ψ1.5, h104** | DENSE-CRN FALSIFIED AS CLAIM | Coarse/frontier-limited runs looked positive, but the dense 21×3 CRN Kaggle audit falsified the Pareto claim. Dynamic: Excel 0.002129, CVaR 5.31e9, resource 0.174, `frac_std=0.198`; dense static `f0.10_S1`: Excel 0.002279, CVaR 4.88e9, resource 0.050. Both Excel and CVaR Pareto flags are false; static dominates dynamic on resilience and resource. The lane remains evidence that the policy learns adaptive resource allocation, but not a publishable win against a dense continuous static frontier. Artifacts: `outputs/kaggle/scresia-preventive-pareto-dense-crn_auto/scresia_preventive_pareto_dense_crn_outputs/decision.json`, `outputs/kaggle/scresia-preventive-pareto-dense-crn_auto/scresia_preventive_pareto_dense_crn_outputs/preventive_pareto_excel_delta_mixed10_60k_dense21_crn9000/summary.json`, older positive coarse-frontier artifact `outputs/kaggle/scresia-preventive-pareto-final/scresia_preventive_pareto_final_outputs/decision.json`. |
| ⭐1b-per-op-buffer | **per_op_buffer × risk_obs+hazard × ReT_excel_delta × war φ4/ψ1.5, h104** | IMPLEMENTED, FIRST SCREEN MIXED | New same-family action contract: `Box([op3_frac, op5_frac, op9_frac, shift_signal])`. This directly follows the S3 audit: S3 overfeeds upstream while the bottleneck is downstream, so the agent may learn low Op3/Op5 buffer + higher Op9 buffer + selective shifts. First 1-seed/10k targeted screen: PPO is ultra-low-resource (resource 0.008) and Pareto-wins only vs statics at <= that resource, but misses the discovered sweet spot `op3=0, op5=0, op9=0.10, S1` (Excel 0.002627, CVaR 3.73e9, resource 0.017). This confirms the per-op frontier is richer and promising, but the learner/reward needs to be steered toward downstream Op9. Code: `supply_chain/continuous_its_env.py::PerOpBufferTrackAEnv`, runner: `scripts/run_per_op_buffer_pareto.py`, artifacts: `outputs/experiments/per_op_buffer_smoke_2026-06-28`, `outputs/experiments/per_op_buffer_signal_1seed_2026-06-28`. |
| ⭐1b-v8-tail | **continuous_its × v8 realized-risk obs × ReT_tail_v2 power γ=1.25 × war φ4/ψ1.5, horizon 104** | TRIED, NOT PROMOTED | 1-seed/20k lost both bars and collapsed near-constant: Excel 0.00216 < 0.00225, CVaR 5.13e9 > 4.94e9, `frac_std=0.005`. Artifact: `outputs/experiments/h104_v8_ret_tail_v2_power125_1seed_20k_2026-06-27`. |
| ⭐1b-kaggle | **Kaggle confirmatory for h104 ReT_excel_delta** | COMPLETE, NULL | Kernel `thomaschisica/scresia-continuous-its-confirm`, dataset `thomaschisica/scres-ia-payload`, profile: 5 seeds, 60k timesteps, `n_envs=4`, 8 eval episodes, `max_steps=104`, free initial static frontier. Result: learned Excel 0.002175 < best static 0.002193; learned CVaR 5.27e9 > best static 4.94e9; primary_win=false. This confirms the old free-static h104 ReT_excel_delta lane is not a claim. Artifacts: `outputs/kaggle/scresia-continuous-its-confirm/scresia_continuous_its_confirm_outputs/decision.json`. |
| ⭐1c | **continuous_its × v6 × ReT_excel_delta × war φ4/ψ1.5, horizon 260** | TRIED, NOT CONFIRMED | 2-seed/120k failed at `max_steps=260`: Excel 0.00238 < 0.00251, CVaR 2.32e10 > 2.28e10; adaptive but below static. Artifact: `outputs/experiments/h260_ret_excel_delta_2seed_120k_2026-06-27`. Claim boundary: the current signal is 104-week only. |
| ⭐2 | **continuous_its × v6 × ReT_tail_v2 power γ=1.25 × war φ4/ψ1.5, init_frac=1.0** | PROMISING SCREEN, NOT CONFIRMED | 1-seed screen won Excel + CVaR: Excel 0.002016 > 0.002000, CVaR 1.37e9 < 1.48e9, adaptive. 3-seed confirm failed: Excel 0.00168 < 0.00193, CVaR 1.67e9 > 1.59e9. Artifacts: `outputs/experiments/continuous_its_screen_2026-06-27/tail_power_phi4_psi1p5`, `outputs/experiments/continuous_its_confirm_fixedinit_ret_tail_v2_power125_v6_phi4_psi1p5_3seed_2026-06-27` |
| ⭐3 | **continuous_its × v6 × ReT_excel_plus_cvar α-sweep × war** | TRIED, NOT CONFIRMED | 1-seed α=0.2 won Excel only; α=0.5 won CVaR only; α=0.8 lost. 2-seed/30k α=0.2 failed vs fine continuous static: Excel 0.00189 < 0.00193, CVaR 1.63e9 > 1.59e9; adaptive but not enough. Artifact: `outputs/experiments/confirm_ret_excel_plus_cvar_alpha02_2seed_30k_2026-06-27` |
| ⭐4 | **continuous_its × v6 × ReT_cd_balanced × war** | TRIED, NOT CONFIRMED | 1-seed screen won Excel+CVaR, but 2-seed/30k failed: Excel 0.00172 < 0.00193, CVaR 1.94e9 > 1.59e9; adaptive but unstable. Artifact: `outputs/experiments/confirm_ret_cd_balanced_2seed_30k_2026-06-27` |
| 5 | continuous_its × v6 × control_v1 × war | tried | constant; loses Excel, ~ties CVaR |
| 6 | Discrete(18) × {v4,v7} × {control_v1,ReT_excel_delta,CD} × war | tried | collapses to constant; ties static, no dynamic win (v7 forecast didn't help) |
| 7 | CD same-bar (ReT_garrido2024) full-cost | tried | PERVERSE — crowns S1_I0 (no buffer). Fix = `variance_log` CD (on `garrido-postfix-reruns`, unported) |
| ⭐8 | **Track B (track_b_v1 downstream) × v6 × ReT_tail_v2/ReT_excel_plus_cvar × faithful/war** | SMOKE WEAK | current: no promote, all policies fill≈0.006. increased: promote=True but fill≈0.007/ret≈0.003, highly compressed. Severe faithful was collapsed. Artifacts: `outputs/experiments/track_b_v6_ret_tail_v2_current_smoke_2026-06-27`, `outputs/experiments/track_b_v6_ret_tail_v2_increased_smoke_2026-06-27` |
| 9 | DMLPA/history (frame-stack + attention) × adaptive_benchmark_v2 × continuous_its | TRIED, MIXED | 2-seed DMLPA slightly improved over MLP and one DMLPA seed beat best constant Excel, but mean did not clearly dominate best constant and one policy was effectively constant. Useful ablation, not the main lever. Artifact: `outputs/experiments/continuous_its_dmlpa_adaptiveV2_h104_2026-06-27`. |
| 10 | Learned initial prepositioning (factorized two-phase) | partial (Codex) | prepositioning+capacity is the real lever; not yet in continuous |
| 11 | RecurrentPPO (memory) × v4 × control_v1 | TRIED 500k×5 | **LOST** to best static (`learned_beats_best_static=False`) → memory not the lever |
| 12 | retention DQN retained-vs-reset (memory) | tried | null (memory Δ≈0) |

## Cross-branch assets (best-of-both)
- **Current branch already has:** `ReT_tail_v2`, `ReT_cvar_cd`, `ReT_ladder_v1`, `continuous_its` (ported), v1–v7 obs, all action contracts.
- **On `origin/codex/garrido-postfix-reruns` (to port if needed):** `variance_log` balanced CD index
  (fixes CD perversity), `ReT_tail_v1` (superseded by v2 here), `--n-envs` parallel training,
  `--norm-reward`, end-of-horizon CD aggregation fix, history/recurrent kernels + results, tail
  decision analyzers, forensic DES-faithful audit suite.

## UNTRIED promising combinations (queue)
1. ⭐ per_op_buffer × risk_obs/hazard × ReT_excel_delta × war: run targeted same-space frontier, then dense/full per-op static frontier if the screen survives.
2. ⭐ Δmemory (`retained-reset`) on the lead continuous lane (`continuous_its + risk_obs/hazard + ReT_excel_delta + war`). This is the strongest remaining test for continuous learning.
3. Generalization of the lead Pareto lane across regimes/horizons after the h104 claim is frozen.
4. continuous_its + learned initial prepositioning (two-phase).
5. DMLPA + frame-stack + masked × continuous_its (Garrido learn-the-disruption).
6. Track B only after fixing/understanding compressed fill scale and 7D heuristic mismatch.
7. variance_log CD eval bar (port from postfix-reruns) to re-open the CD lane honestly.
8. n_envs 4–8 + norm-reward as a learning-stability upgrade on any lane.

## Audit findings (2026-06-27 PM) — is it learning / what / Δmemory / architecture
- **Policy learning audit (h104, ReT_excel_delta, v6, 1 seed):** Pepe IS learning — learned Excel
  0.00241 > random-init 0.00226 > best-constant 0.00216. Policy is **state-dependent** (frac R²~obs
  = 0.75, frac_std 0.25) but **REACTIVE, not anticipatory**: top driver `any_location_down` (+0.62,
  a *current* disruption signal); **frac-vs-forecast lead/lag ≈ 0 at all k → it IGNORES the v6
  forecast.** So the win (where it exists) = reactive buffering, not Garrido "prepare-ahead".
  Artifact: `outputs/experiments/policy_learning_audit_h104_2026-06-27/audit.json` (+ saved model.zip).
- **Δmemory (ReT_retained − ReT_reset):** checked ONLY on Discrete(18) (control_v1/CD, masked) →
  null. **NEVER confirmed on the promising continuous_its+risk_obs+h104 lane.** The runner now has
  a continuous track path; next step is a small smoke/pilot on the lead lane.
- **Architecture:** RecurrentPPO (LSTM memory) TRIED at 500k×5 → LOST. **DMLPA (Transformer-over-
  history) = infrastructure only on the branch, NO recorded result** (notebook outputs cleared); set
  up on discrete [6,3]+v5 (no forecast) = the collapsing setting. Untested on the promising lane.

## Not-yet-ported from `garrido-postfix-reruns` (needed?)
`audit_garrido2024_balance_methods` (variance_log CD fix), `tune_ret_tail_reward` + `reward_surface_audit`
(tail/CVaR reward gate), `run_track_a_continuous_it_s_confirm.py`, `--norm-reward`,
`run_track_a_exhaustion_sweep`, DMLPA `dmlpa_ppo` algo + notebooks. continuous_its / ReT_tail_v2 /
--n-envs already here.

## Preventive-agent design + FAIRNESS FIX (2026-06-27 PM) — built
User flagged the real measurement flaw: **raw Excel/CVaR let a constant static sit at S2/S3+high
buffer every week for FREE.** Built the fix + preventive design (additive, in
`supply_chain/continuous_its_env.py` + `scripts/run_preventive_pareto.py`):
- **Resource-aware Pareto eval** charges BOTH static and dynamic a per-step `resource_composite`
  (0.5·buffer_frac + 0.5·(S-1)/2). Smoke: resource-heavy statics now **dominated** — `f0.25_S3`
  (res 0.625) ties best resilience; `f1.0_S3` (res 1.0) gets the same for +60% cost. Win = dynamic
  strictly Pareto-dominates the CHARGED static frontier.
- **Balanced holding cost** in reward (`holding_cost`,`shift_cost`) so prevention is a real timing
  decision (can't hold max buffer free) — symmetric with the charged eval.
- **History-derived HAZARD obs** (no oracle): `weeks_since_last_R1/R2/R3` (overdue→build) +
  `ewma_risk_rate` + realized `active_/recent_Rxx`. Agent builds its OWN risk expectation.
- Win mechanism: spend only when hazard high, draw down when calm → same resilience at LOWER avg
  resource → dominate the always-on static. Full run chained (`preventive_pareto_hc{0.0,0.002}`).

## Learning-improvement levers (ranked)
1. **Action contract** (continuous buffer) — the one lever that produced adaptivity + a win. Biggest.
2. **Reward alignment** — ReT_excel_delta (Excel), ReT_tail_v2 (CVaR); avoid full-cost CD (perverse).
3. **Optimization** — n_envs>1 (thin-gradient fix; we used n_envs=1), norm-reward, more seeds/timesteps.
4. **Architecture** (history/DMLPA/recurrent) — LAST; recurrent already lost. Only the masked-infer setting is worth a shot.
5. **Track B** — different control problem where dynamics genuinely matter.
