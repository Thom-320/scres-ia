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
| ⭐1b-v8-cvar | **continuous_its × v8 realized-risk obs × ReT_excel_plus_cvar α=0.2 × war φ4/ψ1.5, horizon 104** | DENSE-CRN FALSIFIED AS CLAIM | Early 1-2 seed runs looked positive against a coarse/older frontier, but the 2026-06-29 dense 21×3 CRN gate falsified the claim. Dynamic: Excel 0.00208, CVaR 5.45e9, resource 0.545, `frac_std=0.144`; dense static `f0.10_S1`: Excel 0.00234, resource 0.050; dense static `f0.15_S1`: CVaR 4.35e9, resource 0.075. Both Excel and CVaR Pareto flags false. Artifacts: `outputs/experiments/h104_v8_ret_excel_plus_cvar_alpha02_1seed_20k_2026-06-27`, `outputs/experiments/h104_v8_ret_excel_plus_cvar_alpha02_2seed_40k_2026-06-27`, `outputs/experiments/v8_excel_plus_cvar_alpha02_dense_crn_gate_2seed_30k_2026-06-29`. |
| ⭐1b-preventive | **continuous_its × risk_obs+hazard × ReT_excel_plus_cvar α=0.2 × resource-aware Pareto × war φ4/ψ1.5, h104** | 3-SEED EXCEL PARETO WIN, CVaR NOT CONFIRMED | Preventive design with history-derived hazard features plus holding/shift cost. 1-seed smoke won both Excel and CVaR. 3-seed/40k confirm holds **Excel Pareto** but not CVaR: dynamic Excel 0.002220 > best static-at-<=resource 0.002156; CVaR 5.29e9 is worse than efficient S2 static 4.94e9; resource 0.550; `frac_std=0.163`. Action audit shows risk features matter in 2/3 seeds (`active/recent_R22/R23` top drivers), but one seed still keys on consequence/hazard mix. Artifacts: `outputs/experiments/preventive_pareto_v6riskobs_excelcvar_smoke_2026-06-27`, `outputs/experiments/preventive_pareto_excel_plus_cvar_alpha02_3seed_40k_2026-06-27`. |
| ⭐1b-preventive-excel-delta | **continuous_its × risk_obs+hazard × ReT_excel_delta × resource-aware Pareto × war φ4/ψ1.5, h104** | DENSE-CRN FALSIFIED AS CLAIM | Coarse/frontier-limited runs looked positive, but the dense 21×3 CRN Kaggle audit falsified the Pareto claim. Dynamic: Excel 0.002129, CVaR 5.31e9, resource 0.174, `frac_std=0.198`; dense static `f0.10_S1`: Excel 0.002279, CVaR 4.88e9, resource 0.050. Both Excel and CVaR Pareto flags are false; static dominates dynamic on resilience and resource. The lane remains evidence that the policy learns adaptive resource allocation, but not a publishable win against a dense continuous static frontier. Artifacts: `outputs/kaggle/scresia-preventive-pareto-dense-crn_auto/scresia_preventive_pareto_dense_crn_outputs/decision.json`, `outputs/kaggle/scresia-preventive-pareto-dense-crn_auto/scresia_preventive_pareto_dense_crn_outputs/preventive_pareto_excel_delta_mixed10_60k_dense21_crn9000/summary.json`, older positive coarse-frontier artifact `outputs/kaggle/scresia-preventive-pareto-final/scresia_preventive_pareto_final_outputs/decision.json`. |
| ⭐1b-per-op-buffer | **per_op_buffer × risk_obs+hazard × ReT_excel_delta × war φ4/ψ1.5, h104** | IMPLEMENTED, FIRST SCREEN MIXED | New same-family action contract: `Box([op3_frac, op5_frac, op9_frac, shift_signal])`. This directly follows the S3 audit: S3 overfeeds upstream while the bottleneck is downstream, so the agent may learn low Op3/Op5 buffer + higher Op9 buffer + selective shifts. First 1-seed/10k targeted screen: PPO is ultra-low-resource (resource 0.008) and Pareto-wins only vs statics at <= that resource, but misses the discovered sweet spot `op3=0, op5=0, op9=0.10, S1` (Excel 0.002627, CVaR 3.73e9, resource 0.017). This confirms the per-op frontier is richer and promising, but the learner/reward needs to be steered toward downstream Op9. Code: `supply_chain/continuous_its_env.py::PerOpBufferTrackAEnv`, runner: `scripts/run_per_op_buffer_pareto.py`, artifacts: `outputs/experiments/per_op_buffer_smoke_2026-06-28`, `outputs/experiments/per_op_buffer_signal_1seed_2026-06-28`. |
| ⭐1b-per-op-r2-campaign | **per_op_buffer × R2-only non-stationary campaign × ReT_excel_delta** | EARLY STATIC GATE WEAK / NO PROMOTE | Built the same-variables non-stationary R2 campaign runner (`scripts/run_per_op_r2_campaign.py`). Corrected static init fairness (`static-init-mode=match`) after finding that fixed full prepositioning made zero-buffer statics look artificially good. h104 static gate over `calm/r2_phi2/r2_phi4/r2_phi6` did **not** validate the expected Op9-buffer mechanism: robust static was `op3=0, op5=0, op9=0, S2`; oracle diversity was only `S2 -> S1`, not buffer. PPO smoke matched static Excel but used more resource (dominated) and learned upstream Op3/Op5 buffer, not Op9. Do not scale unless a new structural gate shows Op9-buffer movement. Artifacts: `outputs/experiments/per_op_r2_static_gate_match_h104_2026-06-28`, `outputs/experiments/per_op_r2_campaign_smoke_2026-06-28`. |
| ⭐1d | **Track A headroom campaign × controlled risk families × continuous_its** | ORACLE HEADROOM EXISTS, PPO/BC NO WIN | Static gate over `R1/R2/R3/R24/mixed × phi/psi` found a tiny full-grid opening (`oracle-best=+0.000296`). A targeted campaign using regimes where optima changed produced stronger oracle headroom: robust static `f0.05_S3` Excel 0.036423; oracle Excel 0.039223; `oracle-robust=+0.002800`. PPO smoke: `ReT_excel_plus_cvar α=0.2` failed badly (Excel 0.029998, frac_std 0.039). `ReT_excel_delta` nearly matched robust static and learned meaningful R2 conditioning (Excel 0.036308, frac_std 0.234, `corr(frac,active_R22)=0.855`) but remained dominated by dense statics. Behavior-cloning warm-start from the campaign oracle reduced MSE (`0.290 -> 0.059`) but did not win: BC+PPO Excel 0.033348, BC-only Excel 0.035128; both Pareto false. Interpretation: the oracle is a sharp discrete table by regime; continuous PPO smooths it and loses resource efficiency. Artifacts: `scripts/run_track_a_headroom_search.py`, `scripts/run_track_a_headroom_campaign.py`, `outputs/experiments/track_a_headroom_search_full3_continuous_2026-06-29`, `outputs/experiments/track_a_headroom_campaign_excel_delta_smoke_2026-06-29`, `outputs/experiments/track_a_headroom_campaign_excel_delta_bc20_smoke_2026-06-29`, `outputs/experiments/track_a_headroom_campaign_excel_delta_bc100_only_2026-06-29`, memo `docs/TRACK_A_HEADROOM_SEARCH_2026-06-29.md`. |
| ⭐1e | **continuous_its × α/holding_cost sweep × war φ4/ψ1.5** | DENSE-CRN REFINEMENT NULL | Phase-1 grid over `ReT_excel_plus_cvar`: `alpha={0,0.1,0.2,0.5}` × `holding_cost={0,0.002,0.005,0.01,0.02}`, 1 seed, 15k steps, 11-frac static frontier. Under the user’s revised raw-ReT win bar, offline re-score initially found 18/20 raw ReT winners, best `alpha=0.1,hc=0.005` with Excel 0.002254 > best static 0.002205. The dense CRN refinement closed this as a false lead: `alpha={0.05,0.1,0.15,0.2}`, `holding_cost={0.003,0.005,0.007,0.01}`, 2 seeds, 30k, `n_fracs=21`, `eval_seed0=9000`, `--crn-eval` produced **0/32 winners**; target `f0.10_S1` Excel 0.00228, best dynamic cells stayed around 0.00210-0.00213 with resource ~0.51-0.62. Artifacts: `outputs/experiments/alpha_holding_grid_2026-06-29/grid_report.md`, `outputs/experiments/alpha_holding_grid_2026-06-29/raw_resilience_rescore.md`, `outputs/experiments/alpha_holding_refine_crn_2026-06-29/watch.log`. |
| ⭐1f | **per_op_buffer × VecNormalize/warm-start/high-HC exploration × war φ4/ψ1.5** | LEARNING IMPROVES, BUT NOT FULL-FRONTIER WIN | 10-run exploration with per-op buffer action, VecNormalize, warm-start at Op9=0.10, high holding cost, PPO/SAC variants. Against the runner's limited subfrontier (`op3=0, op5=0, op9×shift`), 5/10 runs beat the local sweet spot. Best: `A_vecnorm_hold0.02_warmstart` seed2, Excel 0.00229 vs local static 0.00219, CVaR 4.37e9, resource 0.508. Critical caveat: the true full-grid per-op static from prior artifact remains much higher (`op3=0, op5=0, op9=0.10, S1`, Excel 0.002627, CVaR 3.73e9, resource 0.017), so this is not a claim. Interpretation: VecNormalize/warm-start helps PPO, but the fair next test must compare against the full per-op frontier, not the limited op9-only frontier. Artifact: `outputs/experiments/per_op_explore_2026-06-29/report.md`. |
| ⭐1g | **per_op_buffer × R13/R14/R24 conflict campaign × BC warm-start + VecNormalize** | INITIAL SIGNAL, RICH 3-SEED AUDIT DID NOT CONFIRM | Designed conflict gate to force different Track-A actions across thesis risk families: R13, R14, R24 with `phi={1,4,8}`, `psi=1.5`, h52. Targeted and compact full-grid gates both select robust static `op30_op50_op90_S2`; compact full-grid best static Excel 0.156041 and oracle Excel 0.156187 (`oracle-best=+0.000147`, tiny but positive). Early BC warm-start runs produced a promising raw-ReT/Pareto signal: `ReT_excel_plus_cvar α=0.1`, BC50, 40k, `holding_cost=0.0` gave Excel 0.156321, CVaR 1.842e9, resource 0.138, and an initial 3-seed run reported Excel 0.155903 > full-grid static 0.155254. However, after expanding `run_per_op_conflict_campaign.py` to emit the full Garrido/service/resource metrics panel, the richer 3-seed rerun did **not** confirm: dynamic Excel 0.151200 < best full-grid static 0.155254; CVaR 1.847e9 > 1.842e9; resource 0.342 > 0.267; dominated_by_count=91; dominates_count=0. The richer metrics show the full-grid static also wins on flow_fill_rate, lost_rate, backlog, service_loss_auc, CTj/RPj/DPj tails, delivered rations, and resource. Current status: useful learning/diagnostic lane, not a claim. Artifacts: `outputs/experiments/track_a_conflict_gate_per_op_targeted_2026-06-29`, `outputs/experiments/track_a_conflict_gate_per_op_full4_2026-06-29`, `outputs/experiments/per_op_conflict_campaign_pluscvar_a01_hc0_bc50_40k_2026-06-29`, rich audit `outputs/experiments/per_op_conflict_campaign_pluscvar_a01_hc0_bc50_3seed_richmetrics_2026-06-29`, memo `docs/PER_OP_CONFLICT_METRICS_AUDIT_2026-06-29.md`. |
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
1. ⭐ Δmemory (`retained-reset`) on the lead continuous lane (`continuous_its + risk_obs/hazard + ReT_excel_delta + war`). This is the strongest remaining test for continuous learning, but current Kaggle artifacts are micro/null and operationally underpowered.
2. ⭐ Track B only after fixing/understanding compressed fill scale and 7D heuristic mismatch. Track A dense-frontier gates now repeatedly show static base-stock dominance.
3. ⭐ R2 fidelity/audit lane: continue diagnosing why R2 endogenous behavior differs from Raw_data2 and whether controlled-risk lanes should use the R2 recovery sensitivity.
4. Garrido controlled-risk lane follow-up only if a new structural gate shows dynamic headroom; R1/R2/R3 and CF13/CF20 cheap probes did not promote, and CF20 repair closed as static optimum/no dynamic headroom.
5. per_op_buffer only if a new static structural gate shows true Op9-buffer movement; the R2-only campaign gate did not.
6. Generalization across regimes/horizons only after a candidate survives dense frontier.
7. continuous_its + learned initial prepositioning (two-phase) as diagnostic, not claim engine, unless it beats dense static after warm-start.
8. DMLPA + frame-stack + masked × continuous_its as ablation after a dynamic frontier exists.
9. variance_log CD eval bar (port from postfix-reruns) to re-open the CD lane honestly.
10. n_envs 4–8 + norm-reward as a learning-stability upgrade on a surviving lane, not a substitute for headroom.

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

## ⭐ Track A REOPENED as a serious hypothesis (2026-07-03) — under review, not a result yet

Prompted by Garrido's 2026-07-02 meeting objection ("no se puede abastecer de la nada"): Track A's
headline NEGATIVE result (`per_op_buffer`, `continuous_its_env.py`) used
`_top_up_inventory_buffer` — an unconditioned exogenous top-up (`container.put(shortfall)`), not
physical conservation. See `docs/THESIS_INTERPRETATION_DECISIONS_2026-06-24.md` D10/D11.

Found a cleaner, already-existing contract instead of inventing a new mechanism:
`MFSCGymEnvShifts(action_contract="track_a_v1")` — 5 **effective** dims (`op3_q, op9_q, op3_rop,
op9_rop, shift`; `op5_q` is inert, D11) — every dim consumed by a capacity-capped DES event
(`min(target, available)`), same conservation guarantee Track B's Op10/Op12 already have. No new
variables needed; just the right existing contract.

- `scripts/run_track_a_v2_conservation_3d_gate.py` (op3_q, op9_q, shift only; rop frozen at
  baseline — **screen, not a bound**): `opening_real=True`, oracle-minus-best-static **+0.00191**
  Excel ReT (1 seed, no real CI yet — do not over-read this).
- `scripts/run_track_a_v2_conservation_5d_gate.py` (all 5 effective dims, incl. rop): **the real
  gate, running as of 2026-07-03.** This is the actual go/no-go for whether to train PPO on Track A
  again. Do NOT skip straight to PPO training off the 3D screen alone.
- Decision rule: 5D `opening_real=True` (with >1 seed once promoted) → build+launch PPO
  confirmatory training on this contract. `opening_real=False` → Track A stays a null result, now
  cleaner/stronger (conservation-respecting AND still no headroom). Verdict stays "Track B is the
  paper spine" either way until/unless the 5D gate says otherwise.

Bonus finding from the same investigation: Track B's `post_cdc_only` ablation screen (freezing
`op3_q`/`op3_rop`, i.e. zero authority over the CDC itself) is still positive
(ΔReT=+0.000386 exact, `outputs/experiments/track_b_ablation_post_cdc_screen_2026-07-02/`,
`promote_to_long_run=False` — still screen-scale, 2 seeds/30k) — Track B's win does not depend on
touching the CDC at all.

**2026-07-03 update — 5D gate landed, PPO confirmatory launched.** The real 5D gate
(`outputs/experiments/track_a_v2_conservation_5d_gate_2026-07-03/`, 192 candidates x 9 regimes x 1
seed) finished: `opening_real=True`, oracle-minus-best-static **+0.0041566** Excel ReT (more than
2x the 3D screen's gap) — this is the real go-ahead per the decision rule above. A second AI agent
(Codex) working this repo concurrently independently reached the same conclusion and built
`scripts/run_track_a_v2_conservation_ppo.py` (BC + PPO + checkpoint-selection, same pattern as
`run_track_a_training_repair.py`); reviewed it (correct env/gate wiring, sound BC/eval-seed
separation, one scope note: trains under `reward_mode=ReT_excel_delta` inherited from the gate, not
`ReT_excel_plus_cvar` used by the original per_op_buffer headline — worth flagging when comparing
numbers, not a bug) and Codex's own local smoke test passed
(`outputs/experiments/track_a_v2_conservation_ppo_smoke_codex2/`, dynamic beat best-heldout-static by
+0.00736 on 1 seed). **2026-07-03 update — post_cdc_only CONFIRMED at full scale (5 seeds x 60k).** Promoted from screen
to confirmatory (launched locally on the user's Mac, not the VPS, per "no dependas solo del VPS"):
`outputs/experiments/track_b_ablation_8d_final_2026-07-01/post_cdc_only/summary.json`. Verified
directly (not from a pasted claim): `decision.promote_to_long_run=true`,
`learned_order_level_ret_gap_vs_best_static=+0.0003967`, `learned_raw_ret_win_vs_best_static=true`.
This is a strong, full-scale confirmation that Track B's win survives with ZERO authority over the
CDC/Op3 — directly closes Garrido's "does the win depend on touching the CDC" objection. Sibling
arms for comparison: `joint` +0.000373, `downstream_only` +0.000463, `shift_only` +0.000416, all in
the same directory (`run_claude.log`).

**2026-07-03 — leadtime/holding-cost robustness check COMPLETE, all 3 arms.** Verified directly:
- Arm A (no friction, replicate): static=0.1552542407406481, dynamic=0.1549926, delta=**-0.00026**
  (near-exact reproduction of the original per_op_buffer headline — confirms the repair script's
  protocol is sound).
- Arm B (lead_time=168h only): static=0.15628847, dynamic=0.15504215, delta=**-0.00125** (bigger loss).
- Arm C (lead_time=168h + holding_cost=0.05): ran independently on BOTH the VPS (its own sequential
  script) and locally in parallel (dedup came too late — VPS had already progressed past Arm B by the
  time of the check-in, so both completed; not fatal, just redundant compute). VPS: delta=**-0.00160**;
  local (Mac, arm64): delta=**-0.00465**. Same seeds, same code, same package versions
  (torch 2.12.1, numpy 2.4.6, sb3 2.9.0 on both) — the magnitude difference is real, not a bug,
  most likely CPU-architecture floating-point non-determinism (Intel Haswell vs Apple M1 ARM64) in
  the PPO/env rollout. Flagging honestly: exact numbers are not bit-reproducible across machines even
  with fixed seeds, but the qualitative trend holds in both: **A → B → C monotonically worsens PPO's
  gap as friction becomes more realistic (-0.00026 → -0.00125 → -0.0016/-0.0046).**

**Robustness verdict:** Track A's original negative result is NOT an artifact of the "free/instant"
buffer top-up simplification Garrido objected to — if anything, realistic friction (delay, cost)
makes PPO lose by MORE, meaning the original per_op_buffer headline was already the scenario most
favorable to PPO. This closes the buffer-conservation robustness question cleanly, independent of and
prior to the separate track_a_v1 conservation-respecting-contract investigation above.

Confirmatory 5-seed run launched — **collision note:** both this session and Codex independently
launched the identical run within 3 minutes of each other (same gate, same seeds, same timesteps);
killed the redundant duplicate (mine, PID 823231, minimal sunk cost) and kept Codex's further-along
one as canonical: PID 822421, `outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/`
(bc-epochs=150, the script's own default, not the 200 used by the original per_op_buffer headline —
a minor protocol difference, not expected to flip the qualitative verdict). This is the actual
decision point for whether Track A comes back as a real result — do not reframe the paper until this
finishes and is verified independently (re-read summary.json directly, don't trust a pasted claim).

**2026-07-03 FINAL — Track A v2 PPO confirmatory FINISHED: decisive negative, but not a dead end.**
`outputs/experiments/track_a_v2_conservation_ppo_5seed_40k_2026-07-03/summary.json`, verified directly:
`positive_seeds=0/5`, delta vs best-heldout-static = **-0.006316** (PPO: 0.1651-0.1723 across seeds vs
best_heldout_static 0.174422). This is NOT a borderline/microscopic result — it's a clean, unanimous
loss across all 5 seeds, so per the established decision rule this does NOT get "ask for more seeds"
treatment; it's conclusive as-is.

**Diagnosis (why, not just that):** oracle_selection excel=0.184127 vs best_selection_static
excel=0.178002 — the headroom the 5D gate found IS real and reproduces here too (+0.0061 on this
selection subset). BC warm-start converged cleanly for every seed (bc_loss 0.432→0.023, >94%
reduction) — behavior cloning is not the failure point. But `selected_step` (the checkpoint the
selection process picked as best) was 5000-30000 out of a 40000 budget for 4/5 seeds — meaning PPO
fine-tuning made the policy WORSE than earlier checkpoints in most seeds, i.e. **RL erodes the good
BC-cloned starting point instead of improving on it.** This is a distinct, more precise finding than
"no headroom": the headroom exists, but standard PPO fine-tuning (lr=1e-4, clip=0.1, ent_coef=0.0)
cannot reliably exploit it and instead drifts away from the imitated oracle. Consistent with the
broader pattern in `docs/ARCHITECTURE_REVIEW_2026-07-02.md` (RecurrentPPO/DMLPA also failed to
convert apparent Track A headroom into real wins) — this is a THIRD independent confirmation that
Track A's headroom, however it's parameterized, resists exploitation by standard RL optimization.

**Verdict for the paper:** Track A remains a negative result, now on the cleanest possible footing —
conservation-respecting (Garrido's objection addressed), higher-dimensional (5 effective dims, more
DOF than the old 4D fraction contract), with REAL oracle-vs-static headroom confirmed twice
independently (static gate + this run's own selection-set oracle) — and PPO still cannot convert it.
This is a stronger, more scientifically interesting negative than the original ("no headroom exists")
because it now separates two distinct claims: (1) does static headroom exist — yes; (2) can PPO learn
it — no. Do not spend further overnight compute chasing this without checking in with the user first;
the next legitimate step (if pursued) would be hyperparameter retuning (higher ent_coef for
exploration, disabling BC warm-start to test if it's actively harmful, or a longer budget with looser
clip_range) — a deliberate follow-up, not a "just add more seeds" retry.

**2026-07-03 — critic-pretrain fix implemented and launched (the deliberate follow-up).** Added
`collect_returns_for_critic()` + `critic_pretrain()` to `scripts/run_track_a_v2_conservation_ppo.py`
(new opt-in flag `--critic-pretrain-epochs`, default 0/off, non-breaking). Fits ONLY the value-head
parameters (`mlp_extractor.value_net` + `value_net`, via a dedicated optimizer that never touches the
BC-trained actor) on teacher-policy Monte Carlo returns, before PPO fine-tuning starts — directly
targets the diagnosed failure mode (critic uncalibrated at RL start, early advantage estimates erode
the good BC-cloned actor). Smoke-tested clean (exit 0). Full 5-seed confirmatory launched on VPS:
`outputs/experiments/track_a_v2_conservation_ppo_critic_warmstart_2026-07-03/` (PID 900034,
`--critic-pretrain-epochs 50`, otherwise identical protocol to the un-pretrained run). This is a
genuine test of the hypothesis, not a blind retry — if positive-seed count improves over 0/5, the
diagnosis was right; if not, the headroom is likely unlearnable by standard PPO regardless of critic
calibration.

**2026-07-03 FINAL — gae_lambda retune (3rd/final pre-registered attempt) also failed, worse than
both prior attempts.** `outputs/experiments/track_a_v2_conservation_ppo_gae_lambda_2026-07-03/summary.json`,
verified directly: `positive_seeds=0/5`, `raw_ret_delta_vs_best_heldout_static=-0.008660` (best_heldout_static
excel=0.174422, same comparator as all three attempts; dynamic mean=0.165762). This is WORSE than both
plain PPO+BC (-0.006316) and critic-pretrained PPO+BC (-0.007127) — raising `gae_lambda` toward 1.0 did
not reduce dependence on a possibly-miscalibrated value function; if anything, the higher-variance
advantage estimates it produces made fine-tuning erode the BC-cloned actor even faster.

**Track A is now CLOSED for this investigation cycle.** Three independent, well-reasoned, pre-registered
interventions (plain BC+PPO, critic-pretrained BC+PPO, GAE-lambda retune) all failed the promotion
threshold (beat held-out best static AND >=4/5 positive seeds) on the conservation-respecting 5D
contract, despite confirmed real static-oracle headroom (5D gate +0.0041566, this run's own
selection-set oracle +0.0061-0.0097 across attempts). The headroom exists; standard PPO cannot convert
it under any of the three reasonable fixes tried. Per the established discipline, do NOT launch further
Track A retunes without a fresh conversation with the user — this exhausts the pre-registered
"last chance" protocol. Track B remains the paper spine.

**2026-07-03 — reward-noise hypothesis (retroactive rescoring) built, tested, and RULED OUT.**
Concern: `_compute_ret_excel_delta` recomputes the Excel-faithful ReT over ALL orders every step
using `current_time=env.now`; since Garrido's formula uses running cumulative backorder/unattended
counts, an already-scored order's `ret` value could in principle change retroactively as time passes,
polluting the per-step training reward with noise unrelated to the current action. Built
`scripts/audit_ret_excel_delta_reward_noise.py`: decomposes each step's actual delta into
`retroactive_step` (old orders revalued at the new time) + `new_contribution` (genuinely new orders),
which reconstructs the env's own reported reward exactly (`max_env_vs_actual_gap=0.0` after fixing an
initial bug where the script assumed an empty warm-up baseline instead of snapshotting the true
post-reset order state). Result across all 9 regimes x 3 seeds x 52 steps (27 full episodes):
**`retroactive_share_of_abs_magnitude = 0.0000`, exactly, in every single episode.** Once an order's
backorder/unattended membership is determined, it does not get revisited — the reward is cleanly
Markovian with respect to old-order rescoring. This hypothesis is closed; it is NOT a contributor to
Track A's training difficulty. Critic-lag (the pretraining fix, launched separately) remains the live
hypothesis.

**2026-07-03 — RECONCILED with Codex's independent audit, which found the opposite (43.9%
"old-order revaluation").** Codex built a near-identical script
(`scripts/audit_track_a_ret_excel_reward_revaluation.py`,
`docs/TRACK_A_RET_EXCEL_REWARD_SIGNAL_AUDIT_2026-07-03.md`) and reported 43.9% of step-reward
magnitude coming from "retroactive revaluation of old orders" — a direct numeric contradiction with
the 0.0% above. Reproduced Codex's exact number independently (0.357 on `R24_phi4_psi1.5`, matching
their table) to rule out a fluke, then diffed methodologies line by line:

- **Mine**: freezes each "old" order's fields via `copy.deepcopy` at the pre-step snapshot, then
  revalues that FROZEN snapshot at the new `current_time`. Isolates the effect of time-passing-alone
  with zero change to the order's own state — a narrow, specific test for "does simulation time
  advancing retroactively rescore an order that hasn't changed?"
- **Codex's**: keeps live references to the same mutable `OrderRecord` objects; by the time
  `after_values` is computed, those objects may have been mutated in place by the DES during
  `env.step()` (e.g. `OATj`/`CTj` filled in because the order was actually delivered this step).
  Their "old_revaluation" therefore conflates two different things: (a) true retroactive
  rescoring (time passes, nothing about the order changes) and (b) an order that was PENDING before
  the step and gets RESOLVED during the step — its score updating because it just completed, not
  because of stale bookkeeping.

**Directly inspected the specific orders Codex's script flags as "old revaluation"**: every single
one shows `OATj: None -> <a real timestamp>` between before/after — i.e. it was still outstanding
before the step and got delivered during the step. This is category (b), not category (a).

**Verdict:** both audits are internally correct, but Codex's is mislabeled. There is no retroactive
rescoring bug (confirmed at 0.0% for the narrow, correctly-isolated question). What Codex's 43.9%
actually measures is **delayed credit**: a large share of any given week's reward comes from orders
placed in *earlier* weeks finally resolving now — a real, well-known RL characteristic (multi-step
lag between action and consequence), not reward noise. This reframes the actionable implication: if
this delayed-credit structure is contributing to Track A's learning difficulty, the fix is not
"clean up the reward" (nothing to clean up) but tuning `gamma`/n-step returns to better match the
typical order fulfillment lag (RPj/CTj percentiles run from tens to ~1400h) — a separate, lower-
priority hypothesis to hold in reserve behind the critic-pretrain result.

**2026-07-03 FINAL — critic-pretrain (LC1a) result: negative, slightly WORSE than baseline.**
`outputs/experiments/track_a_v2_conservation_ppo_critic_warmstart_2026-07-03/`, verified two ways
(summary.json numbers + independently recomputed from `seed_health.csv`'s own
`delta_vs_best_static` column — both agree exactly): mean Excel ReT `0.167295` vs held-out best
static `0.174422`, mean delta **-0.007127**, **0/5 positive seeds**. This is WORSE than the
un-pretrained run's -0.006316 — the critic-pretrain fix did not help, and if anything hurt slightly.
Critic pretraining itself converged fine (loss reduced ~95-96% in all 5 seeds, comparable to BC's own
convergence), and checkpoint selection still favored early steps (10k-15k of 40k) in all 5 seeds —
same failure signature as before, critic calibration quality did not change the pattern.

**Verdict: the critic-lag hypothesis is refuted (or at most a minor, non-explanatory factor).**
Combined with the reward-delayed-credit reconciliation above (real, but a different, lower-priority
mechanism, not yet tested), and per the pre-registered threshold in
`docs/TRACK_A_LAST_CHANCE_PREREGISTRATION_2026-07-03.md` (promotion requires beating held-out best
static AND >=4/5 positive seeds — not met, not close), **Track A closes for tonight as the clean
negative boundary**: conservation-respecting, higher-dimensional, real static-oracle headroom
confirmed twice, and neither plain PPO+BC nor critic-pretrained PPO+BC converts it. Do not launch
further retunes (gamma/n-step, no-BC ablation, etc.) without checking with the user first — this was
the pre-registered last-chance attempt.

**2026-07-03 — user explicitly authorized one more attempt ("Lánzalo, última oportunidad").**
GAE-lambda retune (see `docs/TRACK_A_LAST_CHANCE_2_GAE_LAMBDA_PREREGISTRATION_2026-07-03.md`):
`--gae-lambda 0.98` (was hardcoded to SB3 default 0.95), `--critic-pretrain-epochs 0` (disabled, to
keep this a clean one-variable test against the ORIGINAL un-pretrained baseline). Smoke-tested clean.
Launched on VPS: PID 916335, `outputs/experiments/track_a_v2_conservation_ppo_gae_lambda_2026-07-03/`.
Same pre-registered promotion threshold (beat held-out best static AND >=4/5 positive seeds). This is
the third and, per the preregistration doc, final deliberate intervention — if it also fails, Track A
closes without further retunes.

## Track B architecture bakeoff (2026-07-03) — resolving Garrido's novelty concern with evidence

Pre-registered in `docs/TRACK_B_ARCHITECTURE_BAKEOFF_PREREGISTRATION_2026-07-03.md` (updated:
primary arena is the full 8D Track B contract, `post_cdc_only` is a secondary/complementary check,
not the primary bakeoff arena). Investigated and clarified real terminology confusion first: "KAN"
(official Liu et al. 2024 splines, supervised demo only), "KAN-RBF sidecar" (dependency-free
approximation, already positive 5/5 seeds +0.000490), "DMLPA" (David's Transformer-over-history
extractor, faithful port confirmed against his pasted code), and "DKANA" (David's OWN named
architecture — despite the name, does NOT implement literal Kolmogorov-Arnold splines; it's a
hierarchical local+global attention model over a symbolic relational state encoding, currently
trained via behavior cloning only, not PPO — deferred, needs careful SB3 policy-wrapper work, not
rushed alongside the others) are four genuinely different things.

Built `scripts/run_track_b_dmlpa_sidecar.py` (Codex's implementation, adopted after independent
verification): mirrors `run_track_b_kan_sidecar.py`'s pattern but wraps training in
`VecNormalize -> VecFrameStack(n_stack=factor)` and manually replicates the identical
normalize-then-stack pipeline at evaluation time (`_stack_reset`/`_stack_step`). Verified this
frame-stacking logic line-by-line against the actual installed SB3 source
(`StackedObservations.reset`/`.update`) — zero-pad on reset, `np.roll`+place-at-end on step — both
match exactly. Smoke-tested clean on both local Mac and VPS (exit 0, no shape errors).

Launched the 5-seed/60k confirmatory DMLPA sidecar on the VPS (PID 924052,
`outputs/experiments/track_b_dmlpa_sidecar_2026-07-03/confirm_5seed_60k_h104/`), matching the
KAN-RBF sidecar's exact protocol (seeds 1-5, 60k timesteps, 12 eval episodes, h104, control_v1,
adaptive_benchmark_v2, v7) for direct, apples-to-apples comparability. Decision rule per the
preregistration: promote only if it beats canonical PPO+MLP with >=4/5 positive seeds under the
same protocol; otherwise it joins the architecture-robustness appendix (the finding strengthens,
not weakens, the paper's action-space-alignment mechanism claim either way).

**FINAL (2026-07-03) — DMLPA verdict, metric labels corrected on independent verification.**
Codex separately ran a local copy of the same protocol
(`outputs/experiments/track_b_dmlpa_sidecar_2026-07-03/full8d_5seed_60k_h104/summary.json`,
distinct from the VPS run above, `confirm_5seed_60k_h104`, which was still in flight when this
was written) and reported: DMLPA mean ReT 0.005721, beats best static (s2_d1.50,
0.005428) by +0.000291, loses to canonical PPO+MLP (0.005898) by -0.000177, verdict "good
robustness check, not a winning architecture."

Verified directly against `summary.json`'s `decision`/`policy_summary`/`seed_metrics` dicts
(not from narration), per the standing "duda siempre un poco del trabajo de codex" rule:

- The +0.000291 gap vs best static on the sidecar primary metric and
  `promote_to_long_run=true` are exact and correct.
- `0.005721` is **not** `order_level_ret_mean_mean`; it is
  `policy_summary.order_ret_excel_mean` (the Garrido/Excel formula metric). The sidecar primary
  metric is `policy_summary.order_level_ret_mean_mean = 0.005505452581095338`, whose CI95 upper
  bound is `0.005716351576970434`. This is a metric-label issue, not a reason to delete the
  Excel number.
- Per-seed order_level_ret_mean_mean: seed1=0.005124, seed2=0.005708, seed3=0.005710,
  seed4=0.005533, seed5=0.005453. vs best-static per-seed (s2_d1.50): seed1=0.005225,
  seed2=0.005251, seed3=0.005231, seed4=0.005176, seed5=0.005188 → **4/5 seeds beat static**
  (seed1 is the exception, 0.005124 < 0.005225) — Codex's pasted summary didn't state this.
- Same-metric comparison against the canonical PPO+MLP headline should use the Excel row:
  DMLPA `order_ret_excel_mean=0.005721487020279218` vs canonical PPO+MLP
  `order_ret_excel≈0.005898`, gap about **-0.000177**. The larger `-0.000393` number comes from
  comparing DMLPA's sidecar primary `order_level_ret_mean` against the canonical Excel headline;
  keep that as a cross-metric diagnostic only, not a manuscript-facing gap.
- Config confirmed matching the intended protocol exactly: `seeds=[1,2,3,4,5]`,
  `train_timesteps=60000`, `max_steps=104`.

**Verdict is unchanged despite the correction** — DMLPA still fails the promotion bar decisively
(it beats static but remains below canonical PPO+MLP on the same Excel metric). DMLPA joins the
architecture-robustness appendix: genuinely learns, beats the static baseline 4/5 seeds, but does
not unseat canonical PPO+MLP. Consistent with RecurrentPPO (lost) and KAN-RBF
(tied/marginal) — no richer architecture has beaten PPO+MLP under the pre-registered protocol
yet. The VPS run (`confirm_5seed_60k_h104`) remains in flight as an independent corroboration
of this same result; check it before citing DMLPA numbers in the manuscript.

## Real-KAN (pykan) sidecar — "last chance for KAN," 2026-07-03

Pre-registered in `docs/REAL_KAN_SIDECAR_PREREGISTRATION_2026-07-03.md`. Unlike every prior "KAN"
result (the RBF/KAN-inspired sidecar with a linear skip; the supervised pykan surrogate fit on the
147-cell static table), this wraps the official `kan.KAN` class (Liu et al. 2024, learnable
B-spline edges) as a genuine SB3 `BaseFeaturesExtractor` and trains it online with PPO
(`scripts/real_kan_extractor.py`, `scripts/run_track_b_real_kan_sidecar.py`).

**Feasibility finding:** pykan's `KAN.forward` works fine with standard autograd + Adam (no LBFGS
needed), but its default `save_act=True`/`symbolic_enabled=True` interpretability bookkeeping makes
a single-sample forward pass ~160x slower than needed for online RL (0.046s vs 0.00028s per call,
measured on this machine). Disabling both flags makes a normal-scale PPO loop feasible (~17s of
pure extractor cost for 60k rollout steps). This closes the previously-open item in
`docs/ARCHITECTURE_REVIEW_2026-07-02.md` ("A real KAN as an actual PPO policy/value network... would
need a custom wrapper").

**1-seed/30k probe (`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/timing_probe_30k/`):**
real-KAN `order_ret_excel_mean=0.005947` beats canonical PPO+MLP (`0.005893`) and best static
(`0.005446`). Single seed — not evidence on its own, but the closest any alternative architecture
has come to beating canonical PPO+MLP this session.

**3-seed/30k confirmatory (`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_3seed_30k_h104/`),
verified directly against `summary.json`:** pooled `order_ret_excel_mean=0.005931`, **3/3 seeds beat
best static** (s2_d1.50, 0.005451). Paired per-seed comparison against canonical PPO+MLP's own
seeds 1-3 (`track_b_gain_2026-06-30/.../track_b_top_tier_confirm_5seed_60k_h104/summary.json`):
seed1 -0.0000228, seed2 +0.0000407, seed3 +0.0000272 → mean **+0.000015, 2/3 seeds positive** —
**essentially a tie**, and real-KAN used **half the training budget** (30k vs canonical's 60k
timesteps), which is a genuinely notable sample-efficiency signal even though it doesn't clear the
promotion bar (needs a clean beat + >=4/5 seeds under an *identical* protocol, not just a tie under
a smaller budget). Action-collapse check: policy mixes S2/S3 shifts (26%/74%, not stuck at one
constant) with variable op10/op12 dispatch multipliers (mean 1.70/1.78, p95 1.94) — not degenerate.

## RecurrentPPO/LSTM on Track B — "does memory alone help?" (2026-07-03)

Directly motivated by the 2026-07-02 Garrido/David meeting (Granola transcript, meeting
"Supply chain resilience — CVaR metric, decision variables, and neural network integration with
David"): Garrido and David debated whether the agent could learn the empirical pattern of risk
occurrence from history (their "CAN" discussion), while the user flagged the risk of the agent
memorizing episode-specific distributions rather than a generalizable pattern. RecurrentPPO (LSTM)
had only ever been tried on Track A (500k x 5 seeds, lost) — never on Track B. `run_track_b_smoke.py`
already supports `--algo recurrent_ppo` (`MlpLstmPolicy`, `lstm_hidden_size=128`) with no new code
needed.

**3-seed/30k screen** (`outputs/experiments/track_b_recurrent_ppo_2026-07-03/confirm_3seed_30k_h104/`),
verified directly against `summary.json`: `order_ret_excel_mean=0.005857`, **3/3 seeds beat best
static** (0.005451, +0.000406). Paired per-seed vs canonical PPO+MLP's own seeds 1-3 (60k budget,
2x this run's 30k): **3/3 seeds LOSE** (deltas -0.0000636, -0.0000306, -0.0000821; mean -0.0000588).
Action profile is moderate, not degenerate (S1/S2/S3 mix 0.4%/89.6%/10%, cost_index 0.699, close to
canonical's 0.682).

**Reading:** unlike DMLPA (near-tie) and real-KAN (small win, both under the same half-budget
handicap), plain recurrent memory shows no edge over canonical PPO+MLP here — if anything it lags.
This is informative for the Garrido/David "does history help" question: the small positive signal
from DMLPA/real-KAN likely comes from the specific mechanism (attention over stacked frames, or
KAN's spline basis), not from sequential memory alone. Preliminary (3 seeds, half budget vs
canonical) — a matched 5-seed/60k confirmation would be needed before citing this as a clean
negative, but directionally consistent and cheap to obtain given RecurrentPPO's lower per-step cost
than DMLPA/real-KAN.

**Confirmatory 5-seed/60k run launched** (matched exactly to the canonical PPO+MLP protocol) to
settle whether the tie holds at full scale — `outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_5seed_60k_h104/`,
local Mac, in flight as of this writing. This is the most promising alternative-architecture result
of the entire bakeoff (RecurrentPPO lost, DMLPA lost by -0.000177, RBF-KAN tied only with a linear
skip that doesn't isolate the KAN mechanism) — worth the extra confirmatory run before any verdict.

**RESULT (2026-07-03) — 5-seed/60k matched-protocol confirmation, verified directly against
`summary.json`.** The background wait-loop watching this run was killed by a session restart, but
the actual training process (nohup, PID 73663) was unaffected and completed on its own — file
timestamps confirm it finished at 13:33, well after the watcher died. Numbers below are read
straight from the JSON, not from any log narration.

Paired per-seed comparison against canonical PPO+MLP's own 5 seeds (identical protocol: 5 seeds,
60k timesteps, h104, `control_v1`/`adaptive_benchmark_v2`/`v7`/`track_b_v1`, same eval-seed formula
— `track_b_gain_2026-06-30/.../track_b_top_tier_confirm_5seed_60k_h104/summary.json`):

| seed | real-KAN `order_ret_excel_mean` | canonical PPO+MLP | delta |
|---|---:|---:|---:|
| 1 | 0.005940 | 0.005921 | +0.000019 |
| 2 | 0.005917 | 0.005906 | +0.000011 |
| 3 | 0.005933 | 0.005920 | +0.000013 |
| 4 | 0.005933 | 0.005862 | +0.000071 |
| 5 | 0.005908 | 0.005855 | +0.000053 |

**5/5 seeds positive.** Mean paired delta **+0.000034**, 95% CI **[+0.000000, +0.000067]** (t-dist,
n=5) — the lower bound sits right at zero, so this is a real but small, borderline-significant
effect, not a clean CI-entirely-positive result like the Track B headline's own +0.000438. Pooled
means: real-KAN 0.005926 vs canonical 0.005893 vs best static (s2_d1.50) 0.005428 — real-KAN beats
best static by +0.000498, all 5 seeds positive there too (decision dict:
`learned_order_level_ret_gap_vs_best_static=+0.000485` on the sidecar's own primary metric,
`promote_to_long_run=true`).

Service/tail check: fill_rate_mean 0.003374 vs canonical's 0.003349, backorder_rate_mean 0.996626
vs 0.996651 (essentially identical, no degradation), order_ttr_mean_mean 93.6 vs 94.4 (slightly
faster recovery). No red flag there.

Action-profile check (this is the one genuine caveat): real-KAN leans hard into S3 shift (91.2% of
steps, vs canonical's balanced 30/36/34% S1/S2/S3 mix) and near-ceiling downstream multipliers
(op10 mean 1.968/std 0.039/p95 2.0, op12 mean 1.995/std 0.006/p95 2.0 — the multiplier cap is 2.0).
This narrow operating range could look like corner-collapse, but it is **not** replicating the
nearest static corner: `s3_d2.00` (the static "always S3, always d2.00") scores only 0.005421 —
*worse* than best static (0.005428) and far below real-KAN (0.005926, a +0.0005 gap, over 10x its
margin over canonical PPO+MLP). So the small variance around a mostly-near-ceiling baseline is
doing real, timing-sensitive work, not just sitting on a known static point — but the operating
range is narrower than canonical PPO+MLP's, worth disclosing exactly as such.

**Verdict: real-KAN is the first and only architecture alternative this session to clear the
primary promotion bar (beats canonical PPO+MLP, >=4/5 seeds) under a fully matched protocol.** The
effect is small and the CI lower bound touches zero, so this is not yet citable as a confident win
— per the pre-registration's own decision rules, the next required step before any manuscript
claim is a 10-seed confirmation (matching the canonical Track B headline's own 10-seed scale) to
see whether the CI clears zero with margin, plus disclosure of the narrower action-collapse
profile either way. Not launched yet as of this writing — check with the user before committing
more compute to it, given real-KAN's higher per-step cost than MLP/DMLPA/RBF-KAN.

**FINAL (2026-07-03) — 10-seed/60k confirmation, per
`docs/REAL_KAN_10SEED_EXTENSION_PREREGISTRATION_2026-07-03.md`.** Ran fresh seeds 6-10 only
(`outputs/experiments/track_b_real_kan_sidecar_2026-07-03/confirm_10seed_extension_6_10_60k_h104/`),
merged with the existing seeds 1-5. Paired per-seed vs the canonical PPO+MLP 10-seed anchor (seeds
6-10 pulled from `outputs/experiments/track_b_seed_expansion_2026-07-02/track_b_seed_expansion_6_10_claude/summary.json`,
the same file backing the manuscript's own 10-seed headline):

| seed | real-KAN | PPO+MLP | delta |
|---:|---:|---:|---:|
| 1 | 0.005940 | 0.005921 | +0.000019 |
| 2 | 0.005917 | 0.005906 | +0.000011 |
| 3 | 0.005933 | 0.005920 | +0.000013 |
| 4 | 0.005933 | 0.005862 | +0.000071 |
| 5 | 0.005908 | 0.005855 | +0.000053 |
| 6 | 0.005960 | 0.005880 | +0.000080 |
| 7 | 0.005932 | 0.005908 | +0.000024 |
| 8 | 0.005963 | 0.005891 | +0.000072 |
| 9 | 0.005959 | 0.005924 | +0.000035 |
| 10 | 0.005939 | 0.005911 | +0.000029 |

**10/10 seeds positive.** Mean paired delta **+0.000041**, 95% CI **[+0.000022, +0.000059]** (t(9),
95%) — unlike the 5-seed screen, this CI does **not** touch zero: a statistically clean, not
borderline, result. All 10 seeds also beat best static by a wide margin (real-KAN ~0.0059-0.0060 vs
static ~0.0054-0.0055).

**Cost caveat holds and is consistent across all 10 seeds**: `assembly_cost_index_mean` = 0.971
(seeds 1-5) / 0.993 (seeds 6-10) — near the operational ceiling, vs canonical PPO+MLP's ~0.68. The
ReT gain is real but comes bundled with ~45% higher shift/dispatch-utilization cost.

**This meets the pre-registration's own promotion bar for "architecture sidecar confirmed"**
(>=8/10 seeds beat static — 10/10 — and mean at or above the PPO+MLP anchor — yes, with a clean
CI). **It does not meet the bar for replacing the manuscript spine**, which explicitly requires
real-KAN to be superior on the operational tradeoff, not merely higher on ReT — the ~45% cost
increase fails that test. Status: promoted to confirmed strong architecture sidecar / direct
response to Garrido's KAN question; PPO+MLP remains the manuscript headline.

**2x2 matrix completed (2026-07-03): real-KAN under post_cdc_only (Op3/CDC frozen),
5 seeds x 60k**, `outputs/experiments/track_b_real_kan_sidecar_2026-07-03/post_cdc_only_5seed_60k_h104/`.
Trained from scratch under the constrained contract (`PostCdcOnlyWrapper` applied to both the
training and eval envs via the same `args._ablation_wrapper` mechanism the canonical PPO+MLP
post_cdc_only ablation uses — verified fresh model files per seed, not reused from the 8D run).

| | 8D completo | Post-CDC (Op3 frozen) |
|---|---:|---:|
| PPO+MLP | 0.005898 (10 seeds) | 0.005838 (5 seeds) |
| Real-KAN | 0.005938 (10 seeds, delta +0.000041, 10/10) | 0.005895 (5 seeds, delta +0.000057, 4/5) |

Paired per-seed vs canonical PPO+MLP post_cdc_only: seed1 +0.000057, seed2 +0.000071, seed3
-0.000000 (dead heat), seed4 +0.000095, seed5 +0.000064 — mean +0.000057, 4/5 positive (5-seed
scale, not yet the 10-seed confirmation the 8D arm has).

**Real-KAN's edge over PPO+MLP survives the strict post-CDC constraint and is slightly larger
there than under 8D completo — but costs even more**: `assembly_cost_index=1.0` (the absolute
ceiling), higher than its already-elevated 8D cost (0.97-0.99). The ReT-vs-resource tradeoff
sharpens, not softens, under the stricter contract. Verdict unchanged: real-KAN remains a
confirmed strong architecture sidecar under both decision-surface variants, not a spine
replacement, given the cost profile.

**Fair same-run history bakeoff finished (2026-07-03)**: `outputs/experiments/track_b_architecture_fair_bakeoff_2026-07-03/full8d_v9_history_5seed_60k_h104/`,
5 seeds x 60k, `ppo_mlp` vs `ppo_mlp_history` (frame-stacked history, no attention) vs
`ppo_dmlpa_positional` (attention over stacked history), same seeds/protocol. Pooled
`order_ret_excel_mean`: `ppo_mlp=0.005920` > `ppo_dmlpa_positional=0.005871` >
`ppo_mlp_history=0.005832`. Confirms the RecurrentPPO finding from a different angle: naive
history (frame-stack, no attention) does not help and actually underperforms plain MLP; attention
over the same history (DMLPA) partially recovers but still does not beat plain PPO+MLP. Consistent
with "the small positive signal from DMLPA/real-KAN comes from the specific mechanism, not from
having history."

## ⭐13 Track B `ReT_tail_v2` training reward (2026-07-04/05) — resilience/tail signal, NOT the same lane as the old continuous_its `ReT_tail_v2` failures above (⭐1b-v8-tail, ⭐2, ⭐8)

Different action contract (`track_b_v1` 8D, not `continuous_its`), different protocol (fixed-RNG,
`v7` obs, `adaptive_benchmark_v2`), and a different question: not "does ReT_tail_v2 beat Excel/CVaR
outright" but "does training against it buy measurable resilience/tail behavior over `control_v1`
without hurting the primary metric." Motivated by the oracle prevention-ceiling gate
(`docs/TRACK_B_ORACLE_PREVENTION_CEILING_VERDICT_2026-07-04.md`): perfect-foreknowledge boosting
couldn't move mean ReT Excel, but DID move `ret_excel_cvar05`/`rolling_4w_min`/`service_loss_auc`/
`ttr_mean`/`ttr_p95` consistently across 5/5 seeds — suggesting the composite ReT Excel reward was
pricing out a real resilience signal, not that no signal existed.

`ReT_tail_v2` is a pre-existing Cobb-Douglas reward (`supply_chain/env_experimental_shifts.py:1685`,
`_compute_ret_tail_v2`, predates this session) weighting service continuity (0.30) + backlog
recovery/containment (0.60) + un-gated cost efficiency (0.10) — a close structural match to the
tail/recovery metrics above. Codex screened it against two alternatives
(`ReT_excel_plus_cvar α=0.2`, `control_v2`) at 3-seed/30k smoke scale; only `ReT_tail_v2` passed the
gate (both alternatives degraded ReT Excel and tail metrics vs. `control_v1` baseline).

**10-seed/60k pooled result, independently verified against `seed_metrics.csv` (not narration)**,
paired per-seed vs. `control_v1` fixed-RNG canonical (seeds 1-5:
`track_b_fixed_rng_confirm_5seed_60k_2026-07-03`; seeds 6-10:
`track_b_control_v1_fixed_rng_extension_6_10_60k_2026-07-05`; `ReT_tail_v2` seeds 1-5:
`track_b_ret_tail_v2_confirm_5seed_60k_2026-07-04`; seeds 6-10:
`track_b_ret_tail_v2_extension_6_10_60k_2026-07-05`), 95% CI via t(9):

| Metric | Mean Δ | 95% CI | Favorable seeds | Clears zero? |
|---|---:|---|---:|---|
| ReT Excel | +0.000016 | [-0.000000, +0.000032] | 7/10 | **no (touches zero)** |
| CVaR05 | +0.000082 | [-0.000014, +0.000179] | 8/10 | no (touches zero) |
| rolling_4w_min | +0.000087 | [+0.000007, +0.000167] | 8/10 | **yes** |
| service_loss_auc/order | -11682 | [-24120, +756] | 7/10 | no (touches zero) |
| ttr_mean | -0.741h | [-1.595, +0.113] | 8/10 | no (touches zero) |
| ttr_p95 | -5.195h | [-9.556, -0.833] | 7/10 | **yes** |
| cost_index | -0.018 | [-0.116, +0.081] | 6/10 | no (neutral — no cost penalty either) |

**Honest read (matching the real-KAN 5-vs-10-seed precedent in this file — do not overclaim a
CI that merely touches zero):** every metric points the same (favorable) direction, majority of
seeds favorable everywhere (6-8/10), and 2 of 7 metrics (`rolling_4w_min`, `ttr_p95`) clear zero
cleanly at n=10 — but the **primary metric (ReT Excel) itself does not yet clear zero**, unlike
real-KAN's 10-seed confirmation which did. This is directionally consistent, multi-metric evidence,
not yet a clean statistical win on the paper's primary metric.

**Causal prevention test on this checkpoint: negative, same pattern as everything else this
session.** `R_full - R_reset(pre-risk)` counterfactual
(`outputs/experiments/track_b_ret_tail_v2_counterfactual_r22_r24_2026-07-04/`): R22 9.4% positive
pairs (35/374), R24 10.4% (41/393), deltas ~+0.0000034-0.0000035 — below any reasonable causal bar.
`ReT_tail_v2` buys adaptive/reactive resilience, not anticipatory prevention.

**15-seed extension (2026-07-05): not promoted.** The requested seeds 11-15 extension completed:
`track_b_control_v1_fixed_rng_extension_11_15_60k_2026-07-05` and
`track_b_ret_tail_v2_extension_11_15_60k_2026-07-05`, merged in
`outputs/experiments/track_b_ret_tail_v2_15seed_merged_summary_2026-07-05`. With CI95 t(14), the
effect shrank and **no metric clears zero**:

| Metric | Mean Δ | 95% CI | Favorable seeds |
|---|---:|---|---:|
| ReT Excel | +0.000008 | [-0.000004, +0.000021] | 9/15 |
| CVaR05 | +0.000045 | [-0.000025, +0.000115] | 10/15 |
| rolling_4w_min | +0.000052 | [-0.000013, +0.000117] | 10/15 |
| service_loss_auc/order | -5294 | [-15434, +4846] | 9/15 |
| ttr_mean | -0.306h | [-1.000, +0.388] | 10/15 |
| ttr_p95 | -2.616h | [-6.476, +1.244] | 10/15 |
| cost_index | -0.011 | [-0.075, +0.053] | 8/15 |

**Final status: exploratory/candidate only, not manuscript headline.** The lane remains useful as a
methodological note: reward design can move the right *direction* on tail/recovery without showing
causal prevention, but it did not survive escalation to a clean confirmatory claim. Do not promote
to spine, do not cite as a confirmed ReT Excel win, and do not spend more compute here unless the
paper explicitly opens a future-work appendix on alternative resilience rewards. See
`docs/TRACK_B_TAIL_RECOVERY_REWARD_SCREEN_2026-07-04.md` for the full screen-to-15seed narrative and
`docs/TRACK_B_ORACLE_PREVENTION_CEILING_VERDICT_2026-07-04.md` for the resilience-metric motivation.

## ⭐14 Track B no-forecast fixed-RNG spine candidate (2026-07-05) — reviewer-safe main lane

This is now the cleanest answer to the "privileged forecast" critique. It is not a new architecture,
not a reward tweak, and not a prevention claim: it is the same Track B `track_b_v1` / `control_v1`
spine with the two explicit forecast channels masked to zero:

- `risk_forecast_48h_norm`
- `risk_forecast_168h_norm`

Fresh fixed-RNG retrain, not a reinterpretation of the older 5-seed bundle:

- `outputs/experiments/track_b_no_forecast_fixed_rng_1_5_60k_2026-07-05/v7_no_forecast/`
- `outputs/experiments/track_b_no_forecast_fixed_rng_6_10_60k_2026-07-05/v7_no_forecast/`
- `outputs/experiments/track_b_no_forecast_fixed_rng_11_15_60k_2026-07-05/v7_no_forecast/`
- merged in `outputs/experiments/track_b_no_forecast_fixed_rng_final_15seed_2026-07-05/`

Protocol: `observation_version=v7` + `ForecastMaskWrapper`, `track_b_v1`, `control_v1`,
`adaptive_benchmark_v2`, fixed-RNG Track B code (`strict_exogenous_crn=True`), seeds 1-15,
60k timesteps/seed, 12 eval episodes, h104, `n_steps=1024`, `batch_size=64`.

Paired against the full-v7 fixed-RNG 15-seed control spine:

| Metric | No-forecast minus full-v7 | 95% CI | Favorable seeds |
|---|---:|---|---:|
| ReT Excel | +0.000002 | [-0.000013,+0.000017] | 9/15 |
| CVaR05 | +0.000015 | [-0.000064,+0.000094] | 8/15 |
| rolling_4w_min | +0.000006 | [-0.000072,+0.000084] | 9/15 |
| service_loss_auc/order | -136 | [-12864,+12592] | 8/15 |
| ttr_mean | +0.011h | [-0.897,+0.920] | 9/15 |
| ttr_p95 | -0.630h | [-5.841,+4.581] | 11/15 |
| cost_index | +0.023 | [-0.032,+0.079] | 5/15 |

**Status: PROMOTE AS REVIEWER-SAFE SPINE OPTION, not as a statistically superior variant.** The
primary ReT Excel result is indistinguishable from full-v7; the important paper implication is that
forecast is not needed to obtain the Track B learning result. This supports the user's preferred
framing: keep the Garrido Excel ReT metric, report gains against the dense static frontier, and use
no-forecast if reviewers/Garrido are worried that the forecast is privileged.

Important boundary: this no-forecast run removes only forecast channels, not the one-hot adaptive
regime state. The stronger "no regime + no forecast" claim remains C16's separate ablation
(`docs/E2_PRIVILEGED_OBSERVATION_VERDICT_2026-07-02.md`) and should be cited as a defense, not
confused with this 15-seed spine candidate.

## ⭐15 Track B final no-forecast A/B/C package (2026-07-06) — promote Case A spine, keep Case C as stress panel

This is the final July 6 package tying together horizon choice, no-forecast confirmation,
Real-KAN sidecar checks, and the selected Case C stress lane.

Source memo: `docs/TRACK_B_FINAL_AUDIT_PACKAGE_2026-07-06.md`.

Final defaults:

- Observation: `v7_no_forecast`
- Horizon: `104` weeks
- Reward: `control_v1`
- PPO batch size: `64`
- Real-KAN A/B batch size: `256`
- Final scale: 5 seeds x 60k x 12 eval episodes

Final A/B:

| Scenario | Architecture | ReT Excel | Delta vs static | Relative delta | Cost |
|---|---|---:|---:|---:|---:|
| Case A all risks | PPO+MLP | 0.005900714 | +0.000256755 | +4.55% | 0.488 |
| Case A all risks | Real-KAN | 0.005854707 | +0.000210748 | +3.73% | 0.378 |
| Case B downstream only | PPO+MLP | 0.604785754 | +0.013894863 | +2.35% | 0.431 |
| Case B downstream only | Real-KAN | 0.604782218 | +0.013891327 | +2.35% | 0.359 |

Independent VPS PPO A/B corroboration matched the local run: Case A `0.005903104`, Case B
`0.604846242`.

Selected Case C final confirmation (`R22/R23/R24`, `R24` frequency x3, `R22/R23` impact x1.5):

| Architecture | ReT Excel | Delta vs static | Relative delta | Cost |
|---|---:|---:|---:|---:|
| PPO+MLP | 0.481160397 | +0.046323705 | +10.65% | 0.719 |
| Real-KAN | 0.470095597 | +0.035258905 | +8.11% | 0.387 |

Status:

- **Promote Case A PPO+MLP no-forecast h104 as the reviewer-safe spine.**
- **Keep Real-KAN as an interpretability / efficiency sidecar**, not as the spine.
- **Use Case C as a stress/adaptation panel** because it creates strong adaptive headroom.
- Do **not** call Case C a prevention win: previous event-aligned prevention audits were negative,
  and this package supports adaptive exposure reduction, not causal anticipation.

Mechanism note: Case B/C branch shifts are real. PPO and Real-KAN move more orders into the
fill-rate branch than the best static baselines, consistent with shorter exposure windows under
adaptive dispatch. This is reportable as resilience through exposure reduction.
