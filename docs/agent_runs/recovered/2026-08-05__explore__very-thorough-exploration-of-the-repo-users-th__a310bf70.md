# Explore — Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a310bf702234dca30`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-05T18:03:20.311Z`
- **effort**: `xhigh` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-a310bf702234dca30.jsonl`
- **turns**: 231 · **assistant messages**: 2

## Task given

```text
Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

I need a complete map of the reinforcement-learning and neural-architecture infrastructure that already exists, so I can plan a paper comparing KAN vs MLP vs MPC vs RL.

Find and report, with exact file paths:

1. All RL training code: PPO, RecurrentPPO, stable-baselines3 usage, gym/gymnasium environment wrappers around the DES. Which env classes exist, what are their observation and action spaces, what reward do they use, and where are the training entry points (scripts/*).

2. All KAN (Kolmogorov-Arnold Network) code: implementations, wrappers, any efkan/pykan dependency, and any results comparing KAN vs MLP. Report what was measured and what the outcome was.

3. All MPC / model-predictive-control / belief-MPC / DP / rollout comparator code. Where does the "MPC is cheap and ties RL" evidence live?

4. The measured "neural premium" evidence: search for curvature vs noise (values around 0.076 and 0.317), Delta_N, TOST equivalence, and the claim that an MLP was WORSE than linear. Report exact file paths, numbers and context.

5. Compute/latency instrumentation: is there anything that measures per-decision planning cost, wall-clock per decision, number of DES calls, or parameter counts? The contract contracts/garrido_expanded_des_e_star_v1.json declares an estimand "Delta_efficiency" — find any code that could measure it.

6. The gym-style env: how expensive is one episode (seconds), how many steps, and is there any vectorized/parallel rollout support?

7. Any existing training results directories with learned policies (checkpoints), and their claim_status.

Report exact paths and numbers, no vague summaries.
```

## Final answer

Here is the complete map.

---

# 1. RL training infrastructure

## 1.1 Dependencies

`requirements.txt` / `requirements-pinned.txt`:
- `gymnasium==1.3.0`, `stable_baselines3==2.9.0`, `sb3_contrib==2.9.0`, `torch==2.12.1`, `simpy==4.1.2`, `pykan==0.2.8`

**95 Python files** import `stable_baselines3`/`sb3_contrib`. 516 tracked `.py` under `scripts/`, 69 under `supply_chain/`, 203 under `tests/`.

## 1.2 Gym environment classes (exact spaces)

| Class | File | Obs space | Action space | Reward |
|---|---|---|---|---|
| `MFSCGymEnv` | `<HOME>/Projects/research/scres-ia/supply_chain/env.py:19` | `Box(0, inf, (15,))` | `Box(-1,1,(4,))` — op3_q, op9_q, op3_rop, op9_rop multipliers (`1.25+0.75*a`) | `proxy` or `rt_v0` = `-(α·recovery/46 + β·inventory/1.72e7 + γ·service_loss)` (lines 89–115) |
| **`MFSCGymEnvShifts`** (main RL env) | `supply_chain/env_experimental_shifts.py:242` (3413 lines) | `Box(0, 20, (obs_dim,))`, obs_dim ∈ {15 (v1), 18 (v2), 20 (v3), 24 (v4), 30 (v5), 42 (v6), **52 (v7)**, 79 (v8), 89 (v9), 101 (v10)} — dims at lines 212–221 | Track A `Box(-1,1,(6,))`; Track B (`track_b_v1`) `Box(-1,1,(8,))`; `shift_only` (1,); `shift_q9` (2,) | 20+ reward modes: `ReT_thesis`, `ReT_corrected`, `ReT_unified_v1`, `ReT_seq_v1`, `ReT_ladder_v1`, `control_v1`, `control_v1_pbrs`, `control_v2`, `ReT_garrido2024{,_raw,_train}`, `ReT_cd_v1`, `ReT_cd_sigmoid`, `ReT_cvar_cd`, `ReT_excel_delta`, `ReT_excel_plus_cvar`, `ReT_tail_v2` (dispatch at lines 2901–3049 and 3221–3400) |
| `ProgramORetOnlyEnv` | `supply_chain/program_o_ret_env.py:70` | `Box(0,1,(21,))` — `OBSERVATION_DIM=21` (line 36); fields at 40–64 (on_hand, locked_pipeline, backlog qty/orders/age, in_flight, `belief_c`, `predicted_share_c`, 5-hot previous action, week/7, remaining/8) | `Discrete(4)` — product mix k over 3 weekly batch slots | **zero for decisions 1–7; terminal `ret_excel_request_snapshot_v2` only**; 8 weekly decisions |
| `ProgramOPerBatchEnv` | `supply_chain/program_o_per_batch_env.py:107` | `Box(0,1,(OBSERVATION_DIM,))` | `Discrete(2)` | terminal ReT |
| `GarridoLearningEnv` | `supply_chain/l_program_env.py:427` | `Box(±normalizer.clip, (len(OBSERVATION_FIELDS),))` | `Discrete(3)` | `RewardScales` (line 324) |
| `PreventiveReserveV2Env` | `supply_chain/v2_preventive_env.py:181` | `Box(-inf,inf,(len(OBSERVATION_FIELDS),))` | `Discrete(len(RESERVE_TARGETS))` | preventive-reserve |
| `ProgramEConvoyEnv` | `supply_chain/dra2_policy_env.py:41` | `Box(-10,10,(len(OBSERVATION_KEYS),))` | `Discrete(2)` | convoy dispatch |
| `ProgramLRouteRecourseEnv` | `supply_chain/program_l_route_recourse_env.py:66` | `Box(-10,10,(len(OBSERVATION_KEYS),))` | `Discrete(3)` (2 if recourse off) | route recourse |
| `K3Env` | `scripts/run_k3_ppo.py:36` | `Box(0,10,(6,))` | `Discrete(len(LEVELS))` | `served/D0 − 0.25·after/D0` (line 83) |
| `MaintenanceEnv` | `scripts/run_paper2_rl.py:37` | `Box(0,1,(9,))` | `Discrete(3)` | `−wsl/2564` (line 56) |
| `PerishEnv` | `scripts/run_paper2b_rl.py:33` | `Box(0,5,(L+2,))` | `Discrete(len(ORDER_LEVELS))` | `−(wsl + λ·ww)/D0` (line 51) |
| `GEnv` | `scripts/run_program_g_learner.py:51` | `Box(-5,5,(7,))` | `Discrete(3)` | Program G |
| `TrackBCampaignEnv` | `scripts/run_track_b_campaign.py:122` | inherits from probe env | inherits | campaign |
| `MarkovBanditEnv` | `scripts/sanity_markov_retention.py:33` | `Box(0,1,(3,))` | `Discrete(2)` | `1 if a==z` |

**Wrappers** (all `gym.Wrapper`): `supply_chain/track_bp_env.py:48` (`TrackBPreventiveEnv`, Box action) and `:167` (`FixedBufferTrackBEnv`); `supply_chain/track_c_env.py:54`; `supply_chain/continuous_its_env.py:45` (`ContinuousItsTrackAEnv`, Box), `:273` (`PerOpBufferTrackAEnv`, Box), `:432` (`PerOpBufferMultiDiscreteTrackAEnv`, `MultiDiscrete`); `supply_chain/event_triggered_env.py:30`; `supply_chain/thesis_decision_env.py:24` (`MultiDiscrete([6,3])`) and `:319` (`Discrete18TrackAEnv`); `supply_chain/dkana_env.py:34, :252` (`MultiDiscrete([6,3])` / `[6,6,6,3]` / Box).

**Env factories**: `supply_chain/external_env_interface.py` (1040 lines) — `make_shift_control_env:702`, `make_track_b_env:717` (reward `ReT_seq_v1`, obs `v7`, action `track_b_v1`, risk `adaptive_benchmark_v2`, 168 h step), `make_thesis_aligned_training_env:735`, `make_dkana_*:765/779`, `make_thesis_factorized_track_a_env:793`, `make_discrete18_track_a_env:806`, `make_continuous_its_track_a_env:818`, `make_per_op_buffer_track_a_env:831`.

## 1.3 Training entry points

- **`<HOME>/Projects/research/scres-ia/train_agent.py`** — canonical PPO trainer. `PPO` + `DummyVecEnv` + `VecNormalize`. Defaults: `--n-envs 1`, `lr 3e-4`, `n_steps 1024`, `batch 64`, `n_epochs 10`, `gamma 0.99`, `gae_lambda 0.95`, `clip 0.2`, `ent_coef 0.01`, `eval-episodes 30`, `step-size-hours 168`.
- **`reproduce_paper.sh`** — 7 steps, `SEEDS="11 22 33 44 55"`, `TIMESTEPS=500000`, `MAX_STEPS=260`. Header comment: **"~48h on M3 Pro (all experiments)"**, smoke ~30 min.
- **`scripts/train_program_o_ret_learner.py`** — `sb3_contrib.RecurrentPPO`, one frozen seed per invocation (the Paper-2 Level-3/4 learner).
- **`scripts/train_program_e_ppo.py`** — `MaskablePPO`.
- **`scripts/train_dkana_behavior_clone.py`**, `scripts/run_confirmatory_ppo_ladder.py`, `scripts/run_track_a_v2_conservation_ppo.py`, `scripts/run_thesis_decision_ppo_smoke.py`, `scripts/run_k3_ppo.py`, `scripts/run_program_g_learner.py`, `scripts/run_meta_learner_over_configs_v1.py`, `scripts/run_meta_learner_thesis90_v1.py`, `scripts/benchmark_ppo_thesis.py`, `scripts/benchmark_control_reward.py`.
- RecurrentPPO users (13 files): `scripts/train_program_o_ret_learner.py`, `evaluate_program_o_ret_learner.py`, `run_track_b_smoke.py`, `eval_track_b_cross_scenario.py`, `comprehensive_resilience_eval.py`, `build_david_sandbox_notebook.py`, `build_david_c6b_per_batch_notebook.py`, `compare_garrido_dynamic_vs_static.py`, `export_trajectories_for_david.py`, `benchmark_control_reward.py`, `audit_track_b_all_rewards.py`, `run_thesis_decision_ppo_smoke.py`, `kaggle/track_b_sac_td3_confirm/scresia_track_b_sac_td3_confirm.py`.

## 1.4 The canonical Program-O learner protocol (`contracts/program_o_ret_only_learner_v1.json`)

- `sb3_contrib.RecurrentPPO`, `MlpLstmPolicy`, `lstm_hidden_size=64`, `net_arch=[64,64]`
- lr `3e-4`, `n_steps=512`, `batch_size=64`, γ=0.99, λ=0.95, clip 0.2, ent_coef 0.01
- seeds **8101–8110**, `requested_timesteps=200000`, `executed=200192` (391 rollouts × 512)
- training tapes 748100001–748350250 (250,250 skeletons); comparator calibration 7480001–7480048; virgin confirmation 7480101–7480148
- gate thresholds: `simultaneous_lcb95_minimum_each_cell = 0.01`, `favorable_tapes_minimum = 34/48`, `positive_learner_seeds_minimum = 8`

---

# 2. KAN code and KAN-vs-MLP results

## 2.1 Implementations

| File | What it is |
|---|---|
| `<HOME>/Projects/research/scres-ia/scripts/real_kan_extractor.py:28` | `RealKANFeaturesExtractor` — **official `pykan` `kan.KAN`** as an SB3 `BaseFeaturesExtractor`. `width=[input, hidden_width, features_dim]`, `grid=3`, `k=3`, `grid_range=(-6,6)`, `auto_save=False`, `save_act=False`, `symbolic_enabled=False`. Docstring records **"~160x forward-pass slowdown at batch=1 on this machine"** with the interpretability flags on. |
| `scripts/kan_extractor.py:18` | `RBFKANFeaturesExtractor` — RBF/KAN-*inspired* additive layer with a **linear skip**; docstring explicitly says it avoids the third-party KAN dependency. Not a real KAN. |
| `scripts/build_garrido_fig5_surrogate.py:106` `fit_kan` | supervised `KAN(width=[d,4,1], grid=3, k=3, ...)`, Adam lr 0.01, 600 steps. Paired with `fit_mlp:78` = `Linear(d,16)-Tanh-Linear(16,16)-Tanh-Linear(16,1)`, Adam lr 0.01, 600 steps. **These are the two networks reused by both prediction-premium runners.** |
| `scripts/belief_extractor.py:52` | `RealKANBeliefExtractor` |
| `scripts/ruta_b_aux_extractor.py:58` | `RutaBRealKANAuxFeaturesExtractor` |
| `scripts/run_track_b_kan_sidecar.py:62` `train_ppo_kan`; `scripts/run_track_b_real_kan_sidecar.py:75` `train_ppo_real_kan`; `scripts/run_track_b_risk_belief_sidecar.py:263`; `scripts/run_track_b_belief_encoder_sidecar.py:109` | PPO training entry points for KAN sidecars |
| `scripts/audit_real_kan_interpretability.py`, `scripts/run_kan_scres_demo.py` | interpretability / demo |

## 2.2 KAN vs MLP — what was measured, what came out

**(a) Supervised, Garrido Fig. 5 panel (`results/garrido_wrap_q1/result.json`, `results/garrido_fig5_surrogate/result.json`)** — 90 configurations, 5-fold grouped-by-seed CV, SESOI `0.05`, elapsed 14.9 s:

| Task | linear | backprop MLP | KAN |
|---|---|---|---|
| B1 held-out R² | **0.96975** | 0.98632 (Δ +0.01657, CI95 [+0.00484, +0.02830]) | 0.99133 (Δ **+0.02158**, CI95 [+0.00416, +0.03900]) |
| B2 activation accuracy | 0.7111 | 0.7178 (Δ +0.0067, CI [−0.1392, +0.1525]) | 0.7711 (Δ +0.0600, CI [−0.0166, +0.1366]) |

`q1_decision.decision = "NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL"`, `selected_model_before_gates = "linear_null"`, `eligible_neural_models = []`. Task A (the figure as drawn) is an **identity**: `r2 = 1.0`, `max_abs_identity_error = 3.22e-15`, status `IDENTITY_NOT_A_LEARNING_TASK`.

**(b) Supervised, buffer profile** — see §4.

**(c) Online RL sidecars (Track B):**
- `docs/REAL_KAN_SIDECAR_CONFIRMATORY_VERDICT_2026-07-03.md`: PPO+Real-KAN 5 seeds × 60k, `order_level_ret_mean = 0.005700` vs best static `s2_d1.50` `0.005214` (+0.000485), 5/5 seeds positive. Cost index `0.971` vs static `0.667`.
- `docs/REAL_KAN_10SEED_EXTENSION_VERDICT_2026-07-03.md`: 10-seed merge — Excel-ReT KAN `0.005938` vs best static `0.005440` (+0.000499, 10/10 seeds, CI95 [+0.000483, +0.000515]); **vs canonical PPO+MLP `0.005898` → +0.000041, 10/10 seeds, CI95 [+0.000025, +0.000057]**. Cost caveat: KAN shift-utilization index `0.982` vs PPO+MLP ≈`0.68`. Verdict: strong sidecar, **does not replace the PPO+MLP spine**.
- `docs/PROMISING_LANES_REGISTRY.md:415`: 3-seed/30k paired vs PPO+MLP seeds 1–3 → mean **+0.000015, 2/3 seeds positive — "essentially a tie"**, at half the training budget (30k vs 60k).
- `docs/TRACK_B_ARCHITECTURE_FAIR_BAKEOFF_VERDICT_2026-07-03.md` (5 seeds × 60k, obs v9): `ppo_mlp` **0.005920** > `ppo_dmlpa_positional` 0.005871 > `ppo_mlp_history` 0.005832. DMLPA − MLP = **−0.000049** (2/5 seeds). Plain MLP wins.
- Artifacts: `outputs/experiments/track_b_real_kan_sidecar_2026-07-03/{plumbing_smoke, timing_probe, timing_probe_30k, confirm_3seed_30k_h104, confirm_5seed_60k_h104, confirm_10seed_extension_6_10_60k_h104, post_cdc_only_5seed_60k_h104, lastchance_30k_seeds2_3}`; `track_b_realkan_batch_size_sweep_{32,64,128,256,512}_3seed_30k_2026-07-06`; `track_b_kan_noskip_sidecar_2026-07-03`; `track_b_kan_sidecar_2026-07-02`; `track_bp_kan_cellA_11d_3seed_60k_2026-07-09`; `real_kan_interpretability_2026-07-04`. Checkpoints: `.../models/seedN/ppo_real_kan_model.zip` + `vec_normalize.pkl`.
- Belief trunks: `outputs/experiments/track_b_v10_belief_pretrain_dataset_2026-07-04/{mlp_belief_trunk.pt, real_kan_belief_trunk.pt}`.

**(d) Contract status of KAN**: `contracts/garrido_expanded_des_e_star_v1.json:170` — `"separable_univariate_structure": "KAN_as_a_predeclared_comparator_not_a_rescue"`; line 195 lists `"PPO/MLP/KAN training"` as **forbidden before the Submission-A receipt**.

---

# 3. MPC / DP / rollout comparators

## 3.1 Code

| File | Content |
|---|---|
| `<HOME>/Projects/research/scres-ia/supply_chain/program_o_state_rich.py` (824 lines) | The Paper-2 classical family. `finite_state_rich_configurations():138` returns exactly 10 frozen configs: `base_stock__{1,2}`, `max_pressure__{0,5000}`, `min_cost_flow__{1,2}`, **`belief_mpc__{3,4}`**, `belief_dp__{3,4}`. `_deterministic_mpc_action():235` enumerates `product(range(4), repeat=effective_horizon)` — **4³ = 64 (or 4⁴ = 256) action sequences against a closed-form expected-demand model, zero DES calls**. `_belief_dp_action():292` is a seven-branch weekly-count approximate belief DP. |
| `supply_chain/expanded_contract_controllers.py` | `StaticPosture:96`, `DDMRPController:110` (real DDMRP: decoupled lead time, rolling ADU, red/yellow/green, net-flow, dynamic buffer adjustment), **`ReceedingHorizonMPC:188`** — plans by *re-running the real DES* over prefix+candidate under CRN, and **counts `self.plan_calls` (lines 204/208/216)**. Docstring: "That is expensive and it is the honest cost of planning in a contract where the cheap transducer is invalid." |
| `supply_chain/expanded_contract_controllers_v2.py` | `VectorStaticPosture:59`, `ProjectedDDMRPController:76`; state-conditioned replay MPC lives in the v2 runner |
| `supply_chain/route_horizon_dp.py` | Full-horizon backward-induction DP over `(epoch, convoy_free_at)`; `solve():82`, `evaluate_schedule():146`, `myopic_schedule():175` |
| `supply_chain/replenish_ret.py` | `rollout_policy`, `rollout_actions`, `sS_policy`, `paced_policy`, `periodic_calendars` |
| `scripts/run_k3_strong_mpc.py` (155 lines) | The K3 terminal pre-learner: grid-searches `mpc_grid` = 3×7×5 = 105 `paced_policy(α,β,γ)` params, vs `periodic_static` (2690 calendars), `budgeted_sS` (grid), `inventory_paced` |
| `scripts/run_expanded_contract_comparators.py` / `_v2.py` | Garrido's "step 3: MPC on the expanded contract" runners |
| `scripts/run_program_h_belief_audit.py`, `scripts/run_program_h_information_audit.py`, `supply_chain/program_h_belief.py` | belief policies (regret fitted-Q, belief MPC) |

## 3.2 Where "MPC is cheap and ties RL" lives

**Primary evidence — Program O Level 4** (`papers/paper2/results_table.json`, `results_table.md`, prose in `papers/paper2/03_results_draft.md:49-59`, `04_discussion_draft.md:29-36`, `01_introduction_draft.md:38-44`):

> Δ_N (RecurrentPPO − best frozen classical, per cell):
> - `rho75_share90`: **−0.00165**, LCB95 **−0.00879**, 1/10 seeds beat both comparators
> - `rho90_share75`: **−0.00273**, LCB95 **−0.01400**, 0/10 seeds
> - `rho90_share90`: **−0.00150**, LCB95 **−0.00828**, 2/10 seeds

Source of truth: `results/program_o/ret_only_learner_v1/calibration_run/result.json` → `inference.estimates["<cell>::H_neural"]` (exact SEs 0.0020756 / 0.0032732 / 0.0019720). `terminal_verdict = "PENDING_DIRECT_FULL_DES_REPLAY_AND_INTEGRITY_AUDIT"`; the frozen contract verdict is `STOP_CALIBRATION_NOT_ELIGIBLE`.

The prose claim, verbatim from `04_discussion_draft.md:31-35`: the MPC "replans online at every decision, requires an explicit decision model and belief machinery at run time, and its cost scales with the enumeration horizon; the learned policy is a constant-time function of observations."

**Secondary — Program O H_obs (belief-MPC beats open-loop)**: `research/paper2_exhaustive_search/PROGRAM_O_DUAL_RESOURCE_DIAGNOSTIC_VERDICT_2026-07-15.md:43-48`

| policy | cell | effect | favorable | LCB |
|---|---|---|---|---|
| `belief_mpc__3` | rho75_share90 | 0.06481 | 39/48 | 0.00814 |
| `belief_mpc__3` | rho90_share75 | 0.07173 | 43/48 | 0.00818 |
| `belief_mpc__3` | rho90_share90 | 0.10125 | 39/48 | 0.01617 |
| `belief_mpc__4` | rho75_share90 | 0.06613 | 40/48 | 0.01082 |
| `belief_mpc__4` | rho90_share75 | 0.07014 | 44/48 | 0.00535 |
| `belief_mpc__4` | rho90_share90 | 0.09935 | 40/48 | 0.01370 |

`belief_mpc__3` is the frozen primary (`contracts/program_o_fixed_clock_physical_hobs_validation_v1.json:59`, chosen for "shorter three-week horizon and parsimony").

**K3 (retracted) — MPC beats classical, then PPO collapses to a fixed schedule**:
- `results/k3/strong_mpc_terminal.json`: `strong_mpc` ret_order **0.94095** vs `budgeted_sS` 0.92853, `inventory_paced` 0.92819, `periodic_static` 0.90376. Δ = **+0.01242 [+0.00546, +0.01928]**, all six gates pass, `verdict = PROMOTE_K3_TO_CONFIRMATION`, `ordered_D0` deviation exactly 0.
- `results/k3/ppo_virgin.json`: PPO seeds 0–3 `ret_mean = 0.96979` and beat MPC by +0.01771 [+0.01042, +0.02604]; seed 4 `0.959375` does not.
- **`results/k3/open_loop_confound_audit.json`** — `verdict = "RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND"`. `ppo_seed0_unique_test_sequences = 1`, `ppo_seed0_minus_fixed_ret = [0,0,0]`. Interpretation field: *"PPO learned one fixed eight-week open-loop schedule … the fixed schedule matches PPO exactly and beats the tested MPC under identical resources; therefore neither neural incremental value nor observable adaptive headroom is established."*

**Expanded-contract MPC (Garrido's step 3)** — `results/expanded_contract_comparators/result.json`, `claim_status = "DEVELOPMENT_INSTRUMENT_DIAGNOSTIC_NOT_VALID_MPC_OR_DDMRP_ADJUDICATION"`, elapsed 125.0 s, 6 tapes, 52-week horizon, 4-week epochs. R1r: `mpc_receding_horizon` −6.30e-05 vs best static, `ddmrp_dynamic` −1.65e-04, `static_I168` best. Explicitly reclassified (`reclassified_2026_07_29`): must **not** be claimed as "ddmrp defeated" / "garrido step 3 expanded mpc" — the MPC ignores observed state and plans on unrelated seeds, only 6 of 216 postures searched.

**Ladder as declared** (`contracts/garrido_expanded_des_e_star_v1.json:137-148`):
`best_constant → lookup_or_order_up_to → threshold_or_hysteresis → tree_or_tabular → spline_or_GAM → DP_or_rollout → direct_DES_MPC → robust_or_scenario_MPC → belief_DP_or_belief_MPC → neural_controller_only_after_observable_structured_residual`

---

# 4. The measured "neural premium" evidence

## 4.1 Curvature 0.076 vs noise 0.317 — MLP WORSE than linear

**Artifact**: `<HOME>/Projects/research/scres-ia/results/headroom/buffer_prediction_premium/result.json`
- `self_sha256 = ae1a545bb4274e75859e6665ceacbe129d8683af0d05c025e8f5bdbddc5a787a`
- `claim_status = "NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE"`
- `schema_version = buffer_prediction_premium_v1`, `target = "ret_excel_risk_conditional"`, `n_rows = 1530`, `sesoi = 0.05`, `elapsed_seconds = 198.207`
- Design: 17 buffer levels (0…1344 h step 84) × 3 families (`R1r`, `R2r`, `R1r+R2r`) × 3 escalations (`base`, `freq_x3`, `freq_x5`) × 10 seeds (6800001–6800010) = 1530 episodes; 5-fold CV **grouped by seed**
- Runner: `scripts/run_buffer_profile_prediction_premium.py`; preregistration `docs/PREREGISTRO_PRIMA_PREDICCION_BUFFER_2026-08-01.md` (sha256 `ebb65c9c3043faa6363ddd1ee89fb523ef24b7d7a2f0571b46750f1996d766b7`)

`held_out_r2_mean`:

| model | R² | Δ vs linear | CI95 | passes SESOI 0.05? |
|---|---|---|---|---|
| constant | **−0.00336** | | | |
| **linear** | **0.68261** | — | — | — |
| **backprop (MLP 16-16 tanh)** | **0.55483** | **−0.12778** | **[−0.31568, +0.06012]** (sd 0.21436) | **no — WORSE than linear** |
| KAN | **0.71632** | +0.03371 | [−0.07874, +0.14616] (sd 0.12828) | no |

MLP per-fold spread `[0.7327, 0.4834, 0.7349, 0.5589, 0.2642]` vs linear `[0.7312, 0.6179, 0.6312, 0.7045, 0.7283]`.

**Curvature**: `profile_curvature.mean_one_minus_linear_r2 = 0.07625893719919702` (**0.076**), by cell:
`R1r|base 0.08400`, `R1r|freq_x3 0.08329`, `R1r|freq_x5 0.13429`, `R2r|base 0.02388`, `R2r|freq_x3 0.03708`, `R2r|freq_x5 0.07562`, `R1r+R2r|base 0.05870`, `R1r+R2r|freq_x3 0.07860`, `R1r+R2r|freq_x5 0.11088`.

**Noise**: `1 − 0.6826 = 0.3174` (**0.317**) — unexplained episode-level variance.

**The claim** (`docs/RESULTADO_PRIMA_PREDICCION_BUFFER_2026-08-01.md:39-42`):
> "Una prima neural exige que la curvatura de la superficie supere el ruido que la oculta. Aquí `0,076 ≪ 0,317`, y por eso la capacidad extra no se paga."

`q1_panel_reference` in the same artifact: linear `0.96975`, backprop `0.98632`, KAN `0.99133` — the near-linear ρ→ReT panel this is contrasted against.

All six falsifiers pass: `f1_the_surface_actually_has_curvature`, `f2_no_driver_leakage` (7 features), `f3_folds_are_grouped_by_seed` (5 folds, 10 seeds), `f4_linear_baseline_is_not_a_straw_man`, `f5_sesoi_was_fixed_in_advance`, `f6_seeds_are_virgin` (350 prior seeds scanned, 0 collisions).

## 4.2 ⚠️ The 0.076 ≪ 0.317 claim has been formally WITHDRAWN

`docs/RESULTADO_PRIMA_CD_2026-08-01.md:90`:
> "La afirmación *«la curvatura está por debajo del ruido»* **sigue retirada**: comparaba falta de ajuste sobre medias contra error predictivo sobre episodios."

The buffer artifact is **superseded**: `results/headroom/cd_surface_prediction_premium/result.json` has `supersedes` pointing at `results/headroom/buffer_prediction_premium/result.json`. Any paper that uses 0.076 vs 0.317 must reinstate it on a like-for-like basis (both on episodes, or both on cell means).

## 4.3 The superseding run — Cobb–Douglas surface

`<HOME>/Projects/research/scres-ia/results/headroom/cd_surface_prediction_premium/result.json`
- `self_sha256 = 527227bd96eb8d8d9ed534aebea410da79efb91bb2d1e2aa365e85b5a9f43a99`, `claim_status = "PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED"` (the doc explicitly labels this runner label as **over-claiming**)
- `n_rows = 1224`, `elapsed_seconds = 164.903`, `target = "R_cobb_douglas"`, `sesoi = 0.05`, daily cadence, runner `scripts/run_cd_surface_prediction_premium.py`, preregistration `docs/PREREGISTRO_PRIMA_CD_2026-08-01.md`

`held_out_r2_mean`:

| model | R² |
|---|---|
| `train_cell_mean_comparator` | **0.69308** |
| `spline_buffer` | 0.63652 |
| **`linear_interactions`** (primary baseline) | **0.63059** |
| `tree` (CART) | 0.62247 |
| `linear_additive` | 0.60620 |
| **`kan`** | **0.60186** |
| **`backprop`** | **0.58414** |
| `constant` | −0.01675 |

Contrasts (`t(4) = 2.776`): available margin **+0.0625, CI95 [−0.0606, +0.1856]** (crosses zero); KAN − primary **−0.0287 [−0.1048, +0.0473]**; backprop − primary **−0.0465 [−0.1388, +0.0459]**. **Both networks below the classical baseline; the spline beats both.** Declared defects: `known_defect_baseline_selection` (best_classical chosen looking at test folds), `estimand_declaration` (CD target recalibrated per fold), and `f7` is a post-hoc non-binding amendment.

## 4.4 Δ_N, TOST / equivalence

- **Definition**: `contracts/garrido_expanded_des_e_star_v1.json:133` — `"Delta_N": "V(pi_neural) - V(pi_best_structured) in closed-loop rollout"`. Also `contracts/g3a_asymmetric_claimants_v2.json:95`, `contracts/program_s_product_mix_risk_interaction_gsa_v1.json:187`, `docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:118`.
- **Rules**: `contracts/program_q_frozen_policy_replication_v1.json:83-86` — `Delta_N = learner − max(classical)`; `neural_premium_rule: LCB95(Delta_N) >= 0.01 in every cell`; `equivalence_rule: simultaneous CI95(Delta_N) wholly inside [-0.01,+0.01] in every cell`.
- **Implementation**: `scripts/adjudicate_program_q.py:34-48` — the "TOST" is implemented as a **two-sided simultaneous-CI containment test** (`lcb95 >= -0.01 and ucb95 <= 0.01`), not a literal two-one-sided-t function. Power: `scripts/power_program_q_replication.py:100-130` (studentized simultaneous max-t bootstrap, reselecting the 65,536-calendar and 10-classical maxima inside every resample). Audit: `scripts/audit_program_q_power_preopen.py:35`.
- **Power result** (`results/program_q/power_preopen_v5_20260718/artifacts/program_q_power_v1.json`, mirrored in `research/paper2_exhaustive_search/program_q_power_preopen_v5_verdict_20260718.json:25-28`):

| N/cell | H_OL power | Delta_N_equivalence power | joint |
|---|---|---|---|
| 128 | 1.0 | 0.528 | 0.528 |
| 160 | 1.0 | 0.653 | 0.653 |
| 192 | 1.0 | 0.7595 | 0.7595 |
| **256** | 1.0 | **0.8755** | **0.8755** |

Selected N = 256, reserved block 7490001–7490256, `"opened": false`. Terminal verdicts available: `PASS_Q_NEURAL_PREMIUM` / `PASS_Q_LEARNED_ADAPTATION_CLASSICALLY_EQUIVALENT` / `BOUND_Q_LEARNED_ADAPTATION_ONLY` / `STOP_Q_NO_REPLICATED_LEARNED_ADAPTATION`.
- Tests: `tests/test_program_q_contract.py:47, 201-204, 220`; `tests/test_david_sandbox_notebook.py:67` (`Delta_N_LCB05_dev`).
- Notebook harness that recomputes Δ_N live: `scripts/build_david_sandbox_notebook.py:683-684, 721-725, 757, 773, 878` (hardcodes the historical `Delta_N = −0.001645866…, −0.002732454…, −0.001496518…`).

## 4.5 Other measured "no premium" endpoints

- `docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md` (canonical status doc, 2026-08-01) — WRAP/Q1: no material neural premium, linear explains the surface; G2: `THRESHOLD_RULE_SUFFICES`; Program Q: state-dependent value vs open-loop but "no demostró una prima neural … ni superioridad sobre un belief-MPC específico".
- `results/headroom/g2_autotomy_threshold/result.json` → `claim_status = "THRESHOLD_RULE_SUFFICES"`
- `results/headroom/g3_obs_conversion_v2/result.json` → `"STRUCTURED_CONTROL_SUFFICES_G3_OBS"`
- `results/program_g/terminal_metric_audit/verdict.json` → `"STOP_PROGRAM_G_NO_ROBUST_ADAPTIVE_VALUE_UNDER_STYLIZED_CONTRACT"`
- `results/program_m/hpi_validation_v1/result.json` → `"HPI_VALIDATION_NO_PASSING_CONNECTED_REGION__STOP_BEFORE_HOBS"`
- `results/garrido_neural_headroom_gate_v1/result.json` → `"HOLD_E1_PLACEBO_NOT_OPENED"`
- `results/paper2/rl_convertibility.json`: H_PI `419.1 [275.3, 578.1]`, PPO best H_obs **−4.4**, `any_seed_beats_static = false` (6 seeds). `results/paper2b/rl_convertibility.json`: H_PI `564.0 [420.7, 712.8]`, PPO beats static in **6/6** seeds (H_obs_ppo `204.2 [119.7, 296.0]` etc.), `ppo_minus_basestock` positive.

---

# 5. Compute / latency instrumentation — and `Delta_efficiency`

`Delta_efficiency` appears in **exactly one place**: `contracts/garrido_expanded_des_e_star_v1.json:134` —
> `"Delta_efficiency": "quality_and_guardrails_matched while online latency or DES calls decrease"`

There is **no runner that computes `Delta_efficiency` today**, but every ingredient exists:

| Instrument | File | What it measures |
|---|---|---|
| **Per-decision latency** | `<HOME>/Projects/research/scres-ia/scripts/benchmark_program_q_latency.py` | `benchmark_callable(policy, observations, warmup=100, repeats=1000)` → `median_ms`, `p95_ms`, `mean_ms`, `batch_size=1`, `failures`, `tracemalloc_peak_bytes`, plus `setup_seconds_excluded_from_online_latency`, python/platform/torch/`torch_threads`. Schema `program_q_latency_benchmark_v1`. Refuses to overwrite (`FileExistsError`). Takes any `module:function` factory. **This is the Delta_efficiency instrument.** Tested at `tests/test_program_q_contract.py:220`. |
| **DES / planner calls** | `supply_chain/expanded_contract_controllers.py:204, 208, 216` — `ReceedingHorizonMPC.plan_calls` | counts full-DES rollouts per MPC decision; printed per tape at `scripts/run_expanded_contract_comparators.py:183` |
| **Parameter counts** | `scripts/build_david_sandbox_notebook.py:431-432`, `scripts/build_david_c6b_per_batch_notebook.py:379` — `architecture_audit()` | `total_parameters`, `trainable_parameters`, `policy_class`, `source_sha256`, `history_length`, `observation_shape`, `device`, versions of torch/gymnasium/sb3/sb3-contrib, extractor dummy-forward shapes, `has_layer_norm` |
| **Wall-clock per episode** | `research/paper2_exhaustive_search/war_stress_gsa_h104_benchmark_20260716.json` | `seconds_per_episode = 1.2606108539998786` (H104, 104 weeks / 728 days, seed 94700001), `policy_wall_seconds = [1.2653, 1.2559]` |
| **Compute feasibility gate** | same file, `compute` block | `parent_atlas_full_projected_episodes = 86,749,056`; `projected_workers = 32`; `projected_wall_days = 39.43`; `compute_gate_pass = false` |
| **Compute benchmark harness** | `scripts/benchmark_program_s_s1_compute_v1_1.py` | `WALLTIME_LIMIT_SECONDS = 7*24*3600`, `WORKERS = 2`, writes `results/program_s/s1_compute_benchmark_v1_1/result.json` |
| **Training wall-clock** | `results/program_e/ppo/training_verdict.json` | MaskablePPO **170.44 s for 200,000 timesteps** per seed (9301–9310); smoke 4.97 s for 5,000 steps |
| **KAN forward-pass cost** | `scripts/real_kan_extractor.py` docstring; `docs/REAL_KAN_SIDECAR_PREREGISTRATION_2026-07-03.md:31-42`; `docs/PROMISING_LANES_REGISTRY.md:415` | pykan with defaults **0.046 s vs 0.00028 s per single-sample call = ~160×** at width `[52,32,16]`, batch=1; with flags off ≈ **17 s of pure extractor cost for 60k rollout steps**, ≈**0.011 s per gradient step at batch=256** |
| **Run-level elapsed** | `elapsed_seconds` field written by ~40 runners (`time.perf_counter`) | e.g. `results/headroom/buffer_prediction_premium/result.json` 198.21 s / 1530 episodes ≈ **0.13 s/episode**; `cd_surface_prediction_premium` 164.90 s / 1224 ≈ **0.135 s/episode**; `garrido_meta_learner_h3power_local` 13,373.6 s / 288 configs ≈ **46 s/config** |
| **Declared compute envelope** | `contracts/program_o_relevant_risk_sensitivity_v1.json:45` | "18 configs × 12 tapes × ~45 policy-sequences ≈ 9,720 full-DES episodes; at **1–5 s/episode** ≈ 3–14 h on the local 10-core pool" |
| **Cross-machine calibration** | `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md:35-36` | 1,728 episodes/replica ≈ **126 s on M1 Pro**, ≈**504 s on the VPS** (VPS ~4× slower) |

The paper text already commits to reporting this separately: `papers/paper2/03_results_draft.md:68-70` ("A mandatory computational benchmark (per-decision latency, memory, planner cost at equal hardware)…"), `02_methods_draft.md:110`, `04_discussion_draft.md:29-36`.

---

# 6. Episode cost, step count, vectorization

**Cost per episode**
- Full DES, 104-week horizon: **1.2606 s/episode** (machine-measured, `war_stress_gsa_h104_benchmark_20260716.json`)
- Declared planning range: **1–5 s/episode** (`contracts/program_o_relevant_risk_sensitivity_v1.json:45`)
- Short/reduced runs: ~**0.13 s/episode** (buffer & CD premium runners, `sim.run()` cadence, 52-week horizon)
- Program-O RecurrentPPO: 200,192 timesteps × 8 decisions/episode = **25,024 episodes per seed**; 10 seeds trained on a 250,250-skeleton block

**Steps per episode**
- `MFSCGymEnv` / `MFSCGymEnvShifts`: `max_steps = (SIMULATION_HORIZON − warmup) / step_size_hours`, default `step_size_hours = 168.0` (weekly). Track B canonical `max_steps = 104` (104 weeks); `reproduce_paper.sh` uses `MAX_STEPS=260`.
- Program O-R: exactly **8 weekly decisions** + clearance horizon (`contracts/program_o_ret_only_learner_v1.json` `decision_weeks: 8`, `batch_slots_per_week: 3`).
- `docs/ACTION_PROPAGATION_AND_CADENCE_2026-06-27.md:42`: daily decisions cost **~7× steps/episode**.

**Vectorization / parallel rollout**
- **`SubprocVecEnv` is used in ZERO `.py` files** (verified repo-wide, excluding `.venv`). Only `DummyVecEnv` / `make_vec_env` / `VecNormalize` — i.e. **in-process, serial env stepping**. (One `SubprocVecEnv` string exists in `notebookffc7b2c5ff.ipynb`.)
- `train_agent.py` `--n-envs` default **1**.
- Scripts that use `n_envs=4`: `scripts/run_track_b_architecture_fair_bakeoff.py:87`, `scripts/run_per_op_explore.py`, `scripts/run_ret_refinement.py`, `scripts/run_track_b_campaign_r2_r24.py:150`. `scripts/formal_evaluation.py:79` reads `n_envs` from config (default 1).
- Parallelism outside SB3 exists via process pools in sweep runners and `WORKERS`/`projected_workers` fields (32 workers projected for the war-stress atlas; 2 workers in `benchmark_program_s_s1_compute_v1_1.py`; "local 10-core pool").
- Kaggle/VPS offloading is the actual scaling path: 30 `kaggle/*` job directories, `results/program_o/ret_only_learner_v1/vps_run/` with a full custody harness (`custody/producer_control.json`, `watcher_state.jsonl`, `model_files.sha256`).

**Measure-through-the-pipeline rule**: `CLAUDE.md:78` — "Measure through the pipeline (`arm_runner.py`), never with an ad-hoc script" → `supply_chain/arm_runner.py`.

---

# 7. Training results directories with learned policies

## 7.1 Frozen, contract-governed checkpoints

| Path | Contents | Status |
|---|---|---|
| `<HOME>/Projects/research/scres-ia/results/program_o/ret_only_learner_v1/vps_run/models/` | `recurrent_ppo_seed_8101…8110.zip` (10 RecurrentPPO checkpoints) + `training_manifest_seed_81NN.json` each (`contract_sha256 4471ffd4…`, `executed_timesteps 200192`, `model_sha256`, `git_commit 02aad62f`, `tape_range [748100001, 748125025]`) | training `result.json` → `status = "COMPLETE_PENDING_EVALUATION"`; custody in `vps_run/custody/model_files.sha256` |
| `results/program_o/ret_only_learner_v1/calibration_run/result.json` | Level-3/4 estimands (§3.2, §4.4) | `terminal_verdict = "PENDING_DIRECT_FULL_DES_REPLAY_AND_INTEGRITY_AUDIT"`; amendment gates `base_cells: false`, `base_lcb: false`; contract verdict **`STOP_CALIBRATION_NOT_ELIGIBLE`** |
| `results/program_e/ppo/` | `maskable_ppo_seed_9301…9310.zip` + `training_verdict.json` (`algorithm: MaskablePPO`, 200,000 steps, 170.4 s/seed) | `PROGRAM_E_PPO_TRAINING_COMPLETE`; `results/program_e/ppo_smoke/maskable_ppo_seed_9399.zip` → `PROGRAM_E_PPO_TECHNICAL_SMOKE` |
| Hash freeze for Program Q | `research/paper2_exhaustive_search/program_q_historical_recurrentppo_fallback_freeze_20260717.json` | `historical_candidate_hashes: PASS_10_OF_10`; retraining forbidden |

## 7.2 Development checkpoints under `outputs/experiments/` (575 directories)

- `track_a_v2_conservation_ppo_5seed_40k_2026-07-03/seed_{1..5}_best_model.zip`
- `track_b_real_kan_sidecar_2026-07-03/*/models/seed*/ppo_real_kan_model.zip` + `vec_normalize.pkl`
- `track_b_architecture_fair_bakeoff_2026-07-03/full8d_v9_history_5seed_60k_h104/models/ppo_{mlp,mlp_history,dmlpa_positional}_seed{1..5}/`
- `track_b_ruta_b_permuted_label_control_3seed_30k_2026-07-07/seed{1,2,3}/ruta_b_model.zip`
- `track_b_v10_belief_pretrain_dataset_2026-07-04/{mlp_belief_trunk.pt, real_kan_belief_trunk.pt}`
- `track_b_case_c_v10_no_forecast_belief_pretrain_r22_2026-07-06/mlp_belief_trunk_r22_case_c{,_head}.pt`
- `riskobs_audit_war_h104_2026-06-27/model.zip`, `track_b_kaggle_*_smoke_2026-07-01/models/ppo_seed1_best.zip`

## 7.3 `claim_status` values found across `results/`

`NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE` · `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED` · `THRESHOLD_RULE_SUFFICES` · `STRUCTURED_CONTROL_SUFFICES_G3_OBS` · `CONTENTION_DOES_NOT_OPEN_THE_DOOR` · `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` · `DEVELOPMENT_FIG5_SURROGATE` · `DEVELOPMENT_DRIVER_TABLE` · `DEVELOPMENT_SCREEN_NO_CLAIM` · `DEVELOPMENT_INSTRUMENT_DIAGNOSTIC_NOT_VALID_MPC_OR_DDMRP_ADJUDICATION` · `HALTED_FALSIFIER_FAILED` · `PREFLIGHT_HALTED_FALSIFIER_FAILED` · `STOP_G3C_GUARDRAIL` · `HOLD_E1_PLACEBO_NOT_OPENED` · `HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION` · `REPLAY_OF_BURNED_BLOCK_NO_NEW_CONFIRMATION` · `RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND`

---

# 8. Two governance facts that constrain any paper plan

1. **`contracts/garrido_expanded_des_e_star_v1.json:4`** — `status = "DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT"`; `neural_training_authorized: false`; line 192–197 `forbidden_before_receipt` explicitly includes **`"MPC or DDMRP scientific screens"`, `"PPO/MLP/KAN training"`, `"neural architecture selection"`**. So a fresh KAN-vs-MLP-vs-MPC-vs-RL comparison run is currently contractually blocked; only replays of burned data, synthetic tests, and flags-off harness work are permitted.
2. **`docs/PROGRAMA_PRIMA_NEURAL_2026-08-01.md:164-171`** — the canonical framing already separates the two premium questions: *"MLP, KAN, DeepSets y atención son representaciones; PPO es un algoritmo; RecurrentPPO añade memoria. No se presentan como peldaños de una misma escalera"*, and *"Una eventual prima también puede definirse por eficiencia —misma calidad con menor latencia, menos llamadas al DES o mejor generalización—, pero esa pregunta se preregistra por separado de la prima de calidad."* That is exactly the `Delta_efficiency` paper, and it is the one lane with instruments already built and no fresh-seed requirement.
