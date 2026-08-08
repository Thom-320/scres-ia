# Explore — Repo: /Users/thom/Projects/research/scres-ia.

- **agent_id**: `aacdc7972ec8de620`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-07T16:49:02.251Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-aacdc7972ec8de620.jsonl`
- **turns**: 101 · **assistant messages**: 2

## Task given

```text
Repo: /Users/thom/Projects/research/scres-ia.

GOAL: find every lane that is CLOSE to yielding a claim but did not get there — i.e. where a modest amount of further work could plausibly convert a negative/blocked result into a positive one.

Look for, and report with exact paths and numbers:

1. `docs/PROMISING_LANES_REGISTRY.md` — read it fully and summarise each lane's current state and what would reopen it.
2. Results whose verdict is negative/STOP but whose point estimate is POSITIVE with the interval only just crossing zero, or whose effect just missed a preregistered SESOI/bar. Search `results/**/*.json` for fields like `lcb95`, `sesoi`, `passes_sesoi_and_ci`, `h_regime`, and flag cases where the mean > 0 but lcb95 < 0, or where a measured value is within ~2x of its declared bar.
3. Results explicitly labelled as blocked on ONE fixable thing — grep docs/ and results/ for strings like "HOLD", "PREFLIGHT", "NOT_OPENED", "UNDERPOWERED", "POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT", "PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED", "blocked_domain_fact".
4. Any document listing questions to ask Garrido (search for "Q6", "Q7", "Q11", "Q13", "Q14", "R09", "reopener", "exhaustion certificate"). Report what each reopener would unlock.
5. Environments under `supply_chain/*env*.py` that have NO corresponding results directory — list them, and for each read the module docstring and report in one line what decision it exposes and what its action space is.

OUTPUT: a ranked list of "closest to a claim" lanes. For each: path/evidence, the number and how far it is from its bar, what exactly is blocking it, and an estimate of whether the blocker is (a) compute, (b) an instrument fix, (c) an external fact from Garrido, or (d) a governance/custody decision.

Read-only. Exact paths and numbers, no narrative padding.
```

## Final answer

## 1. `docs/PROMISING_LANES_REGISTRY.md` — 1,016 lines, 89 KB, last appended 2026-08-06

It is a **living append-only tracker**, not a summary. Structure: a lane table (⭐1 … 12), then ~25 dated narrative sections. Note it is **stale relative to the actual frontier** — it stops at the 2026-08-06 curvature lane and does not contain the 2026-08-07 work (`track_b_nonneural`, `dmlpa_variants`, `step3_pooled`, `TABLA_CANONICA_DE_CLAIMS`).

Lane states and what would reopen each:

| lane | state (registry line) | reopener |
|---|---|---|
| ⭐1 / 1b / 1c continuous_its × ReT_excel_delta (h52/104/260) | L27–29, 41–42: 2-seed pilots won, 3/5-seed confirms lost (0.002175 < 0.002193; CVaR 5.27e9 > 4.94e9) | none — Kaggle 5×60k `primary_win=false` |
| ⭐1b-v8 realized-risk obs | L30: Excel 0.00220 < 0.00225 but **CVaR 4.59e9 < 4.94e9** at 1 seed/20k | a CVaR-primary preregistration; never escalated past 1 seed |
| ⭐1b-preventive (hazard obs, Pareto) | L32: **3-seed Excel Pareto win** 0.002220 > 0.002156; CVaR fails (5.29e9 vs 4.94e9) | closest survivor of the family; needs a dense-CRN Pareto re-gate |
| ⭐1b-per-op-buffer | L34: PPO res 0.008 Pareto-wins only below sweet spot `op9=0.10,S1` (0.002627, res 0.017) | reward steering toward downstream Op9 |
| ⭐1d Track A headroom campaign | L36: **oracle−robust = +0.002800** real; PPO 0.036308 vs robust 0.036423; BC MSE 0.290→0.059 | "headroom exists, PPO can't convert" |
| Track A v2 conservation 5D | L146–256: gate `opening_real=True`, **oracle−best-static +0.0041566**; three preregistered PPO attempts all `positive_seeds=0/5` (−0.006316, −0.007127, −0.008660) | **CLOSED**; explicit "do not retune without a fresh conversation" |
| Real-KAN sidecar | L518–552: **10/10 seeds, +0.000041 CI [+0.000022,+0.000059]** — clears zero | already a claim; blocked on cost: `assembly_cost_index` 0.971–0.993 vs MLP 0.68 |
| ⭐13 ReT_tail_v2 | L611–648: at n=10 two metrics clear zero; **at n=15 nothing does** (ReT Excel +0.000008 [−0.000004,+0.000021]) | shrinking effect — do not reopen |
| ⭐14/15 no-forecast spine | L682–694: ReT Excel +0.000002 [−0.000013,+0.000017] — indistinguishable, promoted as *reviewer-safe*, not superior | n/a |
| ⭐16 Ruta B | L748–840: preventive claim **retracted**; prevention closed across 8+6 tiers, most exact zero | "unless the environment itself changes (a real preparation lead time)" |
| ⭐17 Track B-P | L844–975: +0.0285 CI [+0.0158,+0.0412] 5/5 — then **decomposed to a fixed posture** (Op3=0.1531, Op5=0.2480, Op9=0.2068); fixed−11D +0.000440 [−0.000799,+0.001680] | last open gate: "classical held-out per-op posture optimization" |
| Curvature / `H_regime` | L979–1016: Garrido's declared γ=1 → **0.0195 vs bar 0.05**; only convex γ=5/10/20 clears | see #4 below |

---

## 2. Positive point estimate, interval crosses zero / just misses the bar

Ranked by (point estimate ÷ bar):

| # | artifact | number | bar | gap |
|---|---|---|---|---|
| **N1** | `results/citable_risk_attitudes/result_after_anchor_fix.json` — CVaR over regimes, α=0.90/0.95/0.99 | **H = 0.0482** | 0.05 | **96.4 % of bar; short by 0.0018** |
| **N2** | `results/headroom/cd_surface_prediction_premium/result.json` `available_margin_vs_primary` | mean **+0.06248**, CI95 t(4) **[−0.06063, +0.18559]** | SESOI 0.05 | point = **1.25× bar**, CI crosses zero (df=4) |
| N2b | same file, `available_margin_vs_posthoc_best` | +0.05655 [−0.11783, +0.23094] | 0.05 | 1.13× bar, crosses zero |
| **N3** | `results/headroom/g3_obs_conversion_v2/result.json` `FIFO_PARTIAL\|R1r+R2r\|base/H_obs_windowed_minus_constant` | mean 0.02075, **lcb95 = 0.01473 > SESOI 0.01** | 0.01 | **already clears** — blocked on custody (§3) |
| N3b | same, `freq3_imp2` cell | mean 0.01288, lcb95 0.00729 | 0.01 | 73 % of bar |
| **N4** | `results/dmlpa_variants/result.json` `vs_base/dmlpa_nhead4` | +1.589 **[+0.639, +2.625]**, Holm p=0 | >0 | clears; wrong hyperparameters (hueco B2) |
| N4b | `vs_base/dmlpa_1layer` | +1.210 [+0.609, +1.983], Holm p=0 | >0 | clears; same defect |
| N4c | `vs_base/dmlpa_meanpool` | +0.472 **[−0.440, +1.394]** | >0 | positive mean, crosses zero |
| **N5** | `results/track_b_nonneural/result.json` `network_minus_constant` | MLP **+2.176**, KAN +1.949, DMLPA +1.437 | — | but `nonneural_eval_means/threshold_rule = 97.142` vs constant 96.567 → the honest neural margin over the **best non-neural** arm is only **+1.60**, and no CI is computed for that contrast |
| N6 | `results/grid_transfer_ordered_v1/result.json` `contrasts/neuron/vs_marginal_replay` | mean +0.013278, lcb95 **−0.002949** | >0 | 82 % of the way; **but** the virgin-block confirmation `results/grid_transfer_confirmation_v2/result.json` gives neuron **−0.01178 [−0.01849, −0.00484]** → closed against |
| N7 | `results/search_ladder_v5/result.json` `ofat_transfer` vs `neuron_memory` | +0.01071 **[+3.56e−05, +0.02171]** | >0 | excludes zero **by 3.6e−05** — technically a claim, declared "too small to matter" |
| N8 | `results/search_ladder_v5` `ucb1_transfer` | −0.00701 [−0.02444, +0.01408] | >0 | tie |
| N9 | `results/architecture_bakeoff/result.json` `contrasts/DMLPA_minus_KAN` | mean +0.6108, lcb95 −0.3500 | >0 | superseded; runner **never sealed anything** (TABLA_CANONICA §6) |
| N10 | `results/step3_pooled/result.json` `greedy_pi_best_found_v2` | R1r +2.374e−05 **lcb95 +4.32e−06**; R2r +0.001002 **lcb95 +8.83e−05** | >0 | both LCBs clear, but arm is excluded by design (perfect information) |
| N11 | `results/garrido_v0_surface_gates_v1/result.json` `g2.../by_context/R21` | mean +0.011057, lcb95 **−0.001122** | >0 | 91 % there; holdout replicated the STOP (`R21` lcb −0.005834, mean +0.000621) — effect collapsed OOS |
| N12 | `results/headroom/g2_autotomy_threshold/result.json` | mean **−0.0744** [−0.1751, +0.0262] | SESOI 0.05 | wrong sign, far |

---

## 3. Blocked on exactly ONE fixable thing

| id | artifact / doc | verdict string | the one thing |
|---|---|---|---|
| **B1** | `results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json` | `SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE`, `promotion_status = BLOCKED_NO_RETROACTIVE_RESEAL_AND_NO_CONTRACT_CONFORMITY` | The run used **v2 fields** (seeds 7800001–140, `lost_orders` margin 0.5) but **sealed against the legacy contract** (`docs/PREREGISTRO_G3_OBS_CONVERSION_OBSERVABLE_2026-08-01.md`, declared block 5200001–016, margin 0.25). `scope_checks.source_execution_matches_v2_fields = true`; `source_seal_matches_intended_v2 = false`. A positive result (lcb95 0.01473 > SESOI 0.01) that is unusable **for paperwork reasons only** — and the 140-seed block is now burned. |
| **B2** | `results/headroom/g3c_preflight_burned/result.json` | `PREFLIGHT_HALTED_FALSIFIER_FAILED` — **only f2 fails** (`f2_min_dwell_actually_binds_at_3_or_7_days`); f1,f3,f4,f5,f6,f7,f9 all pass | Power is already fine: `n_required_for_sesoi = 22` (base) / `23` (freq3_imp2) vs `budget_max_fresh_seeds = 96`, `within_budget = true`, `n_burned = 16`, `mde_at_n_burned = 0.01150`. Caveat: `primary_hysteresis_vs_myopic` mean is **−0.00073 / −0.00290** — wrong sign, so fixing the dwell binding is necessary but likely not sufficient. |
| **B3** | `results/garrido_neural_headroom_gate_v1/result.json` | `HOLD_E1_PLACEBO_NOT_OPENED`, `placebo_status = NOT_OPENED_BY_E1_CONTRACT`, `training_authorized = false` | `headroom/H_regime = 0.0`, `lcb95 = 0.0`, `ucb95 = 0.0`, `next_environment = NONE_UNDER_CURRENT_GATE`. Ledger reopener (`docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md:15`): "autorizar MLP/PPO". Exact zero — the placebo cannot rescue it. |
| **B4** | `results/garrido_cssu_liveness_gate_v1/result.json` | `GATE_A_PASS_GATE_B_HOLD`, `gate_b.status = HOLD_OP11_PHYSICS_UNSPECIFIED` | Gate A fully passes (6/6 falsifiers, `activation_latency_hours = 24.0`, mass conserved 242,500/424,936). `physics_boundary/op11_handling_hours = 0.0`. Needs the Op11 finite-handling domain fact from Garrido. |
| **B5** | `results/headroom/contention_policy_class_smoke/result.json` | `POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT` | The full run `results/headroom/contention_policy_class/result.json` is `HALTED_FALSIFIER_FAILED` — **only `f7_no_gain_by_abandonment` fails**. Headroom itself is large and clean: `clairvoyant_minus_constant` lcb95 **+0.01692 / +0.01571 / +0.01624** across cells; `incremental_state_value` lcb95 **+0.02117 / +0.02669 / +0.02105**. The win is bought with lost orders (`lost_orders` clairvoyant 1.4375 vs placebo 1.375; 2.4375 vs 2.375 on R24_AGE). |
| **B6** | `results/headroom/cd_surface_prediction_premium/result.json` | `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED` | `docs/RESULTADO_PRIMA_CD_2026-08-01.md:10` explicitly calls this label "del runner que sobreafirma". Needs a preregistered V2 (the committed prereg omits spline, CART, the primary baseline and f7) + more than 5 folds (t(4)=2.776 is what widens the interval). |
| **B7** | `results/headroom_gsa/oos_guardrail_check.json` | `qualifies_new_lane = false` | H_obs **positive and OOS-stable on 3 independent blocks** (0.0131 / 0.0114 / 0.0100, CI95>0) — fails on `worst_cssu_fill_delta ≈ −0.13` vs threshold **−0.02** (6.5× over). Located θ: `signal_q=0.532, lead=2, surge_mult=1.946, commonality=0.887, r22_prob=0.107`. |
| **B8** | `results/step3_pooled/result.json` | `NO_STRUCTURED_CONTROLLER_CONVERTS`; **two** falsifiers fail | `f4`: preregistered guardrail `worst_product_fill` was not persisted, only `flow_fill_rate`. `f6`: `ddmrp_projected_v2` has `n_distinct_postures = 1` — the DDMRP arm is a dead actuator. Its own `ddmrp_domain_note` says the artifact "supports NO claim about DDMRP". Fix estimated at **~5 h** (`docs/REGISTRO_DE_HUECOS_2026-08-07.md` A1). |
| **B9** | `results/garrido_wrap_custody_manifest_v1.json:81`, `docs/GARRIDO_WRAP_CURRENT_STATE_2026-08-01.md:19` | `HOLD_E1_PLACEBO_NOT_OPENED`, `training_authorized=false` | same as B3 |
| **B10** | `research/paper2_exhaustive_search/approach_registry.json` (per `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md:121,137`) | `blocked_domain_fact` on **R03** and **R09** | R09 = mission-expiry + admission authority, `Discrete(0/1/2)`; "Sin deadlines, colapsa a D1" |
| **B11** | `research/seed_custody_registry.json` | `status = BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `new_seed_opening = false`, `scientific_execution_authorized = false` | Exactly **one virgin block** left: `g3a_v2_development`, seeds **7700001–7700120** (120 seeds), `RESERVED_NOT_OPENED`, gated by rule `submission_a_receipt_required_before_g3a_open = true`. This is the single scarcest resource in the repo. |

---

## 4. Documents listing questions for Garrido — and what each reopener unlocks

**Canonical batch:** `research/paper2_exhaustive_search/garrido_face_validation_questions.md` (Q1–Q14).
**Certificate:** `docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md` §3 "Exact Garrido questions (the only reopeners)".
**Email draft (unsent):** `docs/GARRIDO_QUESTIONS_EMAIL_DRAFT_2026-07-16.md` — internal mapping at lines 82–90.
**Answers already drafted:** `docs/GARRIDO_FACE_VALIDATION_RESPONSES_2026-07-13.md` (Q6 at :166, Q7 at :194, Q11 at :273).
**Elicitation instrument:** `docs/GARRIDO_ELICITATION_R03_R09_2026-07.md`.

| reopener | condition that reopens it | what it unlocks | strength |
|---|---|---|---|
| **Q11 / R09 — mission expiry** | hard deadlines with **permanent abandonment**, **tighter** than the 24–120 h R21/R23/R22 recovery times, **AND** doctrinal triage/eviction authority | the only thesis-native path to a positive Paper-2 instance; converts B10's `blocked_domain_fact` into an `ACTIVE` family. Email note (:83): the cap-60 variant is already exhausted by D1 — **only the real temporal deadline + admission authority reopen it** | **strongest** |
| **Q6 / Q7 — integrated shared resource** | ONE named resource mutually-exclusively committed across plant / LOC / theatre; or Maintenance Battalion teams **fewer** than disabled sites (forced serialization) | a second thesis-native family. `GARRIDO_FACE_VALIDATION_RESPONSES:189` warns a positive Q6 **cannot** replace the unresolved exact same-contract bound | OPEN |
| **Q13 — Program O construct** | ≥2 mutually non-substitutable ration classes sharing the Op5–Op7 bottleneck with uncertain, persistent, advance-observable mix | restores representativeness of the **ceiling only** (`H_PI = 0.152`); Program O's H_obs already failed OOS at 26/48 vs required 34 | not a positive |
| **Q14 — freight economics** | fixed-clock reserved vs pay-per-use fleet | only *scopes* a retired development finding | **not decisive** |
| **Q2 / R03 — route recourse** | ≥2 routes, one finite fleet, persistence + pre-dispatch warning | **BAR RAISED** — already has a full-DES negative; would additionally require finite downstream storage. This is what gates §5's `program_l_route_recourse_env` | raised |
| **(5) tail-risk / worst-theatre doctrine** (email lines ~68–72; internal note :86) | "¿existen requisitos vinculantes sobre el servicio del peor teatro… o la medida operativa de aceptación es la resiliencia promedio?" | **double payoff**: (i) licenses the CVaR-over-regimes estimator, whose H=0.0482 sits 0.0018 under the 0.05 bar (N1); (ii) if acceptance is *mean* resilience, the `worst_cssu_fill −0.02` guardrail that kills B7 (at −0.13) and `f7_no_gain_by_abandonment` that kills B5 are both re-specifiable | **highest expected value, unlisted in the certificate** |
| **`sumBt`** (`docs/REGISTRO_DE_HUECOS_2026-08-07.md` A4) | what the column actually is | no convention reconstructs it in >**1.09 %** of **47,780** rows. The Excel formula reproduction is proven; the **behavioural DES reproduction is not**. `blocked_domain_fact` — and the doc says "lo decidimos nosotros si él no responde" | validity of the whole ReT ledger |
| **David's `david_export.json`** (E2) | the 4.2-point discrepancy | probably hueco B2 (hyperparameters), unconfirmed | instrument |

Binding rule (email :88): *"una respuesta 'reopens' NO autoriza entrenamiento — autoriza preregistrar el contrato correspondiente con oracle-first."*

---

## 5. `supply_chain/*env*.py` with **no** results directory

All 15 matched files; grep of `results/` for each module name returns **0** for every one except `env_experimental_shifts` (4). Ranked by how close each is to producing something:

| module | results refs | scripts refs | decision exposed | action space |
|---|---|---|---|---|
| **`supply_chain/program_l_route_recourse_env.py`** | **0** | **0** | when to divert a convoy to a disclosed alternate route that bypasses the Op8 R22 stall, given a noisy pre-departure nowcast | `Discrete(3)` HOLD / ROUTE_1 / ROUTE_2, 21-dim `Box` obs |
| `supply_chain/v2_preventive_env.py` | 0 | 1 | daily order-up-to target for a finite reserve positioned behind the threatened Op10–Op12 corridor, under an explicitly imperfect warning; stock-conserving, lead-time and route aware | `Discrete(len(RESERVE_TARGETS))` |
| `supply_chain/event_triggered_env.py` | 0 | 1 | **whether to act at all today** — HOLD advances the DES one day preserving posture; INTERVENE may change shift immediately and commit one strategic buffer target with a **7-day physical lead** that cannot be cancelled | `Box` (continuous shift + buffer commit) |
| `supply_chain/program_o_per_batch_env.py` | 0 | 1 | at each of 24 real batch-arrival epochs, allocate the arriving 5,000-unit batch to `P_H` or `P_C` | `Discrete(2)` |
| `supply_chain/dra2_policy_env.py` | 0 | 4 | dispatch-or-wait for the finite convoy at Op8, from 19 observable staging/backlog/convoy fields | `Discrete(2)` |
| `supply_chain/track_c_env.py` | 0 | 2 | campaign-regime control with **priced** economics (J_v3, λ frozen from Gate C0): 8D track_b_v1 + 3 lagged buffer fractions, route-aware replenishment | `Box(11)` |
| `supply_chain/track_bp_env.py` | 0 | 5 | 8D reactive dispatch/shift + 3 **lead-time-committed** buffer targets (Op3 rm, Op5 rm, Op9 rations) | `Box(11)` |
| `supply_chain/program_o_ret_env.py` | 0 | 5 | one of four weekly count allocations, canonical ReT terminal reward | `Discrete(4)` |
| `supply_chain/l_program_env.py` | 0 | 8 | categorical S1/S2/S3 assembly-shift request only; no retained inventory/backlog/WIP/RNG across resets (cross-campaign identification) | `Discrete(3)` |
| `supply_chain/continuous_its_env.py` | 0 | 32 | de-discretized Garrido I_{t,S}: common buffer fraction at Op3/Op5/Op9 + continuous shift signal | `Box(2)` (+ per-op `Box(4)` variant) |
| `supply_chain/thesis_decision_env.py` | 0 | 4 | Garrido's literal two thesis decisions, factorized | `MultiDiscrete([6,3])` / `Discrete(18)` |
| `supply_chain/dkana_env.py` | 0 | **0** (47 in `outputs/`) | David's hierarchical local+global attention over a symbolic relational state | `MultiDiscrete([6,3])`, `[6,6,6,3]`, or `Box` |

`supply_chain/env.py`, `env_experimental_shifts.py`, `external_env_interface.py` are the shared substrate, not standalone lanes.

---

## RANKED — closest to a claim

**1. `results/headroom/g3_obs_conversion_v2/result.json` — a positive result blocked purely by paperwork.**
`lcb95 = 0.014726 > SESOI 0.010` on `H_obs_windowed_minus_constant` (base cell), `mde_primary = 0.00918 < 0.010`, `falsifiers/all_passed = true` (8/8), `claim_status = STRUCTURED_CONTROL_SUFFICES_G3_OBS`. Blocked by `contract_scope_adjudication.json`: the run carries **v2 execution fields** but a **legacy seal**. Blocker: **(d) governance/custody** — and the 7800001–140 block is already burned, so a fresh v2 execution needs new seeds, which `research/seed_custody_registry.json` forbids.

**2. `results/track_b_nonneural/result.json` (C1) — `NEURAL_PREMIUM_LIKELY_IN_TRACK_B`, first neural positive of the project.**
MLP 98.743, KAN 98.516, DMLPA 98.003 vs `constant_best` 96.567 → **+2.176 / +1.949 / +1.437**; 4/4 falsifiers pass; constant searched over 160 candidates on a disjoint fit block (seed0 888000, 8 episodes) and evaluated on 777000/24 episodes. Two blockers: **(d)** virgin block requires PI signature (`TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`: `C1_BLOQUE_VIRGEN NO-GO`), and **(b)** hueco A1 — `worst_product_fill` still not persisted. **Hidden caveat worth flagging:** `threshold_rule` reaches 97.142 (**+0.575 [+0.330, +0.821]** over the constant), so the defensible neural margin over the best *non-neural* arm is +1.60, and that contrast has no interval in the artifact.

**3. `supply_chain/program_l_route_recourse_env.py` — the largest untouched positive-headroom lane.**
The stylized screen already passed: `results/paper2_search/program_l_route_recourse_screen.json` headline cell `q0.90_cover4d` gives **H_PI = 0.11045, H_obs = 0.07491, η = 0.678**, placebo −0.02355 (real beats placebo), lost-order guardrail OK. Region map has ≥3 passing cells (cover_days=2: H_obs 0.11711 lcb 0.07068; 0.06976 lcb 0.03586). Two failure points: `resource_ok = false` (trips +0.90, veh_busy −4.2), and the screen is explicitly *"stylized single-destination… NOT full Op1-Op13 DES; NO learner trained."* The full-DES env exists with `Discrete(3)` and 21 obs, has **zero references anywhere in `results/` or `scripts/`**, and `docs/REGISTRO_DE_HUECOS_2026-08-07.md` D1 says the tape blocker was **unblocked on 2026-08-07** (`materialize_tape(..., contract_id="program_e_policy_realizability_v1")`). Blocker: **(a) compute + a runner script**, then **(c)** Q2/R03 face validation before any paper claim.

**4. CVaR over regimes — 0.0018 from the bar.**
`results/citable_risk_attitudes/result_after_anchor_fix.json`: **H = 0.0482** at α=0.90/0.95/0.99 (identical across all three because with six regimes each reduces to the worst regime), vs threshold **0.05** and Garrido's declared-γ identity 0.0195. Every risk-averse attitude moves *away* (CRRA η=0.25→0.0158; η=1→0.0050; η=10→4e−7; CARA a=10→4.5e−5). Blocker: **(c) external fact from Garrido** — whether doctrine has a binding worst-theatre/worst-campaign service floor. If yes, the CVaR estimator is licensed and the threshold is re-derivable; if no, the lane closes by number. Reporting rules bound in `docs/ENMIENDA_REPORTE_H_REGIME_CURVATURA_DECLARADA_2026-08-06.md`.

**5. `results/step3_pooled/result.json` — the cheapest instrument fix in the repo (~5 h).**
`f4` and `f6` are the only failures. Fixing `worst_product_fill` persistence and the dead DDMRP actuator (`n_distinct_postures = 1`) simultaneously (i) makes `NO_STRUCTURED_CONTROLLER_CONVERTS` an adjudicable verdict instead of a development diagnostic, (ii) permits the DDMRP adjudication Garrido asked for, and (iii) clears hueco A1, which is the stated gate on C1 (#2). Also note `greedy_pi_best_found_v2` already clears zero in both families (R1r lcb +4.32e−06, R2r lcb +8.83e−05) — real headroom exists, it just isn't deployable. Blocker: **(b) instrument fix**.

**6. `results/headroom/cd_surface_prediction_premium/result.json` — point estimate 1.25× SESOI, interval crosses zero.**
+0.06248 [−0.06063, +0.18559] vs SESOI 0.05. The interval width is driven by **df = 4** (5 seed-grouped folds, t=2.776). Doubling folds would roughly halve the multiplier's penalty. Secondary robust fact: both networks are *below* the classical baseline (KAN −0.0287, backprop −0.0465). Blocker: **(a) compute** (more folds/seeds) + **(d)** a V2 preregistration, since the committed prereg omits spline, CART, the primary baseline and f7.

**7. `results/dmlpa_variants/result.json` — two contrasts already clear zero under Holm.**
`nhead4` +1.589 [+0.639, +2.625] and `1layer` +1.210 [+0.609, +1.983], both Holm-adjusted p = 0, 5/5 falsifiers pass. Blocked by hueco B2 (`n_steps` 2048 vs 512, `ent_coef` 0.0 vs 0.01) and hueco C2 (development seeds, so the 200k rerun answers "does it hold at convergence?" not "does it replicate on new data?"). Blocker: **(a) compute — already in flight**; then **(d)** for a virgin replication.

**8. `results/headroom/contention_policy_class/result.json` — huge clean headroom, one falsifier away.**
`clairvoyant_minus_constant` lcb95 +0.01692 / +0.01571 / +0.01624 and `incremental_state_value` lcb95 +0.02117 / +0.02669 / +0.02105 across the three cells — all comfortably positive. Only `f7_no_gain_by_abandonment` fails (`lost_orders` clairvoyant 1.4375 vs placebo 1.375; 2.4375 vs 2.375). Blocker: **(c)** — same doctrine question as #4; if abandonment is operationally permitted, this converts. Otherwise **(b)** a differently-specified endpoint.

**9. `results/headroom_gsa/oos_guardrail_check.json` — positive and OOS-stable across three independent blocks, killed by a fairness guardrail.**
H_obs 0.0131 / 0.0114 / 0.0100 with CI95>0 on `ret_order`/`ret_quantity`; `worst_cssu_fill_delta` −0.1435 / −0.1282 / −0.1258 vs threshold −0.02. Blocker: **(c)** — identical to #4/#8. Not fixable by compute.

**10. `results/headroom/g3c_preflight_burned/result.json` — power is already sufficient, instrument is not.**
`n_required_for_sesoi = 22/23`, `budget_max_fresh_seeds = 96`, `within_budget = true`. Only `f2_min_dwell_actually_binds_at_3_or_7_days` fails. Blocker: **(b) instrument fix** + **(d)** custody authorization for 22–23 fresh seeds. Discounted because `primary_hysteresis_vs_myopic` mean is negative (−0.00073 / −0.00290) — the fix is necessary, not sufficient. Superseded by `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md`.

**11. `results/garrido_cssu_liveness_gate_v1/result.json` — Gate A fully passed, Gate B is one domain fact.**
6/6 falsifiers pass, `activation_latency_hours = 24.0`, mass conserved. `gate_b.status = HOLD_OP11_PHYSICS_UNSPECIFIED`, `physics_boundary/op11_handling_hours = 0.0`. Blocker: **(c) Garrido** — multi-CSSU competition with observable allocation authority (email internal note item 5, `HOLD_PENDING_DOMAIN_FACT`).

**12. `supply_chain/event_triggered_env.py` — the only untested *when-to-act* decision.**
`docs/REGISTRO_DE_HUECOS_2026-08-07.md` D3: *"el único que prueba si el óptimo varía en el instante de un choque — distinto de variar entre periodos fijos, que es lo que medimos y cerramos."* Its 7-day non-cancellable buffer commitment is exactly the "real preparation lead time" that the registry (L798) names as the one thing that would reopen prevention, and it directly attacks Track C's fixed-168 h cadence root cause. Countervailing evidence: `PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md` §4 declares within-event timing **un-licensed** (risk-sensitivity door: R2_frequency @ cap 0.5 = 5.0e−05 vs bar 0.01) and *"substantially undercut"* — but explicitly **not refuted**. Blocker: **(d) a new preregistration with independent motivation**, then **(a) compute**.

**13. `results/garrido_neural_headroom_gate_v1/result.json` — HOLD, but with nothing behind it.**
`H_regime = 0.0`, `lcb95 = ucb95 = 0.0`, `next_environment = NONE_UNDER_CURRENT_GATE`. Listed for completeness: the "one fixable thing" (open the placebo) cannot move an exact zero. Blocker: **(d)**, but low expected value.

### Blocker tally
- **(a) compute:** #3 (route recourse full-DES), #6 (more folds), #7 (in flight)
- **(b) instrument fix:** #5 (`worst_product_fill` + DDMRP actuator, ~5 h — highest leverage), #10 (dwell binding)
- **(c) external fact from Garrido:** #4, #8, #9 all hinge on **one** question (worst-theatre service floor vs mean resilience); #11 on Op11 physics; #3 on Q2/R03; plus `sumBt` (A4) and Q11/R09
- **(d) governance/custody:** #1, #2, #12 — all ultimately gated by `research/seed_custody_registry.json` `NO_NEW_SEEDS_AUTHORIZED` and the single remaining virgin block **7700001–7700120**, itself gated by `submission_a_receipt_required_before_g3a_open`
