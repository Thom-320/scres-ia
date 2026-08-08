# Explore — Read-only census in <HOME>/Projects/research/scres-ia. Machine is busy — no pytest, n

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a7fa964d35b2e3bc4`
- **session**: `2c6b3a01-62a0-499d-957d-694424855dff`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T03:57:09.085Z`
- **effort**: `xhigh` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/2c6b3a01-62a0-499d-957d-694424855dff/subagents/agent-a7fa964d35b2e3bc4.jsonl`
- **turns**: 96 · **assistant messages**: 4

## Task given

```text
Read-only census in <HOME>/Projects/research/scres-ia. Machine is busy — no pytest, nothing heavy. Read artifacts, registries and docs.

QUESTION: across EVERYTHING in this repository, what is the maximum set of defensible claims we could publish, and how many papers' worth is it?

1. Read `research/evidence_registry.jsonl` (216 rows). Group every artifact by evidence_grade. For CONFIRMATORY and DEVELOPMENT rows, list the claim_status strings and cluster them into thematic families (e.g. "search/transfer", "metric integrity", "risk sensitivity", "headroom/no-door", "provenance", "demand process"). Report counts per family.

2. Read `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md`, `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` and its ENMIENDA_1..4, and `docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md` plus `results/paper2_search/paper2_exhaustion_certificate_2026-07-15.json`. Report: what is AUTHORISED to be claimed, what is PROHIBITED (list the exact forbidden wordings and retracted figures), and what the exhaustion certificate says is closed.

3. Identify EVERY distinct publishable finding the repo can support, and for each give: the one-sentence claim in its strongest defensible form, the artifact(s), the evidence grade, and which of these three papers it belongs to —
   (P2) "Retained search state before neural architecture" — the outer-loop transfer paper
   (P1) measurement/metric integrity — abandonment, censoring, cadence
   (P3) something else entirely — name it
   Flag any finding that belongs to none.

4. Specifically hunt for findings that are STRONG but currently ORPHANED — sealed, high-grade, and not assigned to any manuscript. Examples to check: the Alzheimer/memory-advantage numbers, the Cobb-Douglas port, the risk-sensitivity screen (garrido_risk_headroom_sensitivity_v1), the demand_process characterisation, contention/Program O headroom, the determinism repair, the cross-architecture reproduction. For each say whether it is a paper section, a supplement, or a separate paper.

5. Read `docs/PROMISING_LANES_REGISTRY.md` if it exists and report which lanes are open vs closed.

6. Finally: is there enough for a THIRD paper, and if so what is its thesis and what is missing? Be blunt if the answer is no.

Report as a dense structured inventory with exact paths, claim_status strings and numbers. Do not summarise vaguely.
```

## Final answer

I have the full picture. Here is the census.

---

# REPO-WIDE CLAIM CENSUS — `<HOME>/Projects/research/scres-ia`

**Bottom line first: two papers are real and nearly written; the third is a real paper whose evidence is one metric-generation stale and can never be confirmed (zero virgin seed blocks remain). Call it 2.5.**

---

## 1. `research/evidence_registry.jsonl` — 216 rows

`<HOME>/Projects/research/scres-ia/research/evidence_registry.jsonl`
Schema keys: `artifact_path, claim_status_as_authored, evidence_grade, grade_rationale, contract_path, content_sha256, self_sha256, dedup_key{,_complete,_hash}, duplicate_of, supersedes, replay_of, merge_basis, branch, created_at`.

| evidence_grade | rows | unique (artifact, claim_status) |
|---|---:|---:|
| DEVELOPMENT | 88 | 88 |
| UNCONTRACTED | 57 | 57 |
| REPLAY | 42 | 42 |
| NEGATIVE_OR_HALTED | 20 | 20 |
| DIAGNOSTIC | 5 | 5 |
| **CONFIRMATORY** | **4** | **3** |

The 4th CONFIRMATORY row is a branch duplicate: `results/garrido_h2_h3_confirmation_v1/result.json` appears twice, once on `HEAD` and once on `codex/paper-b-retained-v5`.

### 1a. CONFIRMATORY — all three, verbatim

| artifact | `claim_status_as_authored` | custody block | contract |
|---|---|---|---|
| `results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | 8200001–8200060 (virgin) | `docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md` |
| `results/garrido_h2_h3_confirmation_v1/result.json` | `CONFIRM_H2_H3_ALL_SIX_PANELS` | 96111336–97836128, 12 virgin roots | (none recorded; grade lives in sibling `completion_receipt.json`) |
| `results/gsa_confirmation/result.json` | `GSA_CONFIRMED_ON_VIRGIN_BLOCK` | 7700001–7700120 (repurposed virgin) | `docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md` |

### 1b. Thematic families (deduplicated, per grade)

| family | CONF | DEV | REPLAY | DIAG | NEG/HALT | UNCONTR | **total** |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Headroom / no-door** | 1 | 47 | 4 | 1 | 12 | 33 | **98** |
| **Search / transfer / architecture** | 1 | 12 | 24 | 0 | 1 | 10 | **48** |
| **Metric integrity / measurement** | 0 | 23 | 14 | 2 | 6 | 14 | **59** |
| **Provenance / instrument / manuscript** | 0 | 6 | 0 | 2 | 1 | 0 | **9** |
| (unclassified: `garrido_h2_h3_confirmation_v1` — physical reproduction) | 1 | | | | | | **1** |

**The distribution is the story: 98 of 215 classified rows are the search for headroom that never opened, and only one of them is confirmatory (and that one is GSA, self-degraded to a one-bit conclusion).**

### 1c. DEVELOPMENT (88) — claim_status strings by family

**Search/transfer/architecture (12):**
`ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` ×6 (`garrido_meta_learner/`, `_h3power_local/`, `_h3power_vps/`, `_h3power_vps_local_replay/`, `_v2/`, `_v2_vps_crosscheck/`) · `NEURON_DOES_NOT_BEAT_THE_NULL` (`garrido_meta_learner_smoke_v2/`) · `SURFACE_REPLAY_MEMORY_EFFECT` (`garrido_meta_learner_thesis90_v2/`) · `H3_PRIME_SUSTAINED_AT_N120` (`garrido_h3_merge_adjudication/`) · `OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE` (`ofat_lcb_reconciliation/`) · `A_DMLPA_VARIANT_SEPARATES` (`dmlpa_variants/`) · `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE` (`headroom/buffer_prediction_premium/`)

**Metric integrity (23):**
`COBB_DOUGLAS_SURVIVES_THE_ABANDONMENT_TEST` (`metric_audit/abandonment_v1/`) · `NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT` (`metric_audit/contention_service_first_v2/`) · `DES_CHANGES_ARE_BEHAVIOURALLY_INERT` ×2 (`des_change_differential/`, `vps_vs_local_differential/`) · `DEVELOPMENT_FIDELITY_SWEEP_NO_CONSTANT_CHANGED` ×4 · `DEVELOPMENT_CONCORDANCE_SCREEN` · `PROSPECTIVE_CORRECTIVE_CONFIRMATION` ×2 (`ret_metric_repair_confirmation_v1/{result,adjudication}.json`) · `V2_METRIC_UNCHANGED_SINCE_FREEZE` · `SERVICE_FIRST_V2_AUDIT_SEALED` · `CSSU_ACTUATOR_SCOPE_DECLARED` · `DEVELOPMENT_PREREGISTERED_{THREE_ARM_AUTOTOMY_TEST, AUTOTOMY_CLOSURE, CROSSING_FALSIFIER_FAILED, TWO_ARM_READING_TEST, R14_SEED_TEST_FALSIFIER_FAILED, TWO_ARM_ONSET_ADMISSION_TEST}` · `SENSITIVITY_ONLY_NOT_ECONOMIC_CALIBRATION` + `INDEPENDENT_RELATIVE_PRICE_SENSITIVITY_NO_MONETARY_CLAIM` (`cobb_douglas/economic_sensitivity_v1,v2/`)

**Headroom / no-door (47):**
`CONTENTION_DOES_NOT_OPEN_THE_DOOR` ×4 (`sensitivity/contention_headroom_v1{,_1,_2,_3}/`) · `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` ×2 · `DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER` (`garrido_risk_headroom_sensitivity_v1/`) · `NO_HEADROOM_EVEN_UNDER_A_SOUND_METRIC` (`headroom/cobb_douglas_v1/`) · `HEADROOM_IS_A_METRIC_DEFECT_NOT_A_DECISION` (`sensitivity/bottleneck_asymmetry_v1/`) · `ARGMAX_MOVES_WITHOUT_VALUE` ×2 · `POLICY_CLASS_WAS_THE_BINDING_CONSTRAINT` · `PREMIUM_WAS_AVAILABLE_AND_NOT_CAPTURED` · `G1_GENERATES_CURVATURE` · `THRESHOLD_RULE_SUFFICES` · `STRUCTURED_CONTROL_SUFFICES_G3_OBS` · `NO_STRUCTURED_CONTROLLER_CONVERTS` ×2 (`step3_pooled/`, `step3_split_pooled/`) · `DEVELOPMENT_INSTRUMENT` ×4 (`step3_split_s1..s4/full/`) · `GSA_QUALIFIES_UNDER_RESILIENCE_ONLY` · `GSA_CONFIRMED_ON_VIRGIN_BLOCK_AS_A_ONE_BIT_CALENDAR_CHOICE` · `GATE_A_PASS_GATE_B_HOLD` · `HOLD_E1_PLACEBO_NOT_OPENED` · `DEVELOPMENT_{HEADROOM_MAP, MIXED_RISK_HEADROOM, MULTI_METRIC_HEADROOM, OBSERVABLE_SWEEP, OP12_CONDITIONED_POLICY, SECOND_ORDER_AND_RISK_SEARCH, ASYMMETRIC_RIGHTS_NO_METHOD_CLAIM}` · Program O: `PASS_METRIC_LIVENESS_ONLY__TRANSDUCER_FREEZE_ALLOWED`, `DIAGNOSTIC_STABLE_SIGNAL_FIXED_CLOCK_ONLY`, `PASS_SCREEN_ONLY__FREEZE_FRESH_VALIDATION_BEFORE_OPENING`, `PASS_VALIDATED_TRANSDUCER_REGION__FULL_DES_FREEZE_ALLOWED`, `STOP_PROGRAM_O_AFTER_CORRECTIVE_VALIDATION`, `TECHNICAL_INVALIDATION_REQUIRES_CORRECTIVE_REPEAT`, `STOP_PROGRAM_O_CLASSICAL_HOBS_VALIDATION`, `PASS_PROGRAM_O_FIXED_CLOCK_PHYSICAL_PREFLIGHT`, `PASS_PROGRAM_O_GATE_O0_FULL_ACTION_TRANSDUCER` ×2, `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND` · `NEURAL_PREMIUM_LIKELY_IN_TRACK_B` (`track_b_nonneural/`) · `UNTRAINED_SIDE_ONLY_NO_CONTRAST_YET` · `PROSPECTIVE_CONFIRMATION` (`q_r1/successor_confirmation_v1/merged/` — **run_role, not verdict; its adjudication is `STOP_REPAIRED_Q_R1_NO_RETAINED_INFORMATION_PASS`**)

**Provenance/manuscript (6):**
`H1_SUPPORTED__H3_NOT_SUPPORTED` ×3 (`manuscript/h1_h3_originales_v3/`, `h1_h3_v2/`, `h1_h3_v2_1/`) · `H2_SUPPORTED_LEARNING_CURVE` (`manuscript/h2_learning_curve/`) · `KEEP_THE_RESERVED_BLOCK` (`custody/confirmation_block_size_audit/`) · `BLOCK_SIZE_STILL_ADEQUATE_UNDER_CORRECTED_SD` · `EVIDENCE_REGISTRY_BUILT`

---

## 2. What is AUTHORISED, what is PROHIBITED, what is CLOSED

### 2a. Authority chain (read in this order — later supersedes earlier)

1. `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` (frozen, never edited)
2. `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md`
3. `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` (`550a253`) + `_ENMIENDA_1` (`d7a205b`) · `_ENMIENDA_2` (generated by `scripts/build_claim_freeze_amendment_v1.py`) · `_ENMIENDA_3` (`78b13d0`) · `_ENMIENDA_4`
4. **`papers/paper2/claim_lock.json`** — declares `supersedes_for_citation` over all six of the above. This is the actual single resolution point for the manuscript. 12 claims, each with `allowed` / `forbidden` / `why_forbidden` / `self_sha256` / `file_sha256`.
5. `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` — says *which file wins*, not what can be claimed.

### 2b. AUTHORISED (exact wordings and numbers)

**Primary metric, frozen:** `auc_regret_norm` under the **PREFIX** normaliser. Reason on record: `runs_to_within_1pct` imputes `budget+1` and is censored at wildly different rates per arm.

**Canonical panel** — `results/garrido_normaliser_audit_v3/result.json`, 6 contexts × 12 replicates:

| arm | AUC regret (prefix) ↓ | runs-to-1% (secondary, censored) |
|---|---:|---:|
| `neuron_memory` | **0.05203** | 7.08 |
| `neuron_reset` | 0.11274 | 12.92 |
| `ofat` | 0.10024 | 12.42 |
| `random` | 0.13851 | 19.54 |

memory − reset **+0.06070**, LCB95 **+0.04556**, UCB95 +0.08020 · memory − OFAT **+0.04821**, LCB95 **+0.03325**.

**Ladder** — `results/search_ladder_v5/result.json`, 15 methods + oracle:
`ucb1_transfer` 0.04502 · `neuron_memory` 0.05203 · `ofat_transfer` 0.06274 · `lookahead_kg_transfer` 0.08018 · `gp_ei_transfer` 0.08390 · `thompson_transfer` 0.08908 · all others ≥ 0.09655. **The six leaders are exactly the six state-retaining arms.** Scope `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`, `run_role = CACHE_ANALYSIS`, seeds 5300001–012 reused.

**Confirmation 1** — `results/grid_transfer_confirmation_v2/result.json`, block 8200001–060, 288 → 4,608 configurations, `transfers = {ucb1: true, neuron: false, gp: false, ofat: false}`:

| family | vs cold start | vs state-blind marginal replay |
|---|---|---|
| UCB1 | +0.05744 [+0.04989] | **+0.03073 [+0.01990]** ✅ |
| neuron | +0.05439 [+0.04290] | **−0.01178 [−0.01849, −0.00484]** ❌ |
| GP-EI | +0.01433 | −0.02160 ❌ |
| OFAT | +0.01422 | −0.02467 ❌ |

**Confirmation 2** — `results/garrido_h2_h3_confirmation_v1/result.json`, 12 virgin roots 96111336–97836128, six panels, Holm p 4.8e−17 … 1.9e−15, `global_confirmation_pass: true`, `development_roots_opened: false`. Boundary travels with it, literal: *"it does not establish learner, feedback, or architectural value."*

**Confirmation 3 (GSA), degraded by self-correction** — `results/gsa_confirmation/result.json` superseded by `results/gsa_confirmation_corrective/result.json`: `GSA_CONFIRMED_ON_VIRGIN_BLOCK_AS_A_ONE_BIT_CALENDAR_CHOICE`. H_PI 0.012926135742150043, H_obs 0.012286775167294721 [LCB95 **0.00867768343802695**], n=120, η **0.9505373773253464**, obs − permuted placebo +0.07210657135008629 [LCB95 +0.049801679249144644], negative control obs − oracle −0.00063936 [−0.00144761, −7.0488e−05], `policy_sequences {AAAA: 31, ABAB: 89}`, `placebo_ties: 74`. Reported-not-blocking: `worst_cssu_fill_delta = −0.12916666666666668`.

**Fig. 5 is an identity** — `results/garrido_fig5_surrogate/result.json` + `results/garrido_wrap_q1/result.json` (`IDENTITY_NOT_A_LEARNING_TASK`): `max_abs_identity_error = 3.22e−15`, R² = 1.0, coefficients `Re_FRt ≈ 1`, `Re_RPj ≈ 1`, and **three of five driver columns identically zero** (`Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`).

**H1–H4 verdicts** (`_ENMIENDA_1` §1):

| | original wording | reformulation |
|---|---|---|
| H1 recovery | **SUPPORTED** +126.0 h [+98.4, +154.5], by **absorption** (875/960 vs 755/960) | H1′ lost service: SUPPORTED |
| H2 learning curve | **SUPPORTED**, slope **+0.042201 [+0.034664, +0.049922]** | — |
| H3 variance across intensities | **NOT SUPPORTED** — opposite sign, CI crosses zero, estimand present | H3′ search-cost variance: SUPPORTED |
| H4 dependence on `L_{t−1}` | **measured** +0.06070 [+0.04556] | — |

H2 null control (random − OFAT) **−0.005088 [−0.015570, +0.005658]** — crosses zero, so the trend is not escalating context difficulty. Per-context advantages: 0.00000 / +0.28275 / +0.19052 / +0.22111 / +0.31709 / +0.26869.

**Curvature and noise** — `results/headroom/buffer_prediction_premium/result.json`: curvature **0.076259** (`profile_curvature.mean_one_minus_linear_r2`, stored); noise **0.317394** (**derived**, `1 − held_out_r2_mean.linear` with linear = 0.682606). Held-out R²: linear 0.6826 · KAN 0.7163 · **backprop 0.5548 (worse than a straight line)** · constant −0.0034. Must always be cited with the derivation shown.

**Locked decisions** (`TABLA_CANONICA` §8):
```
MANUSCRITO_Q1_Q2            GO
Q1_Q2_CERRADAS              NO — bounded to the measured contract
C1_BLOQUE_VIRGEN            NO-GO
MECANISMO                   retained search state, not architecture
INTEGRACION                 outer-loop simulation optimization coupled to the DES
```

### 2c. PROHIBITED — exact forbidden wordings

From `TABLA_CANONICA` §7:

| forbidden phrase | why |
|---|---|
| «Q1 y Q2 están cerradas» / "Q1 and Q2 are closed" | answered only **inside the rehearsed simulation-optimization contract** |
| «la neurona gana» / «hay prima neural» ("the neuron wins" / "there is a neural premium") | tie with UCB1; loses the transfer falsifier |
| «las redes son la familia que imita SCL» | refuted in the measured contracts |
| «aprendizaje organizacional» / «la cadena aprende» | it is an **external between-run loop**, not within-episode control |
| «KAN es mejor política» | CI crosses zero and it is 4.1× slower per decision |
| «C1 está confirmado» | development on a used block, no `worst_product_fill` |
| «está en `main`» | `main` is at `89acc81` (28-Jul) |

Plus, per amendment:
- **`_ENMIENDA_1` §3: it is forbidden to write «excluye el cero» / "excludes zero"** for `neuron_memory` vs `ofat_transfer`. Two sealed artifacts score byte-identical arrangements and give opposite signs: **−2.761e−05** in `search_ladder_v2_ordered`, **+3.565e−05** in `search_ladder_v5`, common mean **+0.01071**; the lower bound is positive in **65% of 40 resampling seeds** (B = 50,000). Mandated wording: *"indistinguishable in regret AUC (mean +0.01071; the lower bound falls on both sides of zero depending on the resample),"* citing **both** sealed bounds. By contrast `neuron_memory` vs `ucb1_transfer` is stable — LCB positive in **0%** of seeds.
- **`_ENMIENDA_3` E1: forbidden to say "three confirmations" (inflates grade) AND forbidden to say "only two" (omits a sealed artifact).** Mandated: *"Two prospective confirmations usable by the retention manuscript, plus a third (GSA) that exists, ran on a **reused** virgin block, and was degraded by its own correction to a one-bit conclusion."*
- **`papers/paper2/claim_lock.json` per-claim `forbidden` lists**, e.g. for the confirmation: `"only UCB1 learns"`, `"the neuron has no memory"`, `"UCB1 is universally superior"`, `"factorized UCB policy"` (*"'policy' would conflate the outer loop with within-episode control"*). For the H2/H3 validation: `"validation of the DES"`, `"the DES is validated"`, `"order-level behavioural replication"`, `"reproduces the Simulink model"`. For Fig. 5: `"Garrido's neuron is absurd"`, `"the proposal is wrong"`, `"the neuron cannot learn"`. For the ladder: `"retention was prospectively confirmed"`, `"retention is confirmed across families"`, `"the top-six ranking shows retention causes"`.
- **`results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json`** (`SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE`) — five literal `prohibited_claims`, ending with **"that this artifact confirms a neural premium."**

### 2d. RETRACTED figures — do not cite

| retracted number | replacement |
|---|---|
| **`7.24 / 13.54 / 12.42`** (memory/reset/OFAT runs-to-1%) | prefix panel `7.08 / 12.92 / 12.42` |
| **`7.90` and `5.43`** — H4 in draft v.0, `+7.90 runs [+6.88, +8.93]` | primary **AUC +0.06070 [+0.04556, +0.08020]**; secondary censored **5.83 runs [+4.44, +7.31]**, always labelled |
| oracle panel `6.99 / 14.89 / 12.42` | citable **only** as sensitivity, always labelled |
| `results/architecture_bakeoff*` as "sealed" | **`run_architecture_bakeoff_v1.py` never sealed anything.** External content-only seal from 2026-08-07 in `results/*/sealed_record.json`; **provenance is not certifiable backwards.** Several of the five external audits built tables on these numbers |
| `results/k3/confirmation.json` | self-retracted, `effective_verdict: RETRACT_...` |
| `q_r1/successor_confirmation_v1` | Program Q compound STOP on `worst_product_fill` |
| neural premium of `track_b_v1` (+1.44 … +2.18) | inside a **±2.4** harness-noise band, no interval in its artifact |
| stylized VoI atlas, 2 positive H_obs cells | `RETRACTED_BY_CORRECTIVE_AUDIT` — **shed-to-win** (theatre starvation, fails worst-CSSU fairness), excluded from inference |
| exhaustion certificate's "pure OAT hypercross / negative only along the axes" | **FALSE and retracted** — `Cf20=(T,T,T,T)` raises all four R2 risks simultaneously and produced no door |
| `_ENMIENDA_1` §E3 "one virgin block remains" | **retracted by `_ENMIENDA_4`** — the block was repurposed by written PI authorisation, opened 18:05Z and closed 18:20Z on 2026-08-07 |

### 2e. Exhaustion certificate — what it says is CLOSED

`docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md` + `results/paper2_search/paper2_exhaustion_certificate_2026-07-15.json`
Status: `PAPER2_SEARCH_BOUNDARY_CERTIFIED__NO_POSITIVE_INSTANCE__EXACT_REOPENERS_NAMED__DISCRETE_DESIGN_SCOPE`, science commit `adbfb8f`.
Primary endpoint throughout: `ret_excel_request_snapshot_v2`. Forbidden: metric substitution, Cobb-Douglas promotion, hand-rolled ReT, temporal-index promotion.

**Terminal finding:** across 4 decision surfaces, 3 independent programs, 2 physics generations, Garrido's own Cf1–Cf20 risk table, and one engineered extension with ~7× more clairvoyant headroom — **no decision contract converts perfect-information headroom into deployable, resource-honest, out-of-sample-stable observable value. Zero PPO seeds were wasted; every gate killed its candidate before training.**

| program | ceiling | outcome |
|---|---|---|
| Full-DES max H_PI (`DRA2b_finite_convoy`) | **0.0221** | H_obs ≈ 0; Program E **0/10 PPO seeds**, tree, heuristic, convex mixture all fail |
| Stylized VoI atlas, 64 cells | H_PI mean 0.0135 / max 0.024595 | H_obs mean **−0.007896**, 79.7% ≤ 0, **η mean −0.7895** — ⚠️ RETRACTED, directional only |
| Track C campaign oracle (R22/R23/R24) | switching headroom **6.5e−05** (oracle 2.8e−05), 24/24 tapes, bar 5% | **NULL** — **hysteresis beats clairvoyance ×2.3**: a non-privileged EWMA detector outperforms the true-state instant switcher |
| L(e−1) program | clairvoyant constant bound +1.6–3.6% | **STOP** — optimal weekly shift = **S1 in 100%** of branched states |
| Route recourse (Program L / R03) | full-DES H_PI **≤ 0.005** at every R22 regime (24h→720h) | **STOP** — buffer intercept; stylized screen showed 0.15, full-DES collapsed **~30×** |
| Garrido risk escalation | **max H_profile_safe 6.931144890998109e−05 vs bar 0.01 → 144× below** | **0/63 cells**, `passing_doors: []`, guardrails all clean |
| **Program O** (two non-fungible products) | **H_PI 0.15151378920653932**, simultaneous safe LCB95 **0.1156159088651399**, exact fungible-null **0.0**, 25,177-episode parity, throughput conserved | **CLOSED** — H_obs failed OOS |

**§2.6, the decisive result:** 45 profiles × 18 constant postures × 6 seeds = **4,860 ten-year evaluations**. *The optimal constant posture is INVARIANT across all 45 risk profiles, at every budget cap.* Risks genuinely degrade physical resilience (ReT **0.53 → 0.20**) but create **zero regime-tailoring headroom**. Custody: result `e4a3d4a0`, execution `01c1d9d` (child of frozen `6794fe6`, contract byte-identical).

**Program O H_obs chain:** label-only HMM **REFUTED** (oracle-true (ρ,share) changed **0/192** trajectories; action is share-magnitude-invariant) → state-rich classical `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND` (`d67ac97a`) → dual-resource diagnostic `DIAGNOSTIC_STABLE_SIGNAL_FIXED_CLOCK_ONLY` (`e48606e7`, belief-MPC beats **all** information placebos over 3 connected cells, incremental state value LCB95 **+0.025…+0.073**; pay-per-use fails) → fixed-clock physical OOS **`STOP_PROGRAM_O_CLASSICAL_HOBS_VALIDATION`** (`09ec3f16`): `rho90_share90` favorable on **26/48 vs required 34**, mean +0.083 but bimodal; other cells 45/48 and 44/48. Sealed tapes 7420049–7420096 opened exactly once. Closed with **no rescues**. Bug disclosed: 69 incompatible-scale estimands merged into one simultaneous critical (23.65M), voiding LCB/placebo/guardrail flags — **verdict independent of the bug** (26/48 is a raw pre-bootstrap count).

**Claim boundary, literal:** full-DES H_PI established **true** (Program O only) · H_obs established **false (nowhere)** · learner authorized **false** · Paper 2 confirmed **false** · **Paper 3 authorized false** · positive instance found **false**.

**The only legitimate reopeners:** Q11/R09 mission expiry (**OPEN — strongest thesis-native reopener**), Q6/Q7 integrated shared resource (**OPEN**), Q13 Program O construct (open, restores ceiling representativeness only), Q14 freight economics (open, **not decisive**), Q2/R03 route recourse (**BAR RAISED**). Residual untested thread disclosed: **within-event intervention timing** — un-licensed, motivation "substantially undercut" by posture invariance.

---

## 3. Every distinct publishable finding, with paper assignment

Assignment key: **P2** = `papers/paper2/` "Retained search state before neural architecture" · **P1** = `papers/paper1_unobserved_frontier/` measurement/metric integrity · **P3** = the third paper, named in §6 below.

### P2 — already locked in `papers/paper2/claim_lock.json` (12 claims, sections 3.1–3.7 drafted)

| # | strongest defensible claim | artifact | grade | section |
|---|---|---|---|---|
| **1** | In a prospective 288→4,608 expansion on a virgin block, **only** a factorized UCB carrier beat both cold start and a state-blind replay of its own search marginals (+0.03073, LCB95 +0.01990, n=60); the neural carrier did not (−0.01178 [−0.01849, −0.00484]). | `results/grid_transfer_confirmation_v2/result.json` (`7bc33823ccd90b5e…`) | **CONFIRMATORY** | RQ2, leads Results |
| **2** | The reconstructed DES prospectively reproduced six thesis-derived comparative panels, 12/12 tapes each under Holm, `generated_orders` exactly zero in every tape. | `results/garrido_h2_h3_confirmation_v1/result.json` (`bc375d3021b64d10…`) | `GRADE_IN_SIBLING_RECEIPT` | Methods / targeted validation |
| **3** | Under the literal reading of Fig. 5, ReT is exactly the sum of the driver contributions supplied as inputs (R²=1.0, max identity error 3.22e−15) — an identity, not a predictive task. | `results/garrido_fig5_surrogate/result.json` | `GRADE_NOT_MACHINE_DISCOVERABLE` | Methods, algebraic proposition |
| **4** | In development replay, the six state-retaining arms occupied the six leading positions of fifteen, and the same approximator fell **rank 2 → rank 12** when its memory was removed. | `results/search_ladder_v5/result.json` | DEVELOPMENT | RQ1 |
| **5** | Retaining state improved the same approximator by **+0.06070 AUC [LCB95 +0.04556]** under a prefix normaliser blind to the unrun surface. | `results/garrido_normaliser_audit_v3/result.json` | REPLAY | RQ1 |
| **6** | Two sealed artifacts place the neuron-vs-OFAT bound on opposite sides of zero over byte-identical replicates; positive in 65% of 40 resampling seeds — not distinguishable. | `results/ofat_lcb_reconciliation/result.json` | `GRADE_NOT_MACHINE_DISCOVERABLE` | RQ1, stated as a limit |
| **7** | At matched parameter budgets (**532 vs 529**) the KAN attained better supervised fit yet searched **worse** (+0.01037, CI95 [+0.00302, +0.01893], p=0.0012); the best searcher of a seven-architecture bake-off was a **five-parameter neuron**. | `results/surrogate_architecture_bakeoff/result.json` | DEVELOPMENT | RQ3b |
| **8** | Under the parameter-matched latent contract, KAN underperformed MLP: 97.58 vs 98.44, paired **−0.862 [−1.605, −0.119]**, 4/5 seeds negative, −0.88% relative, smaller than within-seed eval SD 0.76–1.06. | `results/dmlpa_kan_latent/result.json` | (scope: `DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN`) | Supplement |
| **9** | On a deliberately curved surface the backprop MLP scored **below a straight line** (−0.128 [−0.316, +0.060]); neither approximator reached the preregistered SESOI of 0.05; in-situ curvature 0.0763 against 0.3174 unexplained episode variance. | `results/headroom/buffer_prediction_premium/result.json` | DEVELOPMENT | Discussion |
| **10** | Ramsey RESET rejects linearity in **all six** contexts (F **384–2463**); AIC and seed-grouped held-out R² both select quadratic-with-interactions in all six; non-linearity buys **+0.19 to +0.23** held-out R² in R1r. **The architecture tie is not explained by a linear surface.** | `results/functional_form_diagnostics/result.json` | DIAGNOSTIC | Supplement + RQ3 discussion |
| **11** | Within the thesis-inherited U(2400,2600) demand process: realised weekly **CV 7.1%**, **24.8%** of weeks already exceed single-shift capacity, lag-1 ACF **−0.228** against an iid band of ±0.065. | `results/demand_process/result.json` | DIAGNOSTIC, **falsifiers `all_passed: false`** | Results 3.1 |
| **12** | Across 45 risk profiles escalating the thesis's own recurrent risks (4,860 evaluations), no door passes: max H_profile_safe **6.93e−05 [0, 2.08e−04]** vs a 0.01 bar. The optimum varies across at most three postures; following it buys nothing. | `results/garrido_risk_headroom_sensitivity_v1/result.json` | `GRADE_NOT_MACHINE_DISCOVERABLE` | Discussion |

### P1 — `papers/paper1_unobserved_frontier/`, 3 modules built + 16 admitted artifacts (`_ENMIENDA_2`)

Built already (`numbers.json`, `numbers.tex`, `evidence_receipt.json`, 3 figures):

| module | claim | artifact |
|---|---|---|
| **M1** | **No ReT variant, in either regime, is maximised where service is maximised — 0 hits in 8 cells.** ReT argmax at share 0.1; service and Cobb-Douglas argmax at 0.5. R1r+R2r: fill **0.5139 at ReT optimum vs 0.8058 at service optimum**, lost orders **76.125 vs 0.0**, omitted **135.875 vs 53.375**. R2r: fill 0.4977 vs 0.8496, lost 80.0 vs 0.0. `cobb_douglas_agrees_with_service: true` in both. | `results/metric_audit/abandonment_v1/result.json` (`1d37752cbca3649b…`) |
| **M2** | The recovery endpoint had no estimand until repaired: `HALTED_FALSIFIER_FAILED` → `H1_SUPPORTED__H3_NOT_SUPPORTED`. Repaired H1 hybrid−static **+125.985 h [+98.347, +154.545]**, n=960, p=0.0; hybrid−reset **+74.046 [+51.524, +97.468]**. Absorption **875 / 823 / 755 of 960**; censored at τ=1344h **7.19%**; horizon 6048h; **78.75%** of cells deploy differing configurations. | `results/manuscript/h1_h3_v1/result.json` + `results/manuscript/h1_h3_originales_v3/result.json` |
| **M3** | The oracle normaliser is not path-preserving: it changed the evaluated path in **6/6 contexts for `neuron_memory` and 6/6 for `neuron_reset`, and 0/6 for `ofat` and `random`** — i.e. it leaks exactly to the arms under test. The prefix normaliser changed **0/6 for all four**. | `results/twin_surface_v2/result.json` (`04b8137157e83a61…`) |

Admitted but not yet drafted (`_ENMIENDA_2` §§2–12), each with `self_sha256` **and** `file_sha256` because a design review mislabelled 16 file digests as `self_sha256`:

| P1 § | finding | artifact | grade |
|---|---|---|---|
| §2 | demand process scope (CV 7.1%, ACF −0.228) — **also cited by P2 §3.1; duplication risk** | `results/demand_process/result.json` | ⚠️ falsifiers not all passed |
| §3 | abandonment, the reference | `results/sensitivity/contention_headroom_v1_2/result.json` `CONTENTION_DOES_NOT_OPEN_THE_DOOR` | DEVELOPMENT |
| §4 | the refuted mechanism | `results/sensitivity/contention_headroom_v1_3/result.json` | DEVELOPMENT |
| §5 | context: `NON_SEPARABLE_BUT_CONTEXT_INVARIANT` | `results/surface_gates_v2/result.json` | REPLAY, ⚠️ falsifiers not all passed |
| §8 | **`H_regime` is NOT invariant to monotone reparameterisation.** `A_MONOTONE_RESCALING_SURVIVES_ALL_THREE`, 9/9 falsifiers, declared family K=661, Holm over all 661, exact bootstrap LCB over seeds, signal proxy validated (solvable pairs 0.9530 → 0.0212 under a step), power test passed (planted optimum H=0.10 → LCB 0.0965), clean negative control (**0 of 661 on the 288 grid**). Under Garrido's declared curvature (γ=1, his published σ): **0.0000 on 288 and 0.0195 on the extended grid — both below the 0.05 threshold.** With extra curvature H* goes **+0.1311 → +0.3815, 2.9× from a constant we chose**; the deciding transform is interior, `power(γ≈20)`. | `results/monotone_transform_family_v4/result.json` | REPLAY |
| §9 | **Two ReT defects.** (A) step cadence: **37% dispersion** across cadences on identical trajectories with physics invariant, and **the cadence inverts the recommended policy** (0.9 under `sim.run()`, 0.1 under daily stepping). (B) fulfilment-delay cliff: at `GARRIDO_FULFILLMENT_DELAY_HOURS = 54` vs `LT = 48`, ReT = 0.004424; at 48, **0.980513 — a 221.6× swing over six hours**, with autotomy unreachable. Repaired prospectively to dispersion **exactly 1.0**. | `results/metric_audit/ret_defects_v1/`, `ret_cadence_corrective_v2/`, `ret_repair_variants_v1/`, `ret_metric_repair_confirmation_v1/` (`PROSPECTIVE_CORRECTIVE_CONFIRMATION`, `historical_endpoint_unchanged: true`, `neural_authorization: false`) | UNCONTRACTED + DEVELOPMENT |
| §10 | **The Track B harness is not deterministic, and the preregistered suspicion is refuted.** `DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT`: 8 envs default threads Δ=0.472; 8 envs threads=1 Δ=**1.620**; **1 env, threads=1 Δ=1.539**. `DummyVecEnv` does seed and still does not reproduce. Consequences: the seed is **not** a replication unit; all seed-paired logic is void; `track_b_v1`'s +1.44…+2.18 neural premium falls inside a **±2.4** band. | `results/determinism_diagnostic/result.json` | DIAGNOSTIC |
| §10 | **…and then repaired.** `DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN`, f1/f2/f3 all pass. Artifact's own `consequence` field: *"the ±2.4 band collapses, so the track_b neural premium becomes MEASURABLE. It is still not CONFIRMABLE: no virgin seed block remains."* Recorded false start: the same line exists in `MFSCGymEnv`; repairing only the base class changed nothing because `track_b_v1` uses the subclass. | `results/determinism_repair_control/result.json` | DIAGNOSTIC |
| §12 | **`NO_ENDPOINT_CARRIES_REGIME_HEADROOM`** — 7 endpoints × 2 design spaces, threshold 0.05, none reaches it: `ret_excel_risk_conditional` +0.00380 / +0.02829 · `ret_excel_full_ledger` +0.00028 / +0.00978 · `ret_excel` +0.00050 / +0.00045 · `delivered_rations` +0.00015 / 0 · `flow_fill_rate` +0.00008 / **0** · `demanded_rations` 0 / 0 · `lost_orders` 0 / 0. **The endpoint we had been using is the most favourable of the seven.** | `results/endpoint_headroom_atlas/result.json` | REPLAY |
| §12 | **H2 learning curve, slope +0.042201 [+0.034664, +0.049922]**, 5 falsifiers, 120 replicates, zero seeds, prefix normaliser, with the null control that crosses zero. | `results/manuscript/h2_learning_curve/result.json` (`74b75141241ba763…`) | DEVELOPMENT |

**Additional P1-eligible findings not yet in either paper's manifest** (from `docs/ATLAS_DE_METRICAS_DE_RESILIENCIA_2026-08-06.md`):
- **The two metrics reward abandonment by two different mechanisms.** `ret_excel` **always** picks an extreme (0.1 or 0.9, never 0.5) across 18 cells while every sound endpoint picks 0.5; the share maximising `ret_excel` delivers **50%** of rations, the one minimising it delivers **80%** — 12× more "resilience" bought with 30 points of service, in all six cells. Cobb-Douglas is blind by a different route: an order never served **leaves the backorder queue and stops costing**, and there is no fill-rate term among its five variables.
- **The most uncomfortable number in the repo.** Under fungible resource, `H_regime` of `ret_excel_omitted_n` is **0.375** and of `ret_excel_visible_n` is **0.43**, against **1e−05** for the score itself. **What varies with the regime is not resilience — it is how many orders get excluded from scoring.** (`results/sensitivity/multi_metric_headroom_v1/`, 4,375 runs, 16 metrics + CD; normalised H/SD: risk_conditional 0.131 best, `ret_excel` 0.002, CVaR 0.001 bottom.)

### Findings that belong to NEITHER (flagged)

- **`results/gsa_resilience_only/` + `results/gsa_confirmation_corrective/`** — explicitly excluded by `_ENMIENDA_3` E4: *"C3 (GSA) no entra en la espina de ningún manuscrito; se declara en el censo."* This is a **sealed, five-falsifier, 600-tape, three-block-replicated positive with a virgin-block confirmation attached** (η 0.78–0.91 dev; η 0.9505 conf) — and it is homeless by decree, because the objective was changed to resilience-only by PI decision on the day it ran, and it costs the worst CSSU **−0.129** fill.
- **`results/buffer_saturation_diagnostic/`** — DDMRP adjudicated: **×10 above reference moves the metric exactly 0.000000** with `saturated_upward: true` in all three nodes, while lowering does hurt. "In this chain DDMRP degenerates to a constant at the ceiling, and the ceiling falls in a flat region."
- **`results/step3_expressiveness/`** — `BOTH_STEP3_FALSIFIERS_FAIL_ON_DOMAIN_EXPRESSIVENESS_NOT_ON_A_DEFECT`: the aggregate contract has **a single claimant** (141 orders, `cssu_destination = None`), so `worst_product_fill` **is** `flow_fill_rate` and the preregistered guardrail **is not expressible**.
- **The custody/evidence-grade methodology itself** — rules **R1–R10**, `research/evidence_registry.jsonl`, `research/seed_custody_registry.json`, and `R9: GRADE_NOT_MACHINE_DISCOVERABLE`. This is a genuine meta-methods contribution (a confirmation census that four external audits got wrong for a *diagnosable* reason: `grep '"run_role": "CONFIRMATION"'` finds C1 and C3 and misses C2, because C2's grade lives only in a sibling receipt). Homeless.

---

## 4. STRONG but ORPHANED — verdict on each

| finding | status | verdict |
|---|---|---|
| **Alzheimer / memory-advantage numbers** (`ALZHEIMER_EFFECT_*`, 9 DEV + 3 REPLAY artifacts) | **NOT orphaned.** Locked as P2 claim #5 (`RETENTION_NEURON_VS_RESET`) from `garrido_normaliser_audit_v3`. | **Paper section (P2, RQ1).** But rule **R2** binds: the nine `garrido_meta_learner*` artifacts are **not independent evidence** — dedup unit is `(contract_sha256, execution_commit, seed/tape block, estimand, endpoint)`. Cite one. |
| **Cobb-Douglas port** (`docs/COBB_DOUGLAS_PORT_RESULTS_2026-07-29.md`, `supply_chain/cobb_douglas_resilience.py`, `results/cobb_douglas/score_v1.json`, `economic_sensitivity_v1/v2`, `cobb_douglas_component_headroom{,_extended}`, `cobb_douglas_variant_family`, `headroom/cobb_douglas_v1`) | **Genuinely orphaned.** Only the abandonment slice reaches P1 M1. | **Paper section (P1) + supplement.** Three findings are publishable and unclaimed: (a) **Garrido's published exponents cannot be inverted** — `exp(0.20/0.024) = 4,160` against his stated ζ_max ≈ 3,612, a 15% error from 1.7% rounding; (b) **ζ enters positively — the index does not punish hoarding**, only the holding cost inside κ̇ does, with an exponent ~25× ζ's; (c) **the floor choice can dominate the index** — τ floored at 1e−4 gave that term magnitude 9.9 against a total budget of 0.20. Re-derived exponents: ζ 0.024→0.014200, ε 0.026→0.016990, φ 0.04→0.025582, τ 0.06→**1.075378**, κ̇ 0.1771→0.354673. This is a construct-validity critique of a 2024 IJPR paper. **Not** a separate paper — it is salami against P1. |
| **`garrido_risk_headroom_sensitivity_v1`** (`DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER`) | Assigned twice: P2 claim #12 (Discussion) **and** exhaustion certificate §2.6 (the "decisive new result"). | **Paper section in P3, one-line cross-reference in P2.** Its full weight — 4,860 ten-year evaluations, posture invariance across all 45 profiles at every budget cap, ReT 0.53→0.20, 144× below bar — is wasted as a P2 discussion paragraph. **Note the scope retraction travels with it**: Cf20 raises all four R2 risks together; uncovered are the continuous space, inter-family R1×R2, φ×ψ, and temporal coupling. |
| **`demand_process` characterisation** | Double-assigned: P1 §2 (`_ENMIENDA_2`) **and** P2 Results 3.1 (`claim_lock`). | **Scope paragraph in both — it is a scope declaration, not a finding.** Its `problems` entry is binding: *"falsifiers did not all pass — cite WITH the failure."* Required framing per `_ENMIENDA_3`: `within the thesis-inherited U(2400,2600) demand process`, weekly CV 7.1% — **never "minimal variation."** |
| **Contention / Program O headroom** | Half-assigned: `contention_headroom_v1_2/_3` are P1 §§3–4 *references*; the Program O chain (11 DEV artifacts) is orphaned. | **P3 spine.** Program O H_PI = **0.15151378920653932** (LCB95 0.11562, exact fungible-null 0.0, 25,177-episode parity, throughput conserved) is "the only material perfect-information ceiling in the entire search" and the only one to survive real Op9–Op12 buffering. Its H_obs failure chain (0/192 → resource confound → fixed-clock-only diagnostic → 26/48 vs 34 OOS) is a complete narrative arc. **Caveat that must be disclosed:** ledger convention is `PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION` and Δ_N is negative in all three cells. |
| **Determinism repair** | Assigned: P1 §10 (both diagnostic and repair, `_ENMIENDA_2`). | **Paper section (P1 §10).** Strongest form: *a preregistered mechanistic hypothesis (`make_vec` discarding its seed) was refuted by its own falsifier, the true cause located, the repair sealed with three falsifiers, and the consequence stated in the artifact — the seed is a replication unit again, so the premium is measurable but permanently unconfirmable.* This is a model of instrument accountability. |
| **"Cross-architecture reproduction"** (`results/repro_probe/A`, `/B`) | Orphaned — and **mis-described.** | **Not a separate finding; it is corroborating evidence for P1 §10.** Both probes are **the same architecture** (DMLPA, 187,404 params, seed **9492**, 200k steps, identical entry-script sha `3b231baa…`, same host). They differ by **1.64 ReT units** (94.383 vs 96.023) — an independent replication of the ±2.4 non-determinism band, not a cross-architecture result. Both carry `DEVELOPMENT_ARCHITECTURE_BAKEOFF_NO_ADJUDICATION` / `DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION`, and inherit the **unsealed-runner** defect. |
| **Metric curvature closure** (`citable_risk_attitudes`, `ONLY_RISK_SEEKING_CURVATURE_REACHES_THE_BAR`) | Orphaned. | **Supplement to P1 §8.** Substantively stronger than it was asked to be: CRRA η=0.25 → 0.0158; η=1 → 0.0050; η=10 → 4e−7; CARA a=10 → 4.5e−5, against identity 0.0195 and a 0.05 bar. Only the convex control qualifies (γ 5/10/20 → 0.085/0.159/0.277). **Risk aversion does not merely fail the bar — it moves headroom toward zero**, because concavity compresses the upside and makes one robust configuration look better. CVaR is a separate estimator: `H_regime_cvar = 0.048226736392119496`, identical at α 0.90/0.95/0.99 because six regimes all collapse to the worst. |
| **GSA lane** | Orphaned **by decree** (`_ENMIENDA_3` E4). | **Supplement or a standalone short note — not a paper.** It is a positive with a virgin-block confirmation, but it is one bit (`AAAA` vs `ABAB`, 31/89), the block was repurposed, and it costs the worst CSSU 12.9 points of fill. |
| **Custody / evidence-grade methodology** (R1–R10) | Orphaned. | **Methods appendix, shared across P1/P2/P3.** Cannot carry a paper alone in this venue. |

---

## 5. `docs/PROMISING_LANES_REGISTRY.md` — 89 KB, ~1,016 lines, append-only

**It exists, and it has been formally retired.** `_ENMIENDA_1` §E5, verbatim:

> **Se conserva como registro histórico y deja de intervenir en decisiones.** Su mezcla de resultados vigentes, superseded y señales de dos semillas es un mecanismo eficiente para resucitar claims muertos. La instrucción permanente de «nunca perder un lane» se cumple archivándolo, no consultándolo.

It terminates before `track_b_nonneural`, `step3_pooled` and the canonical table. Status inventory:

**CLOSED / falsified (Track A `continuous_its` family — lanes ⭐1 … 12):** ⭐1 not confirmed (3-seed fixed-init: Excel 0.00186 < 0.00193) · ⭐1b not confirmed at 5 seeds (0.00213 < 0.00218) · ⭐1b-audit — **learns adaptive but not forecast-driven** (`risk_forecast_48h/168h std = 0`; behaviour is temporal ramping) · ⭐1b-v8-cvar **DENSE-CRN FALSIFIED** · ⭐1b-preventive-excel-delta **DENSE-CRN FALSIFIED** (0/32 winners) · ⭐1b-per-op-r2 no promote · ⭐1c h260 failed · ⭐1e **0/32 winners** under dense CRN · ⭐1g rich 3-seed audit did not confirm (`dominated_by_count = 91`, `dominates_count = 0`) · ⭐1b-kaggle **COMPLETE, NULL** (`primary_win = false`) · lane 5, 6 collapse to constant · **lane 7 CD same-bar PERVERSE — crowns S1_I0, no buffer** · lane 11 RecurrentPPO 500k×5 **LOST** → memory is not the lever · lane 12 retention DQN **null (Δ≈0)**.

**OPEN in the file but superseded by later adjudication:** ⭐1b-preventive (3-seed Excel Pareto win, CVaR not confirmed) · ⭐1b-per-op-buffer · ⭐1d (oracle headroom exists, PPO/BC no win) · ⭐1f · ⭐2, ⭐3, ⭐4 (screens not confirmed) · ⭐8 · lane 9 DMLPA mixed · lane 10 partial · **⭐14 `PROMOTE AS REVIEWER-SAFE SPINE OPTION`** · ⭐15 · **⭐16 preventive claim RETRACTED, headroom CLOSED, efficiency claim survives** · **⭐17 Track B-P — "Gates 0/1 PASSED; Gate 2 running. NEVER LOSE THIS LANE."** ← the only lane still marked live, and it is **stale since 2026-07-09**; its family is exactly the one `TABLA_CANONICA` §7 now forbids.

**The one lane that closed cleanly and post-dates the retirement:** metric curvature (2026-08-06) — opened `⭐ ABIERTA` and closed the same file with `ONLY_RISK_SEEKING_CURVATURE_REACHES_THE_BAR`.

**Net: zero lanes are genuinely open.** Every live-looking marker predates a later adjudication that closed it.

---

## 6. Is there enough for a THIRD paper?

**Yes — but only one, and not the one anyone was hoping for. Bluntly: it is a negative-results methods paper, its ceiling numbers were measured on a metric that Paper 1 exists to retract, and it can never be confirmed.**

### The thesis

> **When not to train: a preregistered exhaustion search for deployable adaptive headroom in a validated supply-chain DES.** Across four decision surfaces, three independent programs, two physics generations, the source thesis's own risk-escalation table, and one researcher-disclosed extension engineered with ~7× more clairvoyant headroom, no decision contract converts perfect-information headroom into deployable, resource-honest, out-of-sample-stable observable value. **Zero PPO seeds were spent** — every gate killed its candidate before training. The negative is mechanistically explained, not merely observed: the DES is structurally over-buffered so constants are near-optimal; **the optimal posture is invariant to the risk regime**; and **commitment lags defeat clairvoyance** — a non-privileged EWMA detector beats the true-state instant switcher ×2.3.

Venue per `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md` §15: *Simulation Modelling Practice and Theory* or *Journal of Simulation*, fallback *C&IE*.

### What it has

Six quantitative ceilings with custody hashes (0.0221 / 0.0135–0.0246 / 6.5e−05 / +1.6–3.6% / ≤0.005 / 6.93e−05); Program O's 0.15151 with an exact 0.0 fungible null and conserved throughput; a complete four-stage H_obs failure chain with a disclosed executor bug whose verdict is independent of it; five named falsifiable reopeners; **and — unusually — a self-retraction of its own scope claim inside the artifact** (the OAT/hypercross retraction).

### What is missing — four items, in order of severity

1. **Endpoint generation mismatch, and it is the fatal one.** The certificate's `primary_endpoint` is `ret_excel_request_snapshot_v2` and it explicitly forbids metric substitution. But `results/metric_audit/ret_defects_v1/` (2026-07-29, **two weeks later**) shows that same metric family has **37% dispersion across step cadences with the cadence inverting the recommended policy**, and a **221.6× cliff** across six hours of fulfilment delay. Every number in §2 was measured on the instrument Paper 1 exists to indict. **The partial patch already exists** — `results/endpoint_headroom_atlas/` (7 endpoints, 2 design spaces, all below 0.05) and `results/sensitivity/multi_metric_headroom_v1/` (4,375 runs, 16 metrics + Cobb-Douglas) independently show the *negative* survives every endpoint tried. That defends the conclusion but not the tabulated ceilings. **The certificate must be re-adjudicated section by section under the repaired metric, or its ceilings restated as instrument-conditional.**
2. **Program O's ledger convention is unsigned.** `PROVISIONAL_PENDING_GARRIDO_SIMULINK_CONFIRMATION`, with Δ_N negative in all three cells. This is the paper's headline positive number and it depends on a domain sign-off that has not happened.
3. **No confirmation is possible, ever, without new PI authorisation.** `research/seed_custody_registry.json` = `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `new_seed_opening: false`. Enumerated over all **35** block rows: `USED_DEVELOPMENT_NOT_VIRGIN` 26 · `BURNED_CONFIRMATION_COMPLETE` 3 · `ATTEMPTED_NO_SEALED_ARTIFACT` 2 · `ARTIFACT_PRESENT_PENDING_MERGE` 2 · `ARTIFACT_PRESENT_PENDING_CANONICAL_CUSTODY` 1 · `BURNED` 1. **Zero available.** Corroborated inside an artifact: `results/dmlpa_kan_latent/result.json` carries `scope: DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN`. And note the registry now holds **35** rows where `_ENMIENDA_4` §E3 enumerated **33** — the amendment is already stale.
4. **The two strongest reopeners are questions, not data.** Q11/R09 (hard deadlines with permanent abandonment plus doctrinal triage authority) and Q6/Q7 (one named resource mutually-exclusively committed across plant/LOC/theatre) are unanswered domain facts. A reviewer will ask why they were not asked before certifying exhaustion. `_ENMIENDA_1` §E6 item 5 already reduces the ask to five questions for Garrido — **that conversation is the cheapest thing on this list and it has not happened.**

### The alternative P3, and why it fails

"Resilience metrics disagree about what to do while agreeing there is nothing to learn" is a genuinely strong, fully-owned, self-contained thesis: four independently constructed metrics (Garrido's Excel ReT, its full-ledger variant, his 2024 Cobb-Douglas index, and a service-first endpoint) **agree there is no regime-dependent headroom and disagree completely on what to do**, with two of them rewarding non-service by different mechanisms — censoring in one, absence of a lost-order price in the other. It lands directly on Bruckler et al. 2024 (C&IE 192, 110176), which Garrido cites, and on his own §6.2 gap.

**But it is Paper 1.** `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md` §14 already ruled on it: *"Descartado: partir el atlas en 2 papers. Salami."* Splitting it out is the single most likely way to get both papers rejected.

### Blunt count

| | |
|---|---|
| **P1** measurement/metric integrity | **Real. Nearly done.** 3 modules built, 16 artifacts admitted, 3 figures generated, evidence receipt sealed. Needs §§2–12 prose and the Cobb-Douglas port folded in. |
| **P2** retained search state before neural architecture | **Real. Nearly done.** 12-claim lock with per-claim allowed/forbidden wording, sections 3.1–3.7 drafted, two prospective confirmations, one against its own hypothesis. |
| **P3** when-not-to-train exhaustion | **Real, but stale and unconfirmable.** One generation behind on the endpoint, one unsigned ledger convention, zero seeds, two unasked domain questions. |

**Two papers' worth of confirmed-and-current science; a third paper's worth of material that needs a re-adjudication pass and a one-hour conversation with Garrido before it can be defended.**

### Three discrepancies found during the census, worth fixing before anything ships

1. **`docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` §1 is stale** — its table says `CONFIRMACIÓN … cuántos hay: **2**` and §"Las dos confirmaciones, y son sólo dos" lists C1 and C2. `_ENMIENDA_3` E1 supersedes this with a three-artifact census and **forbids the phrase "only two."** The index still contains a forbidden formulation.
2. **`research/seed_custody_registry.json` has 35 block rows; `_ENMIENDA_4` §E3 enumerates 33.** Rule R10 says custody state is read from the registry row, never the contract — so the amendment's own count is now the stale artifact.
3. **`results/repro_probe/A` and `/B` are the same architecture and the same seed (9492)** — a determinism probe, not a cross-architecture reproduction. If anything downstream describes them as cross-architecture, that is wrong, and both inherit the `run_architecture_bakeoff_v1.py` unsealed-runner defect.
