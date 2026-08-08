# SEALED-EVIDENCE INVENTORY — C&IE manuscript, "Retained Search State Before Neural Architecture"

Repo `<HOME>/Projects/research/scres-ia`, branch `codex/expanded-contract-comparators-v2`, HEAD `9b1afa9`. All digests below are **sha256 of the file** (`shasum -a 256 <path> | cut -c1-16`), which is what `claim_lock.json` calls `file_sha256`. Where an artifact also carries an internal `self_sha256` I give it separately, because the two differ and the repo has confused them before.

---

## A. MASTER ARTIFACT TABLE

| # | Path | claim_status | run_role / scope | Grade | Seed block (n) | Endpoint | n | Falsifiers | file sha256[:16] / self_sha256[:16] |
|---|---|---|---|---|---|---|---|---|---|
| A1 | `results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | `CONFIRMATION` / `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER` | **CONFIRMATORY** | `garrido_grid_transfer_v2_confirmation` 8200001–8200060 (60, virgin) | `auc_regret_norm` | 60 seeds × 6 contexts, budget 24 | **6/6 PASS** (`f1_null_subgrid`, `f2_new_factors_move`, `f3_transfer_beats_marginal_replay`, `f4_seed_custody`, `f_budgets_matched`, `f_source_manifest_identical`) | `7bc33823ccd90b5e` / `eceb9ee97613e172` |
| A2 | `results/retention_contrasts/result.json` | `RETENTION_LOWERS_REGRET_IN_6_OF_6_FAMILIES` | *(no `run_role`)* / `DEVELOPMENT_REANALYSIS_OF_A_SEALED_REPLAY_NO_SEEDS_NO_ADJUDICATION` | **REPLAY / DEVELOPMENT** | `garrido_q2_des288` 5300001–5300012 (12, burned) | `auc_regret_norm`, paired AUC(reset)−AUC(retained) | 12 per family × 6 families | **5/5 PASS** (`f1_twin_labelling`, `f2_seed_vector_alignment`, `f3_sealed_contrast_reproduces_bitwise`, `f4_pairing_carries_information` 5/6 families, `f5_no_seed_outside_declared_replay`) | `80a0a500de621675` / `e9132c95c4dda0bc` |
| A3 | `results/search_ladder_v5/result.json` | `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` | `CACHE_ANALYSIS` / `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` | **DEVELOPMENT** | `garrido_q2_des288` 5300001–5300012 (12) | `auc_regret_norm` | 12 seeds, 16 arms, budget 24, 6 contexts, 288 configs | **all_passed True**; f1–f6 PASS, `f7_no_fresh_seeds` = `not_applicable` (declared replay) | `f648a1da5aefaf2f` / `f7dfb1e8bc1c036b` |
| A4 | `results/surrogate_architecture_bakeoff/result.json` | `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP` | `CACHE_ANALYSIS` / `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` | **DEVELOPMENT** | `garrido_q2_des288` 5300001–5300012 (12) | `auc_regret_norm` (lower better) | 12 seeds × 7 arms | **f1–f5 PASS**, `f6_no_fresh_seeds` n/a | `f96e5b6ff0489932` / `965c477d91ebc322` |
| A5 | `results/dmlpa_kan_latent/result.json` | `KAN_LATENT_HURTS` | *(no `run_role`)* / `DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN` | **GRADE_NOT_MACHINE_DISCOVERABLE** | 9491–9495 (5, open) | `ret_mean_track_b_v1` (**higher** better) | 5 paired seeds, 24 eval eps, 100k steps | **5/5 PASS** | `b301833949b4386b` / `f79ca248bb983bd4` |
| A6 | `results/garrido_fig5_surrogate/result.json` | `DEVELOPMENT_FIG5_SURROGATE` | *(none)* / *(none)* | **GRADE_NOT_MACHINE_DISCOVERABLE** | fixed `seed: 20260731`, 90 configs / 30 groups | R², identity error | 90 rows, 5 grouped folds | **5/5 PASS** | `58d4c8a071cec86a` / `40bf64852c6e6f44` |
| A7 | `results/garrido_wrap_q1/result.json` | `DEVELOPMENT_FIG5_SURROGATE` | *(none)* | **DEVELOPMENT** — *not in claim_lock* | seed 20260731 | `B1_held_out_R2` (primary), `B2_activation_accuracy` | 90 rows / 5 folds | **5/5 PASS** (same set as A6) | `fd5d63a508531362` / `dc2e431338f605b3` |
| A8 | `results/garrido_h2_h3_confirmation_v1/result.json` (+ `completion_receipt.json`, `tape_level_deltas.json`) | `CONFIRM_H2_H3_ALL_SIX_PANELS`; receipt `COMPLETE_VALID_CONFIRMATION_AGGREGATE` | **absent from result.json** (`run_role`, `scope`, `claim_status`, `self_sha256` all null) — grade lives in the sibling receipt | **GRADE_IN_SIBLING_RECEIPT** (confirmatory) | 12 confirmation tape roots `96111336 … 97836128`; `development_roots_opened: false` | `flow_fill_rate` primary + delivered/full-ledger/unresolved concordant | 1,080 rows; 12 tapes × 6 panels | **6/6 panel gates confirmed**, all six subsidiary gates true in every panel; Holm 6/6 pass; `neutral_shift_checks` 108/108 equal | result `bc375d3021b64d10`; receipt `d4305bcf6bf5209d`; tape deltas `e12f3cf944c7ac0f` (72 rows) |
| A9 | `results/twin_surface_v2/result.json` | `PREFIX_NORMALISER_IS_BLIND_TO_THE_UNRUN_SURFACE` | `CACHE_ANALYSIS` / `DEVELOPMENT_ON_BURNED_TAPES_STRUCTURAL_SPY_TEST` | **DEVELOPMENT (control)** — *not in claim_lock* | 5300001–5300012, single-seed probe 5300001 | path identity under surface twin | 4 arms × 6 contexts | **1 substantive PASS** (`f6_surface_twins_have_identical_prefix_paths`), `f_no_fresh_seeds` n/a | `cbefeb716d1eda3f` / `04b8137157e83a61` |
| A10 | `results/garrido_normaliser_audit_v3/result.json` | `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER` | `BURNED_REPLAY_AUDIT` / `DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER` | **REPLAY** | `garrido_q2_des288` 5300001–5300012 (12) | `auc_regret_norm` under **prefix** and **oracle** normalisers | 12 repeats × 4 strategies × 2 normalisers × 6 contexts, 288 configs | **f1–f4, f6 PASS**; `f5_no_fresh_seeds` n/a | `fd617753949947e6` / `b1cf3705a070a191` |
| A11 | `results/frozen_path_equivalence/result.json` | `FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_SOURCE` | *(no run_role)* / `PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS` | **PROVENANCE** | seeds touched 5300001–12 + 8200001–60 (no new) | `cell_level_exact_reproduction` | 600 cells (300 + 300), 6 contexts | **5/5 PASS** incl. planted-1e-9 mutation control | `f162f581dbaf4e02` / `15bf9f046ba92af3` |
| A11b | `results/frozen_path_equivalence_v2/shards/` | **RUNNING NOW** — 8 `--phase surface --surface ext` shards + `--phase chain`; 24/… ext R1r shards written, each `cells: 4608, mismatches: 0, max_abs_delta: 0.0` | contract `docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md` | not yet sealed | 8200001–8200060 | same | full 4,608-wide grid + downstream verdict replay | pending `--phase seal` | no `result.json` yet |
| A12 | `results/demand_seasonal_engine/result.json` | **`ENGINE_PARTIAL`** | `DIAGNOSTIC` / `DEVELOPMENT_ENGINE_CHARACTERISATION_NO_ADJUDICATION_NO_LEARNER` | **DIAGNOSTIC, PARTIAL** | 8600001–8600012 (12) | `weekly_cv_and_seasonal_acf` | 12 episodes × ~954 weeks | **`all_passed: False`** — g1,g2,g3 PASS; **g4 FAIL**, **g5 FAIL** | `49cf0a6674efab38` / `f6eea651e931f57f` |
| A13 | `results/manuscript/h1_h3_originales_v3/result.json` | `H1_SUPPORTED__H3_NOT_SUPPORTED` | *(no run_role)* / `DEVELOPMENT_ALREADY_OPEN_BLOCK_NO_VIRGIN_SEEDS_NO_ADJUDICATION` | **DEVELOPMENT** — *not in claim_lock* | 6000001–6000120 (120, already open) | H1 `restricted_ttr` hours; H3 variance of service-loss AUC | H1 960 cells, H3 360 cells | **7/7 PASS** | `39061791dd37eef4` / `dc46ce6069755a28` |
| A14 | `results/manuscript/h2_learning_curve/result.json` | `H2_SUPPORTED_LEARNING_CURVE` | *(no run_role)* / `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION` | **DEVELOPMENT** — *not in claim_lock* | 6000001–6000120 (120) | OLS slope of (reset − memory) AUC on context ordinal 1..6 | 120 replicates | **5/5 PASS** incl. an order-confound null | `2894e525dc360c8f` / `74b7514124 1ba763` |
| A15 | `results/garrido_h3_merge_adjudication/result.json` | `H3_PRIME_SUSTAINED_AT_N120` | *(no run_role)* | **ADJUDICATED MERGE** — *not in claim_lock* | 6000001–6000090 (local) ⊎ 6000091–6000120 (VPS), disjoint | variance of search cost **across contexts**, memory − reset | 120 | **4/4 PASS** | `e06c53c136a89bcb` / `1ac02efa1618e5a9` |

### Support artifacts the manuscript already cites but which were not on the task list

| # | Path | Status | Grade | Seeds | file sha256[:16] |
|---|---|---|---|---|---|
| B1 | `results/demand_process/result.json` | `DEMAND_PROCESS_CHARACTERISED` | DIAGNOSTIC, **`all_passed: False`** (`f3_lag1_acf_inside_iid_band` FAILS — that failure *is* the finding) | 8600001–8600012 | `cb4f88398c4f93a4` |
| B2 | `results/surface_gates_v2/result.json` | `NON_SEPARABLE_BUT_CONTEXT_INVARIANT` | `CACHE_ANALYSIS` / DEVELOPMENT, **`all_passed: False`** (`g1_context_adaptation_is_worth_something` FAILS by design) | 5300001–12 | `5abd006f27be0d55` |
| B3 | `results/headroom/buffer_prediction_premium/result.json` | `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE` | 6/6 falsifiers PASS incl. `f6_seeds_are_virgin` | 6800001–6800010 | `54bf5fa2594262bd` |
| B4 | `results/functional_form_diagnostics/result.json` | `FUNCTIONAL_FORM_DIAGNOSTICS_REQUESTED_BY_DOMAIN_EXPERT` | DIAGNOSTIC, 4/4 PASS | none opened (sealed base cache, 60 seeds) | `504589fc230b86a0` |
| B5 | `results/ofat_lcb_reconciliation/result.json` | `OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE` | DEVELOPMENT re-analysis, 3/3 PASS | none | `4a383abeceda9d66` |
| B6 | `results/search_surrogates/result.json` | `APPROXIMATOR_IS_NOT_THE_INGREDIENT_RETENTION_IS` | `CACHE_ANALYSIS` / DEVELOPMENT | 5300001–12 | self `7a543f6b2066f82e` |
| B7 | `results/gsa_confirmation/result.json` `1f487d91900e2ea4` + `gsa_confirmation_corrective` `5e393b64b8ab950a` | third confirmation, **repurposed** virgin block, self-downgraded to a one-bit calendar choice | census only, enters no claim | — | as shown |

---

## B. THE NUMBERS, BY SECTION

### RQ2 — CONFIRMATION (A1, `results/grid_transfer_confirmation_v2/result.json`)
Design: 288 → **4,608** configurations (16×), budget 24, 6 contexts (`R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`), n = 60 virgin seeds. `transfers = {ucb1: true, neuron: false, gp: false, ofat: false}`.

| Family | vs cold: mean | LCB95 | UCB95 | **vs marginal replay: mean** | LCB95 | UCB95 | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `ucb1` | +0.05743819277092614 | +0.049888584439231054 | +0.06480569983808135 | **+0.03073311127302739** | **+0.019896866431745156** | +0.042561504429982196 | ✔ |
| `neuron` | +0.054393629143446906 | +0.04289557823009438 | +0.06687098572184685 | **−0.011782983814352867** | −0.01848914143725492 | −0.004835955987826106 | ✘ |
| `gp` | +0.01432635783389502 | +0.008787735338250295 | +0.02056083569699176 | **−0.021595031378833847** | −0.03050640445741258 | −0.012267301800787132 | ✘ |
| `ofat` | +0.014218869453841312 | +0.008002602681115948 | +0.019318535198958688 | **−0.024672161316045497** | −0.032580438401241286 | −0.01666244793292766 | ✘ |

Absolute mean AUC (12 arms, ready for a grouped bar figure):
`ucb1_transfer 0.07268348890154588` · `ucb1_marginal 0.10341660017457327` · `ucb1_cold 0.130121681672472` · `neuron_transfer 0.07980427883032978` · `neuron_marginal 0.06802129501597692` · `neuron_cold 0.1341979079737767` · `gp_transfer 0.09341379718626547` · `gp_marginal 0.0718187658074316` · `gp_cold 0.10774015502016045` · `ofat_transfer 0.09404105138601099` · `ofat_marginal 0.06936889006996547` · `ofat_cold 0.10825992083985228`.

Falsifier evidence worth putting in a methods table:
- `f1`: 103,680 cells checked against the 288 cache, **0 mismatches, max_abs_delta 0.0**.
- `f2` mean within-base-config spread from the two new factors: R1r 4.62656798833088e-04, R2r 1.682970557955972e-03, R1r+R2r 4.986128089778619e-04, R1r|esc 4.55390891544663e-04, R2r|esc 7.997609689765407e-04, R1r+R2r|esc 4.7228375006935774e-04.
- `f4` custody caveat, quote verbatim if the seed table is shown: registry is `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, so this is **NO_KNOWN_COLLISION, not a proof of virginity**; `foreign_registry_conflicts: []`, `sealed_artifact_overlap: []`.
- Runtime 3,802.4 s; contract `docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md` `1abbb2df6526f82b` (verified on disk).

### RQ1 — REPLAY, six paired within-family contrasts (A2, `results/retention_contrasts/result.json`)
Estimand: **AUC(reset) − AUC(retained)** per seed, positive = retention helps. n = 12 each. Bootstrap percentile, 5,000 draws, rng 20260806.

| Family | reset arm | retained arm | mean AUC reset | mean AUC retained | **Δ mean** | LCB95 | UCB95 | seeds favouring retention |
|---|---|---|---:|---:|---:|---:|---:|---:|
| `neuron` | `neuron_reset` | `neuron_memory` | 0.11273605766343564 | 0.05203274760040456 | **+0.06070331006303109** | +0.04568372925950388 | +0.07952764953166233 | **12/12** |
| `ucb1` | `ucb1` | `ucb1_transfer` | 0.09655146536034616 | 0.04502300698358852 | **+0.051528458376757603** | +0.03583075965748686 | +0.06593053388065225 | 11/12 |
| `ofat` | `ofat` | `ofat_transfer` | 0.10024197702481123 | 0.06274306709376977 | **+0.03749890993104145** | +0.02919537000775833 | +0.046747871800931845 | 12/12 |
| `lookahead_kg` | `lookahead_kg` | `lookahead_kg_transfer` | 0.11479477721224257 | 0.08018189890560519 | **+0.034612878306637375** | +0.02609869898961568 | +0.04314644488751498 | 12/12 |
| `gp_ei` | `gp_ei` | `gp_ei_transfer` | 0.10661346477746912 | 0.08390483487553962 | **+0.02270862990192948** | +0.012764700492337935 | +0.034099512485183496 | 12/12 |
| `thompson` | `thompson` | `thompson_transfer` | 0.10893184647329195 | 0.08907820238728746 | **+0.019853644086004468** | +0.01021780840209468 | +0.029558465192416398 | 10/12 |

`excludes_zero: true` in all six. `f3` reproduces the sealed neuron contrast **bit-identically** (mean `0.06070331006303109` = sealed; intervals differ only in the last decimals: sealed `[0.04590626893881853, 0.07997188775380663]`). `f4` shuffled-pairing widths: paired intervals narrower in **5 of 6** families (`ucb1` is the exception: paired 0.030099774223165385 vs shuffled 0.026742099303340608).

**Limits that travel with the result, from the artifact itself** (copy into the caption): contexts are averaged inside each seed before storage (`run_search_comparator_ladder_v5.py:218`), so **within-context contrasts are NOT recoverable and would require a re-run**; the 12 seeds are a declared re-execution of the burned block; six families is not a random sample of search methods.

### RQ1 support — the 15-method ladder (A3, `results/search_ladder_v5/result.json`)
Budget 24, 12 seeds, 288 configs, 6 contexts. Primary `auc_regret_norm`, oracle = 0.0.

| Rank | Arm | Mean AUC | Stateful | Paired vs `neuron_memory`: mean [LCB95, UCB95] |
|---:|---|---:|:--:|---|
| 1 | `ucb1_transfer` | 0.045023006983588520 | ✔ | −0.007009740616816028 [−0.024441318883473704, +0.014076439977282206] |
| 2 | `neuron_memory` | 0.052032747600404560 | ✔ | — |
| 3 | `ofat_transfer` | 0.062743067093769770 | ✔ | +0.01071031949336522 [+3.564844184833131e-05, +0.02171366105332719] |
| 4 | `lookahead_kg_transfer` | 0.080181898905605190 | ✔ | +0.028149151305200626 [+0.012845676382162115, +0.04191154996295932] |
| 5 | `gp_ei_transfer` | 0.083904834875539620 | ✔ | +0.03187208727513507 [+0.019229943011605136, +0.042145755333430956] |
| 6 | `thompson_transfer` | 0.089078202387287460 | ✔ | +0.037045454786882905 [+0.02038444229207778, +0.05185218017696919] |
| 7 | `ucb1` | 0.096551465360346160 | ✘ | +0.04451871775994159 [+0.03518653818669935, +0.05447320053251466] |
| 8 | `ofat` | 0.100241977024811230 | ✘ | +0.04820922942440667 [+0.03292031155918523, +0.0634593535389054] |
| 9 | `gp_ei` | 0.106613464777469120 | ✘ | +0.05458071717706455 [+0.035851158494365966, +0.07257890214069028] |
| 10 | `thompson` | 0.108931846473291950 | ✘ | +0.05689909887288736 [+0.04068412119715085, +0.07242185714216451] |
| 11 | `lhs_local` | 0.109489… | ✘ | +0.05745607517703233 [+0.03514962730858726, +0.08190136753893594] |
| 12 | **`neuron_reset`** | 0.112736057663435640 | ✘ | +0.06070331006303109 [+0.04590626893881853, +0.07997188775380663] |
| 13 | `lookahead_kg` | 0.114794777212242570 | ✘ | +0.06276202961183801 [+0.043346493918471785, +0.08081807702258073] |
| 14 | `random` | 0.139794561343199900 | ✘ | +0.08776181374279535 [+0.07033476374620021, +0.10375751133298124] |
| 15 | `annealing` | 0.174204… | ✘ | +0.12217146726844114 [+0.09547940053705108, +0.14881565146767145] |

Falsifier facts usable as method text: `f2` provokes a `LookupError` on unvisited cells (not asserted); `f3` shows KG ≠ EI on the first-context visit sequence; `f4` KG wins 5 of 6 on a synthetic single-peak surface (proves the arm can search); `f5` transfer arms end with **144 retained observations** each; `f6` all eleven v4 arms reproduce with **max_drift 0.0**.

### RQ1 support — honest normaliser (A10, `garrido_normaliser_audit_v3`)
`neuron_memory − neuron_reset`, **prefix** normaliser: **+0.06070331006303109 [+0.04556081327092027, +0.08019782091887094]**, n = 12. Under the **oracle** normaliser the same contrast is +0.09014692583412813 [+0.07419195970001444, +0.10802658070883876] — the honest normaliser costs about a third of the effect. Censoring rates on the retired `runs_to_within_1pct` estimand: memory 0.0556 / reset 0.1528 / ofat 0.2222 / random 0.6111 (prefix). **`claim_lock` forbids quoting 7.90, 7.24, 13.54, 12.42, 5.83 runs.**
Companion control A9 (`twin_surface_v2`): under a surface twin, the **prefix** normaliser leaves every path unchanged for all four arms × six contexts, while the **oracle** normaliser changes `neuron_memory` and `neuron_reset` paths in all six — the required-fail leg fires, so the test is not vacuous.

### RQ3a — the Fig-5 identity (A6/A7)
`R² = 1.0`, `max_abs_identity_error = 3.219646771412954e-15`. Identified coefficients: `Re_RPj = 0.9999999999999677`, `Re_FRt = 0.9999999999999998`. Degenerate all-zero columns: `Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`. Status `IDENTITY_NOT_A_LEARNING_TASK`. 90 configurations, 30 groups, 87 pairs, seed 20260731.

Task B (A7 `garrido_wrap_q1` only, `q1_decision`) — the part 03_results.md does **not** yet use:
- **B1 held-out R² (5 grouped folds)**: kan 0.9913280603236345 (sd 0.007366637), backprop 0.9863153335912018 (sd 0.008158610), **linear 0.9697483885147611** (sd 0.016506538), constant 0.0.
- Paired vs linear: backprop +0.016566945076440742 [+0.0048352801397191485, +0.028298610013162334]; kan +0.02157967180887328 [+0.0041640998372037795, +0.03899524378054278]. **SESOI 0.05; `passes_sesoi_and_ci: false` for both.**
- **B2 activation accuracy**: kan 0.7711111111111111, backprop 0.7177777777777778, linear 0.7111111111111111, majority 0.3333333333333334. Paired vs linear: kan +0.06000000000000003 [−0.016566172558918603, +0.13656617255891867]; backprop +0.006666666666666665 [−0.1392088748907709, +0.15254220822410422]. Both fail.
- `decision: NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`, `eligible_neural_models: []`, `selected_model_before_gates: linear_null`, `promotion_eligible: false`, blockers `["HOLD_WRAP_BEHAVIORAL_FIDELITY", "HOLD_METRIC_PROVISIONAL"]`.
- `verdict` rule ("beats the linear baseline by more than one between-fold SD of that baseline"): B1 true for both, **B2 false for both**.

### RQ3b — architecture bake-off as a search surrogate (A4)
Parameters: **kan 532, mlp_matched 529**. `kan_minus_matched_mlp` = **+0.01036905263158129, CI95 [+0.0030182183544574972, +0.018926078870915616], p_two_sided = 0.0012** (lower AUC better ⇒ interval entirely against the KAN).

| Arm | Mean AUC regret | % of ceiling | vs reference (mean [LCB95, UCB95], Holm p) |
|---|---:|---:|---|
| `neuron_5p` (**best**) | 0.05203274760040456 | 99.49913906574433 | — (reference) |
| `mlp_matched` | 0.08852563418951176 | 99.62982164346218 | +0.0364928865891072 [+0.024450252001980696, +0.04767301535079317], 0.0 |
| `spline_poly` | 0.09753786839357298 | 99.21492873120197 | +0.04550512079316843 [+0.02911623672727571, +0.0601788748707236], 0.0 |
| `kan` | 0.09889468682109304 | 98.53874073085433 | +0.04686193922068849 [+0.030074046837576664, +0.06308709092485791], 0.0 |
| `gbt` | 0.10832338755446090 | 97.75007865502914 | +0.05629063995405633 [+0.035820670072635105, +0.07553940359241366], 0.0 |
| `gp_matern` | 0.11379409899242587 | 96.99465513902659 | +0.061761351392021324 [+0.039200591454080495, +0.08288424578433401], 0.0 |
| `random` | 0.13979456134319990 | 93.38116637966432 | +0.08776181374279535 [+0.07158389607292293, +0.10385834513626549], 0.0 |

Harness-sensitivity leg (`f5`): best arm beats random by **−0.08776181374279535 [−0.1043089713335704, −0.07131315406601826]**, p = 0.0. `f4`: 7 distinct visit traces for 7 arms. **`scope_note` should be quoted**: "This does NOT live in the MPC lane… Here nothing is deployed: a configuration is chosen, the oracle is exact, and there is no pending service guardrail."

Efficiency, from B6 `search_surrogates` (the only place per-decision cost exists): `neuron_memory` 5 params, median 1.7917000150191598e-05 s/decision, total 0.03253584104095353 s; `surrogate_mlp` 369 params, 1.9035449986404274e-04 s, total 0.3420224959409097 s; `surrogate_kan` 380 params, 6.026664996170439e-04 s, total 1.0251607240224985 s. Neither surrogate beats the 5-parameter neuron: kan −0.0010122492317514702 [−0.006130431615626372, +0.004259958023086185]; mlp −0.001561219707701832 [−0.005942054454332779, +0.002597813540918106].

### RQ3c — KAN in a PPO latent, DIFFERENT contract (A5)
Endpoint `ret_mean_track_b_v1`, **higher is better**. `dmlpa_mlp 98.44314398434753` vs `dmlpa_kan 97.5808931153368`. Paired `kan − mlp` = **−0.8622508690107338 [−1.605043533470689, −0.11945820455077864]**, n = 5, **4 of 5 seeds negative** (per-seed: 9491 +?, see rows — mlp 98.918/97.479/97.978/99.131/98.709 vs kan 98.239/97.384/98.114/97.326/96.841). Relative **−0.88 %**, against within-seed evaluation SD **0.762–1.060**. Parameter match: kan 199,082 (hidden_dim 10) vs mlp 200,052 (hidden_dim **152**) — a **15×** width confound the artifact states up front (`result_before_training`). Determinism control `f4`: two replicas, delta 0.0. Declared uncovered confound (`not_covered`): normalisation order differs from the external version; only `latent_rw` was tested. Runtime 8,850.9 s. `claim_lock` **forbids the bare "−0.86225"** without endpoint, orientation, absolute means and the 15× confound.

### DES support — six thesis-derived comparative panels (A8)
1,080 rows, 12 confirmation tape roots, `development_roots_opened: false`, code_commit `9829084de18d0e1bf57d0da31d54beca56dd6997`, contract `1d3c80bd48feac4c`.

| Panel | Δ flow fill rate [LCB95, UCB95] | Δ delivered rations | Δ lost orders | Δ unresolved | Δ ret_excel_full_ledger | Δ ret_continuous | Holm p (rank/threshold) | pos tapes |
|---|---|---:|---:|---:|---:|---:|---|---|
| R1r · H2 buffer | **+0.0875118332520174** [+0.08700638248972684, +0.08801728401430797] | +614,350.51 | −180.24166666666667 | −237.14166666666665 | +9.342096476229296e-04 | +0.02795048443410669 | 6.618352850350777e-17 (2 / 0.01) | 12/12 |
| R1r · H3 shift | **+0.06746753306233107** [+0.06703512419865507, +0.06789994192600707] | +473,644.13 | −142.975 | −182.83333333333334 | +6.610789328672795e-04 | +0.019109208201903823 | 1.895791510834798e-15 (6 / 0.05) | 12/12 |
| R2r · H2 buffer | **+0.0919034…** [+0.0900042…, +0.0938025…] | +687,126 | −227.375 | −265.6 | +0.144738 | +0.0737255 | 1.297614470374752e-15 (5 / 0.025) | 12/12 |
| R2r · H3 shift | **+0.0618811…** [+0.0607973…, +0.0629648…] | +461,152 | −148.06666666666666 | −178.175 | +0.097323 | +0.0424779 | 4.80521239899934e-16 (3 / 0.0125) | 12/12 |
| R3 · H2 buffer | **+0.0170291…** [+0.0166778…, +0.0173804…] | +242,161 | −36.51666666666667 | −93.20833333333333 | +0.0234177 | +0.00310749 | 4.7554109189133694e-17 (1 / 0.008333…) | 12/12 |
| R3 · H3 shift | **+0.0124305…** [+0.0121131…, +0.0127479…] | +176,768 | −28.366666666666667 | −68.04166666666667 | +0.0175876 | +0.00236466 | 1.2176835466651612e-15 (4 / 0.016666…) | 12/12 |

`generated_orders`: mean 0.0, sd 0.0, **zero_tapes 12/12 in every one of the six panels**. Every panel passes all six gates: `primary_holm`, `fill_lcb_positive`, `delivered_lcb_positive`, `full_ledger_lcb_positive`, `unresolved_ucb_negative`, `generated_orders_exact_zero`. `neutral_shift_checks: {all_equal: true, count: 108}`. **Per-tape figure data**: `tape_level_deltas.json` (`e12f3cf944c7ac0f`), 72 rows = 6 panels × 12 tapes, each carrying all 12 delta columns.

**Claim boundary, verbatim from the artifact**: *"Confirmation applies only to H2/H3 resource interventions in the frozen thesis-grounded reconstructed DES; it does not establish learner, feedback, or architectural value."*
**Forbidden by claim_lock**: "validation of the DES", "the DES is validated", "order-level behavioural replication", "reproduces the Simulink model". Reason on file: `sumBt` is unreconstructed — `docs/PREGUNTAS_GARRIDO_2026-08-07.md:15` and `docs/REGISTRO_DE_HUECOS_2026-08-07.md:26` both state **no convention reconstructs the column in more than 1.09 % of 47,780 rows**.

### Provenance (A11 / A11b)
Sealed v1: 600 cells (300 in `garrido_transfer_confirmation_v2_ext` over 290 configs × 60 seeds; 300 in `wrap288_v1` over 198 configs × 12 seeds), **n_differing 0, max_abs_delta 0.0**, six contexts each. Declared drift = exactly `supply_chain/supply_chain.py` + `supply_chain/arm_runner.py`, `undeclared: []`. Mutation control plants **1e-9** at `…_v2_ext/R1r/8200049.json` config 1230 and `wrap288_v1/R1r/5300003.json` config 145, on key `delivered_rations`, and detects both while a clean cell still matches. Self-stated limit: *"does not certify unsampled cells, nor any artifact whose surface is not one of these caches, nor that the 2026-08-07 edits are harmless for FUTURE runs."*
v2 in flight closes exactly that: every cell of both surfaces plus a **downstream chain replay** of the confirmation verdict itself. Current shards: 24 written, 4,608 cells each, 0 mismatches, 0.0 max delta.
**Cross-architecture reproducibility** (`docs/HALLAZGO_REPRODUCIBILIDAD_CROSS_ARCH_2026-08-08.md`, commit `0488de6`): 105 slices, **30,240 cells, 0 mismatches, max_abs_delta exactly 0.0**, arm64/macOS-27.0/Python 3.11.15 → x86_64/Linux glibc 2.43/Python 3.14.4. **Measured over `ssh`, not through the pipeline — the doc itself says the sealed artifact must come from the v2 `--phase seal`, and forbids hand-writing a `result.json` for it.**

### Seasonal demand engine (A12) — ENGINE_PARTIAL, do not present as ready
Seasonal contract: period 12 weeks, `trough_weeks 1`, `trough_scale 0.35`, `forecast_seed_periods 36`, `double_trend false`; profile = eleven 1.059090909090909 values + one 0.35; `seasonal_profile_cv = 0.19598237397554638`.
Gates: **g1 PASS** (native path byte-identical, n = 5,724 / 5,725 on two seeds); **g2 PASS** weekly CV **0.1774649610307387** inside band [0.15, 0.28] vs `garrido_figure3_implied_cv 0.2130431067107785` and `thesis_uniform_cv 0.07130914239210204`; **g3 PASS** `acf_at_seasonal_lag = 0.8388902012880438` vs iid band 0.0647523908238176 (`acf1` 0.0061965294996763585, `acf_half_season` −0.11466299994600222, thesis-uniform seasonal acf 0.015270315100640202); **g4 FAIL** — α covers [0.086, 0.999] but **γ only [0.0198, 0.6913]**, mean 0.22420107553935156; **g5 FAIL** — forecast correlation with next-week realised is **−0.23418487217933096** (SE 0.046956848619133) with MAPE 0.23775924824850533, i.e. the forecast is anti-informative.
Three declared source ambiguities to quote if the engine appears at all: Eq (1) taken literally double-counts the trend (`F+2d`); Figure 3's reported kurtosis −1.88 with skewness 5.19 is **impossible** (kurtosis ≥ skewness² + 1), so calibration used mean/sd/min/max; the Makridakis 36-value seed series is untranscribed, so the period-12 profile is *our* reconstruction at *our* scale.

### Appendix material — H1/H2/H3 (A13/A14/A15)
- **H1** `restricted_ttr = min(TTR, τ)`, τ = 1,344 h, horizon 6,048 h, 960 cells: hybrid vs reset **+74.04583333333333 h [+51.523828125, +97.46838541666664]**; hybrid vs static **+125.98541666666667 h [+98.346796875, +154.54471354166665]**; both Holm-rejected at 0.05 with p_raw 0.0. Levels: hybrid 75.7 h, reset 149.74583333333334 h, static 201.68541666666667 h. On the 756 differing cells only: +159.9814814814815 [+123.99262566137567, +195.61435185185184].
- **H3 (original)** horizon 8,736 h, 360 cells: hybrid vs reset −1.1094468323605106e+15 [−3.648590718211967e+15, +1.2929658238053142e+15], p_one_sided 0.8208, Holm 1.0 → **NOT supported**; hybrid vs static −1.0497340472798898e+14, p 0.5248 → **NOT supported**.
- Declared endpoint redefinition, quote it: *"H1 uses restricted_ttr = min(TTR, tau) with a paired placebo, not system_ttr. It is a different estimand, written 2026-08-06 for the v0 lane and before this preregistration, not a loosened version of the one that returned 1.000."* Regime note: under recurrent R11–R24 at 52 weeks events merge into one cluster that never ends, so H1 is measured under **isolated shocks**.
- **H2 learning curve**: primary slope of (reset − memory) on context ordinal = **+0.04220147575193482 [+0.03466393985914079, +0.04992205677530306]**, n = 120. Order-confound null (`random − ofat`) = −0.0050880611149445275 [−0.015570216666852871, +0.0056579986049085955] — straddles zero, so the trend is not the ordering. `f4` shows the estimator can return negative: min slope −0.0564365828492133, 22 of 120 negative. Mean AUC by arm: `neuron_memory 0.17022357795416482`, `ofat 0.2619659335765739`, `neuron_reset 0.38358395746120394`, `random 0.4434164832167541`. Advantage by context: R1r **0.0**, R1r+R2r 0.1905162315685694, R1r|esc 0.22111385282764517, R1r+R2r|esc 0.26868607643557035, R2r 0.2827524501250933, R2r|esc 0.31709366608535716.
- **H3′ merge**: memory − reset variance-across-contexts **+9.314444444444444 [+2.3491250000000012, +16.34740972222222]**, n = 120; memory − ofat **+16.220277777777778 [+9.61070833333333, +22.740534722222215]**. Mean variance: memory 44.264166666666675, reset 53.57861111111111, ofat 60.48444444444444, random 67.50333333333333. Per-slice diagnostic: local n=90 +10.265925925925927 [+2.3608518518518533, …]; **VPS n=30 +6.459999999999998 [−7.30877777777778, +20.32813888888888] — straddles zero**, disclose it.

### Surface property (B2) and demand scope (B1)
`H_regime = 0.003802243800697269`, **LCB95 1.3519330929717688e-05, UCB95 0.014856419878462748** (5,000 boot), threshold 0.05 ⇒ `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. Best common config `{buffer_hours 1344, shifts 2, op9_rop 12, op12_rop 12}`; 4 distinct argmaxes across 6 contexts, all at `buffer_hours 1344, op9_rop 12`, differing only in shifts (1/2/3) and one `op12_rop 24` in R2r. Separability gain per context (interaction share) 0.041–0.169.
Demand: 65,835 regular orders, min 2400 / max 2600 / mean 2500.2114528746106; weekly mean 15,626.899895178198, weekly sd 1,114.3408297723865, **weekly CV 0.07130914239210204**; weeks over S1 capacity **0.24799091544374563**; weekly capacity S1 15,384 / S2 30,768 / S3 46,152; rations per shift 2,564; contingent share of quantity 0.08004842713742029 (2,863 contingent orders, mean 5,001.887879846315). ACF: lag-1 **−0.22834947925105253** (SE 0.003766206607563948), lag-2 −0.16984908640397386, lag-4 −0.01511092888364519, iid band ±0.0647523908238176.

### Discussion support (B3, B4)
Curvature: `mean_one_minus_linear_r2 = 0.07625893719919702`; per cell 0.02388177309515338 (R2r|base) … 0.13428686933720857 (R1r|freq_x5). Held-out R² (seed-grouped, 5 folds, 1,530 rows, 10 virgin seeds 6800001–10): **kan 0.716317881832606, linear 0.682606466674648, backprop 0.5548262724413864, constant −0.0033610961798922643**. Paired vs linear: kan +0.03371141515795817 [−0.07873510733477114, +0.14615793765068746]; **backprop −0.12778019423326165 [−0.31567766641387396, +0.06011727794735067]**. SESOI 0.05, neither passes.
Functional form (B4, requested by Garrido 2026-08-07; 17,280 rows and 60 seeds per context):

| Context | Ramsey RESET F (power 2) | p | linear held-out R² | quad+int held-out R² | gain | ΔAIC(linear − best) |
|---|---:|---:|---:|---:|---:|---:|
| R1r | 2234.2085322266794 | 0.0 | 0.5913446402417829 | 0.8212174554911373 | +0.2299 | 14,415.807922952343 |
| R1r+R2r | 2463.181672676075 | 0.0 | 0.5934341308263082 | 0.8014163836208287 | +0.2080 | 12,545.480808219028 |
| R1r\|esc | 1928.2 | 0.0 | 0.5598 | 0.7932 | +0.2334 | 13,178.8 |
| R1r+R2r\|esc | 1008.7901845097854 | 3.263464573241074e-215 | 0.5715306996778922 | 0.7638450080873902 | +0.1923 | 10,494.696166553156 |
| R2r | 427.6 | 0.0 | 0.1697 | 0.2276 | +0.0578 | 1,299.5 |
| R2r\|esc | 383.7 | 0.0 | 0.2778 | 0.3670 | +0.0892 | 2,361.5 |

AIC and held-out R² both select `quadratic_interactions` in **6/6**. Range quoted in claim_lock ("F 384–2463", "+0.19 to +0.23 in the R1r family") checks out.

---

## C. EXISTING PROSE — DO NOT REWRITE

**`papers/paper2/03_results.md` (209 lines, 2026-08-07 20:32) is current and near-complete.** Sections and their evidence:
- §3.1 demand scope (B1) · §3.2 six DES panels (A8) · §3.3 Fig-5 identity (A6) · §3.4 ladder + `H_regime` (A3 + B2) · §3.5 **the central RQ2 result** (A1) · §3.6 fit-vs-search (A4 + B3) · §3.7 five explicit prohibitions + pointer to "Appendix A" reconciling H1–H4.
It already carries the frozen framing, the vocabulary rules, and the boundary quote. **Every headline number in it verifies against the artifacts** (I checked all six panels, all four RQ2 contrasts, all seven bake-off AUCs, both curvature figures).

**`papers/paper2/claim_lock.json` (17 KB, 2026-08-07 21:49, `generated_at_commit: ccaa759`)** — 12 claims with `allowed` sentence, artifact, digests, endpoint, estimand, evidence_grade, `forbidden` phrase list and `why_forbidden`. It is declared authoritative over the six `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07*` documents for citation resolution. Its own `problems` list has one entry: `DEMAND_PROCESS_SCOPE: falsifiers did not all pass -- cite WITH the failure`.

**Superseded, retain but do not mine:** `01_introduction_draft.md`, `02_methods_draft.md`, `03_results_draft.md`, `04_discussion_draft.md`, `results_table.md/.json`, `build_results_table.py` — all 2026-07-18, all written for the **Program O / Program Q four-level ladder** ("Learning Adaptive Control Without a Neural Premium"), a closed programme. `03_results.md` says so in its own header. `docs/manuscript_draft/` and `docs/manuscript_notes/` are June-era and older still.

**Reusable prose elsewhere:** `docs/MANUSCRIPT_MODEL_VALIDATION_SECTION_2026-07-31.md` is a finished, self-corrected model-validation section (CTj lattice, the 54 h vs 48 h floor, autotomy unreachable, `ret_above_one_share`, the retracted RPj-saturation claim). `docs/RESULTADO_DIAGNOSTICOS_FORMA_FUNCIONAL_2026-08-08.md` and `docs/HALLAZGO_REPRODUCIBILIDAD_CROSS_ARCH_2026-08-08.md` are written up and ready.

---

## D. PRIMARY EVIDENCE PER SECTION, AND THE TABLE THAT GOES IN IT

| Section | Primary artifact | Table columns | Figure |
|---|---|---|---|
| **RQ1** (retention improves search) | **A2 `results/retention_contrasts/result.json`** — this replaces rank position, which is not a causal estimand (claim_lock says so explicitly) | family · reset arm · retained arm · mean AUC reset · mean AUC retained · Δ · LCB95 · UCB95 · seeds favouring retention (6 rows above) | six paired dot-and-interval rows, one per family; or 12 paired slopes per family. **A3** supplies the 15-arm ranking as a secondary/context figure |
| **RQ2** (survives 16× expansion + state-blind replay) | **A1 `results/grid_transfer_confirmation_v2/result.json`** | family · vs-cold mean/LCB/UCB · vs-marginal-replay mean/LCB/UCB · transfers bool (4 rows) | 4-family × 3-arm (cold / marginal / transfer) grouped bars from `mean_auc`, with the marginal-replay contrast as the annotated delta. This is the paper's Fig. 1 |
| **RQ3** (is the carrier neural) | **A4** for 3b, **A6/A7** for 3a, **A5** for 3c | 3a: model · B1 R² · Δ vs linear [CI] · B2 acc · Δ vs linear [CI] · passes SESOI. 3b: 7 arms × mean AUC · % of ceiling · Δ vs neuron_5p [CI, Holm p] · parameters. 3c: arm · params · hidden_dim · ret_mean · paired Δ [CI] · seeds negative | one fit-vs-search scatter (held-out R² on x, AUC regret on y) is the single most persuasive figure for the KAN point, but **see the flag below — the two axes do not currently come from one artifact** |
| **DES support** | **A8 + `tape_level_deltas.json`** | panel · Δ fill [LCB, UCB] · Δ delivered · Δ lost · Δ unresolved · Δ full-ledger ReT · generated orders · Holm p · favourable tapes (6 rows) | 6-panel dot plot of the 72 per-tape deltas, one panel per row, with the LCB marked. Direct structural answer to Ding et al.'s convergence-curve grids |
| **Provenance / reproducibility** | **A11** now, **A11b** when sealed; `HALLAZGO_REPRODUCIBILIDAD_CROSS_ARCH` for the cross-arch sentence | cache · configs · seeds · contexts · cells · mismatches · max_abs_delta · mutation perturbation detected | none needed; a boxed reproducibility statement |
| **Positioning vs Ding et al. (2026)** | A1's `f3` + A2's `f4` + A11's `f3` | our Table 1 row can claim what their Table 1 cannot: a **non-neural comparator**, a **state-blind marginal replay control**, **seed custody with its own incompleteness disclosed**, **falsifiers that state why they can fail**, and **intervals on every headline** | — |

---

## E. FLAGS — numbers a manuscript wants that are NOT in a sealed artifact

1. **RQ1's primary artifact is not in `claim_lock.json`.** `claim_lock` was generated at `ccaa759`; `results/retention_contrasts/` landed at `22e9d2a`, a descendant. The lock has **no `RQ1_RETENTION_SIX_FAMILIES` entry**. Regenerate `scripts/build_paper2_claim_lock_v1.py` before submission or RQ1 cites outside the lock.

2. **The six retention contrasts were never preregistered in a document that describes them.** `results/retention_contrasts/result.json` names `docs/ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md` (`8e78b344…`, hash verified) as both `contract_path` and `preregistration`, but that document is about a **deduplicated evidence registry** — it contains no estimand, no arms, no falsifiers for this analysis. Grep across `docs/` finds no doc preregistering these contrasts. It is honest REPLAY re-analysis of a sealed artifact with a bit-identical reproduction check, but "preregistered" must not be claimed for it.

3. **`03_results.md` §3.4 uses a phrase `claim_lock` explicitly forbids.** Line 114–115: *"`ofat_transfer` **+0.0107 [+0.0000356, +0.0217]** — the latter excludes zero…"*. `claim_id: OFAT_CONTRAST_IS_RESAMPLING_UNSTABLE` lists `"excludes zero"` as forbidden, because the two sealed ladders give **opposite-signed bounds on byte-identical arrays** (`search_ladder_v5` stored LCB +3.564844184833131e-05 vs `search_ladder_v2_ordered` stored LCB −2.761381942678142e-05, shared mean 0.01071031949336522, positive in only **26 of 40** rng seeds = 0.65). The artifact's `how_to_report_it` gives the required wording. Fix before submission.

4. **"The KAN attains better supervised fit on held-out partitions yet searches worse" splices two artifacts.** `results/surrogate_architecture_bakeoff/result.json` contains **no fit metric at all** — no R², no held-out score, only AUC regret. The fit half comes from A7 (`kan 0.9913 > backprop 0.9863`, 90-row wrap panel) or B3 (`kan 0.7163 > backprop 0.5548`, 1,530-row buffer panel), neither of which is the 532/529-parameter pair the search half compares. A one-artifact fit-vs-search scatter **does not exist**. Either qualify the sentence or run a fit measurement inside the bake-off contract.

5. **"0.3174 of unexplained episode-level variance" is derived, not stored.** It is `1 − held_out_r2_mean.linear = 1 − 0.682606466674648`. Arithmetically fine; say "1 − R²" rather than implying a sealed field.

6. **`H_regime = 0.00380` is cited in §3.4 with no path or digest.** It lives in `results/surface_gates_v2/result.json` (`5abd006f27be0d55`, self `27244b4ee4505bbc`), whose `falsifiers.all_passed` is **False** because `g1_context_adaptation_is_worth_something` fails — which is the finding, but must be stated that way. Also note the standing memory that H_regime is **not curvature-invariant**: a monotone rescaling moved the extended-grid figure 0.0195 → 0.0742. The 288-grid zero is transform-proof; this 0.00380 number needs its normalisation ("per-context min-max of the seed-averaged surface") stated.

7. **A8 has no machine-readable grade.** `run_role`, `scope`, `claim_status`, `self_sha256` are all absent from `result.json`; the confirmation status lives only in `completion_receipt.json`. `ENMIENDA_3` §E2 documents this as the root cause of five miscounted external audits. Cite result **and** receipt together, always.

8. **Per-context breakdowns do not exist for RQ1 or RQ2.** Both `search_ladder_v5` and `grid_transfer_confirmation_v2` average the six contexts inside each seed before storage (`run_search_comparator_ladder_v5.py:218`). Any reviewer question of the form "does UCB1 transfer in R2r specifically?" requires a **re-run**. `retention_contrasts` states this limit explicitly.

9. **`frozen_path_equivalence` v1 certifies 600 of ~1.66 M cells.** Its own `what_this_does_not_certify` says unsampled cells are not covered. The v2 job now running closes it (full grid + downstream verdict chain); until `--phase seal` produces a `result.json`, the manuscript's provenance paragraph rests on a 600-cell sample plus the 30,240-cell cross-arch measurement — and the latter **was taken over ssh, not through the pipeline**, and the finding doc forbids fabricating an artifact for it.

10. **`demand_seasonal_engine` is `ENGINE_PARTIAL` with `all_passed: False`.** g4 (γ does not cover the unit interval, max 0.6913) and g5 (forecast correlation **−0.234**, i.e. anti-informative) both fail. It cannot support any seasonal-generalisation claim. If the paper needs a "does this survive non-stationary demand" answer, §3.1 already scopes it out ("not established here… subject of a separate study") — keep it that way.

11. **The census sentence about confirmations is a frozen formulation.** `ENMIENDA_3` forbids **both** "three confirmations" and "only two". Authorised wording: two prospective confirmations usable by this manuscript, plus a third (`gsa_confirmation`, `1f487d91900e2ea4`) that ran on a **repurposed** virgin block and was downgraded by its own corrective (`5e393b64b8ab950a`) to a one-bit calendar choice.

12. **Seed custody is disclosed as incomplete.** Every custody falsifier in this corpus returns `registry_status: BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED` and `registry_is_complete: false`. A1's `f4` states plainly that its result is **NO_KNOWN_COLLISION, not a proof of virginity**. Reproduce that caveat in the methods section rather than letting a reviewer find it.

13. **No number exists for the H1/H2/H3 → RQ mapping promised in §3.7 ("Appendix A").** A13/A14/A15 supply the H-level numbers, but no artifact or document performs the reconciliation. That appendix is unwritten prose, not missing data.