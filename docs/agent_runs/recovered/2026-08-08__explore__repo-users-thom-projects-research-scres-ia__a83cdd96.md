# Explore — Repo: <HOME>/Projects/research/scres-ia

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a83cdd966dedfb36c`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T00:50:10.388Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-a83cdd966dedfb36c.jsonl`
- **turns**: 63 · **assistant messages**: 3

## Task given

```text
Repo: <HOME>/Projects/research/scres-ia

GOAL: produce the definitive list of POSITIVE, citable results — the ones a paper could lead with — with exact numbers, intervals, artifact paths and evidence grade.

Method:
1. Read `research/evidence_registry.jsonl` (one JSON per line). Filter to rows where `duplicate_of` is null. Report every row whose `evidence_grade` is CONFIRMATORY, with its artifact path and authored claim_status.
2. Then, for each of these artifacts, open it and extract the headline numbers (effect, CI/LCB, n, seed block):
   - results/grid_transfer_confirmation_v2/result.json
   - results/gsa_confirmation_corrective/result.json
   - results/garrido_h2_h3_confirmation_v1/result.json
   - results/manuscript/h1_h3_originales_v3/result.json
   - results/manuscript/h2_learning_curve/result.json
   - results/manuscript/h1_h3_v2_1/result.json
   - results/garrido_h3_merge_adjudication/result.json
   - results/garrido_normaliser_audit_v3/result.json
   - results/search_ladder_v5/result.json
   - results/surrogate_architecture_bakeoff/result.json
   - results/garrido_fig5_surrogate/result.json and results/garrido_wrap_q1/result.json
   - results/headroom/buffer_prediction_premium/result.json
   - results/step3_split_pooled/result.json
   - results/determinism_repair_control/result.json
3. Also check whether `results/dmlpa_kan_latent/result.json` exists yet (a run may be in flight) and report its state.
4. For each, state clearly: is the headline POSITIVE (an effect with an interval excluding zero), NEGATIVE (a measured null/stop), or DIAGNOSTIC (an instrument property).

OUTPUT: one table of POSITIVE results (effect, interval, n, artifact, grade), one table of NEGATIVE/boundary results, one of DIAGNOSTIC findings. Exact numbers only — do not round beyond what the artifact stores, and do not infer numbers that are not present. Read-only.
```

## Final answer

## 1. Registry pass — `research/evidence_registry.jsonl`

216 rows; 211 with `duplicate_of == null`. Grade distribution over all 216: DEVELOPMENT 88, UNCONTRACTED 57, REPLAY 42, NEGATIVE_OR_HALTED 20, DIAGNOSTIC 5, CONFIRMATORY 4.

**Every CONFIRMATORY row with `duplicate_of == null` (3 of the 4; the 4th is a self-duplicate of the third):**

| artifact_path | claim_status_as_authored | seed_block (dedup_key) | contract | grade_rationale |
|---|---|---|---|---|
| `results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | 8200001–8200060 | `docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md` | confirmation role over custody block 8200001-8200060 |
| `results/gsa_confirmation/result.json` | `GSA_CONFIRMED_ON_VIRGIN_BLOCK` | 7700001–7700120 | `docs/PREREGISTRO_CONFIRMACION_GSA_2026-08-07.md` | confirmation role over custody block 7700001-7700120 |
| `results/garrido_h2_h3_confirmation_v1/result.json` | `CONFIRM_H2_H3_ALL_SIX_PANELS` | 96111336–97836128 | `contract_path: null` (sha `1d3c80bd…`), branch `codex/paper-b-retained-v5` | confirmation role over custody block 96111336-97836128 |

**Two registry facts that change how you cite these:**

1. The CONFIRMATORY grade sits on `results/gsa_confirmation/result.json`, **not** on `results/gsa_confirmation_corrective/result.json`. The corrective is registered `DEVELOPMENT` and carries `supersedes → gsa_confirmation` with reason "f4 demanded an unsatisfiable property and f6 tested the data instead of the estimator; both were my specification errors". The graded-CONFIRMATORY file itself has `falsifiers.all_passed = false` (f4 failed). So the only artifacts that are both CONFIRMATORY and internally all-pass are grid_transfer_confirmation_v2 and garrido_h2_h3_confirmation_v1.
2. `garrido_h2_h3_confirmation_v1` is registered twice; the second row sets `duplicate_of` to its own path.

Sign conventions below: `ret_excel`/`ret_continuous`/`flow_fill_rate` — higher is better, positive delta = intervention helps. `auc_regret_norm` — lower is better, so a positive "X vs neuron_memory" means X is **worse**. `restricted_ttr` contrasts are hours saved.

---

## 2. POSITIVE results (interval excludes zero)

| # | Result | Effect (mean) | 95% interval | n | Artifact | Grade / status |
|---|---|---|---|---|---|---|
| P1 | H2 buffer, R2r (Δ`ret_excel`) | 0.1284415546074628 | [0.12384565157279213, 0.13303745764213346], p=1.297614470374752e-15 | 12 tapes | `results/garrido_h2_h3_confirmation_v1/result.json` | **CONFIRMATORY** / `CONFIRM_H2_H3_ALL_SIX_PANELS` |
| P2 | H3 shift, R2r (Δ`ret_excel`) | 0.07582819033358885 | [0.07334974955636112, 0.0783066311108166], p=4.80521239899934e-16 | 12 | same | CONFIRMATORY |
| P3 | H2 buffer, R3 | 0.03623302776727328 | [0.03527370395484829, 0.037192351579698274], p=4.7554109189133694e-17 | 12 | same | CONFIRMATORY |
| P4 | H3 shift, R3 | 0.02359898015235028 | [0.02275943950123066, 0.0244385208034699], p=1.2176835466651612e-15 | 12 | same | CONFIRMATORY |
| P5 | H2 buffer, R1r | 0.0005354201229473343 | [0.0005208109332238583, 0.0005500293126708102], p=6.618352850350777e-17 | 12 | same | CONFIRMATORY |
| P6 | H3 shift, R1r | 0.00036415833276666384 | [0.00035066979556362535, 0.0003776468699697023], p=1.895791510834798e-15 | 12 | same | CONFIRMATORY |
| P7 | **UCB1 grid transfer beats its own marginal replay** (Δ`auc_regret_norm`) | 0.03073311127302739 | [0.019896866431745156, 0.042561504429982196] | 60 | `results/grid_transfer_confirmation_v2/result.json` | **CONFIRMATORY** / `GRID_TRANSFER_CONFIRMED__UCB1` |
| P8 | UCB1 transfer vs cold | 0.05743819277092614 | [0.049888584439231054, 0.06480569983808135] | 60 | same | CONFIRMATORY |
| P9 | GSA observed headroom `H_obs` (η = 0.9505373773253464 of `H_PI` = 0.012926135742150043) | 0.012286775167294721 | corrective: [0.00867768343802695, 0.016043205900984406]; original: [0.00867686813365007, 0.01615513451498119] | 120 tapes | `results/gsa_confirmation_corrective/result.json` (DEVELOPMENT, supersedes) / `results/gsa_confirmation/result.json` (CONFIRMATORY) | see caveat below |
| P10 | GSA obs minus permuted placebo | 0.07210657135008629 | [0.049801679249144644, 0.0959939565344071] | 120 | `results/gsa_confirmation_corrective/result.json` | DEVELOPMENT |
| P11 | **H1** hybrid vs static, restricted TTR hours saved | 125.98541666666667 h | [98.346796875, 154.54471354166665], p_one_sided 0.0, Holm 0.0 | 960 cells | `results/manuscript/h1_h3_originales_v3/result.json` | DEVELOPMENT / `H1_SUPPORTED__H3_NOT_SUPPORTED` |
| P12 | H1 hybrid vs reset | 74.04583333333333 h | [51.523828125, 97.46838541666664], p_one_sided 0.0, Holm 0.0 | 960 cells | same (levels: hybrid 75.7, reset 149.74583333333334, static 201.68541666666667; τ=1344.0 h) | DEVELOPMENT |
| P13 | **H2** learning-curve slope (reset−memory AUC vs context ordinal 1..6) | 0.04220147575193482 | [0.03466393985914079, 0.04992205677530306] | 120 replicates | `results/manuscript/h2_learning_curve/result.json` | DEVELOPMENT / `H2_SUPPORTED_LEARNING_CURVE` |
| P14 | **H3′** variance of search cost across contexts, memory vs ofat | 16.220277777777778 | [9.61070833333333, 22.740534722222215] | 120 | `results/garrido_h3_merge_adjudication/result.json` | DEVELOPMENT / `H3_PRIME_SUSTAINED_AT_N120` |
| P15 | H3′ memory vs reset (the preregistered primary; rule LCB95>0 → sustained) | 9.314444444444444 | [2.3491250000000012, 16.34740972222222] | 120 | same | DEVELOPMENT |
| P16 | **H1′** service-loss AUC, primary all cells (ration·hours) | 61349290.03333334 | [14432891.773750002, 121804658.42541659] | 72 cells × 5 seeds | `results/manuscript/h1_h3_v2_1/result.json` | DEVELOPMENT |
| P17 | H1′ secondary, differing cells only | 105170211.48571427 | [28008967.28071429, 204916732.27571425] | 42 cells | same | DEVELOPMENT |
| P18 | **Alzheimer effect under honest (prefix) normaliser**, memory vs reset `auc_regret_norm` | 0.06070331006303109 | [0.04556081327092027, 0.08019782091887094] | 12 | `results/garrido_normaliser_audit_v3/result.json` | REPLAY / `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER` |
| P19 | same, memory vs ofat / vs random | 0.048209229424406676 / 0.08647353724461697 | [0.03325231524458697, 0.06319700028838643] / [0.0672894240322004, 0.1052117627365469] | 12 | same | REPLAY |
| P20 | **Neuron holds against lookahead**: lookahead_kg_transfer minus neuron_memory | 0.028149151305200626 | [0.012845676382162115, 0.04191154996295932] | 12 | `results/search_ladder_v5/result.json` | REPLAY / `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` |
| P21 | same vs thompson_transfer / gp_ei_transfer / ofat_transfer | 0.037045454786882905 / 0.03187208727513507 / 0.01071031949336522 | [0.02038444229207778, 0.05185218017696919] / [0.019229943011605136, 0.042145755333430956] / [3.564844184833131e-05, 0.02171366105332719] | 12 | same | REPLAY |
| P22 | **KAN searches worse than a matched MLP** (Δ`auc_regret_norm`, KAN − MLP; params 532 vs 529) | 0.01036905263158129 | [0.0030182183544574972, 0.018926078870915616], p_two_sided 0.0012 | 12 | `results/surrogate_architecture_bakeoff/result.json` | REPLAY / `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP` |
| P23 | neuron_5p beats every surrogate arm (all six `vs_reference` LCBs > 0, Holm-adjusted p = 0.0), best = 0.05203274760040456 mean AUC regret, 99.49913906574433 % of ceiling | e.g. vs gp_matern 0.061761351392021324 | [0.039200591454080495, 0.08288424578433401] | 12 | same | REPLAY |

Caveat on P9: the corrective artifact reframes the same numbers as a **one-bit calendar choice** — the belief policy emitted only 2 distinct action sequences across 120 tapes (`AAAA` 31, `ABAB` 89), both members of the periodic-calendar comparator set, and `best_static_calendar = ["A","B","A","B"]`. Cite it as "a one-bit per-tape choice between two fixed calendars captures 95.05 % of the perfect-information ceiling", not as "adaptive beats static".

Caveat on P20/P21: the headline is *not* that neuron_memory is best. `ucb1_transfer` beats it (mean AUC regret 0.04502300698358852 vs 0.05203274760040456), and that contrast **crosses zero**: −0.007009740616816028 [−0.024441318883473704, 0.014076439977282206], n=12. `ranking_best_first` puts `ucb1_transfer` first, `neuron_memory` second.

---

## 3. NEGATIVE / boundary results (measured null or stop)

| Result | Effect | Interval | n | Artifact | Grade / status |
|---|---|---|---|---|---|
| **H3 (original construct) not supported** — hybrid vs reset, variance of service-loss AUC across intensities | −1109446832360510.6 | [−3648590718211967.0, 1292965823805314.2], p_one_sided 0.8208, Holm 1.0 | 360 cells | `results/manuscript/h1_h3_originales_v3/result.json` | DEVELOPMENT |
| H3 hybrid vs static | −104973404727988.98 | [−3299057815968604.5, 2939228658503357.5], p_one_sided 0.5248, Holm 1.0 | 360 cells | same | DEVELOPMENT |
| H3 (v2.1 construct) memory vs ofat | 18.291666666666668 | [−5.334236111111105, 41.27805555555555] | 72 cells | `results/manuscript/h1_h3_v2_1/result.json` | DEVELOPMENT |
| H3 (v2.1) memory vs reset | 7.858333333333331 | [−14.842291666666663, 28.80916666666665] | 72 cells | same | DEVELOPMENT |
| **No neural premium on a curved buffer surface** — KAN − linear held-out R² | 0.03371141515795817 | [−0.07873510733477114, 0.14615793765068746], sd 0.12828472863632748, SESOI 0.05, `passes_sesoi_and_ci: false` | 5 folds / 10 seeds / 1530 rows | `results/headroom/buffer_prediction_premium/result.json` | DEVELOPMENT / `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE` |
| same, backprop − linear | −0.12778019423326165 | [−0.31567766641387396, 0.06011727794735067], sd 0.21436302071235958 | 5 folds | same (R² means: kan 0.716317881832606, linear 0.682606466674648, backprop 0.5548262724413864, constant −0.0033610961798922643) | DEVELOPMENT |
| **NO_GO neural premium in the wrap panel (Q1)** — B1 kan − linear: CI excludes zero but sits under SESOI 0.05 → boundary, not a win | 0.02157967180887328 | [0.0041640998372037795, 0.03899524378054278], `passes_sesoi_and_ci: false` | 5 folds | `results/garrido_wrap_q1/result.json` | UNCONTRACTED / `DEVELOPMENT_FIG5_SURROGATE`, `decision: NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`, `eligible_neural_models: []` |
| same, B1 backprop − linear | 0.016566945076440742 | [0.0048352801397191485, 0.028298610013162334], under SESOI | 5 folds | same | UNCONTRACTED |
| same, B2 activation kan − linear | 0.06000000000000003 | [−0.016566172558918603, 0.13656617255891867] | 5 folds | same | UNCONTRACTED |
| **No structured controller converts** — `winners: {R1r: [], R2r: []}`; DDMRP R1r | −0.0003029240528163556 | [−0.00034234897043016653, −0.00026613035568597736], 0/12 tapes positive | 12 tapes | `results/step3_split_pooled/result.json` | DEVELOPMENT / `NO_STRUCTURED_CONTROLLER_CONVERTS` |
| same, DDMRP R2r (degenerate) | 0.0 | [0.0, 0.0] | 12 | same | DEVELOPMENT |
| same, replay-MPC R1r / R2r | −2.1383599774512426e-05 / −0.0009908485520556398 | [−4.635861152125526e-05, 4.31420167334875e-06] / [−0.0037005969730673867, 0.0006825842011338549] | 12 each | same | DEVELOPMENT |
| **Run in flight, no result yet**: `results/dmlpa_kan_latent/result.json` | — | — | — | directory does not exist | see §5 |

---

## 4. DIAGNOSTIC findings (instrument properties, not effects)

| Finding | Number | Artifact | Grade |
|---|---|---|---|
| **Determinism repaired** — root cause: `MFSCGymEnvShifts.reset` passed `seed=None` to `MFSCSimulation` on unseeded resets. Two full PPO runs (seed 9492, 8 envs, 20000 steps) now land identically: replicas [93.973236562416, 93.973236562416], delta 0.0. Pre-fix fixed-seed spread 2.363, across-seed spread 2.102. | delta = 0.0 | `results/determinism_repair_control/result.json` | **DIAGNOSTIC** / `DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN`; supersedes `determinism_diagnostic` (`DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT`). Its own text: "still not CONFIRMABLE: no virgin seed block remains." |
| **Garrido Fig. 5 as drawn is an identity, not a learning task** — ReT is exactly the sum of driver contributions; three of five driver columns are identically zero. | R² = 1.0, max_abs_identity_error 3.219646771412954e-15; identified coefficients Re_FRt 0.9999999999999998, Re_RPj 0.9999999999999677 | `results/garrido_fig5_surrogate/result.json` and `results/garrido_wrap_q1/result.json` (both `DEVELOPMENT_FIG5_SURROGATE`, UNCONTRACTED) | UNCONTRACTED |
| Q1 panel surface is near-linear (why the buffer-premium re-ask was needed): linear R² 0.9697483885147611 vs backprop 0.9863153335912018, kan 0.9913280603236345 | — | same two | UNCONTRACTED |
| Curvature premise for the buffer lane was verified, not assumed | mean(1 − linear R²) = 0.07625893719919702, per-cell range 0.02388177309515338 (R2r\|base) to 0.13428686933720857 (R1r\|freq_x5) | `results/headroom/buffer_prediction_premium/result.json` | DEVELOPMENT |
| **Oracle normaliser leaks; prefix does not** — twin-surface falsifier: under `oracle`, both neuron arms' paths change in all 6 contexts (`all_paths_unchanged: false`); under `prefix`, `all_paths_unchanged: true`. Censoring of `runs_to_within_1pct` is arm-dependent (memory 0.0556, reset 0.1528, ofat 0.2222, random 0.6111), which is why `auc_regret_norm` is primary. | — | `results/garrido_normaliser_audit_v3/result.json` | REPLAY |
| **GSA estimator can return a negative** — obs minus perfect-information oracle | −0.0006393605748553234 [−0.001447611268211855, −7.048799550112284e-05], n=120 | `results/gsa_confirmation_corrective/result.json` (f3) | DEVELOPMENT |
| **Step-3 DDMRP arm is degenerate** — projected DDMRP emits exactly one posture (1344, 1344, 504); f6 fails, so the artifact supports no DDMRP claim in either direction. Also: preregistered guardrail `worst_product_fill` was never persisted; only `flow_fill_rate` was, "strictly weaker than the one preregistered". | `n_distinct_postures: 1`; `falsifiers.all_passed: false` | `results/step3_split_pooled/result.json` | DEVELOPMENT |
| Step-3 perfect-information ceiling arm (excluded from verdict by f5) does beat the incumbent: R1r 2.3737781128803516e-05 [4.3232264695715615e-06, 4.92942600925825e-05]; R2r 0.0010016228531676784 [8.827830742864223e-05, 0.002202324971160838] | — | same | DEVELOPMENT |
| **Grid transfer: only UCB1 transfers.** `transfers: {gp: false, neuron: false, ofat: false, ucb1: true}`. GP/OFAT/neuron all lose to their own marginal replay (neuron −0.011782983814352867 [−0.01848914143725492, −0.004835955987826106]). Grid 288 base → 4608 extended configs, budget 24. | — | `results/grid_transfer_confirmation_v2/result.json` | CONFIRMATORY |
| **H1 endpoint had to be redefined**: `restricted_ttr = min(TTR, τ)`, τ=1344.0 h, because under the recurrent R11–R24 regime at 52 weeks events merge into one cluster with no return-to-normal — "a property of Garrido's risk regime, not of the instrument". `system_ttr` returned 1.000. | absorbed fraction 0.8517361111111111, censored-at-τ 0.071875 | `results/manuscript/h1_h3_originales_v3/result.json` | DEVELOPMENT |
| **GSA seed block was repurposed, not fresh**: block `g3a_v2_development` 7700001–7700120, `repurposed_from: G3a asymmetric-claimant development`, authorised by `docs/AUTORIZACION_PI_REPROPOSITO_BLOQUE_7700001_2026-08-07.md`. | — | `results/gsa_confirmation/result.json` | CONFIRMATORY |
| **Seed custody is declared incomplete** — grid transfer f4 evidence: `registry_status: BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `registry_is_complete: false`. Verdict is NO_KNOWN_COLLISION, "not a proof of virginity". | — | `results/grid_transfer_confirmation_v2/result.json` | CONFIRMATORY |

---

## 5. `results/dmlpa_kan_latent/result.json` — state

**Does not exist. The run is in flight right now.** `results/dmlpa_kan_latent/` is not present under `results/`. But `ps` shows a live parent process plus 8 workers:

```
pid 34427  .venv/bin/python scripts/run_dmlpa_kan_latent_v1.py --contract docs/PREREGISTRO_DMLPA_KAN_LATENT_2026-08-07.md
```

started 6:13 PM (108+ min CPU at 171 %), with 8 child processes started 7:44 PM. Only the runner and the preregistration exist on disk; the closest sealed artifact is the unrelated `results/dmlpa_variants/result.json` (7 Aug 06:50).

Its preregistered reading rule, from `docs/PREREGISTRO_DMLPA_KAN_LATENT_2026-08-07.md`: primary is mean ReT, seed-paired contrast `kan − mlp`, bootstrap over seeds; LCB95 > 0 → `KAN_LATENT_HELPS`, UCB95 < 0 → `KAN_LATENT_HURTS`, CI crossing zero → `KAN_LATENT_INDISTINGUISHABLE`. Design: 100000 steps, seeds 9491–9495 (already open, zero new seeds), 24 eval episodes, 8 envs, CPU. Parameter-matched arms: `dmlpa_mlp` hidden_dim 152 → 200052 params (0.03 % deviation) vs `dmlpa_kan` hidden_dim 10 → 199082 params (0.46 %). Scope is declared development — it adjudicates nothing in the manuscript and authorises no confirmation, since no virgin blocks remain.

One citable pre-training fact is already fixed by the design: at matched parameter budget the KAN only affords `hidden_dim = 10` against the MLP's 152 — roughly fifteen times narrower.

---

## 6. What a paper can actually lead with

- **Strongest, cleanest**: the six-panel H2/H3 resource confirmation (P1–P6) — CONFIRMATORY, 12/12 positive tapes per panel, all six Holm gates passed, 1080 rows, code commit `9829084de18d0e1bf57d0da31d54beca56dd6997`, freeze receipt `352a4dcaa4635c4aeeafa582c783588af2e156a58ee73f3c45ec75f7ebae1a0f`. Its own declared boundary: "does not establish learner, feedback, or architectural value."
- **Second**: UCB1 grid transfer (P7–P8) — the only arm of four that beats its state-blind marginal replay, on a reserved block, with no RL and no neural learner in the loop.
- **The manuscript trio**: H1 supported (P11–P12), H2 supported (P13), H3 supported only as the reformulated H3′ (P14–P15); the original H3 construct is a clean measured null.
- **The honest negatives are strong material in their own right**: no neural premium even where curvature was measured, Fig. 5 as drawn is an algebraic identity, KAN searches measurably worse than a parameter-matched MLP, and no structured controller converts.
