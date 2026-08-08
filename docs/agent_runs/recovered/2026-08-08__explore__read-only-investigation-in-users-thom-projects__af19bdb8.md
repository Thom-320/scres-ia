# Explore — Read-only investigation in /Users/thom/Projects/research/scres-ia. The machine is busy wit

- **agent_id**: `af19bdb8cdb10991e`
- **session**: `2c6b3a01-62a0-499d-957d-694424855dff`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T03:56:50.880Z`
- **effort**: `xhigh` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/2c6b3a01-62a0-499d-957d-694424855dff/subagents/agent-af19bdb8cdb10991e.jsonl`
- **turns**: 114 · **assistant messages**: 2

## Task given

```text
Read-only investigation in /Users/thom/Projects/research/scres-ia. The machine is busy with a long verification job — do NOT run pytest or anything heavy. Read artifacts and docs only.

There is a v.0 draft of a paper with a research question and four hypotheses. I need to know, for EACH, exactly what sealed evidence answers it, at what evidence grade, with the exact numbers, and what is NOT answered.

THE v.0 RESEARCH QUESTION:
"How can the integration of neural network–based learning mechanisms into simulation models enhance the dynamic operationalization and predictive accuracy of Supply Chain Resilience (SCRES) under recurring and heterogeneous disruption scenarios?"

THE v.0 FORMAL CLAIM: traditional simulation assumes R_t = f(S_t, D_t); the contribution is introducing L_{t-1} (accumulated learned knowledge) as an endogenous state variable, so R_t = f(S_t, D_t, L_{t-1}).

THE FOUR HYPOTHESES:
- H1 Learning Effect: hybrid simulation–neural models achieve significantly SHORTER RECOVERY TIMES than static simulation models.
- H2 Adaptation: neural-enabled SCRES models improve under SUCCESSIVE disruptions (learning-curve effect).
- H3 Volatility Reduction: learning-enabled models reduce performance VARIANCE across heterogeneous disruption intensities.
- H4 Path Dependency: resilience at time t is positively influenced by accumulated learning from prior disruptions.

FOR EACH of RQ, the L_{t-1} formal claim, and H1..H4, report:
1. Which sealed artifact(s) adjudicate it. Give exact paths.
2. claim_status, scope string, evidence grade, seed block, n, endpoint.
3. The exact numbers with intervals.
4. Whether the estimand actually tested is the one the hypothesis states, or a REDEFINED one — and if redefined, when the redefinition was written relative to the run (this matters enormously: an estimand written after seeing data reads as endpoint shopping).
5. Which falsifiers passed and which failed.
6. The honest verdict: supported / not supported / supported-but-different-estimand / unanswerable.

Then a separate critical question: the v.0 draft's whole framing is that NEURAL learning is the mechanism. Our evidence says the effective carrier is NOT neural. Enumerate every sealed artifact that bears on "is the carrier neural", with its numbers, and state how much of the v.0 framing survives contact with that evidence.

Also check: docs/ has files like ENMIENDA_* and TABLA_CANONICA_* that adjudicate these. Read them and report what they already say about H1-H4, including any RETRACTED figures or forbidden wordings tied to these hypotheses.

Be exhaustive and exact. Quote claim_status strings verbatim.
```

## Final answer

## Where things live

- v.0 draft text: `/Users/thom/Projects/research/scres-ia/tmp/v0_draft.txt` (RQ at l.35–38, L_{t−1} claim l.68–77, H1–H4 at l.78–90); DOCX renders in `/Users/thom/Projects/research/scres-ia/deliverables/v0_neuralNet-scres_DES_section_updated.docx`
- Adjudicating docs: `/Users/thom/Projects/research/scres-ia/docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, `…_ENMIENDA_1.md`, `/Users/thom/Projects/research/scres-ia/docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` + `_ENMIENDA_1..4.md`, `/Users/thom/Projects/research/scres-ia/docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md`, `/Users/thom/Projects/research/scres-ia/docs/TABLA_CANONICA_GENERADA.md`
- Machine-derived evidence grades: `/Users/thom/Projects/research/scres-ia/research/evidence_registry.jsonl` (216 artifacts, 211 distinct experiments; grade is **derived**, never copied from `claim_status`). Repo-wide: `CONFIRMATORY` n=3, `DEVELOPMENT` 87, `UNCONTRACTED` 57, `REPLAY` 39, `NEGATIVE_OR_HALTED` 20, `DIAGNOSTIC` 5. **Virgin seed blocks remaining: 0.**

---

# 1. The RQ

> "How can the integration of neural network–based learning mechanisms into simulation models enhance the **dynamic operationalization** and **predictive accuracy** of SCRES under recurring and heterogeneous disruption scenarios?"

The RQ decomposes into two measurable halves, and **both are answered negatively for the neural mechanism**.

### 1a. "Dynamic operationalization" = does knowing the regime buy a different posture?

**Artifacts:**
- `/Users/thom/Projects/research/scres-ia/results/surface_gates/result.json` — `claim_status`: **`"NON_SEPARABLE_BUT_CONTEXT_INVARIANT"`**; scope `"DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER"`; `run_role: CACHE_ANALYSIS`; grade **REPLAY**; seeds 5300001–5300012 (n=12, declared replay of `garrido_q2_des288`).
- `/Users/thom/Projects/research/scres-ia/results/surface_gates_v2/result.json` — same status, 5000 bootstrap reps.
- `/Users/thom/Projects/research/scres-ia/results/endpoint_headroom_atlas/result.json` — `claim_status`: **`"NO_ENDPOINT_CARRIES_REGIME_HEADROOM"`**; grade REPLAY.
- `/Users/thom/Projects/research/scres-ia/results/garrido_v0_surface_gates_v1/result.json` — `claim_status`: **`"STOP_NO_RECOVERY_LEARNING_HEADROOM"`**; grade **NEGATIVE_OR_HALTED**; seeds 5300001–006.
- `/Users/thom/Projects/research/scres-ia/results/garrido_v0_surface_gates_holdout/result.json` — `claim_status`: **`"OPENED_AGAINST_PREREGISTRATION_STOP_NO_RECOVERY_LEARNING_HEADROOM"`**; grade NEGATIVE_OR_HALTED; seeds 5300007–012.

**Exact numbers:**
- `H_regime = 0.003802243800697269`, LCB95 `1.0824674490095292e-16` (n_boot 1000) / `1.3519330929717688e-05` (n_boot 5000), UCB95 `0.014413327648688967` / `0.014856419878462748`, against a **preregistered gate of 0.05** → **gate fails**.
- Atlas over 7 endpoints × 2 caches: `material_endpoints = []` for both. Best endpoint is `ret_excel_risk_conditional` at H_regime `0.02829448338047058` [`0.016181349111474213`, `0.043337093026439164`] — still under 0.05. `flow_fill_rate`, `lost_orders`, `delivered_rations`, `demanded_rations` all at H_regime `0.0`.
- v0 recovery surface gates: **G2 passes in 1 of 6 contexts (4 required)** — only R24 (dev mean `+0.24692032069726086` [`+0.16751470880347807`, `+0.320518494384289`]; holdout `+0.22783410791531603` [`+0.12743229808021625`, `+0.31349863790427596`]). **G3 fails at exact zero:** `ttr_gain_hours` mean `0.0`, LCB95 `0.0`, UCB95 `0.0`, n=6, threshold `minimum_lcb95_hours = 24.0`. The same posture `[0, 672, 168]` is selected in all six contexts and all six folds, in both development and holdout.

**Falsifiers:** surface_gates `all_passed: false` (the `f_no_fresh_seeds` is `not_applicable`/declared replay; the *gate* `g1_context_adaptation_is_worth_something` fails). v0 gates: f1–f4 pass in development; in the holdout `f1_only_development_seeds_are_loaded` is **`passed: false`** by construction, and `all_passed: false` — that is the point of the `OPENED_AGAINST_PREREGISTRATION_` prefix.

**Estimand redefinition / timing:** the holdout opening is documented in `/Users/thom/Projects/research/scres-ia/docs/APERTURA_HOLDOUT_V0_CONTRA_EL_PREREGISTRO_2026-08-07.md`, marked **"Escrito ANTES de correr"**, and states explicitly that the opening violates the preregistration's own stop rule twice, that the block `5300007–012` is burned, and that **"si el holdout contradice el STOP … NO es un positivo citable."** The block is registered as `BURNED_OPENED_AGAINST_PREREGISTRATION`. This is the cleanest possible handling of a rule violation, but it does mean the holdout adds no authority.

**Verdict: NOT SUPPORTED for the neural mechanism, and largely unanswerable as posed.** There is no measurable regime-adaptive headroom on any endpoint in this chain, so "dynamic operationalization" has almost nothing to operationalize. What *is* supported is a strictly weaker and different thing: retained **search** state finds the (invariant) optimum faster.

### 1b. "Predictive accuracy"

**Artifacts:**
- `/Users/thom/Projects/research/scres-ia/results/garrido_wrap_q1/result.json` — `claim_status`: **`"DEVELOPMENT_FIG5_SURROGATE"`**, `q1_decision.decision`: **`"NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL"`**; grade **UNCONTRACTED** (`contract_path: null` — "no contract hash: nothing fixes what this run was allowed to claim"). n=90 configurations, 5 grouped folds, seed 20260731.
- `/Users/thom/Projects/research/scres-ia/results/garrido_fig5_surrogate/result.json` — same numbers, same grade UNCONTRACTED.
- `/Users/thom/Projects/research/scres-ia/results/headroom/buffer_prediction_premium/result.json` — `claim_status`: **`"NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE"`**; grade **DEVELOPMENT**; seeds 6800001–6800010 (virgin, `f6` scanned 350 prior seeds, zero collisions); n_rows 1530.

**Exact numbers (near-linear panel, B1 held-out R²):** linear `0.9697483885147611`, backprop `0.9863153335912018`, KAN `0.9913280603236345`. Paired vs linear: backprop `+0.016566945076440742` [`+0.0048352801397191485`, `+0.028298610013162334`]; KAN `+0.02157967180887328` [`+0.0041640998372037795`, `+0.03899524378054278`]. **SESOI `0.05` fixed in advance; `passes_sesoi_and_ci: false` for both.** B2 activation accuracy: linear `0.7111111111111111`, backprop `0.7177777777777778` (`+0.006666666666666665` [`−0.1392088748907709`, `+0.15254220822410422`]), KAN `0.7711111111111111` (`+0.06000000000000003` [`−0.016566172558918603`, `+0.13656617255891867`]) — **CI crosses zero for both**; `eligible_neural_models: []`.

**Exact numbers (curved panel):** constant `−0.0033610961798922643`, linear `0.682606466674648`, **backprop `0.5548262724413864` (worse than a straight line)**, KAN `0.716317881832606`. Paired: backprop `−0.12778019423326165` [`−0.31567766641387396`, `+0.06011727794735067`]; KAN `+0.03371141515795817` [`−0.07873510733477114`, `+0.14615793765068746`]. Both `passes_sesoi_and_ci: false`. Curvature `0.07625893719919702` (stored field `profile_curvature.mean_one_minus_linear_r2`); noise `0.317394` = **derived** as `1 − 0.682606466674648`.

**Falsifiers:** all six pass in `buffer_prediction_premium` (`f5_sesoi_was_fixed_in_advance` explicitly: "choosing the threshold after seeing the difference is how a negligible gain becomes a headline"). All five pass in `garrido_wrap_q1`/`fig5_surrogate`.

**Verdict: NOT SUPPORTED.** No neural predictive premium at any preregistered SESOI, on either a near-linear or a measurably curved surface. `TABLA_CANONICA` §5 requires the noise figure always be cited with its derivation.

---

# 2. The L_{t−1} formal claim (`R_t = f(S_t, D_t, L_{t−1})`)

**Adjudicating artifact:** `/Users/thom/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json`

- `claim_status`: **`"GRID_TRANSFER_CONFIRMED__UCB1"`**
- `scope`: **`"CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER"`**
- `run_role`: `"CONFIRMATION"`; **evidence grade `CONFIRMATORY`** (one of exactly 3 in the whole repo)
- Seed block `garrido_grid_transfer_v2_confirmation`, seeds **8200001–8200060**, n=60
- Endpoint: AUC of normalised regret, budget 24, 288 → **4608** configurations, 6 contexts
- `transfers = {"gp": false, "neuron": false, "ofat": false, "ucb1": true}`

**Exact numbers (positive = transfer helps):**

| family | vs cold start | **vs state-blind marginal replay** |
|---|---|---|
| `ucb1` | +0.05743819277092614 [+0.049888584439231054, +0.06480569983808135] | **+0.03073311127302739 [+0.019896866431745156, +0.042561504429982196]** ✅ |
| `neuron` | +0.054393629143446906 [+0.04289557823009438, +0.06687098572184685] | **−0.011782983814352867 [−0.01848914143725492, −0.004835955987826106]** ❌ |
| `gp` | +0.01432635783389502 [+0.008787735338250295, +0.02056083569699176] | −0.021595031378833847 [−0.03050640445741258, −0.012267301800787132] ❌ |
| `ofat` | +0.014218869453841312 [+0.008002602681115948, +0.019318535198958688] | −0.024672161316045497 [−0.032580438401241286, −0.01666244793292766] ❌ |

**Falsifiers:** `all_passed: true` — f1 (103,680 cells reproduced against the 288-cache, `max_abs_delta: 0.0`, 0 mismatches), f2, **f3_transfer_beats_its_marginal_replay** (the falsifier the neuron fails on numbers while the falsifier itself "passes" as a test), f4_seed_custody (`NO_KNOWN_COLLISION`, registry declares itself incomplete), f_budgets_are_matched, f_source_manifest_is_identical.

**Estimand:** exactly the one the claim needs, and the falsifier separates "I transferred surface structure" from "I revisited configurations that already worked." Written in `/Users/thom/Projects/research/scres-ia/docs/ENMIENDA_CONFIRMACION_BLOQUE_LIMPIO_2026-08-05.md` (contract sha `1abbb2df…`), **before** the run of 2026-08-06T09:39.

**Verdict: SUPPORTED-BUT-DIFFERENT-CARRIER.** `L_{t−1}` as an endogenous state variable is confirmed prospectively — but `L` is a **factor-level bandit statistic**, not a neural weight vector. The neural `L` beats cold start and **fails** the demanding counterfactual with the entire interval on the wrong side. `L` also lives in the **outer loop between DES runs**, not inside the event kernel — `GARRIDO_Q1_Q2_CLAIM_FREEZE` §2 lists `"embedded inside the DES transition kernel"` and `"closed-loop adaptive control within the episode"` as **forbidden phrasings**.

---

# 3. H1 — Learning Effect (shorter recovery times)

**Artifact:** `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_originales_v3/result.json`

- `claim_status`: **`"H1_SUPPORTED__H3_NOT_SUPPORTED"`**
- `scope`: **`"DEVELOPMENT_ALREADY_OPEN_BLOCK_NO_VIRGIN_SEEDS_NO_ADJUDICATION"`**
- Evidence grade: **`DEVELOPMENT`** ("contracted, but not a confirmation over a virgin block")
- Preregistration `/Users/thom/Projects/research/scres-ia/docs/PREREGISTRO_H1_H3_ORIGINALES_V3_2026-08-07.md`, contract sha `24a1fe62fcaab9ad547bd85aa702cd60982168fc1804b40659957ff9c7fc4d0d`; `self_sha256` `dc46ce6069755a28929b5b107196232a399dd374820b31dc1b186c70e6ef807e`
- Seed block **6000001–6000120** (n=120 replicates), already open; cell = (seed × shock context), **n_cells = 960 per arm**
- Endpoint: `restricted_ttr_hours = min(TTR, τ)`, τ = 1344 h (8 weeks), horizon 6048 h, with a **paired shock-free placebo**
- Arms: hybrid = `neuron_memory`, reset = `neuron_reset`, static = `ofat`

**Exact numbers:**
- Levels: hybrid **75.7 h**, reset **149.74583333333334 h**, static **201.68541666666667 h**
- `H1_hybrid_vs_static`: mean **+125.98541666666667 h**, LCB95 **+98.346796875**, UCB95 **+154.54471354166665**, `p_one_sided = 0.0`, Holm `p_holm = 0.0`, rejected at .05
- `H1_hybrid_vs_reset`: mean **+74.04583333333333 h** [**+51.523828125**, **+97.46838541666664**], Holm `p = 0.0`
- Secondary descriptive, on the 756 differing cells only: **+159.9814814814815** [+123.99262566137567, +195.61435185185184]
- **Mechanism is absorption, not restoration speed:** absorbed hybrid **875/960**, reset 823/960, static **755/960**; censored at τ 37 / 75 / 95; recovered 923 / 885 / 865

**Falsifiers: all seven pass** (`all_passed: true`) — f1 differing configs (756/960 = 0.7875 vs 0.30 threshold), f2 endpoint has range, f3 placebo shock-free (426 placebo episodes), **f4 sealed surface still reproduces (`max_abs_delta: 0.0` exact on 24 sampled cells, tolerance 1e-12 — this closes gap A2, the `supply_chain.py` drift)**, f5 intensity ladder escalates, f6 variance axis, f7 no new seeds.

### Estimand: REDEFINED, and the redefinition timing is clean
The artifact carries a dedicated field, verbatim:

> `"endpoint_redefinition_declared": "H1 uses restricted_ttr = min(TTR, tau) with a paired placebo, not system_ttr. It is a different estimand, written 2026-08-06 for the v0 lane and before this preregistration, not a loosened version of the one that returned 1.000."`

Corroboration of the timing:
- The prior attempt `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_v1/result.json` is **`"HALTED_FALSIFIER_FAILED"`** (grade NEGATIVE_OR_HALTED): `f1` failed (hybrid and static deployed the identical config) **and** `f3` failed with `censored_fraction_by_arm = {hybrid: 1.0, reset: 1.0, static: 1.0}` — `system_ttr` gave `0.0 ± [0.0, 0.0]` by vacuity.
- `restricted_ttr` was built in `supply_chain/garrido_v0_recovery.py` for the v0 lane, preregistered in `/Users/thom/Projects/research/scres-ia/docs/PREREGISTRO_GARRIDO_V0_RECOVERY_GATE_2026-08-06.md` (file mtime 2026-08-06 20:47, committed `ab9d5f5` 2026-08-06 20:50:59).
- The H1 preregistration was committed `189c2c8` at **2026-08-07 10:57:00**; the run created at **2026-08-07T16:09:05Z**. So: instrument written 2026-08-06 for a different lane → preregistration 2026-08-07 10:57 → run 2026-08-07 16:09. **Preregistration precedes the run by ~5h; the instrument precedes the preregistration by ~14h.**
- The preregistration itself flags the exposure honestly (§1): the 2026-08-01 document listed "fix `system_ttr`" as the *last* of three options "precisely because changing an instrument after seeing it return 1.000 is suspicious", and then states: **"Aun así, es una redefinición del endpoint y el manuscrito la presenta como tal."**

**Two caveats you should carry:**
1. **Regime restriction, declared in the artifact:** `"Under the recurrent R11-R24 regime at 52 weeks the events merge into one cluster that never ends, so no return-to-normal exists to time. H1 is therefore measured under isolated shocks."` The v.0 RQ says *recurring* disruptions. **H1 is answered under isolated shocks only** — the opposite of the RQ's regime.
2. Minor discrepancy worth checking: the preregistration §1 cites "21 / 43 / 33 / 87 distinct configurations"; the sealed artifact reports `distinct_configurations_deployed = {hybrid: 20, reset: 37, static: 21}`. Different counting basis, but the numbers don't line up as written.

**Verdict: SUPPORTED-BUT-DIFFERENT-ESTIMAND (declared, pre-dated, and honestly labelled), at DEVELOPMENT grade, under isolated shocks, by absorption rather than by faster restoration.** Not confirmatory. Also see §7 — the "hybrid" arm here is a 5-parameter logistic unit.

**Reformulation H1′ (also supported):** `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_v2_1/result.json`, `claim_status` **`"H1_SUPPORTED__H3_NOT_SUPPORTED"`**, grade DEVELOPMENT, seeds 6200001–6200005 (virgin, f7 scanned 314 prior seeds), 12 repeats, 72 cells. Endpoint `service_loss_auc_ration_hours`. Primary all cells **+61,349,290.03** [**+14,432,891.77**, +121,804,658.43]; hybrid_vs_reset +33,547,084.65 [+3,893,251.25, +81,838,777.71]; on the 42 differing cells +105,170,211.49 [+28,008,967.28, +204,916,732.28]. Levels: hybrid 47,444,828.28 / reset 80,991,912.93 / static 108,794,118.32. All seven falsifiers pass. The artifact self-labels: `"what_it_is_not": "not a recovery TIME: it is the integral of lost service, which mixes magnitude with duration"`.

---

# 4. H2 — Adaptation / learning curve

**Artifact:** `/Users/thom/Projects/research/scres-ia/results/manuscript/h2_learning_curve/result.json`

- `claim_status`: **`"H2_SUPPORTED_LEARNING_CURVE"`**
- `scope`: **`"DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION"`**
- Evidence grade: **`DEVELOPMENT`**
- `self_sha256` `74b75141241ba7639f629e2ba8db0ef28bff480fe706489f8977dcd4d4bf48fa`; preregistration `/Users/thom/Projects/research/scres-ia/docs/PREREGISTRO_H2_CURVA_DE_APRENDIZAJE_2026-08-07.md`, contract sha `7741309cf93fa4e0a49995f0900b1062bce986eb292fc0dc7b66fbc782345cfb`
- Seed block **6000001–6000120**, n=120; zero seeds consumed (pure re-analysis); normaliser = **prefix**
- Endpoint: `auc_regret_norm`; estimand string verbatim: `"OLS slope of (reset AUC - memory AUC) against the context ordinal 1..6, per replicate. A large but FLAT advantage supports H4, not H2."`

**Exact numbers:**
- **Primary slope: mean `+0.04220147575193482`, LCB95 `+0.03466393985914079`, UCB95 `+0.04992205677530306`, n=120**
- Null control (random − OFAT, neither retains anything): mean `−0.0050880611149445275` [`−0.015570216666852871`, `+0.0056579986049085955`] — crosses zero, so the trend is **not** context-order difficulty
- Advantage by context (reset − memory): R1r `+0.0` (structurally exact — with nothing to carry, memory and reset are the same arm), R2r `+0.2827524501250933`, R1r+R2r `+0.1905162315685694`, R1r|esc `+0.22111385282764517`, R2r|esc `+0.31709366608535716`, R1r+R2r|esc `+0.26868607643557035`
- Arm means on this block: `neuron_memory` 0.17022357795416482, `neuron_reset` 0.38358395746120394, `ofat` 0.2619659335765739, `random` 0.4434164832167541

**Falsifiers: all five pass** — f1 source is the contracted pair at n=120 (90 local seeds 6000001–090 + 30 VPS 6000091–120, both sealed against `PREREGISTRO_H3_POTENCIA_2026-08-01`), **f2 normaliser is prefix not oracle** (this is the falsifier that prevents reintroducing the leak that retired the original H2 figure), **f3 order confound absorbed** (the one that could have blocked reading H2 at all), f4 slope can be negative (`min_slope_observed −0.0564365828492133`, `max 0.1477783164540918`, **22 of 120 replicates negative**), f5 no new seeds.

**Estimand: SHARPENED, and written before the run.** The preregistration is headed "Escrito y commiteado ANTES de mirar los datos", committed `093a38a` at **2026-08-07 13:39:38**; the run created **2026-08-07T18:39:39Z** — five hours later. It fixes the reading rule in advance with four named outcomes including `H2_NOT_SUPPORTED_ADVANTAGE_IS_FLAT`, and states: **"si sale plana no se reescribe como «la memoria ayuda en todos los contextos» — eso ya lo dice H4."** This is a slope-not-level estimand, which is strictly *stricter* than the draft's wording, not looser. **No endpoint shopping here.**

**Retracted predecessor figure (explicitly named):** the circulating H2 form "**ventaja +0,00 → +10,00 entre contextos**" is **RETIRED** — it came from the runner with the normaliser leak documented in `/Users/thom/Projects/research/scres-ia/docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` (a `f5_no_context_leakage` falsifier with `"passed": True` **hardcoded**; the searcher was reading driver vectors of *unrun* configurations). The zero at the first context survives as a structural check.

**Verdict: SUPPORTED, on the hypothesis's own estimand, with the order confound explicitly controlled — but at DEVELOPMENT grade on an already-open block, and on search-cost AUC, not on operational resilience.** Carrier caveat in §7 applies.

---

# 5. H3 — Volatility Reduction

**Artifact:** `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_originales_v3/result.json` (same run as H1)

- `claim_status`: **`"H1_SUPPORTED__H3_NOT_SUPPORTED"`**; same scope, grade DEVELOPMENT, same seed block 6000001–6000120
- Endpoint: **`service_loss_auc_ration_hours`** (preregistration §4 forbids `ret_excel` as primary because it "rewards abandonment"); cell = (seed × base context), variance ddof=1 across the four intensity rungs ×1…×4; **n_cells = 360 per arm**; horizon 8736 h

**Exact numbers:**
- Variance levels: hybrid **1.6107490760087546e+16**, reset **1.4998043927727036e+16**, static **1.6002517355359556e+16** → **the hybrid is the MOST volatile of the three**
- `H3_hybrid_vs_reset`: mean **−1.1094468323605106e+15** [**−3.648590718211967e+15**, +1.2929658238053142e+15], `p_one_sided = 0.8208`, Holm `p = 1.0`, not rejected
- `H3_hybrid_vs_static`: mean **−1.0497340472798898e+14** [**−3.2990578159686045e+15**, +2.9392286585033570e+15], `p_one_sided = 0.5248`, Holm `p = 1.0`, not rejected
- Secondary (reported with warning, never decisive): `ret_excel` variance hybrid 4.005122444264847e-06, reset 4.170787267301283e-06, static 4.348030563274829e-06 — points the other way, which is exactly why the preregistration barred it

**Falsifiers: all seven pass**, including `f5_the_intensity_ladder_escalates` (mean risk events 256.65 → 308.66 → 354.72 → 398.79 across ×1…×4) and `f6_variance_is_across_intensities_not_within` (across-intensities hybrid 1.6107e16 vs wrong-axis control 7.0836e32). **The instrument works; the effect is absent.** This is the key upgrade over the 2026-08-01 status.

**Estimand: the one the hypothesis states, unmodified.** No redefinition field on H3 in this artifact.

**Verdict: NOT SUPPORTED — with the sign on the wrong side and 78.75% of cells deploying different configurations.** `TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §6: *"ya no es «sin estimando», es «sin efecto»."* That is a meaningful strengthening of the negative: on 2026-08-01 H3 was `NO EVALUABLE`; it is now refuted with a live estimand.

**Reformulation H3′ (supported, different construct):** `/Users/thom/Projects/research/scres-ia/results/garrido_h3_merge_adjudication/result.json`, `claim_status` **`"H3_PRIME_SUSTAINED_AT_N120"`**, grade DEVELOPMENT, seeds 6000001–6000120 (n=120), contract `/Users/thom/Projects/research/scres-ia/docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md`, `reading_rule: "LCB95 > 0 -> sustained; LCB95 <= 0 at n=120 -> refuted with power"`. Estimand verbatim: `"variance of search cost ACROSS CONTEXTS, per replicate; memory minus reset."` Numbers: memory_vs_ofat **+16.220277777777778** [**+9.61070833333333**, +22.740534722222215]; memory_vs_reset **+9.314444444444444** [**+2.3491250000000012**, +16.34740972222222]. Variance by strategy: neuron_memory 44.264166666666675, neuron_reset 53.57861111111111, ofat 60.48444444444444, random 67.50333333333333. All four merge falsifiers pass. **Per-slice caveat you should know:** local_90 +10.2659 [+2.3609, +18.1927] but **vps_30 +6.46 [−7.3088, +20.3281] crosses zero** — the n=120 merge is what carries it.

The predecessor `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_v2_1/result.json` states the redefinition in the artifact itself: `"the manuscript's H3 is variance of performance across disruption intensities, which is untestable here because the optimum does not move; this is a different construct, not a repair"`. **H3′ is a different construct, and it is labelled as one.**

---

# 6. H4 — Path Dependency (`L_{t−1}`)

**Primary artifact:** `/Users/thom/Projects/research/scres-ia/results/garrido_normaliser_audit_v3/result.json`

- `claim_status`: **`"ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER"`**
- `scope`: **`"DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER"`**; `run_role: "BURNED_REPLAY_AUDIT"`; `replay_of: "garrido_q2_des288"`
- Evidence grade: **`REPLAY`**
- Seeds **5300001–5300012**, n=12 replicates, 6 contexts, budget 24, 288 configurations
- Primary endpoint `auc_regret_norm` under the **prefix** normaliser; `primary_rationale` verbatim: `"runs_to_within_1pct imputes budget+1 and is censored at very different rates per arm, so its mean is not comparable across arms."`

**Exact numbers (prefix, canonical):**
- Arm means: `neuron_memory` **0.05203274760040456**, `neuron_reset` **0.11273605766343564**, `ofat` **0.10024197702481125**, `random` **0.1385062848450215**
- **memory_vs_reset: +0.06070331006303109 [+0.04556081327092027, +0.08019782091887094], n=12** ← the canonical H4 number
- memory_vs_ofat: +0.048209229424406676 [+0.03325231524458697, +0.06319700028838643]
- memory_vs_random: +0.08647353724461697 [+0.0672894240322004, +0.1052117627365469]
- Secondary, **always labelled censored**: `runs_to_within_1pct` prefix memory_vs_reset **+5.833333333333332 [+4.444444444444445, +7.3055555555555545]**; memory_vs_ofat +5.333333333333333 [+3.25, +7.125]. Prefix means 7.083333333333333 / 12.91666666666667 / 12.416666666666666 / 19.541666666666668. Censoring rates 0.0556 / 0.1528 / 0.2222 / 0.6111.

**Falsifiers: all pass** — f1 harness reproduces the sealed `garrido_meta_learner_v2` (`abs_delta = 0.0` on all four arms), f2 visit sequences scale-invariant, f3 non-neural arms untouched, f4 censoring reported, f5 no fresh seeds (declared replay), **f6 surface twins do not change prefix paths** (`prefix.all_paths_unchanged: true`; **`oracle.all_paths_unchanged: false`** — the oracle normaliser's path changes for both neuron arms in all six contexts when hidden tails move, which is the leak).

### The v.0 draft's H4 figure is FORBIDDEN
`TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md` §2, verbatim heading: **"`H4` — la cifra del borrador está prohibida."**

- The draft cites **`+7,90 corridas [+6,88, +8,93]`**. That is `memory_vs_reset` on `runs_to_within_1pct` under the **oracle** normaliser. The artifact's oracle panel: mean **7.902777777777778** [**6.8194444444444455**, **8.930555555555555**].
- **Mandatory substitution:** primary **AUC +0.06070 [+0.04556, +0.08020]**; secondary **5.83 runs [+4.44, +7.31]**, always labelled censored.
- The amendment notes it is easy to miss "porque el borrador cita el número **sin nombrar su normalizador**."
- Also on the retired list (`TABLA_CANONICA` §1 and `CLAIM_FREEZE` §5): **`7,24 / 13,54 / 12,42`** and **`7,90 / 5,43`**. The oracle panel `6,99 / 14,89 / 12,42` may be cited **only** as sensitivity and **always** with that label. The prefix panel is `7,08 / 12,92 / 12,42`.

**Estimand:** the one H4 states — memory retained vs memory reset, with everything else held identical. This is the *one* place in the project where `L_{t−1}` is isolated cleanly (`docs/ESTADO_CONTRA_GARRIDO_2026-07-31.md`: "es el único sitio del proyecto donde `L_{t−1}` se aísla"). The `CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` leak retired the memory-vs-OFAT and memory-vs-random numbers but explicitly preserved the H4 estimand: *"el contraste que la fuga no distingue"* — both arms read the same leaked information, so the leak cancels in the paired contrast.

**Verdict: SUPPORTED as measured, at REPLAY grade on burned tapes, on a search-cost endpoint — but the number in the draft must be replaced.** `TABLA_CANONICA_ENMIENDA_1` §1 grades it as "**medida** +0,06070 [+0,04556]", deliberately not "SOSTENIDA", because it is a replay, not an adjudication.

**Related but distinct — do not conflate:** `/Users/thom/Projects/research/scres-ia/docs/H4_RETAINED_VS_RESET_VERDICT_2026-07-02.md` is **Paper 2's** H4 (Track B RL retained-vs-reset, seeds 8101–8110, obs_hidden +0.0000493 [+0.0000167, +0.0000819]). Same letter, different paper, different estimand, third-decimal-order effect. It is not the v.0 H4.

---

# 7. The critical question: is the carrier neural?

**Short answer: no.** The v.0 framing — "integration of *neural network–based* learning mechanisms" — is not what the evidence carries. Here is every sealed artifact that bears on it.

### 7.1 The single confirmatory test says the carrier is a bandit, not a net
`/Users/thom/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json` — `GRID_TRANSFER_CONFIRMED__UCB1`, CONFIRMATORY, virgin block 8200001–8200060, n=60. `neuron` vs its own state-blind marginal replay: **−0.011782983814352867 [−0.01848914143725492, −0.004835955987826106]** — the *entire* interval on the wrong side. `ucb1` **+0.03073311127302739 [+0.019896866431745156, +0.042561504429982196]**. `transfers.neuron = false`.

### 7.2 The neuron ties with, or loses to, classical methods with matched memory
`/Users/thom/Projects/research/scres-ia/results/search_ladder_v5/result.json` — `claim_status` **`"THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH"`**, `run_role: CACHE_ANALYSIS`, `scope: "DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION"`, grade REPLAY, seeds 5300001–012, 15 methods + oracle.
- `ucb1_transfer` **0.04502300698358852** ranks #1; `neuron_memory` **0.05203274760040456** ranks #2; `ofat_transfer` 0.06274306709376977 #3; then `lookahead_kg_transfer` 0.08018189890560519, `gp_ei_transfer` 0.08390483487553962, `thompson_transfer` 0.08907820238728746; everything else ≥ 0.09655146536034616. **The six memory arms are exactly the top six.** `neuron_reset` falls to #12 at 0.11273605766343564.
- vs `neuron_memory`: `ucb1_transfer` **−0.007009740616816028 [−0.024441318883473704, +0.014076439977282206]** → tie, point to UCB1; `ofat_transfer` +0.01071031949336522 [+3.564844184833131e-05, +0.02171366105332719].
- All falsifiers pass; f7 is `not_applicable` (declared replay, `passed: null`).

`/Users/thom/Projects/research/scres-ia/results/ofat_lcb_reconciliation/result.json` — `claim_status` **`"OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE"`**, grade DEVELOPMENT, B=50,000, 40 resampling seeds, `self_sha256` `a35bb6ec721d683892000e74aae4a1c08b901aa13d36ee1f7aa5c956536e93ba`. The two sealed ladders score **identical arrays** (`f1` passes) and give opposite signs: `search_ladder_v2_ordered` LCB `−2.761381942678142e-05`, `search_ladder_v5` LCB `+3.564844184833131e-05`, shared mean `+0.01071031949336522`. **`lcb_positive_fraction_over_rng_seeds = 0.65`.** The artifact's own instruction, verbatim: `"Report the contrast as indistinguishable from zero and quote both sealed bounds. Do NOT write 'excludes zero'."` By contrast, `neuron_memory` vs `ucb1_transfer` has `lcb_positive_fraction = 0.0` — **"the neuron does not beat UCB1 with transfer" is robust.**

### 7.3 The neuron that "wins" is a 5-parameter single logistic unit
Source: `scripts/run_meta_learner_normaliser_audit_v1.py:129` (`class Fig5Neuron`) and `scripts/run_search_comparator_ladder_v2.py:331` (`make_neuron_arm`). It is `rho` (one weight vector, `np.zeros(dim)`), `predict = sigmoid(rho @ x)`, `update: rho += 0.35 * (y − predict(x)) * x`. **No hidden layer. It is logistic regression trained by the delta rule.** `results/surrogate_architecture_bakeoff/result.json` names it `neuron_5p` and reports `parameters = {kan: 532, mlp_matched: 529}` against it. The v.0 draft's own gap statement says "ANN, RNN, or Reinforcement Learning, guys, we have to figure it out!" — none of those three is what the winning arm is.

### 7.4 Real networks lose to a 5-parameter unit and to each other
`/Users/thom/Projects/research/scres-ia/results/surrogate_architecture_bakeoff/result.json` — `claim_status` **`"KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP"`**, grade REPLAY, seeds 5300001–012, all six falsifiers pass, parameter-matched 532 vs 529.
- AUC regret: `neuron_5p` **0.05203274760040456**, `mlp_matched` 0.08852563418951176, `spline_poly` 0.09753786839357298, `kan` 0.09889468682109304, `gbt` 0.1083233875544609, `gp_matern` 0.11379409899242587, `random` 0.1397945613431999
- `kan_minus_matched_mlp` **+0.01036905263158129 [+0.0030182183544574972, +0.018926078870915616]**, p = 0.0012 (lower is better → wholly against the KAN)
- vs the 5-parameter reference: `mlp_matched` **+0.0364928865891072 [+0.024450252001980696, +0.04767301535079317]**, Holm p = 0.0. **A 529-parameter MLP loses to a 5-parameter unit by an interval that excludes zero.**

### 7.5 As a control policy, architecture buys nothing
`/Users/thom/Projects/research/scres-ia/results/architecture_bakeoff/result.json` — `claim_status` **`"DEVELOPMENT_ARCHITECTURE_BAKEOFF_NO_ADJUDICATION"`**, scope `"DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION"`, **evidence grade `UNCONTRACTED`** (no contract hash), 5 seeds 9491–9495, 60,000 steps.
- KAN 97.04885864117553 (sd 0.6336, 204,816 params, **2.8155 ms/decision**), MLP 97.52395958347986 (sd 0.8135, 199,215 params, 0.6865 ms), DMLPA 97.65966171837127 (sd 0.8585, 187,404 params, 1.5970 ms)
- `KAN_minus_MLP` **−0.47510094230432853 [−1.5483579317025629, +0.5981560470939057]**; `DMLPA_minus_MLP` +0.13570213489141453 [−0.5692526511049414, +0.8406569208877706]; `DMLPA_minus_KAN` +0.6108030771957431 [−0.3499751596514674, +1.5715813140429535] — **all three cross zero.** KAN is 4.1× slower per decision.
- `TABLA_CANONICA` §6 records the defect nobody caught: **`run_architecture_bakeoff_v1.py` never sealed anything.** `results/architecture_bakeoff/sealed_record.json` is content-only, `claim_status: "CONTENT_SEALED_PROVENANCE_NOT_CERTIFIABLE"` — **provenance is not certifiable backwards.** Several external audits built tables on these figures.

`/Users/thom/Projects/research/scres-ia/results/dmlpa_kan_latent/result.json` — `claim_status` **`"KAN_LATENT_HURTS"`**, scope **`"DEVELOPMENT_OPEN_SEEDS_NO_CONFIRMATION_POSSIBLE_NO_VIRGIN_BLOCKS_REMAIN"`**, all five falsifiers pass, params matched 199,082 vs 200,052. `kan_minus_mlp` **−0.8622508690107338 [−1.605043533470689, −0.11945820455077864]**, n=5. Arm means dmlpa_kan 97.5808931153368 vs dmlpa_mlp 98.44314398434753. At equal budget the KAN affords hidden_dim 10 vs the MLP's 152.

### 7.6 The residual neural premium in Track B is small, uncontracted, and was inside the instrument's noise
`/Users/thom/Projects/research/scres-ia/results/track_b_nonneural/result.json` — `claim_status` **`"NEURAL_PREMIUM_LIKELY_IN_TRACK_B"`**, scope `"DEVELOPMENT_NO_CUSTODY_SEEDS_NO_ADJUDICATION"`, grade DEVELOPMENT, 24 eval episodes, all four falsifiers pass.
- trained_mlp 98.74303791918337 (**+2.176012221662134** over best constant), trained_kan 98.51622795481971 (+1.9492022572984808), trained_dmlpa 98.0037408940151 (+1.4367151964938643), untrained_net 72.20152694219925 (−24.36549875532198)
- `threshold_rule` 97.14154227552704, **+0.5745165780057876 [+0.3304118174965454, +0.8214864274274412]** over `constant_best` 96.56702569752123
- **The three networks span 0.739 among themselves.** `CLAIM_FREEZE` §6: *"elegir arquitectura no compra nada; usar una compra poco."*
- Artifact's own `why`: `"track_b_v1 had no non-neural arm, so 'neural premium' was untestable there. The ordering was wrong: this should have preceded the architecture comparisons."`

`/Users/thom/Projects/research/scres-ia/results/determinism_diagnostic/result.json` — `claim_status` **`"DEEPER_THAN_BOTH_ENVIRONMENT_LIMIT"`**, grade DIAGNOSTIC. Replicate deltas: A (8 envs, default threads) 0.47235723844329414; B (8 envs, threads=1) 1.6198479289422352; C (1 env, threads=1) 1.5388935190847377, at tolerance 1e-9. `TABLA_CANONICA_ENMIENDA_1` §6: the +1.44…+2.18 neural premium **fell inside a ±2.4 band** — "no es «sin confirmar»: el instrumento no la resuelve."
**Update you must carry:** `/Users/thom/Projects/research/scres-ia/results/determinism_repair_control/result.json`, `claim_status` **`"DETERMINISM_REPAIRED_SEED_IS_A_REPLICATION_UNIT_AGAIN"`**, created 2026-08-07T19:49Z, root cause `"MFSCGymEnvShifts.reset passed seed=None straight to MFSCSimulation on unseeded resets"`. Its `consequence` field, verbatim: `"the seed is a replication unit again and the +-2.4 band collapses, so the track_b neural premium becomes MEASURABLE. It is still not CONFIRMABLE: no virgin seed block remains."` The preregistered suspicion (`make_vec` discarding `seed`) was **refuted**, and that is reported as such.

### 7.7 The figure the whole framing rests on is an algebraic identity
`/Users/thom/Projects/research/scres-ia/results/garrido_fig5_surrogate/result.json` and `/Users/thom/Projects/research/scres-ia/results/garrido_wrap_q1/result.json` — both grade **UNCONTRACTED**. `f1_task_A_is_an_identity`: R² = 1.0, `max_abs_identity_error = 3.219646771412954e-15`, `Re_FRt = 0.9999999999999998`, `Re_RPj = 0.9999999999999677`, and **three of the five driver columns identically zero** (`Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`). `status: "IDENTITY_NOT_A_LEARNING_TASK"`. **ReT is exactly the sum of the drivers handed to the neuron as dendrites — the network as drawn has nothing to learn.**

### 7.8 The measured precondition for a neural premium is not met
`/Users/thom/Projects/research/scres-ia/results/headroom/buffer_prediction_premium/result.json` — `NO_NEURAL_PREMIUM_EVEN_WITH_MEASURED_CURVATURE`. Curvature **0.076259** vs episodic noise **0.317394**. Backprop **0.5548** — worse than a straight line at 0.6826. The frozen formulation: *"Una prima neural requiere que la curvatura de la superficie supere el ruido que la oculta."*

### How much of the v.0 framing survives

| v.0 element | survives? |
|---|---|
| Resilience is a learning-dependent, path-dependent property | **Yes** — H4 measured, H2 supported |
| `R_t = f(S_t, D_t, L_{t−1})` with `L` endogenous and retained across runs | **Yes, prospectively confirmed** — but `L` is a bandit statistic in an outer loop |
| **NEURAL networks are the mechanism** | **No.** Fails the one confirmatory counterfactual; ties or loses to UCB1/OFAT with matched memory; a 5-parameter logistic unit beats a 529-parameter MLP; the drawn Fig. 5 is an identity; no predictive premium at any preregistered SESOI |
| "Hybrid simulation–neural" as the paper's title claim | **No, as stated.** The winning "hybrid" arm is not a network in any interesting sense, and the sealed evidence for the *effective* carrier is non-neural |
| "Dynamic operationalization … under recurring and heterogeneous disruptions" | **No.** H_regime 0.0038 vs gate 0.05; no endpoint carries regime headroom; H1's only estimand exists under *isolated* shocks, not the recurring regime the RQ names |
| Learning improves over successive disruptions | **Yes** — H2 slope +0.0422 [+0.0347, +0.0499], with the order confound controlled |

The frozen replacement sentence, from `GARRIDO_Q1_Q2_CLAIM_FREEZE_ENMIENDA_3` §E4, verbatim:

> *"State retention ranked above memoryless search during development, but prospective transfer was carrier-specific: factor-level UCB1 outperformed both cold start and state-blind marginal replay, whereas the neural carrier did not."*

And from `_ENMIENDA_1` §E1: **`Q1_REFUTED_LOCALLY  NEURAL_MEMORY_FAILS_MARGINAL_REPLAY  (A2, UCB95 −0,00484)`**.

---

# 8. What the ENMIENDA_* / TABLA_CANONICA_* docs already say about H1–H4

### 8.1 The status table (`TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md` §1, verbatim structure)

| | original wording | declared reformulation |
|---|---|---|
| **H1** recovery | **SOSTENIDA** +126,0 h [+98,4, +154,5], by **absorption** (875/960 vs 755/960) | H1′ lost service: SOSTENIDA |
| **H2** learning curve | **SOSTENIDA** slope +0,0422 [+0,0347, +0,0499] | — |
| **H3** variance across intensities | **NO SOSTENIDA** — opposite sign, CI crosses zero, **with an estimand present** | H3′ variance of search cost: SOSTENIDA |
| **H4** dependence on `L_{t−1}` | **medida** +0,06070 [+0,04556] | — |

Followed by: **"Tres de cuatro se sostienen. No se redondea a cuatro."**

### 8.2 RETRACTED figures tied to these hypotheses

| retracted figure | tied to | replacement |
|---|---|---|
| **`+7,90 corridas [+6,88, +8,93]`** (cited in the draft **without naming its normaliser**) | **H4** | AUC **+0,06070 [+0,04556, +0,08020]** primary; **5,83** runs [+4,44, +7,31] secondary, always labelled censored |
| **`7,24 / 13,54 / 12,42`** runs-to-optimum | H4 panel | prefix panel **7,08 / 12,92 / 12,42** |
| **`7,90 / 5,43`** | H4 contrasts | as above |
| **"ventaja +0,00 → +10,00 entre contextos"** | **H2** | slope **+0,042201 [+0,034664, +0,049922]** |
| memory-vs-OFAT `+5,18` and memory-vs-random `+12,31` (runs) | H4 family | retired for leakage (`CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md`) |
| "memoria vale +0,0515 a UCB1 con LCB95>0" | H4/Q1 | report arm-mean difference 0,09655 → 0,04502 **without an LCB**; no paired-CI artifact exists for UCB1's memory effect |
| A1 "% del techo" rankings (99,70 etc.) | ladder | `search_ladder_v5` contains no `percent_of_ceiling`; cite A5 instead |
| `system_ttr` `0,00 ± [0,00, 0,00]` | H1 v1 | vacuous — 100% censored in all three arms |

### 8.3 FORBIDDEN wordings

From `TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §7:
- "Q1 y Q2 están cerradas"
- **"la neurona gana" / "hay prima neural"**
- **"las redes son la familia que imita SCL"**
- **"aprendizaje organizacional" / "la cadena aprende"** — it is an external loop *between runs*, not control within the episode
- "KAN es mejor política"
- "C1 está confirmado"
- "está en `main`"
- any of the retired §1 figures

From `_ENMIENDA_1` §3: **"prohibido escribir «excluye el cero»"** for `neuron_memory` vs `ofat_transfer`. Mandated phrasing: *"indistinguibles en AUC de arrepentimiento (media +0,01071; la cota inferior cae a ambos lados del cero según el remuestreo)"*, quoting **both** sealed bounds.

From `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` §2, allowed vs forbidden vocabulary: use "coupled to the DES", "outer-loop sequential simulation optimization", "persistent search state across runs", "`L_k` operacionalizado como estado de búsqueda"; **never** "embedded inside the DES transition kernel", "closed-loop adaptive control within the episode", "the supply chain learns", "organizational learning demonstrated".

From §7 items 9–16, prohibited assertions include: that networks best imitate SCL; that a confirmed neural premium exists; that organizational learning was demonstrated; that the chain learns within a campaign; **that memory materially improves per-regime adaptation (`H_regime` fails its gate)**.

### 8.4 Structural placement decision — this one matters for the v.0 draft

`GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_3.md` §E4:

> **`H1–H4 originales` | fuera de la espina, a apéndice de reconciliación**

The frozen central claim of the manuscript is the retention/carrier sentence in §7 above — **not** H1–H4. H1–H4 are to be reported in a reconciliation appendix.

### 8.5 Custody facts that cap everything above

- `TABLA_CANONICA_GENERADA.md`: **virgin blocks remaining = 0**; `new_seed_opening: False`; registry status `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`. **No H1–H4 result can be upgraded to confirmatory without new PI-authorised seeds.**
- Exactly **3 CONFIRMATORY** artifacts exist repo-wide: `grid_transfer_confirmation_v2` (the one that says the carrier is UCB1, not the neuron), `garrido_h2_h3_confirmation_v1` (DES physics reproduction — boundary, literal: *"does not establish learner, feedback, or architectural value"*; note its H2/H3 are **Garrido's thesis panels**, buffer and shift effects, **not the draft's H2/H3**), and `gsa_confirmation` (self-degraded to a one-bit calendar choice). **None of the draft's H1–H4 has confirmatory grade.**
- `_ENMIENDA_4` is a retraction of a custody claim made in `_ENMIENDA_1` §E3, and establishes **R10: custody state comes from `research/seed_custody_registry.json`, never from the contract.**
- 191 of 216 experiment keys (88%) are incomplete, so the corpus cannot be fully deduplicated; `ENMIENDA_REGISTRO_DE_EVIDENCIA_2026-08-07.md` §1 records that a naive sweep counted **439 positive intervals** which are mostly "dos familias de cintas re-analizadas bajo diez nombres de directorio."

---

## Bottom line per item

| item | verdict | grade | key number |
|---|---|---|---|
| **RQ (dynamic operationalization)** | **not supported / largely unanswerable as posed** | REPLAY + NEGATIVE_OR_HALTED | H_regime 0.003802 [1.08e-16, 0.014413] vs gate 0.05; v0 G3 = 0.0 exact |
| **RQ (predictive accuracy)** | **not supported** | DEVELOPMENT + UNCONTRACTED | no model clears SESOI 0.05; backprop 0.5548 < linear 0.6826 |
| **`L_{t−1}` claim** | **supported — different carrier** | **CONFIRMATORY** | ucb1 +0.03073 [+0.01990, +0.04256]; neuron −0.01178 [−0.01849, −0.00484] |
| **H1** | **supported, redefined endpoint (declared, pre-dated), isolated shocks only, by absorption** | DEVELOPMENT | +125.985 h [+98.347, +154.545], Holm p=0.0, n_cells 960 |
| **H2** | **supported, on its own estimand, order confound controlled** | DEVELOPMENT | slope +0.042201 [+0.034664, +0.049922], n=120; null −0.005088 [−0.015570, +0.005658] |
| **H3** | **not supported — wrong sign, live estimand** | DEVELOPMENT | −1.109e15 [−3.649e15, +1.293e15], Holm p=1.0, n_cells 360 |
| **H4** | **measured/supported; draft's number forbidden** | REPLAY | +0.06070331 [+0.04556081, +0.08019782], n=12 — **not** +7.90 |
| **"the carrier is neural"** | **refuted where it was testable; untestable where it wasn't** | CONFIRMATORY (against) | neuron −0.01178 [−0.01849, −0.00484]; 5p unit beats 529p MLP by +0.03649 [+0.02445, +0.04767] |
