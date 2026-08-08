# Plan — Repo: /Users/thom/Projects/research/scres-ia. Target journal: Computers & Industrial Engin

- **agent_id**: `ad5124e4e938018a3`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-07T17:23:01.866Z`
- **effort**: `medium` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-ad5124e4e938018a3.jsonl`
- **turns**: 117 · **assistant messages**: 4

## Task given

```text
Repo: /Users/thom/Projects/research/scres-ia. Target journal: Computers & Industrial Engineering. Task: design the SPINE of a manuscript, not the prose.

CONTEXT — the two questions the paper must answer (Garrido, Pongutá & Adarme, ICCL 2024, LNCS 15168 pp.80-94):
Q1: what category of AI algorithms best mimics the supply-chain-learning (SCL) attribute?
Q2: how can that family be integrated into the internal structure of a DES model for SCRES assessment?
Their Fig.2 marks nodes 3 (data gathering) and 8 (verification/validation) as the two ends of an OPEN-LOOP supply chain; the missing link between them they call the "Alzheimer's effect". Their Fig.5 proposes a neuron whose dendrites are four SCRES drivers weighted by rho.

There is also a v.0 draft with four hypotheses: H1 shorter recovery times, H2 learning-curve adaptation, H3 volatility reduction across heterogeneous disruption intensities, H4 path dependency (R_t = f(S_t, D_t, L_{t-1})).

THE EVIDENCE THAT EXISTS (read these files to get exact numbers):
- docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md — the canonical claims table, frozen today
- docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md — artifact table A1-A9 with hashes, retired figures, claim ladder
- docs/RESULTADO_H1_H3_ORIGINALES_V3_2026-08-07.md and results/manuscript/h1_h3_originales_v3/result.json
- docs/RESULTADO_H1_H3_V2_2026-08-01.md (H1' supported), results/garrido_h3_merge_adjudication/result.json (H3' sustained n=120)
- results/search_ladder_v5/result.json (the 15-arm ladder; six stateful arms are the top six)
- results/grid_transfer_confirmation_v2/result.json (the ONLY confirmation-grade artifact: virgin block 8200001-60, GRID_TRANSFER_CONFIRMED__UCB1; the neuron LOSES against a state-blind replay of its own marginals)
- results/garrido_fig5_surrogate/result.json and results/garrido_wrap_q1/result.json (Fig.5 as drawn is an algebraic IDENTITY, not a learning task)
- results/surrogate_architecture_bakeoff/result.json (KAN fits better, searches worse)
- results/track_b_nonneural/result.json (the non-neural floor; networks beat best constant by +1.44..+2.18 but this is DEVELOPMENT and NOT authorised for confirmation)
- docs/REGISTRO_DE_HUECOS_2026-08-07.md — 16 open gaps
- docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md — the large negative: no within-episode headroom in the thesis-native envelope

HARD CONSTRAINTS on what may be claimed (from the claim freeze — do not violate):
- Q1/Q2 are answered WITHIN the tested simulation-optimisation contract, never "closed".
- No neural premium is confirmed. The neuron ties with ucb1_transfer and LOSES the transfer falsifier.
- The integration is an OUTER LOOP between DES runs, never within-episode control, never "organizational learning".
- Figures 7.24/13.54/12.42 and 7.90/5.43 are RETIRED (leaky normaliser). Primary metric is auc_regret_norm under the PREFIX normaliser.
- Only ONE virgin seed block remains in the whole project (7700001-7700120); no new seeds are authorised.

WHAT I NEED FROM YOU:
1. A section-by-section manuscript skeleton (title, abstract thesis sentence, sections with the SPECIFIC artifact + number that goes in each). Say explicitly which figure/table each section carries.
2. A mapping table: every claim -> the artifact that supports it -> its evidence grade (confirmation / development / diagnostic) -> whether it is safe to state.
3. A decision on how to handle the v.0 draft's H1-H4: which survive as stated, which become declared reformulations (H1', H3'), and how to present the fact that H3 fails in its original wording while H3' succeeds. This must be honest, not spun.
4. Where the big NEGATIVE (no within-episode headroom, exhaustion certificate) belongs — is it a section of this paper, or a second paper?
5. The minimum set of additional runs (if any) that would materially strengthen the paper, ranked by (value added / effort), given that no new seeds are authorised.

Be concrete and cite file paths and numbers. Do not write prose for the paper
```

## Final answer

I have everything I need. Here is the manuscript spine.

---

# Manuscript spine — C&IE, Garrido Q1/Q2

## 0 · The one-sentence decision that shapes everything

The paper's centre is **`results/grid_transfer_confirmation_v2/result.json`** (sha256 `7bc33823ccd90b5e`, `run_role: CONFIRMATION`, virgin block 8200001–8200060, n=60). It is the *only* confirmation-grade artifact in the project, and it says the neuron **loses**. A paper built around "our neuron works" has one confirmation that contradicts it. A paper built around **"what transfers is a factorisation, not an approximator"** has one confirmation that *is* its headline. Build the second paper.

---

## 1 · Title, thesis sentence, contribution set

**Working title**
> *What transfers between simulation runs is a factorisation, not an approximator: an outer-loop learning layer for DES-based supply chain resilience assessment*

**Abstract thesis sentence** (the load-bearing one — everything else is scaffolding)
> Within the tested simulation–optimisation contract, the supply-chain-learning attribute is reproduced by **retention of search state between DES runs**, not by any family of function approximators; and under a prospective confirmation on a reserved seed block, the only carrier that beats a state-blind replay of its own visit marginals across a 16× design-space expansion is a **factor-level bandit** (+0.03073, LCB95 +0.01990, n=60), while the neural unit — which beats cold start (+0.05439) — **fails** that same falsifier with its entire interval on the unfavourable side (−0.01178, [−0.01849, −0.00484]).

**Five contributions, in the order a C&IE reviewer will accept them**
1. An outer-loop operationalisation of Garrido's Fig. 2 nodes ③↔⑧ that is *executable* and whose reads of unexecuted cells are impossible by construction (`Surface.value_of_visited` raises `LookupError`), not forbidden by convention.
2. The first prospective confirmation in this line, with a falsifier (marginal replay) that separates "transferred surface structure" from "revisited a lookup table".
3. A diagnosis of Fig. 5 as drawn: ReT is an algebraic identity of its dendrites (`max_abs_identity_error = 3.22e-15`), so the learning task must be relocated, not abandoned.
4. Two instrument findings with reach beyond this model: a normaliser leak detectable only by a twin-surface test, and the fact that **time-to-recovery is undefined under recurrent risk regimes**.
5. A quantitative precondition for a neural premium — curvature must exceed the noise hiding it (0.0763 vs 0.3174).

---

## 2 · Section-by-section skeleton, with the artifact + number each section carries

| § | Title | Artifact(s) | Number that must appear | Figure / Table it carries |
|---|---|---|---|---|
| 1 | Introduction — the open loop | Garrido 2024 Fig. 2 (p. 86) | node ③ = *Decision variables, ρ (experiment design)*, node ⑧ = *Metric of SCRES*; `L = {0+ℓ₁}, {0+ℓ₂}…` accumulates **across runs and configurations** | **Fig. 1** — Garrido's Fig. 2 redrawn with our node numbering and the closed ③↔⑧ arc superimposed. No results. |
| 2 | Background | `docs/REVISION_LITERATURA_KAN_2026-08-06.md`, `docs/LITERATURE_POSITIONING_MATRIX_2026-03-30.md` | the seven C&IE references Garrido cites ([8] Bruckler 2024 C&IE 192:110176, [9], [26], [28], [34], [38], [41]); arXiv:2407.16674 is still a preprint, no journal-ref, no external DOI | **Table 1** — positioning matrix. Novelty gap stated *auditably* (date, databases, query), never as universal absence. |
| 3 | The DES substrate | `results/garrido_reproduction/reproduction.json`; `docs/DES_MODEL_DOCUMENTATION.md`; `supply_chain/garrido_thesis_design.py` | max *j* 2,845 vs thesis 2,834–2,841 (**0.4 %**); only **72.7–75.7 %** of his orders are scored; ReT level ×0.49–0.94 (R1r), ×1.56–3.94 (R2r); **all six** moderation hypotheses H2a–c/H3a–c reproduce in direction; Cf1/Cf2/Cf5 quarantined (19.84 y not 10) | **Fig. 2** — SCUA topology (13 ops, 4 echelons). **Table 2** — reproduction vs Cf1–Cf20. **Hard rule:** the words "one-to-one reproduction" do not appear; the behavioural fidelity gate is blocked. |
| 4 | Method: the outer loop | A2, A1 | interface `x_k → DES(x_k) → y_k, d_1..d_4 → L_k → x_{k+1}`; budget B=24; 6 contexts; CRN | **Fig. 3** — the loop, with Garrido's node numbers on the arcs. |
| 4.2 | Operationalising Fig. 5 | `docs/CORRECCION_META_APRENDIZ_FUGA_2026-07-31.md` | the four-row mapping: dendrites *dᵢ* → **update signal**; weights ρ → **what is retained** (= `L_{t−1}`); activation "ReT(x)>ReT(x−1)?" → the gradient; **model input = decision variables** | **Table 3** — figure-to-implementation mapping. This section states plainly that feeding episode drivers to predict *unseen* configurations was our own leak, caught by a falsifier that was **hardcoded `passed: True`**, and that the fix made us *more* faithful to Fig. 5, not less (his dendrites are labelled `simulation decision variables`, p. 90). |
| 4.3 | Metric and normaliser | A3 `results/garrido_normaliser_audit_v3/result.json` (`fd617753949947e6`) | primary = `auc_regret_norm` under the **prefix** normaliser. Quote the rationale verbatim: *"`runs_to_within_1pct` imputes budget+1 and is censored at very different rates per arm, so its mean is not comparable across arms."* Censoring rates 0.056 / 0.153 / 0.222 / 0.611 | none |
| 5 | Experimental design and evidence grading | `research/seed_custody_registry.json`; the nine artifacts | three grades: **confirmation** (1 artifact), **development** (7), **diagnostic**; `new_seed_opening: false`; `status: BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED` | **Table 4 — the artifact register, in the body, not the appendix.** A1 `f648a1da5aefaf2f` · A2 `7bc33823ccd90b5e` · A3 `fd617753949947e6` · A4 `58d4c8a071cec86a` · A5 `f96e5b6ff0489932` · A6 `54bf5fa2594262bd` · A7 `954ac48301ff1234` · A8 `12e5f2562684655a` · A9 `d641ab0a54ecf2c8`. *(I re-computed all nine file hashes today; they match the freeze table.)* |
| **6.1** | Is there a search problem at all? | `results/surface_gates_v2/`, `results/surface_gates_extended_v2/` | ΔCV-R² non-separability **5/6 contexts at 288** (0.072–0.159), **6/6 at 4,608** (0.112–0.190, LCB95 to +0.179) vs threshold 0.05 | **Fig. 4** = `fig_b_surface_gates` (exists). Premise section: OFAT is *not* optimal by construction, so the search problem is real. |
| **6.2** | Q1a — the fifteen-method ladder | **A1** `results/search_ladder_v5/` | the six `memory_arms` occupy **exactly** ranks 1–6: ucb1_transfer 0.04502 · neuron_memory 0.05203 · ofat_transfer 0.06274 · lookahead_kg_transfer 0.08018 · gp_ei_transfer 0.08390 · thompson_transfer 0.08908; rank 7+ ≥ 0.09655; `neuron_reset` **rank 12** (0.11274) | **Table 5** — all 15 arms + oracle. **Fig. 5** = `fig_c_comparator_ladder` (exists), regenerated for 15 arms. |
| **6.3** | Q1b — the price of memory, per family | A1 / R4 slice | ucb1 +0.0515 [+0.0362] · neuron +0.0607 [+0.0461] · ofat +0.0375 [+0.0293] · gp_ei +0.0227 [+0.0128]. **Four unrelated families all gain materially from crossing state between runs.** | **Fig. 6** = `fig_d_memory_price` (exists). Caveat that must ride along: the freeze retires *"memory is worth +0.0515 to UCB1 with LCB95>0"* as a paired-CI claim — it is an arm-mean difference (0.09655→0.04502), reported as such, **without an LCB**. |
| **6.4** | Q1c — the approximator is not the ingredient | **A5** `results/surrogate_architecture_bakeoff/` | neuron_5p 0.05203 (best) · mlp_matched 0.08853 · kan 0.09889 · spline_poly 0.09754 · gbt 0.10832 · gp_matern 0.11379; parameters KAN 532 / MLP 529; `kan_minus_matched_mlp` **+0.01037 [+0.00302, +0.01893], p=0.0012** — interval entirely against the KAN | **Table 6** + **Fig. 7** = `fig_e_delta_efficiency` with its explicit equivalence band. Claim: **fit quality does not determine sequential-decision quality.** |
| **6.5** | **The falsifier that decides** — Q1 answered, prospectively | **A2** `results/grid_transfer_confirmation_v2/` | 288→4,608 configs, n=60, virgin 8200001–060. vs cold: ucb1 +0.05744 [+0.04989] · neuron +0.05439 [+0.04290] · gp +0.01433 · ofat +0.01422. **vs state-blind marginal replay:** ucb1 **+0.03073 [+0.01990, +0.04256]** ✅ · neuron **−0.01178 [−0.01849, −0.00484]** ❌ · gp −0.02160 ❌ · ofat −0.02467 ❌. `transfers = {ucb1:true, neuron:false, gp:false, ofat:false}`. Raw means worth quoting: `neuron_marginal` 0.06802 **beats** `neuron_transfer` 0.07980 | **Fig. 8 — NEW.** Forest plot, two panels (vs cold / vs marginal replay), four families, with the zero line. **This is the paper's single most important figure.** Also report `f1` anchor: the null sub-grid reproduces the 288 cache over **103,680 cells, max_abs_delta = 0.0**. |
| **6.6** | Q2 — where the learning actually is | **A4** `results/garrido_fig5_surrogate/` + `results/garrido_wrap_q1/` | Task A: R²=1.0, `max_abs_identity_error` **3.22e-15**, `Re_FRt`≈1, `Re_RPj`≈1, **three of five driver columns identically zero** (`Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`) → `IDENTITY_NOT_A_LEARNING_TASK`. Real held-out task, grouped folds (30 groups / 90 rows): backprop +0.0166 [+0.0048,+0.0283], KAN +0.0216 [+0.0042,+0.0390] vs linear, **both below the preregistered SESOI of 0.05** → `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL` | **Fig. 9 — NEW.** Two panels: (a) the identity residual on a log axis against the driver spread; (b) held-out R² with the SESOI band drawn. **Table 7** = B1/B2 vs SESOI. Tone: this **relocates** Garrido's proposal, it does not diminish it. |
| **6.7** | Scope — the limit that must travel with Q1 | **A7** `results/surface_gates/` + extended | `H_regime` = **0.003802**, LCB95 1.08e-16, UCB95 0.014413, against a preregistered gate of **0.05** → the gate **fails**; `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. Extended 4,608 grid: **+0.0283 [+0.0147]**, ×7.4 larger — Garrido's own 28-July instruction, measured, **in the direction he predicted** — and still 1.8× below the gate. Argmax moves in 4/6 contexts but only in `shifts` | **Table 8.** The sentence this section exists to license: **search-transfer value ≠ operational-adaptation value.** Memory avoids *re-deriving* an almost-common posture; it is *not* shown to tailor policy to regime. |
| **6.8** | The v.0 draft's four hypotheses | §3 below | see the hypothesis table | **Table 9** (original vs reformulation, verdict, artifact) + **Fig. 10 — NEW**, the H1 absorption mechanism. |
| **6.9** | Where a neural premium would have to live | **A6** `results/headroom/buffer_prediction_premium/` | 1,530 episodes, 17 buffer levels × 3 families × 3 scalings × 10 seeds, seed-grouped CV, 6 falsifiers. Held-out R²: constant −0.0034 · **linear 0.6826** · **backprop 0.5548 (worse than a straight line)**, Δ −0.1278 [−0.3157,+0.0601] · KAN 0.7163, +0.0337 [−0.0787,+0.1462]. Neither reaches SESOI 0.05. Curvature **0.076259** (= `profile_curvature.mean_one_minus_linear_r2`); noise **0.317394** (= 1 − 0.682606, **derived — say so in the sentence**) | **Fig. 11 — NEW**, R² panel with SESOI band. Closes "your surface was too easy": **this is the curved one.** |
| **7.1** | Instrument finding I — the normaliser leak, measured | **A3** `results/garrido_normaliser_audit_v3/` + `results/twin_surface_v2/` | twin-surface falsifier: contexts whose search path is unchanged when two **never-visited** cells are altered — oracle: ofat 6/6, random 6/6, **neuron_reset 0/6, neuron_memory 0/6**; prefix: **6/6 for all four**. `all_paths_unchanged`: oracle `false`, prefix `true`. Prefix contrasts: memory−reset **+0.06070 [+0.04556]**, memory−OFAT **+0.04821 [+0.03325]**, memory−random +0.08647 [+0.06729]. Status `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER` | **Fig. 12** = `fig_a_normaliser_leak` (exists). **Table 10 — the retired-figures ledger:** 7.24 / 13.54 / 12.42 and 7.90 / 5.43 retired; replacements 7.08 / 12.92 / 12.42 (prefix), always labelled *censored secondary*. The falsifier that has **a mandatory PASS and a mandatory FAIL** is a methodological contribution in its own right. |
| **7.2** | Instrument finding II — TTR is undefined under recurrent regimes | `results/manuscript/h1_h3_originales_v3/` `regime_note` | under R11–R24 at 52 weeks the events **merge into one cluster that never ends**; `system_ttr` censoring = **1.000 in all three arms** — because there is no return-to-normal to time. Not an instrument defect | none (prose + a timeline inset). **This speaks directly to C&IE [8] Bruckler et al. 2024, the SCRES-metrics review Garrido himself cites.** Half the SCRES literature computes time-to-recovery over regimes where the perturbation never ceases. |
| **7.3** | Instrument finding III — the metric rewards abandonment, and the contract is saturated upward | `results/metric_audit/abandonment_v1/`, `results/buffer_saturation_diagnostic/`, `results/ddmrp_unprojected_v1/` | ×10 buffer moves the metric **exactly +0.000000** at all three nodes in both families; →0 hurts (−0.0026 op9/R1r, **−0.0508** op9/R2r). Unprojected DDMRP ends with **+1.02 M (R1r) / +1.27 M (R2r)** extra units and a **bit-identical** `ret_excel_full_ledger` — a million units buying exactly zero resilience | **Fig. 13 — NEW**, the saturation curve. This is the *mechanism* behind §8. |
| 8 | Why the loop is outer, not inner (the bounded negative) | see §4 below | one page + one table, cited as companion | **Table 11**, condensed. |
| 9 | Discussion | — | the three sentences: (i) what mimics SCL is *what retains*, not *what approximates*; (ii) the representation that transfers is the one **factorised like the design space** — one statistic per factor level — because ρ lives on coordinates whose meaning shifts and the GP's points all sit at (…,0,0); (iii) fit quality ≠ decision quality | none |
| 10 | Limitations and threats to validity | claim freeze §7, gaps registry | seven named, below | none |
| 11 | Conclusion | — | the two boxed answers **with their scope labels attached inside the box** | none |
| App. A | Artifact register, falsifier inventory, runtime freeze | `docs/REPRODUCIBILITY.md`, `docs/DES_VALIDATION_DOSSIER.md` | Python 3.11.15 · SimPy 4.1.2 · NumPy 2.4.6 · Torch 2.12.1; every artifact path + SHA-256 + falsifier list | Table A1 |
| **App. B** | **Corrections and retractions ledger** | `docs/TABLA_CANONICA_DE_CLAIMS`, `CORRECCION_META_APRENDIZ_FUGA`, `RESULTADO_H1_H3_V2` addendum | five leaked/hardcoded-falsifier incidents, the seed collision 5800001–05, the `f6` self-comparison tautology, the OFAT exhaustion bug (`"idx" not in locals()`), the `distinct_configs_deployed: 42` mislabel (42 *pairs*, 23 unique configs) | Table B1. **Unusual, and it is this paper's integrity signature.** Do not cut it to save pages. |

**Figure budget:** 13 figures. Five already exist and regenerate from `result.json` with zero hardcoded numbers (`scripts/build_cie_outer_loop_figures.py`). Six are new. Two are prose-only insets. For C&IE, cut Fig. 13 to supplementary if pressed; **never** cut Fig. 8.

---

## 3 · Claim → artifact → grade → safe to state

Grades: **C** = confirmation (prospective, reserved block) · **D** = development (burned tapes / already-open block, no adjudication) · **X** = diagnostic (instrument property, no comparative claim).

| # | Claim, as it would be written | Artifact | Grade | Safe? |
|---|---|---|---|---|
| 1 | Only `ucb1_transfer` beats a state-blind replay of its own marginals across 288→4,608: +0.03073 [+0.01990, +0.04256], n=60 | A2 `7bc33823ccd90b5e` | **C** | ✅ **This is the headline.** |
| 2 | The neuron beats cold start (+0.05439 [+0.04290]) but **loses** the marginal-replay falsifier: −0.01178 [−0.01849, −0.00484] | A2 | **C** | ✅ **Mandatory.** Omitting it is the one thing that would sink the paper on review. |
| 3 | All six state-retaining arms occupy ranks 1–6 of a 15-method ladder; the same neuron without memory falls to rank 12 | A1 `f648a1da5aefaf2f` | D | ✅ with `DEVELOPMENT_ON_BURNED_TAPES` label and seeds 5300001–012 declared as replay |
| 4 | The neuron does not beat `ucb1_transfer`: −0.00701 [−0.02444, +0.01408] — a tie, with the point estimate against us | A1 | D | ✅ — say "tie", never "UCB1 wins", never "the neuron wins" |
| 5 | The neuron does not beat `ofat_transfer`: +0.01071, CI boundary sitting on zero | A1 (+3.56e-05) vs `search_ladder_v2_ordered` (−2.76e-05) | D | ⚠️ **Only as a tie, with the sign flip disclosed.** The two sealed ladders disagree on the LCB's sign at the 5th decimal. Report it as *"the lower bound is indistinguishable from zero; bootstrap resamples place it on either side across ladder versions"* and cite both. Do **not** write "excludes zero". |
| 6 | Retention is worth +0.06070 [+0.04556] AUC to the neuron itself | A3 `fd617753949947e6` | D | ✅ |
| 7 | Four unrelated families gain from retention: ucb1 +0.0515, neuron +0.0607, ofat +0.0375, gp_ei +0.0227 | A1/R4 | D | ✅ as arm-mean gains; the **ucb1 paired CI does not exist** — no LCB for that row |
| 8 | The Alzheimer effect survives a normaliser that cannot see unexecuted cells; twin-surface: prefix 6/6 unchanged, oracle 0/6 for both neural arms | A3 + `twin_surface_v2` | D/X | ✅ strong |
| 9 | ReT is an exact identity of Fig. 5's dendrites (3.22e-15); 3 of 5 driver columns are identically zero | A4 `58d4c8a071cec86a` | X | ✅ |
| 10 | On the genuine held-out task, neither backprop nor KAN clears the preregistered SESOI of 0.05 | `garrido_wrap_q1` | D | ✅ |
| 11 | KAN searches worse than a parameter-matched MLP: +0.01037 [+0.00302, +0.01893], p=0.0012 | A5 `f96e5b6ff0489932` | D | ✅ |
| 12 | A 5-parameter unit is the best searcher of the bake-off (0.05203), 74× fewer parameters, 30× cheaper per decision | A5 + `search_surrogates` | D | ✅ — and it is a **result in Garrido's favour**; say so |
| 13 | KAN as a *control policy*: −0.475 [−1.548, +0.598], no separation; 4.1× slower per decision | A9 `d641ab0a54ecf2c8` | D | ⚠️ **Conditionally.** `run_architecture_bakeoff_v1.py` **never sealed anything**; provenance is not certifiable backwards, and several external audits built tables on it. Either (a) re-run under a sealing harness on development seeds, or (b) demote to a footnote with the defect disclosed. Do not carry it as a main-text table without one of the two. |
| 14 | A neural premium requires curvature above the noise hiding it: 0.0763 vs 0.3174; the MLP is worse than a straight line (0.5548 vs 0.6826) | A6 `54bf5fa2594262bd` | D | ✅ — **always** with the derivation spelled out (noise = 1 − held-out linear R²) |
| 15 | Q2: the integration is a persistent outer loop between DES runs, `L_{t−1}` = retained search state; grid transfer confirmed for a non-neural method | A2 + A1 | C (transfer) / D (rest) | ✅ under `Q2_OUTER_LOOP_INTEGRATION_IMPLEMENTED_AND_GRID_TRANSFER_CONFIRMED_FOR_UCB1` |
| 16 | The surface is non-separable (5/6 at 288; 6/6 at 4,608) — OFAT is not optimal by construction | `surface_gates_v2/_extended_v2` | D | ✅ |
| 17 | Knowing the regime is worth 0.0038 [LCB 1.08e-16] against a 0.05 gate — the gate **fails** | A7 `954ac48301ff1234` | D | ✅ and **required** as the scope clause on every Q1 sentence |
| 18 | Adding upstream buffer variables multiplies `H_regime` ×7.4 to +0.0283 [+0.0147] — Garrido's own instruction, in his predicted direction — and still fails | `surface_gates_extended_v2` | D | ✅ |
| 19 | H1 (original wording) supported: restricted TTR 75.7 h vs 201.7 h, +126.0 h [+98.4, +154.5], Holm p<0.0001, 960 cells | `h1_h3_originales_v3` (`39061791dd37eef4`) | D | ⚠️ only with *"confirmation on a virgin block has not been performed"*, per the artifact's own §6 |
| 20 | H1's mechanism is **absorption, not restoration speed**: 875/960 (91.1 %) vs 823 (85.7 %) vs 755 (78.6 %) shocks absorbed | same | D | ✅ — this is the most quotable sentence in the hypotheses section |
| 21 | H3 (original wording) **not supported**: −1.05e14 [−3.30e15, +2.94e15], p=0.52, 78.7 % of cells deploying different configurations | same | D | ✅ **Mandatory.** |
| 22 | H1′ supported: +61.35 M ration-hours [+14.43 M, +121.80 M], 72 cells | `h1_h3_v2_1` (`5ea481d5897d2f34`) | D | ✅ as a **declared reformulation**, with *"`service_loss_auc` is not a recovery time"* in the same paragraph |
| 23 | H3′ sustained at n=120: memory−OFAT +16.22 [+9.61, +22.74]; memory−reset +9.31 [+2.35, +16.35] | `garrido_h3_merge_adjudication` (`e06c53c136a89bcb`) | D | ✅ as a declared reformulation — **and disclose the per-slice diagnostic**: local_90 +10.27 [+2.36], vps_30 +6.46 **[−7.31, +20.33]** crosses zero |
| 24 | H4: retention is worth +0.06070 AUC [+0.04556], ≈ **5.83** runs saved (censored secondary, prefix) | A3 | D | ⚠️ **The draft's +7.90 [+6.88, +8.93] is a RETIRED number** — it is `memory_vs_reset` under the *oracle* normaliser (14.889 − 6.986). It must not appear. See §3 below. |
| 25 | Time-to-recovery is undefined under Garrido's recurrent regime: `system_ttr` censoring = 1.000 in all three arms | `h1_h3_originales_v3` | X | ✅ — a genuine contribution |
| 26 | The inherited `ret_excel` metric rewards abandonment | `metric_audit/abandonment_v1`, gap B4 | X | ✅ |
| 27 | The contract is saturated upward: ×10 buffer → exactly +0.000000; unprojected DDMRP carries +1.02/+1.27 M units for a bit-identical metric | `buffer_saturation_diagnostic`, `ddmrp_unprojected_v1` | X | ✅ |
| 28 | Reproduction: family separation (~100×) reproduced without tuning; order generation agrees to 0.4 %; all six moderation hypotheses reproduce in direction | `garrido_reproduction` | D | ✅ **only** with "distributional, not per-order; seed 375 does not reproduce his stream; no 1:1 claim" |
| 29 | No within-episode headroom across 4 decision surfaces / 3 programs / 2 physics generations | `PAPER2_EXHAUSTION_CERTIFICATE` | D | ⚠️ **cite as companion, do not re-freeze.** Gap A2: `supply_chain.py` has 12+ commits since the 14-July manifest, including `64b75ce` (145 orders to zero) and `ea246ac`. |
| 30 | Networks beat the best constant by +1.44…+2.18 in Track B | A8 `12e5f2562684655a` | D | ❌ **Not a claim.** Report as an open residual with `C1_VIRGIN_BLOCK = NO-GO` and the reason: the preregistered blocking guardrail `worst_product_fill` was never persisted, so the estimand is undefined. State it as *"we decline to adjudicate this"* — which is stronger than adjudicating it. |
| 31 | "Q1 and Q2 are closed" | — | — | ❌ **Never.** Only `..._WITHIN_TESTED_SIMULATION_OPTIMISATION_CONTRACT`. |
| 32 | "The supply chain learns" / "organizational learning" | — | — | ❌ **Never.** The carrier lives in the experiment loop, not the event kernel — and a reviewer will open the code. |

---

## 4 · The v.0 hypotheses: what survives, what is reformulated, and how to say H3 honestly

### The verdict table (this is manuscript **Table 9**)

| | Original wording (v.0 §3.1) | Verdict on the original | Declared reformulation | Verdict on the reformulation |
|---|---|---|---|---|
| **H1** Learning Effect | "significantly shorter recovery times" | ✅ **SUPPORTED** — restricted TTR 75.7 vs 201.7 h, **+126.0 h [+98.4, +154.5]**, Holm p<0.0001, 960 cells | **H1′** cumulative lost service | ✅ SUPPORTED — +61.35 M [+14.43 M, +121.80 M] |
| **H2** Adaptation / learning curve | "improved performance under successive disruptions" | 🟡 **NOT YET ADJUDICATED** — the "+0.00 → +10.00" curve form was **retired** with the leaky runner; only the structural zero at the first context survives | — | — |
| **H3** Volatility Reduction | "reduce performance variance across heterogeneous disruption intensities" | ❌ **NOT SUPPORTED** — variance 1.61e16 (hybrid) vs 1.60e16 (static); Δ **−1.05e14 [−3.30e15, +2.94e15]**, p=0.52 | **H3′** variance of *search cost* across contexts | ✅ SUSTAINED at n=120 — +16.22 [+9.61, +22.74] vs OFAT |
| **H4** Path Dependency | "R_t positively influenced by accumulated learning" | ✅ **SUPPORTED** — retention worth **+0.06070 AUC [+0.04556]**, ≈5.83 runs saved (prefix, censored secondary) | — | ⚠️ the draft's **+7.90** must be replaced |

### The three decisions, and their justification

**(a) H1 and H4 survive as stated; H3 does not; H2 is not yet adjudicated.** Three of four is an honest headline. Do not round it to four.

**(b) How to write H3 — the paragraph structure that is honest rather than spun.**

The temptation is "H3 fails but H3′ succeeds, so learning reduces volatility after all." That is spin, because H3 and H3′ measure **different things** and the one that failed is the one the draft's theory actually predicts. Write it in this order, and do not reorder:

1. **State the failure first, in its own sentence, with the number.** H3 as written is not supported: the sign is on the wrong side and the interval spans zero.
2. **Kill the excuse before a reviewer offers it.** At n=120 with **78.7 %** of cells deploying *different* configurations, the earlier explanation — "there is no estimand here" — no longer applies. **The estimand exists and the effect is absent.** This is a refutation with power, not a null from a weak design. Say exactly that.
3. **Explain the mechanism, because it is the same one as §6.7.** The optimal posture is nearly invariant across contexts (`H_regime` 0.0038); the arms therefore deploy configurations that perform similarly under escalating intensity. There is nothing for learning to stabilise *in the deployed system*.
4. **Then, and only then, introduce H3′ as a reformulation — with the construct change named.** The variance that learning does reduce is the variance of the **cost of searching**, not the variance of **deployed performance**. Quote the artifact's own words: *"the manuscript's H3 is variance of performance across disruption intensities… this is a different construct, not a repair."*
5. **Disclose H3′'s own weakness.** The two merged slices disagree: local_90 gives +10.27 [+2.36], the vps_30 slice gives +6.46 **[−7.31, +20.33]**, crossing zero. The n=120 merge is sustained; the 30-seed slice alone would not be.
6. **Close with the sentence the draft has to swallow:** *the only variance the learner reduces is the variance of searching, not the variance of what it deploys.*

**(c) H4's headline number must be replaced before anything is written.** The draft's `+7.90 runs [+6.88, +8.93]` is `memory_vs_reset` on `runs_to_within_1pct` under the **oracle** normaliser (14.889 − 6.986) — i.e. it is exactly the retired 7.90 in the freeze's prohibited list. Its prefix replacements: **AUC +0.06070 [+0.04556, +0.08020]** as primary, **5.83 runs [+4.44, +7.31]** as censored secondary, always labelled. This substitution is not optional and is easy to miss, because the draft cites the number without naming its normaliser.

**(d) H2 — the cheapest real win in the whole project.** See §5, item 1.

---

## 5 · Where the big negative goes

**Verdict: a bounded one-page section in *this* paper (§8), full treatment in a *separate* paper. Not both, not neither.**

**Why it cannot be the spine of this paper.** It is a **different estimand**. This paper measures retention between *configuration searches*; the exhaustion certificate measures a neural premium for a *policy within an episode*. The claim freeze §6 already says these two "do not share an abstract", and it is right. Splicing them produces a paper whose two halves answer different questions and whose reviewer cannot tell which one the title refers to.

**Why it cannot be omitted.** It is the *reason the loop is outer*. Without it, "we put the learner between runs instead of inside the episode" reads as a convenience. With it, that placement is a **measured finding**: adaptive control is not warranted inside this system, so the only place learning can pay is between runs. It is also the mechanism behind §6.7 (`H_regime` fails) and §7.3 (saturation) — three sections that are currently three separate negatives and should be presented as one.

**What §8 contains — one page, one table, hard limits:**
- The mechanism, in four lines: the DES is structurally over-buffered → constants are near-optimal; the optimal posture is invariant to the risk regime; commitment lags defeat clairvoyance (a non-privileged EWMA detector beats the true-state instant switcher ×2.3); where clairvoyant headroom exists it fails observability or resource-honesty.
- **One number, and it is the decisive one:** using Garrido's own escalation table — exact Cf1–Cf20 + one-at-a-time R11–R24 + impacts 1×/1.5×/2×, 45 profiles × 18 postures × 6 seeds = **4,860 ten-year evaluations** — the optimal constant posture is **invariant across all 45 risk profiles at every budget cap**; max `H_profile_safe` **6.93e-05 against a bar of 0.01, i.e. 144× below**, `passing_doors = []`. And critically: **Cf20 escalates all four R2 risks simultaneously and still produced no door.**
- One line that must be there because it is the honest bound: risks genuinely degrade physical resilience (ReT 0.53 → 0.20 under escalation) but create **zero regime-tailoring headroom** — this is not a guardrail rejection, all guardrails were clean; the tailoring simply has no value.
- **Table 11**, four rows, ceilings vs observable outcomes.
- **Two mandatory caveats.** (i) Scope: the screen covered a *discrete* subset — not the continuous space, not inter-family R1×R2, not φ×ψ, not temporal coupling. *(A prior revision of that certificate wrongly claimed a pure one-factor-at-a-time design and was retracted; do not reintroduce the retracted framing.)* (ii) **Provenance**: gap A2 — `supply_chain.py` has drifted 12+ commits since the 14-July manifest. Cite the certificate as a companion finding under the physics of its date; **do not re-freeze it and do not recompute it here**.

**The second paper**, when the PI decides: *"When not to train: a certified negative for adaptive control in an over-buffered supply chain"* — carrying Program O's H_PI = 0.15151 (simultaneous safe LCB95 0.11562, exact fungible-null 0.0), the Program E 0/10 PPO seeds, Track C's hysteresis-beats-clairvoyance ×2.3, and the two falsifiable reopeners (Q11/R09 mission expiry, Q6/Q7 integrated shared resource). That paper needs gap A2 resolved first — either re-run under current physics, or withdraw the anchored claims in writing. Both are PI decisions, not analyst decisions.

---

## 6 · Minimum additional runs, ranked by value ÷ effort

No new seeds are authorised (`new_seed_opening: false`, `scientific_execution_authorized: false`). Items 1–5 respect that absolutely — they touch **zero** seeds.

| # | Run | Effort | What it buys | Verdict |
|---|---|---|---|---|
| **1** | **Adjudicate H2 from `per_context.regret_curve`** in the already-sealed artifacts (`garrido_meta_learner_v2`, `search_ladder_v5`). Preregister the estimand *before* looking: e.g. regret-AUC as a function of context ordinal 1…6, memory minus reset, LCB95>0. | ~3 h, **0 seeds** | Turns the **fourth** of the draft's four hypotheses from "measured, form retired" into adjudicated. H2 is the *learning-curve* hypothesis — the one the entire framing rests on — and it is currently the only one with no verdict. `DONDE_PODEMOS_SER_LAXOS` §5.2 already identifies the data as present. | 🥇 **Do first.** Highest value-to-effort in the project. |
| **2** | **Re-express H4 under the prefix normaliser** and sweep the draft for the retired pairs 7.24/13.54/12.42 and 7.90/5.43. | ~1 h, 0 seeds | Removes a **prohibited number from the paper's own hypothesis table**. Purely mechanical, and a reviewer who spots it after publication would be entitled to question every other figure. | 🥇 **Not optional — a correctness fix.** |
| **3** | **Reconcile the `ofat_transfer` LCB.** Recompute the paired bootstrap on the sealed per-arm arrays at B=50,000 with a declared RNG seed; report the interval and its instability. | ~2 h, 0 seeds | Resolves the one place where two frozen documents state opposite signs (−2.76e-05 vs +3.56e-05) at the 5th decimal. A careful reviewer *will* find it. Better to own it as "indistinguishable from zero, and here is the resampling sensitivity". | 🥈 **Do.** |
| **4** | **Re-seal the KAN-as-policy bake-off (A9)** under the sealing harness, on the same development seeds. | ~4 h + compute, 0 new seeds | Repairs the defect none of five external audits caught: `run_architecture_bakeoff_v1.py` never sealed anything, so A9's provenance is not certifiable backwards. | 🥈 **Do — or drop A9 to a footnote.** Either is defensible; carrying it unsealed in a main table is not. |
| **5** | **Generate the six new figures** through `scripts/build_cie_outer_loop_figures.py`'s existing no-hardcoded-numbers pattern (every value loaded from its `result.json`). | ~6 h, 0 seeds | Guarantees no figure can drift from the artifact it claims to show. Already the house pattern; extend, don't reinvent. | 🥈 **Do.** |
| **6** | **Confirm H1 on the last virgin block** (7700001–7700120, id `g3a_v2_development`, `RESERVED_NOT_OPENED`). | 120 seeds — **the last block in the project** | Would upgrade H1 from development to confirmation. | 🚫 **Recommend against.** H1 is not this paper's spine; the paper already has its one confirmation (A2) and it is on the *central* claim. Spending the last virgin block on a secondary hypothesis, when C1 is `NO-GO` on nine unmet preconditions and gap A1 is unclosed, is bad portfolio management. State H1 as development with *"confirmation not performed"* — exactly as the artifact itself does — and keep the block. **PI decision, and it should be an explicit one, not a default.** |
| **7** | Close gap A1 (`worst_product_fill` persisted and blocking) and re-run step 3. | ~5 h + compute, 0 new seeds | Does **not** help this manuscript — it defines the estimand for the *other* paper (C1). If the residual disappears after A1, a whole confirmatory block is saved. | ⏸️ **Defer.** Do it before the second paper, not this one. |
| **8** | Triage the 20 red tests (`docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md`). | ~1 day, 0 seeds | Does not change a number, but a red suite is how the custody defect survived in the first place — and reviewers of reproducibility-forward papers do run the suite. | ⏸️ **Before submission, not before drafting.** |

**Net:** items 1–5 are roughly **two days of work, zero seeds**, and they take the paper from "one confirmation plus a prohibited number in its hypothesis table" to "one confirmation, four adjudicated hypotheses, no retired figures, every figure traceable to a hash". That is the whole delta worth buying.

---

## 7 · Three things that will decide the review, stated plainly

1. **Lead with the loss.** The abstract should contain the neuron's failed falsifier. A paper that reports its own central method losing a falsifier it designed is trusted; a paper where a reviewer finds that in the supplement is not.
2. **The scope label is part of every Q1/Q2 sentence, not a caveat at the end.** `H_regime = 0.0038` against a 0.05 gate means memory avoids *re-deriving* a near-common posture — it does not adapt to regime. Every sentence that says "learns" must survive substituting "avoids re-deriving".
3. **Appendix B is the paper's argument, not its apology.** Five caught leaks, a hardcoded falsifier, a seed collision, a tautological seal check, and a retired figure set — disclosed with the falsifier that caught each one — is a stronger claim to reliability than a clean-looking paper with none of it. C&IE reviewers of simulation work read exactly this.

---

### Critical files for implementation
- `/Users/thom/Projects/research/scres-ia/docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` — the artifact register A1–A9 and the claim ladder; the spine's authority.
- `/Users/thom/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json` — the only confirmation-grade artifact; §6.5 and Fig. 8.
- `/Users/thom/Projects/research/scres-ia/results/manuscript/h1_h3_originales_v3/result.json` — H1/H3 originals, the absorption mechanism, and the TTR-undefined regime note.
- `/Users/thom/Projects/research/scres-ia/results/garrido_normaliser_audit_v3/result.json` — the prefix normaliser, the twin-surface falsifier, and the correct H4 replacement numbers.
- `/Users/thom/Projects/research/scres-ia/scripts/build_cie_outer_loop_figures.py` — the no-hardcoded-numbers figure pipeline the six new figures must extend.
