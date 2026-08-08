# Plan — Repo: <HOME>/Projects/research/scres-ia. Target venue: Computers & Industrial Enginee

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a6b9614b65faa729d`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T01:01:59.129Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-a6b9614b65faa729d.jsonl`
- **turns**: 157 · **assistant messages**: 2

## Task given

```text
Repo: <HOME>/Projects/research/scres-ia. Target venue: Computers & Industrial Engineering (C&IE, Elsevier).

TASK: design TWO manuscripts, section by section, with the exact artifact and number behind each section. Do not write prose — design structure and evidence mapping, and judge acceptance risk honestly.

CONTEXT — what exists:
- A guide-audited C&IE submission chassis on branch `origin/codex/submission-a-program-q` at `papers/submission_a_program_q/` (cover letter, HIGHLIGHTS.txt, TITLE_PAGE.tex elsarticle, CIE_GUIDE_AUDIT_2026-07-29.md, RELEASE_AND_SUBMISSION_CHECKLIST.md, GENERATIVE_AI_DISCLOSURE_DRAFT.md, scripts/build_submission_a_cie_review_bundle.py). Its science is superseded; its mechanics are reusable.
- `scripts/build_cie_outer_loop_figures.py` generates 5 figures from result.json with no hardcoded numbers, but points at SUPERSEDED artifacts (search_ladder_v2, twin_surface, surface_gates). Must be repointed to search_ladder_v5, twin_surface_v2, surface_gates_v2.
- Frozen claim text: `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` + `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` + `_ENMIENDA_1.md`. Read these — they contain the permitted and FORBIDDEN vocabulary.

PAPER 1 — "the measurement paper" (design this first; it is the safety play).
Thesis: widely-used SCRES simulation metrics measure the wrong thing in ways that are demonstrable and correctable. Evidence to map:
- `ret_excel` rewards abandonment (the split maximising it delivers 50% fill vs 80% for the split minimising it) — find the artifact
- time-to-recovery is UNDEFINED under recurrent risk regimes: system_ttr censoring = 1.000 in all arms because events merge into one never-ending cluster (`results/manuscript/h1_h3_originales_v3/result.json`), and the repair is `restricted_ttr = min(TTR, tau)` with a paired placebo
- the oracle normaliser leaks and the prefix normaliser does not, proven by a twin-surface falsifier (`results/garrido_normaliser_audit_v3/result.json`)
- H_regime is not invariant to monotone reparametrisation (`results/monotone_transform_family_v4/`)
- ret_excel is step-cadence dependent — find the artifact
- the harness was non-deterministic and the root cause was one line (`results/determinism_repair_control/result.json`)
Venue fit: Garrido 2024 cites Bruckler et al. 2024, C&IE 192:110176, a review of SCRES metrics. This paper is a measured critique extending exactly that review. It needs NO new compute and is immune to the objection that our demand process is nearly deterministic.

PAPER 2 — "retention before architecture" (the Q1/Q2 paper).
Thesis: what operationalises supply-chain learning in a DES is retained search state between runs, not a neural approximator; and in the only prospective confirmation the carrier that survives is a factor-level bandit.
Evidence: `results/grid_transfer_confirmation_v2/` (CONFIRMATORY, virgin block, UCB1 +0.03073 [+0.01990] vs its own state-blind marginal replay; the neuron LOSES −0.01178 [−0.01849,−0.00484]); `results/garrido_h2_h3_confirmation_v1/` (CONFIRMATORY validation, six panels, Holm p 4.8e-17..1.9e-15, 12 virgin roots — its boundary forbids any learner claim); `results/search_ladder_v5/`; `results/surrogate_architecture_bakeoff/` (KAN searches worse than matched MLP +0.01037 [+0.00302,+0.01893]); `results/garrido_fig5_surrogate/` (Fig.5 as drawn is an algebraic identity, max error 3.22e-15); the four draft hypotheses H1 supported (+126.0h [+98.4,+154.5]), H2 supported (slope +0.0422 [+0.0347,+0.0499]), H3 NOT supported, H4 measured (+0.06070 [+0.04556]).
KNOWN THREAT to state in limitations: the demand process is uniform discrete U(2400,2600), inherited from Garrido-Rios 2017 — about +-4% variation. Garrido has asked for Holt exponential smoothing with Monte-Carlo random alpha/gamma (his own IJPR 2024 method). Our central negative could be an artifact of near-deterministic demand.

FOR EACH PAPER GIVE ME:
1. Title, one-sentence thesis, 5 highlights (<85 chars each, C&IE format).
2. S
```

## Final answer

I read the claim-freeze pair, both amendments, and opened every artifact behind the evidence you listed. All SHA-16s below are computed from the working tree today; the five in the freeze (`f648a1da…`, `7bc33823…`, `fd617753…`, `f96e5b6f…`, `58d4c8a0…`) all verify.

---

# PAPER 1 — the measurement paper

## 1. Title, thesis, highlights

**Title:** *What discrete-event resilience metrics actually reward: six measured failure modes in DES-based SCRES measurement, and the falsifiers that catch them*

**Thesis (one sentence):** In a discrete-event supply-chain resilience study, the scored population, the normalising scale, the observation cadence and the replication unit are all endogenous to the policy being scored, and we show six specific, reproducible consequences — including a canonical resilience index that rates a rationing split 12× higher while it delivers 30 points less fill — each detectable by a stated falsifier before any effect is reported.

**Highlights** (all ≤85 chars, verified):

```
A resilience index rates a split 12x higher while it delivers 30 points less fill.   (82)
Time-to-recovery has no estimand under recurrent risk: censoring is exactly 1.000.   (82)
A leaky normaliser inflates the reported effect by 48%; a twin-surface test finds it. (85)
Regime headroom is not invariant: a monotone rescaling moves it 0.020 to 0.278.      (79)
Six falsifiers, each with a stated failure mode, are supplied as a reusable protocol. (85)
```

## 2. Section skeleton — artifact and number per section

All paths below are under `<HOME>/Projects/research/scres-ia/`.

| § | Content | Artifact + SHA-16 | The number it carries | Figure/Table |
|---|---|---|---|---|
| 1 | Introduction: Bruckler et al. 2024 C&IE **192**:110176 reviewed SCRES metrics; Garrido et al. 2024 call DES-based SCRES metrics "inadequate or incomplete" and ask for credibility/validity work. This paper supplies the measurement. | — | none | — |
| 2 | The instrument: reconstructed military food supply chain, ReT ledger, risk regimes R11–R24 | `calibration_provenance_v1` block (identical in every artifact) + `results/demand_process/result.json` `cb4f88398c4f93a4` | fulfilment delay 54.0 h vs LT promise 48.0 h; 168 h/week; `autotomy_reachable: false`. Demand: weekly CV **0.0713**, lag-1 ACF **−0.2283**, exactly **6.0** orders/week (sd 0.0), **24.8 %** of weeks over one-shift capacity | **T1** provenance + demand |
| 3 | **Failure 1 — the metric rewards abandonment** | `results/sensitivity/contention_headroom_v1_2/result.json` `a47e89831fe6f130` (5,184 episodes); replicated on virgin seeds in `results/metric_audit/abandonment_v1/result.json` `48746e7c386782e5` | FIFO_PARTIAL, non-fungible, R1r+R2r base: share 0.1 → ReT **0.0047313**, fill **0.5069**, forfeited **318,621** rations; share 0.5 → ReT **0.0004149**, fill **0.7953**, forfeited **0.0**; share 0.9 → ReT **0.0050600**, fill **0.4969**, forfeited **335,869**. Ratio **12.2×**. Repeats in all six cells. Virgin replication: `best_share_by_ret = 0.1` vs `best_share_by_service = 0.5` in both regimes | **F1 (new)** inverted-U twin axis; **T2** 9 shares × 4 quantities |
| 4 | Failure 1b — the mechanism is *not* the one we first proposed, and neither is the repair | `results/sensitivity/contention_headroom_v1_3/result.json` `60537d88bbc2b1a9`; `results/metric_audit/abandonment_v1` `branch_explains_the_u: false` | Omitted orders 136 / 58 / 138; lost 76 / 1.6 / 78. `full_ledger` does **not** remove the U (0.0024/0.0003/0.0025); `visible_clipped_0_1` is **worse** (R2r 0.367→0.027→0.353, 14×). **H_regime of `ret_excel_omitted_n` = 0.375 [0.111, 0.722]** and `ret_excel_visible_n` = **0.4306** — against ~1.5e-04 for the score itself. What varies with regime is *who gets scored* | **T3** four repair variants |
| 5 | **Failure 2 — time-to-recovery has no estimand under recurrent risk** | `results/manuscript/h1_h3_v1/result.json` `12abb332afe73c85` (`HALTED_FALSIFIER_FAILED`) | `censored_fraction_by_arm = {hybrid 1.0, reset 1.0, static 1.0}`; `system_ttr_mean = 0.0` for every arm by vacuity; `max_censoring_allowed 0.999`. The run stopped itself | — |
| 6 | Failure 2b — the repair: `restricted_ttr = min(TTR, τ)` with a paired placebo | `results/manuscript/h1_h3_originales_v3/result.json` `39061791dd37eef4` | τ = **1,344 h**; f2 range check: absorbed 0.8517, censored-at-τ 0.0719; f3: **426** placebo episodes, every one checked shock-free. Levels 75.7 / 149.75 / 201.69 h; hybrid−static **+125.99 h [+98.35, +154.54]**, 960 cells, Holm p = 0.0. Mechanism is absorption: **875/960** vs **755/960**, not restoration speed. Artifact field `endpoint_redefinition_declared` timestamps the redefinition *before* the preregistration | **F2 (new)** censoring/absorption; **T4** levels + contrasts |
| 7 | **Failure 3 — the normaliser leaks** | `results/twin_surface_v2/result.json` `cbefeb716d1eda3f`; effect size in `results/garrido_normaliser_audit_v3/result.json` `fd617753949947e6` | Twin-surface f6: perturbing only never-run tail cells changes the oracle-normalised search path in **6/6** contexts for both neuron arms and **0/6** under prefix. Cost of the leak: memory-vs-reset **+0.09015 [+0.07419]** under oracle vs **+0.06070 [+0.04556]** under prefix — a **48 % inflation, biased toward the promoted method**. Censoring on the secondary differs 0.056/0.153/0.222/0.611 across arms | **F3 = REUSE** `fig_a_leak`, repointed; **T5** oracle vs prefix panel |
| 8 | **Failure 4 — the headroom statistic is not reparametrisation-invariant** | `results/monotone_transform_family_v4/result.json` `e7514c85e7be3141` | Identity (= Garrido's published σ(Σ sign·a_x·ln x), γ=1): H_regime **0.019501 [0.017992]**, Holm p 1.0, below the 0.05 gate; on the 288 grid exactly **0**. Deciding transform `power(γ=19.95)`: **0.27764 [0.25713]**, Holm p 0.0 — **14.2×**, from a strictly monotone rescaling of the same 4,608 cells. K = 661, Holm over the whole family. Floor curve **0.3815 → 0.1311** across signal floors 0.80→0.99. Negative control f5: on a grid with one globally optimal configuration, **0** transforms pass | **F4 (new)** H* vs γ with gate line and identity marked |
| 9 | **Failure 5 — the score depends on the observer's step cadence** | `results/metric_audit/ret_cadence_corrective_v2/result.json` `03025ec734721ee8`; supporting `results/metric_audit/ret_defects_v1/result.json` `37208e3c141270a3`, `results/metric_audit/ret_repair_variants_v1/result.json` `b9ac35565f47cb9a` | One identical trajectory: ReT 0.004369 (1-step, 672 h) → 0.005623 (24 h) → 0.005981 (1 h), **37 % spread, monotone**, while fill 0.99650 and delivered 689,182 are identical and **RPj differs in 175 of 311** scored orders. Root cause named to line (`_op_down_since` reset at `supply_chain.py:1856,:1865` → `:5743` → `:5811`). After the immutable-onset repair: **0.004424198300 at all five cadences, spread exactly 1.0, 0 orders differing.** Adjacent: a 6 h change in one constant (`config.py:119`) moves ReT **221.6×** (0.004424 → 0.980513); 7 orders in 3,108 inflate an R2r family mean by 6 %, and all seven principled repairs flip that verdict (Δ +0.0120…+0.0131, CI above zero) while being exact no-ops in R1r (+0.000000) | **T6** five-cadence before/after |
| 10 | **Failure 6 — the replication unit was not the seed** | `results/determinism_diagnostic/result.json` `2126f260a20a9953`; `results/determinism_repair_control/result.json` `9e988da25ee8c9f5` | Diagnostic: three configurations all diverge (0.472 / 1.620 / 1.539), preregistered suspicion **refuted**, cause not named. Repair: vec envs seed only the first reset; `MFSCGymEnvShifts.reset` passed `seed=None` onward. Episode 2: 48.674 vs 51.820. At ~1,900 episodes/run that one argument was the whole **2.363**-point fixed-seed dispersion — larger than the **2.102** the bake-off showed *between* seeds. Controls: 5 episodes bit-identical, different seed → different tape, full 20k×8-env PPO **93.973236562416 / 93.973236562416, Δ = 0.0 exact**. Consequence: a reported premium of +1.44…+2.18 sat inside a ±2.4 noise band | **T7** diagnostic + repair controls |
| 11 | **The protocol** — the six falsifiers generalised, each with its literal `why_it_can_fail` | all six artifacts | This is the transferable contribution and the reason to cite the paper | **T8** protocol table |
| 12 | What survives the tests | `results/metric_audit/abandonment_v1` `48746e7c386782e5`; `results/endpoint_headroom_atlas/result.json` `14317d68c9bebf49` | Cobb-Douglas **passes** the abandonment test (`best_share = 0.5` = service, both regimes, `COBB_DOUGLAS_SURVIVES_THE_ABANDONMENT_TEST`). Seven-endpoint atlas: `flow_fill_rate` raw spread **0.539** and `lost_orders` **108 orders** of lever authority with H_regime **exactly 0** | **T9** seven-endpoint atlas |
| 13 | Limitations, 14 Conclusion | | | |

## 3. Figures

- **Reusable: 1 of 5.** `fig_a_leak` in `<HOME>/Projects/research/scres-ia/scripts/build_cie_outer_loop_figures.py:64` — repoint `twin_surface/result.json` → `twin_surface_v2/result.json`. Schema is identical; verified.
- **New: 4.** F1 abandonment inverted-U (ReT and fill on twin axes, forfeited rations annotated); F2 TTR censoring→absorption; F4 H_regime vs γ over the 661-transform family with the 0.05 gate and identity marked; and the five-cadence panel if you prefer it as a figure over T6.
- `fig_b_surface_gates`'s **right panel only** (H_regime against its bar) could seed F4's "before" state, but its left panel (separability by context) belongs to Paper 2. Recommend not sharing the figure across papers.

## 4. Acceptance risk — honest

**What a C&IE reviewer attacks first, in order:**

1. **"This is a corrigendum on one reconstruction of one thesis."** *Highest desk-reject risk.* The answer is §11: five of six failures are properties of the metric definition or of ordinary simulation practice, not of this chain. Censoring is in the workbook formula (reproduction verified against 47,546 rows with no formula discrepancy); TTR non-identification is a property of the *recurrent risk regime*, stated literally in the artifact's `regime_note`; oracle normalisation is generic benchmarking practice; monotone non-invariance of a min–max headroom statistic is algebraic; cadence dependence is a bug *class* in DES metric accumulators. Only Failure 6 is genuinely local. **The paper must lead with the test, not the number** — if §11 does not land, this reads as a technical report.
2. **"Six items is a list, not a contribution."** The unification must be stated in the abstract: failures 1, 3 and 5 are *scored-population* endogeneity, 2 is *horizon* endogeneity, 4 is *scale*, 6 is *replication unit*. Every one is "a quantity the analyst thought was fixed is actually a function of the policy."
3. **"You found the metric broken after you got a null — motivated reasoning."** The strongest paragraph in the paper answers this: §9 reports that every principled repair *reverses* an R2r verdict **in our own favour**, and we decline to claim the reversal because the preregistered endpoint was `ret_excel` as specified. Reinforced by §4, where our own proposed mechanism (`branch_explains_the_u: false`) and our own proposed repair (`visible_clipped_0_1`, 14× worse) were both refuted by our own audit. Contract SHAs predate every run.
4. **"Where is the replacement metric?"** Deliberately absent. The paper diagnoses and shows one endpoint (Cobb-Douglas) that passes the abandonment test; it does **not** propose a new index. Proposing one without a service-loaded validation would be a fifth defect. Say so.
5. **Novelty vs Bruckler.** Bruckler is a taxonomy/review; this is measurement of what those metrics do when applied. That distinction has to be in the first paragraph.

**Verdict:** moderate. A genuine shot at major revision on first pass, contingent on the unification landing and on the co-authorship framing (below). This is the safety play as you said, but it is not a lock.

---

# PAPER 2 — retention before architecture

## 1. Title, thesis, highlights

**Title:** *Retention before architecture: retained search state, not the approximator class, carries learning across a discrete-event supply chain's design space*

**Thesis:** What operationalises the history-dependent attribute of supply-chain learning in a DES is search state retained between runs rather than the class of function approximator — and in the only prospective confirmation, the sole carrier that beats both cold start and a state-blind replay of its own marginals is a factor-level bandit, not a neural memory.

**Highlights** (pick 5 of these 6; all ≤85 chars, verified):

```
Six state-retaining methods take the top six places in a 15-method benchmark.        (77)
Only a factor-level bandit beats both cold start and its own state-blind replay.     (80)
The neural carrier loses to its own marginal replay: -0.01178 [-0.01849, -0.00484].  (83)
A KAN searches worse than a parameter-matched MLP: +0.01037 [+0.00302, +0.01893].    (81)
Retention's advantage grows with successive disruptions: slope +0.0422 [+0.0347].    (81)
Learning enters the DES as an outer loop over runs, not inside the event kernel.     (80)
```

## 2. Section skeleton

| § | Content | Artifact + SHA-16 | Number | Figure/Table |
|---|---|---|---|---|
| 1 | Introduction: the gap in Garrido et al. 2024's own words — three named candidates (backprop NN, KAN, simulation-optimisation as a form of RL); Fig. 2 nodes ③ `Decision variables, ρ` ↔ ⑧ `Metric of SCRES`; L accumulates across **runs and configurations** | — | none | — |
| 2 | Formulation: the outer loop `x_k → DES(x_k) → y_k, d_1..d_4 → L_k → x_{k+1}`. Carries the permitted-vocabulary constraint verbatim | freeze §2 vocabulary table | none | **F1 (new)** loop schematic. Do **not** reuse `figure1_closed_loop.png` from the Program Q chassis — different loop |
| 3 | DES, design space, demand | `results/demand_process/result.json` `cb4f88398c4f93a4` | 288 base / 4,608 extended configurations, 6 contexts, budget 24. CV 0.0713, ACF1 −0.2283, 6.0 orders/week | **T1** |
| 4 | Endpoint and normaliser: primary `auc_regret_norm`, prefix normaliser | `results/garrido_normaliser_audit_v3/result.json` `fd617753949947e6` | `runs_to_within_1pct` censored 0.056/0.153/0.222/0.611 across arms → not comparable; cite companion paper for the twin-surface proof | one paragraph |
| 5 | Is there a search problem at all? | `results/surface_gates_v2/result.json` `5abd006f27be0d55` | Non-separable: held-out ΔR² 0.0719–0.1586, LCB above 0.05 in **5 of 6** contexts. H_regime **0.003802 [1.35e-05, 0.014856]** vs a preregistered 0.05 gate → `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`. This section **pre-empts** the "your memory doesn't adapt" attack by conceding it first | **F2 = REUSE** `fig_b_gates`, repointed to v2 |
| 6 | The ladder (development) | `results/search_ladder_v5/result.json` `f648a1da5aefaf2f` | Places 1–6 are exactly the six `memory_arms`: 0.04502, 0.05203, 0.06274, 0.08018, 0.08390, 0.08908; place 7 = `ucb1` 0.09655; `neuron_reset` falls to **12th** at 0.11274. Scope label `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` printed in the caption | **F3 = REUSE** `fig_c_ladder`, repointed to v5; **T2** 15-row ladder |
| 7 | Retention pays every family | same, `per_arm` | 6 transfer/no-transfer pairs (v5 adds `lookahead_kg` and `thompson`) | **F4 = REUSE** `fig_d_memory`, repointed and extended 4→6 pairs |
| 8 | The neuron is not special | same `vs_neuron_memory`; `results/ofat_lcb_reconciliation/result.json` | `ucb1_transfer` **−0.00701 [−0.02444, +0.01408]** = tie. `ofat_transfer` **+0.01071**, bounds **−2.761e-05** and **+3.565e-05** across two sealed artifacts scoring identical arrangements, positive in 65 % of 40 reseeds (B=50,000) → report as *indistinguishable*, cite both bounds, **never write "excludes zero"**. `neuron_reset` +0.06070 [+0.04556, +0.07997] | **T3** |
| 9 | **THE CONFIRMATION** (paper centre) | `results/grid_transfer_confirmation_v2/result.json` `7bc33823ccd90b5e` | `run_role: CONFIRMATION`, reserved virgin block 8200001–060, n=60, 288→4,608. ucb1 **+0.05744 [+0.04989]** vs cold and **+0.03073 [+0.01990, +0.04256]** vs state-blind marginal replay; neuron **+0.05439 [+0.04290]** vs cold but **−0.01178 [−0.01849, −0.00484]** vs its own marginal replay; gp −0.02160 [−0.03051, −0.01227]; ofat −0.02467 [−0.03258, −0.01666]. `transfers = {ucb1: true, neuron/gp/ofat: false}` | **F5 (new, most important figure in either paper)** 4 families × 2 counterfactuals, dumbbell with zero line |
| 10 | Why architecture is the wrong axis, part A — Fig. 5 as drawn is an identity | `results/garrido_fig5_surrogate/result.json` `58d4c8a071cec86a` | R² = 1.0, max identity error **3.22e-15**, `Re_RPj` 0.999999999999968, `Re_FRt` 0.999999999999999, three all-zero driver columns. And on task B2 — the activation question the figure actually poses — neither network beats the logistic baseline by one between-fold SD (backprop 0.7178, KAN 0.7711, linear 0.7111, sd 0.0824): `verdict.B2_activation = {backprop: false, kan: false}` | **T4** |
| 11 | Part B — parameter-matched bake-off | `results/surrogate_architecture_bakeoff/result.json` `f96e5b6ff0489932` | KAN 532 / MLP 529 params. `kan_minus_matched_mlp` **+0.01037 [+0.00302, +0.01893]**, p = 0.0012, lower-is-better → interval entirely against the KAN. Best arm is the **5-parameter** unit at 0.05203; `random` at 0.13979 with `best_vs_random` LCB −0.1043 proving the harness separates skill | **F6 (new)** quality-vs-cost, *only if* timings are added (see 5b) |
| 12 | The learning curve | `results/manuscript/h2_learning_curve/result.json` `2894e525dc360c8f` | Slope **+0.042201 [+0.034664, +0.049922]**, n=120; null control (random − OFAT) **−0.005088 [−0.015570, +0.005658]** crosses zero, so the trend is not rising context difficulty; the exact zero at context 1 is structural | **F7 (new)** advantage vs ordinal with fitted slope and null control |
| 13 | Physical validation, boundaried | `results/garrido_h2_h3_confirmation_v1/result.json` `bc375d3021b64d10`, receipt `d4305bcf6bf5209d` | Six panels confirmed, Holm p **4.76e-17 … 1.90e-15**, 12 virgin confirmation tape roots, `development_roots_opened: false`. **First sentence must state** `learners_authorized = false`, `architectural_claim_authorized = false` | **T5** |
| 14 | Limitations | see 4 below | | |
| 15 | Conclusion | | | |

The four draft hypotheses land as: **H1 supported** +126.0 h [+98.4, +154.5] by absorption 875/960 vs 755/960; **H2 supported** slope +0.0422; **H3 not supported** (sign the wrong way, CI crossing zero, 78.75 % of cells deploying different configurations, so "no effect", not "no estimand"); **H4 measured** +0.06070 [+0.04556]. Three of four. Do not round to four. **Note the cross-paper dependency:** H1's endpoint *is* Paper 1's repair (`restricted_ttr`), so H1 is only citable in Paper 2 if Paper 1 has established it — this is an independent argument for the submission order below.

## 3. Figures

- **Reusable: 3 of 5** — `fig_b_gates` → `surface_gates_v2`, `fig_c_ladder` → `search_ladder_v5`, `fig_d_memory` → `search_ladder_v5`. Schemas verified compatible. Two code changes beyond paths: `LABEL` needs the four new v5 arms (`lookahead_kg`, `lookahead_kg_transfer`, `thompson`, `thompson_transfer`), and the hardcoded `MEMORY_ARMS` set at line 51 should read the artifact's own `memory_arms` list instead.
- **`fig_e_delta_efficiency` is NOT reusable as drawn.** It loads `results/search_surrogates/result.json` (`e9ee4d7d64140f63`), whose arms are 369- and 380-parameter surrogates — i.e. exactly the artifact whose parameter matching drew the objection the bake-off was built to answer (`f1`: "David's objection was 532 parameters against 31 called matched"). The canonical A5 bake-off has `parameters` and `percent_of_ceiling` but **no timing fields**. Either add timings (minutes; the artifact's own `no_gpu` note says CPU is enough) or drop the efficiency claim.
- **New: 4** — F1 loop schematic, F5 confirmation dumbbell, F7 learning curve, F6 conditional on timings.

## 4. Acceptance risk — honest

1. **"Your demand process is nearly deterministic; the null is an artifact."** *The exposure.* You now have a measured answer rather than a concession: `results/demand_process/result.json` gives weekly CV **0.0713**, lag-1 ACF **−0.2283**, exactly 6 orders/week, and **24.8 %** of weeks over one-shift capacity. So variation exists and bites at the capacity boundary, but it is *negatively* autocorrelated — it offers little state to condition on, which is a structural reason a state-blind constant is hard to beat *from the demand side*. That defends the claim in this process; it does **not** defend a general claim, and the paper must say so and name Holt exponential smoothing with Monte-Carlo α/γ (Garrido's own IJPR 2024 method) as the successor design. Expect the first-round revision request to be exactly this.
2. **"This is all development on burned tapes."** Seeds 5300001–012 appear in `search_ladder_v5`, `garrido_normaliser_audit_v3`, `surrogate_architecture_bakeoff`, `twin_surface_v2`, `monotone_transform_family_v4` and `endpoint_headroom_atlas`. Under your own R2 rule that is **one** evidential unit, not six. The paper must state this itself, and structure so that the *one* CONFIRMATION on virgin block 8200001–060 is the claim and everything else is captioned as development. Reviewers rarely see a paper concede this; conceding it is the strongest available move.
3. **"H_regime fails your own 0.05 gate — so what is the search buying?"** Sharpest attack. Answer must be in the abstract, not §14: *search-transfer value ≠ operational adaptation value*. Memory avoids **re-discovering** a near-common posture across contexts; it is not shown to adapt policy materially per regime. If the abstract doesn't say it, the referee will.
4. **"You are attacking a co-author's figure."** §10 says Fig. 5 as drawn has nothing to learn. Framed as self-correction with Garrido as an author, it is a strength; framed any other way it is a landmine. Human-gated.
5. **"KAN is interpretable."** Forbidden by the freeze: one partition, no CV, no shape stability across folds; the curves are response slices, not internal edge functions. Say so before a reviewer does.
6. **A negative headline about neural methods** in a venue that publishes a lot of successful deep-RL applications. The novelty that survives this is the **state-blind marginal replay counterfactual** — the falsifier that separates "I transferred structure" from "I revisited configurations that already worked." A referee has almost certainly not seen it. That single idea should be the abstract's second sentence.

**Verdict:** harder than Paper 1. Realistically reject-or-major-revision on first pass, with the revision request being the demand process — a six-month item, not a revision item.

---

# 5. What is genuinely missing before submission

### (a) Analyst work — no new simulation

**Both papers**
1. **`ENMIENDA_2` to the claim freeze.** Fourteen artifacts Paper 1 depends on are **not in the freeze's citable table**, and the freeze's own rule says a number without a row does not circulate. Rows needed (SHA-16 computed today): `contention_headroom_v1_2` `a47e89831fe6f130`, `contention_headroom_v1_3` `60537d88bbc2b1a9`, `metric_audit/abandonment_v1` `48746e7c386782e5`, `manuscript/h1_h3_v1` `12abb332afe73c85`, `manuscript/h1_h3_originales_v3` `39061791dd37eef4`, `manuscript/h2_learning_curve` `2894e525dc360c8f`, `twin_surface_v2` `cbefeb716d1eda3f`, `surface_gates_v2` `5abd006f27be0d55`, `monotone_transform_family_v4` `e7514c85e7be3141`, `metric_audit/ret_cadence_corrective_v2` `03025ec734721ee8`, `ret_defects_v1` `37208e3c141270a3`, `ret_repair_variants_v1` `b9ac35565f47cb9a`, `determinism_diagnostic` `2126f260a20a9953`, `determinism_repair_control` `9e988da25ee8c9f5`, `endpoint_headroom_atlas` `14317d68c9bebf49`, `demand_process` `cb4f88398c4f93a4`.
2. **Resolve the A7 conflict.** The freeze cites `results/surface_gates/result.json` (`954ac48301ff1234`) with H_regime 0.003802, LCB95 **1.08e-16**, UCB95 **0.014413**. `results/surface_gates_v2/result.json` (`5abd006f27be0d55`) gives the same point estimate but LCB95 **1.352e-05**, UCB95 **0.014856**. The figure script can only point at one. Declare which and supersede the other.
3. **Repoint `<HOME>/Projects/research/scres-ia/scripts/build_cie_outer_loop_figures.py`:** line 66 `twin_surface` → `twin_surface_v2`; line 92 `surface_gates` → `surface_gates_v2`; lines 135 and 160 `search_ladder_v2` → `search_ladder_v5`; extend `LABEL`; replace the hardcoded `MEMORY_ARMS` with the artifact's own list; decide `fig_e`.
4. **Parameterise `<HOME>/Projects/research/scres-ia/scripts/build_submission_a_cie_review_bundle.py`** — `PAPER_ROOT` is hardcoded to `papers/submission_a_program_q`; both new paper roots need it as an argument, and `IDENTITY_TOKENS` needs any new author's tokens added before the guard means anything.
5. **Branch/tag decision.** Science is on `codex/expanded-contract-comparators-v2`, the chassis on `origin/codex/submission-a-program-q`, `main` far behind. The checklist item "clean-room build succeeds from the tagged release" cannot be satisfied while they are separate. Per R3, never hand-write the drift number.

**Paper 1 only**
6. Two claims in `docs/ATLAS_DE_METRICAS_DE_RESILIENCIA_2026-08-06.md` I could not trace to a sealed field and which must be located or dropped: (i) "the cadence **inverts** the recommended policy (0.9 under `sim.run()`, 0.1 under daily stepping)"; (ii) the "18 cells" stability claim for `ret_excel` always choosing an extreme.
7. **Labelling precision.** The headline abandonment table's ReT row lives in the artifact under the keys `ret_excel_risk_conditional` / `ret_excel_visible_clipped_0_1`, not a key literally named `ret_excel`. A referee who opens the JSON will find the mismatch. Label exactly as the artifact does.

**Paper 2 only**
8. Import A10 (`garrido_h2_h3_confirmation_v1`) into the canon with its literal boundary — ordered in `ENMIENDA_1` §E6 item 3 and not done.
9. Write the `ofat_transfer` caveat verbatim, citing both sealed bounds.

### (b) Compute

- **Paper 1: none.** All six claims are already sealed. Optional and *not* recommended before submission: a service-loaded replacement endpoint. Diagnose plus "here is one endpoint that passes the test" is the defensible scope; proposing a new index without validation is a seventh defect.
- **Paper 2: one small job** — add `delta_efficiency` timing fields to the 529/532-parameter matched pair inside `surrogate_architecture_bakeoff`. CPU, minutes. Without it, drop F6 and the efficiency claim.
- **Explicitly not runnable:** any new virgin block. `research/seed_custody_registry.json` is `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED` with 94 uninventoried branches. Neither paper may open a new confirmation.
- **Holt/MC-α demand:** a new prospective study with a new preregistration, not a pre-submission item. Name it in Limitations; do not hold either paper for it.

### (c) Human-gated

1. **Garrido's written consent to the framing** — the hardest gate, and the reason to start today. `papers/submission_a_program_q/GARRIDO_FACE_VALIDATION_REQUEST.md` covers DES face validation. Paper 1 needs *more*: written agreement to co-author a paper that audits his metric, and Paper 2 needs the same for the Fig. 5 identity result. Without it, neither is submittable at any technical readiness.
2. **Author order, CRediT roles, affiliations, corresponding author** — every field in `TITLE_PAGE.tex` is `PENDING`, × 2 papers.
3. **Funding sources and sponsor role; competing-interest declarations-tool `.doc/.docx`; generative-AI disclosure wording** — all three open in `RELEASE_AND_SUBMISSION_CHECKLIST.md`.
4. **The five domain questions.** Question 2 is load-bearing for Paper 1: *is there a binding worst-product / worst-theatre floor, or is doctrine evaluated on average resilience?* If the floor binds, the abandonment finding is a doctrinal violation and the paper is materially stronger. Under R4 the answer defines a new contract either way — it never retroactively rescues anything.
5. **Security/permissions review of public military-model wording and assets** (`SECURITY_AND_DISCLOSURE_CHECKLIST.md`).
6. **Archival DOI + anonymous-review routing decision; independent language review; Q1 status recheck on the submission date.**

---

# 6. Submission order — recommendation

**Paper 1 first, alone. Paper 2 second, citing Paper 1 as a companion under review.** Do not submit simultaneously.

Why:

1. **Paper 1 needs zero new compute and Paper 2 needs one small job.** Sequencing costs nothing.
2. **Paper 1 is immune to Paper 2's biggest exposure.** The demand-process objection cannot touch a paper whose claims are about censoring, normaliser leakage, reparametrisation and cadence — those hold under any demand process.
3. **Paper 1 buys Paper 2 its defence.** Paper 2's null is far more defensible once "we measured it on the standard endpoint, showed that endpoint rewards abandonment, and measured it on six others besides" is a citable paper rather than an appendix. And Paper 2's H1 result depends on `restricted_ttr`, which Paper 1 establishes — that dependency is one-directional and settles the order on its own.
4. **Paper 1 needs the least from Garrido** — face validation of the reconstruction, not endorsement of a learner claim.
5. **A first submission is a cheap test of the chassis.** Per the guide audit, C&IE's submission-to-first-decision is ~4 days. A desk reject on Paper 1 tells you whether the chassis, anonymity guard and bundle builder work before you spend them on the paper you actually care about.
6. **Simultaneous submission invites the salami charge.** Two manuscripts sharing one DES with one confirmed result, likely landing with overlapping referees, is exactly the pattern editors flag. Sequential with an explicit companion cross-reference is the standard defensible move.

**The cost, stated:** Paper 1 gives away part of Paper 2's limitations section in public. Mitigate by keeping the evidence sets disjoint — Paper 1 reports **no** AUC, regret, ladder or learner number; Paper 2 reports **no** ReT-defect number. The only shared material should be the `calibration_provenance` block and the demand characterisation, and both should be cited across rather than duplicated.

---

### Critical Files for Implementation

- `<HOME>/Projects/research/scres-ia/scripts/build_cie_outer_loop_figures.py` — the figure pipeline; four artifact paths to repoint, `LABEL`/`MEMORY_ARMS` to extend, `fig_e` to decide
- `<HOME>/Projects/research/scres-ia/docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07_ENMIENDA_1.md` — the frozen claim wording, permitted/forbidden vocabulary and custody rules both manuscripts must obey; successor `ENMIENDA_2` must admit the fourteen Paper-1 artifacts
- `<HOME>/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json` — the only prospective confirmation in either paper; Paper 2's centre and its new F5
- `<HOME>/Projects/research/scres-ia/results/metric_audit/abandonment_v1/result.json` — Paper 1's only virgin-seed replication of the headline abandonment result, plus the refutation of our own proposed mechanism
- `<HOME>/Projects/research/scres-ia/scripts/build_submission_a_cie_review_bundle.py` — the anonymity-guarded bundle builder; `PAPER_ROOT` and `IDENTITY_TOKENS` must be parameterised for two new paper roots
