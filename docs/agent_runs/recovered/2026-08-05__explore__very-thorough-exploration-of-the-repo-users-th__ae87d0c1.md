# Explore — Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `ae87d0c15e99cc52c`
- **session**: `091a3905-ebf9-481b-beaa-fc646c3902d5`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-05T18:01:18.633Z`
- **effort**: `xhigh` · **version**: `2.1.219`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/091a3905-ebf9-481b-beaa-fc646c3902d5/subagents/agent-ae87d0c15e99cc52c.jsonl`
- **turns**: 153 · **assistant messages**: 2

## Task given

```text
Very thorough exploration of the repo <HOME>/Projects/research/scres-ia.

I need a complete map of the MANUSCRIPT state and the CLAIM LEDGER, to plan a submission to Computers & Industrial Engineering.

Find and report with exact file paths:

1. Every manuscript draft or paper skeleton in the repo: look in papers/, deliverables/, docs/, paper_results_package/. For each, report the title, target journal, current section structure, and how complete it is (word counts, TODO markers).

2. The claim ledger / current state documents: docs/GARRIDO_WRAP_CLAIM_LEDGER*, docs/GARRIDO_WRAP_CURRENT_STATE*, docs/GARRIDO_WRAP_ENVIRONMENT_REGISTRY*, docs/PAPER2_EXHAUSTION_CERTIFICATE*, docs/PROMISING_LANES_REGISTRY.md. Quote the status of every lane and every claim that is currently ADJUDICATED / SUSTAINED / positive.

3. Every POSITIVE result the project has (things that are not STOP/HALTED/negative). I specifically want: the H3' result (search-cost variance, LCB95 +2.3491, n=120), the Alzheimer memory-neuron result, the G3-obs STRUCTURED_CONTROL_SUFFICES result, and the DES validation result (-4.43% on ECS vs Garrido's thesis). For each, report the doc, the result json, the headline numbers and the exact scope/caveats.

4. The figures/tables assets: is there any figure generation code (matplotlib) or a figures directory for a paper?

5. The DES model description document that was delivered to Garrido (a DOCX describing the DES model). Where is it, what does it cover?

6. Any document describing what we would tell Garrido, attribution matrices, or author/contribution planning.

Report exact paths, quoted status strings and numbers. Be specific and complete.
```

## Final answer

## 1. Manuscript drafts and paper skeletons

### A. The compiled Elsevier manuscript (most complete artifact in repo)

`<HOME>/Projects/research/scres-ia/docs/manuscript_current/submission/elsevier/`

- **`main.tex`** — title: *"When Is Reinforcement Learning Warranted in a Supply Chain DES? Decision-Rights Eligibility, Comparator Design, and Oracle-to-Policy Conversion"*. Uses `elsarticle` with `\IfFileExists` fallback → **Elsevier-formatted, i.e. already targeted at a C&IE-family journal**. Compiled PDF exists: `main.pdf` (661 KB, dated 9 Jul).
- Section files actually `\input`-ed by `main.tex`:
  - `sections/01_introduction.tex` (1,223 w)
  - `sections/02_related_work.tex` (959 w) — 5 subsections incl. "Gap: action-space alignment and bottleneck authority"
  - `sections/03_eligibility_framework.tex` (556 w) — `\section{Decision-right eligibility framework}`
  - `sections/04_mfsc_case.tex` (448 w)
  - `sections/05_results.tex` (919 w) — subsections: comparator scope reverses headline / decision families fail at different rungs / Program G terminal spatial audit / resilience functionals induce different rankings / Program H belief-state audit
  - `sections/06_discussion.tex` (521 w)
- **Orphaned older-version sections still in the same directory but NOT included** (this is the previous, much larger draft): `sections/03_methodology.tex` (2,943 w), `sections/04_results.tex` (6,148 w, 10 subsections incl. `4.1 DES validation`, Track A negative, Track B positive, causal ablation, same-contract static challenge), `sections/05_discussion.tex` (1,590 w), `sections/06_conclusion.tex` (467 w).
- Total across all .tex: **16,966 words**; the currently-included spine is ≈5,800 words + front matter.
- **TODO markers (exact strings), all in `main.tex`:**
  - line 52: `[TO COMPLETE before submission: author names and roles, e.g. Conceptualization; Methodology; Software; Validation; Formal analysis; Writing -- original draft; Writing -- review and editing; Supervision.]` (this is the CRediT block)
  - line 62: `[TO COMPLETE: funding statement, ...]`
  - line 73: `\emph{[TO UPDATE: re-pin to the camera-ready commit before final submission]}` (currently pinned to commit `10c7de9`, branch `codex/garrido-replication-experiments`)
  - line 80: `[TO COMPLETE: acknowledge A. Garrido for the thesis materials and ...]`
- Abstract headline numbers already in it: PPO advantage `+0.000486` vs restricted static frontier, reversing to `-0.000018049`, CI95 `[-0.000028615,-0.000008087]`; finite-convoy oracle headroom `+0.02212`; 400 locked tapes spatial audit.
- A duplicate/stale build tree exists at `<HOME>/Projects/research/scres-ia/tmp/manuscript_compile/` (older section set: 01,02,03_methodology,04_results,05_discussion,06_conclusion + main.pdf).

### B. Paper 2 skeleton (markdown, complete-ish, different paper)

`<HOME>/Projects/research/scres-ia/papers/paper2/`
- `01_introduction_draft.md` (622 w) — **Working title:** *"Learning Adaptive Control Without a Neural Premium: Exhaustive Open-Loop, Belief-MPC, and Recurrent-RL Comparisons in a Full-DES Military Supply Chain"*. Status line: `calibration-evidence framing (Program Q replication pending)`. References `docs/PAPER2_CLAIM_LADDER_2026-07-18.md` — **that file does not exist in the repo** (dangling pointer).
- `02_methods_draft.md` (1,156 w), `03_results_draft.md` (814 w), `04_discussion_draft.md` (789 w)
- `results_table.md` / `results_table.json` / `build_results_table.py` — machine-generated master table with per-cell source hashes; L1 `H_PI = 0.15151` (LCB95 0.11562), L2 classical `H_obs` LCB95 +0.06595/+0.04303/+0.05860, L3 learned `H_OL` +0.07576/+0.06261/+0.10455, L4 neural premium `Δ_N` all negative. Last row: `| Q prospective replication | all | PENDING | contract frozen: N=128/cell, block 7490001+ |`.
- **No TODO/TBD markers anywhere in papers/paper2** — the drafts are prose-complete but the paper itself is not authorized (see exhaustion certificate below).

### C. v0 draft fragments (the Garrido/David shared Word document)

- `<HOME>/Projects/research/scres-ia/docs/manuscript_draft/section_3_3_rl_formulation.md` (8.9 KB) — "*replaces the placeholder text in the v0 under 3.3 Hybrid Simulation Neural Model*", written 2026-03-24, 15-D observation (now obsolete).
- `<HOME>/Projects/research/scres-ia/docs/manuscript_draft/section_4_2_hybrid_results.md` (5.7 KB) — "*replaces the placeholder text in the v0 under 4.2*".
- `<HOME>/Projects/research/scres-ia/docs/for_team/v0_additions_for_thom.md` — paste-in blocks; explicit note: "*Do NOT change David's sections (3.3, 4.2) or Garrido's framing (Intro, Hypotheses)*".
- `<HOME>/Projects/research/scres-ia/docs/manuscript_notes/` — 17 notes (source-of-truth for §4.2/§4.3, claim language, backlog).

### D. The C&IE writing contract (authoritative outline, no prose)

`<HOME>/Projects/research/scres-ia/docs/CIE_MANUSCRIPT_REWRITE_MAP_GARRIDO_WRAP_2026-08-01.md`
- `**Paper lane:** Garrido–WRAP/v0`, `**Target:** *Computers & Industrial Engineering*`, `**Status:** HOLD_RESULTS_BEFORE_MANUSCRIPT_CLAIMS`
- Contains a 7-item evidence order, a "Claim language currently permitted" list and a "Claim language prohibited until gates close" list, plus a "Required figures/tables after the 288 rerun" list (6 items).
- Final line: *"The v0 DOCX should be rewritten only after the pending H3' and DES-288 artifacts are sealed. Until then, this file is the authoritative outline for §4.2 and §4.3."*

### E. `paper_results_package/` — historical (April 2026), superseded

- `README.md` (2,216 w) — `**Target:** IJPR (primary), C&IE (backup)`; title *"When Does RL Help? Action-Space Alignment as a Prerequisite for Adaptive Supply Chain Resilience Control"*.
- `PUBLISHABILITY_ASSESSMENT.md` (1,465 w) — carries a 2026-07-01 header note: "*This file is historical... Track B is now treated as an 8D operational extension, not '7D'; 'perfect fill' is not the headline*".
- `BRIEF_PARA_GARRIDO_Y_DAVID.md` (1,014 w) — see §6.

---

## 2. Claim ledger / current-state documents

### `docs/GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md`
Frontier of claims for the active Garrido–WRAP/v0 article. Every row verbatim:

| claim | artifact | **estado** |
|---|---|---|
| Cobertura de fuentes | `results/garrido_wrap_source_audit/result.json` | `DEVELOPMENT_SOURCE_AUDIT` |
| Figura 5 literal | `results/garrido_wrap_q1/result.json` | `DEVELOPMENT_FIG5_SURROGATE` |
| Q1 held-out | `results/garrido_wrap_q1/result.json` | `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL` |
| Q2 smoke | `results/garrido_wrap_q2_smoke_2016h/result.json` | `DEVELOPMENT_Q2_CLOSED_LOOP` |
| Replay thesis90 | `results/garrido_meta_learner_thesis90_v2/result.json` | `SURFACE_REPLAY_ONLY` |
| Q2 DES-288 | `results/garrido_meta_learner_v2/result.json` | `ARTIFACT_PRESENT_CANONICAL_CUSTODY_PENDING` |
| G3a v2 asimetría N=2 | `docs/PREREGISTRO_G3_ASIMETRIA_V2_2026-08-01.md` + `contracts/g3a_asymmetric_claimants_v2.json` | `DESIGN_ONLY_NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT` |
| E1 neural headroom | `results/garrido_neural_headroom_gate_v1/result.json` | `HOLD_E1_PLACEBO_NOT_OPENED` |
| CSSU Gate A | `results/garrido_cssu_liveness_gate_v1/result.json` | `GATE_A_PASS_GATE_B_HOLD` |
| Contención constante | `results/metric_audit/contention_service_first_v2/result.json` | `NEGATIVE_EXTENDS_TO_THE_SOUND_ENDPOINT` |
| Expedición | `results/sensitivity/expedite_headroom_v2/result.json` | `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` |
| H1 original | `docs/RESULTADO_H1_H3_V2_2026-08-01.md` | `NO_EVALUABLE` |
| H1′ | same artifact | `DEVELOPMENT_SIGNAL` (report as `service_loss_auc`) |
| H2 | meta-aprendiz v1 retirado | `RETIRED_LEAKAGE` |
| H3′ | `results/garrido_meta_learner_h3power_local/result.json` + `..._vps/result.json` | `ARTIFACTS_PRESENT_MERGE_PENDING` |
| H4 | Q2 retained–reset | `PENDING_CONFIRMATION` |
| PPO/RecurrentPPO de carriles previos | `docs/PAPER_FINDINGS_REGISTRY.md` | `RETIRED_CONTEXT_DRIFT` |
| CSSU con acción muerta/fungible | preauditoría | `RETIRED_PHYSICS_OR_INTERFACE` |

Custody rule: *"Un claim sólo pasa a manuscrito cuando su artefacto contiene contrato, hash de código, semillas, falsadores, estado y ruta de reproducción."*

**IMPORTANT — this ledger is one day stale.** It was written 2026-08-01 18:11 and the H3′ row (`ARTIFACTS_PRESENT_MERGE_PENDING`) was superseded on 2026-08-02 by `H3_PRIME_SUSTAINED_AT_N120` (see §3).

### `docs/GARRIDO_WRAP_CURRENT_STATE_2026-08-01.md`
- `El artículo activo es Garrido–WRAP/v0 para *Computers & Industrial Engineering*. Program Q es un carril separado y no aporta claims a este manuscrito.`
- `Estado global: HOLD_WRAP_BEHAVIORAL_FIDELITY / DEVELOPMENT_ONLY`; contract `garrido_wrap_scres_ai_v1`; `thesis_1to1` frozen.
- Retired figures explicitly named: `+6.31`, `+5.18`, `+12.31` and the old H2 curve — "*no son evidencia. El motivo es la fuga de drivers al rankear candidatos no ejecutados.*"
- H3′ seeds: local `6000001…6000090`, VPS `6000091…6000120`; VPS slice at `HOLD_SOURCE_AUDIT`.

### `docs/GARRIDO_WRAP_ENVIRONMENT_REGISTRY_2026-08-01.md`
Per-environment status (verbatim):
- **E0 / thesis-native WRAP** → `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`; global behavioral fidelity still `HOLD`
- **Figura 5 literal** → invalidated as a planning task; `drivers → ReT` is identity/leakage
- **Replay thesis90** → `SURFACE_REPLAY_ONLY`
- **DES-288 Q2** → `ARTIFACT_PRESENT_CANONICAL_CUSTODY_PENDING`
- **E1 / CSSU split** → Gate A `PASS`; Gate B `HOLD`; observed headroom `0` but neural placebo not opened
- **Expedición** → `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST`
- **Program O** → high development headroom, own physics/metric; does not transfer to WRAP
- **Program Q / Track A/B/PPO** → separate lanes; contribute no claims
- **G3a prospectivo** → design-only; `NOT_AUTHORIZED_UNTIL_SUBMISSION_A_RECEIPT`
- **E2 prospectivo** → not opened; only a new preregistration can open it
Plus a 6-condition list of where a neural premium could exist.

### `docs/PAPER2_EXHAUSTION_CERTIFICATE_2026-07-15.md`
`**Status:** PAPER2_SEARCH_BOUNDARY_CERTIFIED__NO_POSITIVE_INSTANCE__EXACT_REOPENERS_NAMED__DISCRETE_DESIGN_SCOPE`
- Carries a self-**RETRACTION** of the earlier "hypercross/OAT" claim.
- Terminal finding: *"no decision contract converts perfect-information headroom into deployable, resource-honest, out-of-sample-stable observable value... Zero PPO seeds were wasted."*
- Claim boundary table: full-DES H_PI established **true** (Program O only, 0.152); H_obs established **false**; learner authorized **false**; Paper 2 confirmed **false**; Paper 3 authorized **false**; positive instance found **false**.
- Decisive result §2.6: 45 profiles × 18 postures × 6 seeds = 4,860 evaluations, `max H_profile_safe 6.93e-05 vs bar 0.01 → 144× below`, `0/63 cells`, `passing_doors=[]`. ReT degrades 0.53 → 0.20 under escalation but zero tailoring headroom.
- Only reopeners: **Q11/R09 mission expiry (OPEN — strongest)**, **Q6/Q7 integrated shared resource (OPEN)**, Q13 Program O construct (OPEN, ceiling only), Q14 freight economics (OPEN, not decisive), Q2/R03 route recourse (`BAR RAISED`).

### `docs/PROMISING_LANES_REGISTRY.md` (86 KB, 2026-07-18)
Living tracker. Only one row still carries a non-terminal status string: lane ⭐1 `continuous_its × v6 × ReT_excel_delta × war φ4/ψ1.5` = `PROMISING BUT NOT CONFIRMED` (rigorous fixed-init 3-seed rerun failed: Excel 0.00186 < 0.00193, CVaR 1.65e9 > 1.59e9). The tail of the file is lane ⭐17 (preventive/temporal-commitment), whose **final identified status** is: the whole `+0.0285` increment is learned per-op *constant reserve levels*; `schedule_value NULL [−0.0009,+0.0012]`, `exante NULL`, `reactive NULL`; fixed-posture 8D `0.340605` vs dynamic 11D `0.340164` (`+0.000440`, CI95 `[−0.000799,+0.001680]` — equivalent); fixed − no-buffer `+0.028928`, CI95 `[+0.016283,+0.041574]`, 5/5 seeds. Conclusion: *"Dynamic buffer control is unnecessary and mildly harmful."*

### Related, also load-bearing
- `docs/GARRIDO_WRAP_RECONCILIATION_2026-08-01.md` — `**Status:** RECONCILIATION_COMPLETE_FASE4_90_DONE_288_PENDING`; evidence-disposition table; Cobb–Douglas boundary; 5 immediate gates.
- `docs/GARRIDO_WRAP_IMPLEMENTATION_STATUS_2026-07-31.md` — `HOLD_WRAP_BEHAVIORAL_FIDELITY` / `DEVELOPMENT_ONLY`; Q1 table: Linear R² 0.9697, MLP 0.9863 (+0.0166 [0.0048,0.0283]), KAN 0.9913 (+0.0216 [0.0042,0.0390]) — neither reaches `SESOI_R2 = 0.05`.
- `docs/GARRIDO_WRAP_SCRES_AI_CONTRACT_V1.md` — `IMPLEMENTED_SCAFFOLD_HOLD_WRAP_BEHAVIORAL_FIDELITY`; defines H1–H4.
- `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` — **the single best "what can we say for C&IE" table** (see §3).
- `docs/PAPER2_PAPER3_BOUNDARY_CERTIFICATE_AND_STRATEGY_2026-07-13.md` — `OPEN_ACTIVE_BOUND_REQUIRED__MTR_KEY_V4_SIGNED_REDUCED_GATE_PENDING`; `PAPER3_NOT_AUTHORIZED__DEPENDENCY_PAPER2_LEARNED_VALUE_UNMET`.

---

## 3. Every POSITIVE result

### 3.1 H3′ — search-cost variance, SUSTAINED at n=120 ✅ (the strongest positive)

- **Doc:** `<HOME>/Projects/research/scres-ia/docs/RESULTADO_H3_ADJUDICACION_FUSION_2026-08-02.md`
- **Artifact:** `results/garrido_h3_merge_adjudication/result.json`, `self_sha256 = 1ac02efa1618e5a9cd4d910e87819c5c6da25fc7b37a2654214323338d3a41bf` (git-tracked)
- **Contract:** `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md` (`576d02b5de7609eb…`); runner `scripts/adjudicate_h3_merge_v1.py`; **no new seeds**
- `claim_status = H3_PRIME_SUSTAINED_AT_N120`
- `reading_rule = "LCB95 > 0 -> sustained; LCB95 <= 0 at n=120 -> refuted with power"`

| contrast | mean | CI95 | n |
|---|---:|---|---:|
| memory vs reset | **+9.314444444444444** | **[+2.3491250000000012, +16.34740972222222]** | 120 |
| memory vs OFAT (thesis design) | +16.220277777777778 | [+9.61070833333333, +22.740534722222215] | 120 |

Mean variance by strategy: `neuron_memory 44.264166…`, `neuron_reset 53.578611…`, `ofat 60.484444…`, `random 67.503333…`
Per-slice diagnostic (not the adjudication): local n=90 `+10.2659 [+2.3609, +18.1927]`; VPS n=30 `+6.46 [−7.3088, +20.3281]` — **the 30-seed slice alone does not reach**.
All 4 merge falsifiers pass: disjoint seeds (90+30), identical design, identical source (7 module hashes + entry script + contract), n=120 exactly.

**Scope / caveats (must be carried into the paper):**
- The estimand is `variance of search cost ACROSS CONTEXTS, per replicate; memory minus reset. NOT the Alzheimer mean (reset - memory in runs), which is a different quantity that this contract does not adjudicate` (verbatim from the JSON).
- Per `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §3: H3′ ≠ the v0 draft's H3. Draft H3 = variance of *performance/resilience*; H3′ = variance of *search cost*. "*En el manuscrito debe escribirse **H3′ sostenida**, nunca «H3 probada», y con su definición al lado.*"
- `f6 = DECLARED_REPLAY` on both slices → not an independent second confirmation.
- VPS original snapshot source identity "*no se demostró y ya no es reconstruible*".
- "*La fusión no autoriza nada sobre prima neural.*"
- Precursor doc where H3′ was **not** sustained at n=12: `docs/RESULTADO_H1_H3_V2_2026-08-01.md` (memory vs reset +7.86 `[−14.72, +28.31]`) — a power limitation, later resolved.

### 3.2 Alzheimer effect / memory-neuron — measured price ✅ (sealed, but development)

Three generations of this number; **do not mix them**:

| doc | artifact | `reset − memory` | CI95 | status |
|---|---|---:|---|---|
| `docs/RESULTADO_META_APRENDIZ_2026-07-31.md` | `results/garrido_meta_learner/result.json` (`230a0074a10f12ee…`) | **+6.31** | [+5.18, +7.49] | **RETIRED_LEAKAGE** — driver leak; explicitly withdrawn |
| `docs/RESULTADO_H3_AMBAS_REBANADAS_CONTRATADAS_2026-08-02.md` | `results/garrido_meta_learner_h3power_h3_contract_local_v2/result.json` (`e768e0f69f187272…`), n=90 | **+7.2704** | [6.7519, 7.7760] | sealed, development |
| same | `results/garrido_meta_learner_h3power_h3_contract_vps_v2/result.json` (`ad240fbb9dda507a…`), n=30 | **+7.6111** | [6.6110, 8.6556] | sealed, development |
| `docs/RESULTADO_WRAP288_RECONCILIACION_F3_F4_2026-08-02.md` | `results/garrido_q2_des288_reconciled_v2/result.json` (`67473d2590f03ee6…`) | **+7.9028** | [+6.8750, +8.9306] | `RUNTIME_F3_F4_RECONCILIATION_NOT_A_NEW_CONFIRMATION` |

All these JSONs carry `claim_status = ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE`.
Secondary estimands in the same artifacts: memory vs OFAT `+5.04 / +4.92 / +5.43 [+4.01,+6.78]`; memory vs random `+12.19 / +11.69 / +12.56 [+10.65,+14.56]`.
Original (retired) run-count table: memory 7.24 runs to 1% of optimum, OFAT 12.42, reset 13.54, random 19.54; 288 configurations × 6 contexts, 20,736 surface episodes.

**Scope / caveats:**
- `docs/RESULTADO_WRAP288_RECONCILIACION_F3_F4_2026-08-02.md` §5: "*El efecto Alzheimer sigue sin entrar al manuscrito por esta vía. Aquí vale como valor de desarrollo reconciliado; el citable con potencia es el de H3′ a n = 120.*" Block `5.300.001–12`, n=12, exploratory.
- `docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §4 walks back the H4 mapping: "**Apoyo estrecho a H4:** la memoria entre campañas reduce de forma medible el coste de encontrar una buena configuración. La resiliencia entregada no se midió como función de la historia."
- Notable methodological asset: the reconciliation doc documents a **tautological `f3` negative control** that was fixed and demonstrated to now fail on mutants (`negative_control_detected True`, `budget_tamper_rejected True`, `trace_tamper_rejected True`).

### 3.3 G3-obs — `STRUCTURED_CONTROL_SUFFICES_G3_OBS` ✅/❌ (two-sided, "both faces publishable")

- **Doc:** `<HOME>/Projects/research/scres-ia/docs/RESULTADO_G3_OBS_V2_2026-08-02.md`
- **Artifact:** `results/headroom/g3_obs_conversion_v2/result.json`, seal `317daf920579ec6e…` (git-tracked)
- Seed block `7.800.001–140` (70 development / 70 test disjoint); **8 falsifiers pass**, 3 guardrails pass; `primary_metric = worst_claimant_fill`; SESOI 0.010

Positive side — observable conversion exists:
| cell | `H_obs` mean | LCB95 | MDE(90%) |
|---|---:|---:|---:|
| `base` | +0.02074664585684144 | **+0.014726107824371892** | 0.009180290145707292 |
| `freq3_imp2` | +0.012880831201565662 | **+0.007287753081355917** | 0.008488645671320991 |

Placebos: `uninformed_placebo` −0.0049 / −0.0085; `wrong_claimant` −0.2449 / −0.1978.
Negative side — no residual over a 2-branch `if`:
| cell | `residual_over_simple` | CI95 |
|---|---:|---|
| `base` | −0.0022005394969836587 | [−0.004852304287817252, +0.00030231108258477706] |
| `freq3_imp2` | +0.0010 | [−0.0010560223247456392, +0.0031] |

Realism cost: 3-day delay `+0.0080/+0.0054`, noise σ=0.30 `+0.0052/+0.0033`, 14-day window `+0.0055/+0.0037`; 4 of 6 exclude zero.
Doc §6 (the C&IE framing): "*Eso responde la Q1 de Garrido con más precisión que «no hay prima»: **la propiedad que importa es el feedback dependiente del estado, no la capacidad del aproximador.***"

**Serious caveat — the formal custody adjudication:** `<HOME>/Projects/research/scres-ia/docs/RESULTADO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md` (+ `docs/CONTRATO_ADJUDICACION_ALCANCE_G3_OBS_2026-08-02.md`, receipt `results/headroom/g3_obs_conversion_v2/contract_scope_adjudication.json`, seal `eee480febd401873…`):
```
claim_status:     SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE
promotion_status: BLOCKED_NO_RETROACTIVE_RESEAL_AND_NO_CONTRACT_CONFORMITY
audit_status:     CONTRACT_SCOPE_ADJUDICATION_NO_NEW_SEEDS_NO_DES_RERUN
```
The run was sealed against the *old* contract (which declares seeds `5.200.001–016` and `lost_orders` margin 0.25) but executed the v2 block and margin 0.50. Prohibited: calling it a v2 execution, calling it fully confirmatory, calling it virgin/independent. It survives as **development evidence with a scope limitation**.
Also relevant: `docs/RESULTADO_AUDITORIA_G3_OBS_F2_2026-08-02.md`, `docs/PREREGISTRO_G3_OBS_V2_CONFIRMACION_2026-08-02.md`, `docs/REVISION_G3_OBS_CONFIRMACION_2026-08-02.md`.

### 3.4 DES validation −4.43% vs the thesis ECS ✅

- **Primary docs:** `<HOME>/Projects/research/scres-ia/docs/PRESENTACION_GARRIDO_2026-03-25.md` (lines 109–124, 368) and `<HOME>/Projects/research/scres-ia/docs/meetings/2026-03-26_GARRIDO_DEMO.md` (line 21) and `<HOME>/Projects/research/scres-ia/docs/REPRODUCIBILITY.md` (lines 46–49).

| Comparison | Hrs/yr | Our model | Thesis ECS | Relative gap |
|---|---|---|---|---|
| Thesis basis (official) | 8,064 | **733,621 rations/yr** | **767,592** | **−4.43%** |
| Gregorian basis (diagnostic) | 8,760 | 796,940 | 767,592 | +3.82% |

Deterministic Cf0 (S=1, no risks, Table 6.10): production 734,458/yr, delivery 733,621/yr, **fill rate 99.3%**, **41 backorders in 20 years**, weekly capacity ~17,948 vs demand ~17,500.

**Critical caveat — the ±15% threshold was retracted.** The March 2026 docs say "*El gap de −4.43% está dentro del umbral aceptado de +/-15%*", but `docs/REPRODUCIBILITY.md` (2026-07-09 provenance fix) states: "*the thesis's own historical calibration dispersion spans $-21.6\%$ to $+14.1\%$ — **there is no formal $\pm 15\%$ acceptance threshold in the source***". Same correction is baked into the figure code (`scripts/build_manuscript_figures.py:840-880`, docstring: "*provenance fix 2026-07-09*").

- **Data source:** `outputs/validation/validation_table_dual_basis.csv` (per-year, dual basis)
- **Figure:** `fig12_des_validation` (PDF+PNG) — panel (a) year-by-year fidelity with RMSE and avg gap; panel (b) degradation: deterministic 733,621 / fill 99.3% / 41 BO → current risk 677,750 / 68.3% / 1,825 BO → increased risk 549,250 / 45.6% / 3,132 BO.
- Broader validation state: `<HOME>/Projects/research/scres-ia/docs/DES_VALIDATION_DOSSIER.md` (Gate 0 v3 PASS, Gate 1 v3 complete, Gate 2 v3 `STOP_NO_DEPLOYABLE_ADAPTIVE_HEADROOM`; runtime freeze with SHA256s) and `docs/DES_MODEL_DOCUMENTATION.md` (**overall fidelity 97%, 102/105 items match**).

### 3.5 H1′ — sustained (a positive you did not list)

- **Doc:** `<HOME>/Projects/research/scres-ia/docs/RESULTADO_H1_H3_V2_2026-08-01.md`
- **Artifact:** `results/manuscript/h1_h3_v2/result.json`, seal `312f91a548d8639c…`, `claim_status = H1_SUPPORTED__H3_NOT_SUPPORTED`; **7 falsifiers pass**; contract `docs/PREREGISTRO_H1_H3_V2_2026-08-01.md`

Cumulative lost service (`service_loss_auc_ration_hours`, lower is better): hybrid **45,358,777**; reset 79,746,843; static/OFAT **107,033,239**.
| contrast | hybrid advantage | CI95 | n |
|---|---:|---|---:|
| primary — all 72 cells | **+61,674,462** | [+38,932,760, +87,248,013] | 360 |
| secondary — 42 differing-config cells | +105,727,650 | [+68,638,033, +148,946,420] | 210 |
| hybrid vs reset | +34,388,066 | [+18,072,049, +53,864,222] | 360 |

Caveat, verbatim: "*`service_loss_auc` **no es un tiempo de recuperación**... no es el estimando que `H1` enuncia.*" Ledger status: `DEVELOPMENT_SIGNAL`.

### 3.6 Other non-negative items

- **CSSU Gate A:** `results/garrido_cssu_liveness_gate_v1/result.json` → `GATE_A_PASS_GATE_B_HOLD`. Permitted claim: "*The CSSU reassignment interface is computationally live; finite physical Op11 handling has not been validated.*"
- **Thesis-90 replay:** `results/garrido_meta_learner_thesis90_v2/result.json` → `SURFACE_REPLAY_MEMORY_EFFECT`; ledger `SURFACE_REPLAY_ONLY`. Validates search/replay logic, not independent DES behavior.
- **Q1 Fig-5 surrogate:** `results/garrido_wrap_q1/result.json` → `DEVELOPMENT_FIG5_SURROGATE`; linear R²=0.9697 on the grid (0.982 in continuous space per `docs/ESTADO_CONTRA_GARRIDO_2026-07-31.md`). This is a **positive methodological finding against Garrido's intuition**.
- **The ReT metric defect** (flagged as "probably the most citable thing in the article" in `docs/ESTADO_CONTRA_GARRIDO_2026-07-31.md` §5): `ret_excel` rewards abandoning a claimant — the allocation that maximizes it delivers **50%** of rations, the one that minimizes it delivers **80%**. Doc: `docs/RESULTADO_RET_PREMIA_EL_ABANDONO_2026-07-31.md`.
- **G3c reopened** (2026-08-05): `docs/SUPERSESION_CIENTIFICA_G3C_2026-08-05.md` → `G3C_REOPENED_AS_ORTHOGONAL_EXTENSION_PREFLIGHT_ONLY`. Preflight on burned tapes only; PI authorization 2026-08-05. Terminal outcomes: `PREFLIGHT_POWERED_PENDING_AUTHORITY` or `STOP_G3C_UNDERPOWERED`. Latest: `docs/RESULTADO_G3C_PREFLIGHT_BURNED_V2_2026-08-05.md`, `docs/RESULTADO_G3C_PREFLIGHT_REJILLA_VIVA_2026-08-05.md`.
- **Latest run is a NEGATIVE** (2026-08-05, most recent science in repo): `docs/RESULTADO_ESTAR_CAPACIDAD_V2_REPLAY_2026-08-05.md` → `HALTED_FALSIFIER_FAILED`; `f6_no_gain_by_abandonment` **fails** (`freq3_imp2`: lost_orders harm 0.500, UCB95 1.500 vs margin 0.50; worst_claimant_fill UCB95 0.030 vs margin 0.010). Artifact `results/headroom/estar_capacity_sweep_v2_replay_20260805/result.json`.

### 3.7 The single best summary table for a C&IE submission plan

`docs/FRONTERA_DE_CLAIMS_H3_Y_AUTORIZACION_2026-08-02.md` §7 — verbatim:

| afirmación | estado defendible |
|---|---|
| Q1 — qué IA imita el SCL | **condicional**: en cuatro contratos separados los controles estructurados capturaron el valor que las redes no capturaron. No universal |
| Q2 — cómo integrarla en el DES | interfaz DES → política → estado retenido **especificada y parcialmente probada**; WRAP-288 y E\* no son validación completa |
| H1 | **no evaluable**: TTR censurado. H1′ es otro constructo |
| H2 | curvas de aprendizaje **descriptivas**, sin adjudicación |
| H3′ | **SOSTENIDA, n = 120**, LCB95 +2,3491 — como reducción de varianza del **coste de búsqueda** |
| H4 | **apoyo estrecho** vía memoria; no path-dependency general |
| G3-obs | **subpotenciado, NO negativo** — ahora autorizado a repetirse con potencia |
| G3c | bloqueado, 2 bloqueadores |
| E\* | sólo diseño |

Closing line: "*La contribución fuerte para C&IE no es «la red ganó». Es una integración DES–aprendizaje con memoria, trazabilidad completa y un resultado positivo estrecho sobre estabilidad de búsqueda...*"

---

## 4. Figures / tables assets

**Yes — there is a full matplotlib figure pipeline, in two independent sets.**

### Set A — the Elsevier manuscript figure pipeline (17 figures)
- **Code:** `<HOME>/Projects/research/scres-ia/scripts/build_manuscript_figures.py` (~1,350 lines). Docstring: *"Publication-grade manuscript figures for the Q1 submission... Okabe-Ito colorblind-safe palette, serif typography, vector PDF (for LaTeX) + 300-dpi PNG (for the Word port)."* Output dir hard-coded: `docs/manuscript_current/submission/elsevier/figures`. Numbers are **hard-coded deliberately** "*so the figure script is self-contained and auditable against the claims registry*".
- **Functions:** `fig1_bottleneck_alignment`, `fig2_mfsc_topology`, `fig3_gap_decomposition`, `fig4_pareto_ret_tail_ctj`, `fig5_generalization_heatmap`, `fig6_action_space_ablation`, `fig7_ret_metric_lineage`, `fig8_ret_branch_timeline`, `fig9_prevention_ceiling`, `fig10_efficiency_architecture`, `fig11_no_forecast_defense`, `fig12_des_validation`, `fig13_track_a_boundary`, `fig14_dispatch_cost_sensitivity`, `fig15_learning_curves`, `fig16_reward_sensitivity`, `fig17_control_loop`.
- **Rendered assets:** `docs/manuscript_current/submission/elsevier/figures/` — 17 × (.pdf + .png). Duplicated in `tmp/manuscript_compile/figures/`.

### Set B — the deliverables/DOCX figure pipeline (7 figures)
- `<HOME>/Projects/research/scres-ia/deliverables/figures/build_figures.py` → `fig1_mfsc_flow`, `fig2_framework`, `fig3_ladder`, `fig4_results`, `fig5_ret_tree`, `fig6_timeline` (each .png + .pdf, 300 dpi)
- `<HOME>/Projects/research/scres-ia/deliverables/figures/build_fig_oracle_metric.py` → `fig7_oracle_metric.png/.pdf`
- `<HOME>/Projects/research/scres-ia/deliverables/figures/build_review_pdf.py` → `deliverables/REVIEW_figures_and_formulas_2026-07-23.pdf` (982 KB, a figures+formulas review package)
- `deliverables/figures/thesis_fig62_reference.png`, `deliverables/fig1_flow.png`

### Other figure builders
- `scripts/generate_control_reward_figures.py`, `scripts/build_action_design_note_figures.py`, `scripts/build_track_b_q1_stats.py`, `scripts/build_garrido_meeting_package.py`, `scripts/generate_proof_of_learning_artifacts.py`, `scripts/audit_real_kan_interpretability.py`
- `docs/garrido_meeting_2026-06-24/make_figures.py` + `make_bottleneck_figures.py`
- `docs/garrido_track_b_alternatives_2026-07-03/make_figures.py`
- Figure directories: `docs/action_space_design_note/figures/`, `docs/garrido_meeting_decision_variables_2026-07-03/figures/`, `archive_untracked/figures/`

### Tables
- `papers/paper2/build_results_table.py` → `results_table.md` / `results_table.json` (per-cell source hashes; machine-generated, "do not edit")
- `docs/track_b_q1_stats_2026-07-02_final_10seed/` and siblings — frozen JSON stat bundles feeding the figures

**Gap for a C&IE Garrido–WRAP submission:** there is **no figure code yet for the new results** (H3′ n=120, G3-obs conversion, Alzheimer price). The 6 required figures are only listed as prose in `docs/CIE_MANUSCRIPT_REWRITE_MAP_GARRIDO_WRAP_2026-08-01.md` §"Required figures/tables after the 288 rerun".

---

## 5. The DES model description DOCX delivered to Garrido

**`<HOME>/Projects/research/scres-ia/deliverables/Program_Q_DES_Model_Description_CIE.docx`** (153 KB, 23 Jul 2026, ~5,054 words, 593 text paragraphs)
- **Generator:** `<HOME>/Projects/research/scres-ia/deliverables/build_des_model_doc.py` (66 KB — builds both the DOCX via `python-docx` and `fig1_flow` via matplotlib; note its `OUTPUT` currently points at `Program_Q_DES_Model_Description_CIE_reviewed.docx`, which is not on disk, so the shipped file is one build behind the script).
- **Structure (Heading1s, verbatim):** Editorial control note · Recommended paper position · 1. Model genealogy and scope · 2. Baseline 0 and the fair comparison ladder · 3. The thirteen-operation discrete-event system · 4. Material flow, queues, and conservation · 5. Time, warm-up, and episode boundaries · 6. Demand process and partial observability · 7. Program Q decision contract · 8. Same-time event convention · 9. Observation supplied to the controller · 10. Resilience metric and reward · 11. The learning context · 12. Why RecurrentPPO with an MLP-LSTM policy is the executed baseline · 13. DMLPA and KAN as matched architecture tests · 14. How learning is identified · 15. Resource equality and the end-of-horizon artifact · 16. Sensitive assumptions and validity checks · 17. Supported and unsupported manuscript statements · 18. Manuscript-ready methods narrative · Appendix A. Coauthor integration checklist · Appendix B. Internal source traceability
- **Central claim boundary callout (verbatim):** "*Program Q outperforms every open-loop calendar in its disclosed two-product extension. / Program Q is practically equivalent to the strongest tested structured feedback family. / Program Q does not establish a neural premium, worst-product safety, cumulative learning, or improvement under active Garrido-native risks.*"
- **Editorial control note (verbatim):** "*The two supplied v0 DOCX files are byte-identical copies of the same unfinished manuscript. The v0 manuscript contains placeholders, an obsolete five-dimensional action space, an obsolete fifteen-dimensional observation, and claims that precede the executed Program Q evidence. This insert should replace the v0 model and learning-environment prose instead of being appended to it without reconciliation.*"

**Two companion DOCX files (the v0 manuscript with our sections injected):**
- `<HOME>/Projects/research/scres-ia/deliverables/v0_neuralNet-scres_DES_section_updated.docx` (2.2 MB, 23 Jul, ~4,509 words of body text) — built/updated by `deliverables/update_v0_model_section.py`
- `<HOME>/Projects/research/scres-ia/deliverables/v0_neuralNet-scres_DES_and_oracle_metric.docx` (2.5 MB, 26 Jul, ~6,063 words) — the later one, adds the oracle-metric section via `deliverables/insert_oracle_metric_section.py` (+ `deliverables/fix_v0_oracle_table_header.py`)

**Markdown counterpart:** `<HOME>/Projects/research/scres-ia/docs/DES_MODEL_DOCUMENTATION.md` (56 KB, 1,151 lines) — *"DES Model Documentation: 1:1 Thesis Verification"*, audit date 2026-03-18, **overall fidelity 97% (102/105 items match)**. 5 parts: function docs (Op1–Op13, risk processes), parameter docs (config.py, Tables 6.2/6.3/6.10/6.25), environment docs (env.py, env_experimental_shifts.py), external interface docs, and **§5 Discrepancy analysis with 5 named bugs** (R12 not gating Op2, Op8 event-triggered, R14 defects discarded, warmup triggers on production not Op9 receipt, PT rounding).

---

## 6. What we would tell Garrido / attribution / authorship

### Attribution & authorship planning
- **`deliverables/Program_Q_DES_Model_Description_CIE.docx` → Appendix A "Coauthor integration checklist"** — this is the closest thing to an attribution matrix. Marked "*Remove this appendix before submission... an editorial control surface for the shared working document.*" Two-column table (Editorial action → Required change) covering: Replace v0 Section 3.2 / Replace v0 Section 3.3 / Baseline wording / Architecture wording / Metric wording / End-horizon wording / Fidelity wording / Results wording / Literature wording / KAN wording. E.g. "*Call Garrido the static reference and call the 65,536-calendar frontier the matched static benchmark*"; "*Do not claim first-of-kind novelty until the current review is complete.*"
- **Appendix B "Internal source traceability"** — a provenance matrix mapping 7 repo files to the controlled fact each governs (`research/paper2_exhaustive_search/source_reconstruction.md`, `contracts/program_o_full_des_hpi_translation_v1.json`, `contracts/program_o_ret_only_learner_v1.json`, `supply_chain/program_o_ret_env.py`, `supply_chain/ret_thesis.py`, `papers/submission_a_program_q/source_of_truth.json`, `docs/GARRIDO_FIDELITY_AUDIT_2026-07-10.md`). **Note: `papers/submission_a_program_q/source_of_truth.json` does not exist in this repo** — dangling reference.
- **CRediT block:** `docs/manuscript_current/submission/elsevier/main.tex:51-55` — `\section*{CRediT authorship contribution statement}` is an unfilled `[TO COMPLETE...]` placeholder listing the roles to assign.
- **Section ownership rule:** `docs/for_team/v0_additions_for_thom.md` — "*Do NOT change David's sections (3.3, 4.2) or Garrido's framing (Intro, Hypotheses)*" ⇒ implicit author map: Garrido = Intro/Hypotheses, David = §3.3/§4.2, Thom = DES (§3.2/§4.1).

### "What we would tell Garrido" documents
- **`<HOME>/Projects/research/scres-ia/docs/ESTADO_CONTRA_GARRIDO_2026-07-31.md`** — the definitive one. Section-by-section state of the v0 draft (Abstract ❌ literal placeholder; §1/§2 ✅; §3.2 ✅; §3.3.1–3.3.5 ✅; **§4.2 ❌ EMPTY**; **§4.3 ❌ EMPTY**; §5 🟡 skeleton). Flags a journal conflict: "*el destino declarado en el encabezado es IEEE TAI / Journal of Simulation, no C&IE — hay que reconciliarlo antes de dar formato.*" H1–H4 vs what we have. §4: "*Nosotros llevamos meses atacando un problema más duro... no es la que él formuló.*" §6: 5-step plan.
- **`<HOME>/Projects/research/scres-ia/docs/DONDE_PODEMOS_SER_LAXOS_2026-08-02.md`** — where to loosen vs not; establishes that `contracts/authority_ladder_v1.json` self-declares `"status": "DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY"` and is therefore not a hard block. Key calendar note: "*el bloqueo real de Submission A es **editorial y humano**, no experimental.*"
- **`<HOME>/Projects/research/scres-ia/docs/DECISIONES_SIN_GARRIDO_2026-07-30.md`** — `DECISION_DOCUMENT_NOTHING_APPLIED`; answers the two questions we were going to ask Garrido directly from the 2017 thesis (lead time 48 h per §6.8.2 p.111; `CTj = LTj` vs our `<=`), with a self-retraction banner.
- `<HOME>/Projects/research/scres-ia/docs/LECTURA_GARRIDO_2026-07-30.md`, `docs/GARRIDO_2024_AI_ALIGNMENT_2026-07-31.md`
- `<HOME>/Projects/research/scres-ia/paper_results_package/BRIEF_PARA_GARRIDO_Y_DAVID.md` — *"Para: Alexander Garrido, David / De: Thomas / Fecha: 4 de abril 2026"* (historical, superseded)
- **Meeting/presentation packages:** `docs/PRESENTACION_GARRIDO_2026-03-25.md` (contains the −4.43% table and 5 discussion questions), `docs/meetings/2026-03-26_GARRIDO_DEMO.md`, `docs/meeting_packages/garrido_presentation.html`, `docs/meeting_packages/garrido_2026-03-11/`, `docs/meeting_packages/garrido_2026-03-13/`, `docs/garrido_meeting_2026-06-24/reporte_garrido.pdf|.tex`, `docs/garrido_meeting_decision_variables_2026-07-03/main.tex|main_es.tex`, `docs/garrido_track_b_alternatives_2026-07-03/`
- **Briefs:** `docs/briefs/garrido_thesis_alignment_audit.md` (12 KB), `docs/briefs/garrido_control_reward_memo.md`, `docs/briefs/garrido_track_b_meeting_script_2026-04-03.md`, `docs/briefs/garrido_track_b_audit_brief_2026-04-03.md`, `docs/briefs/thesis_decision_learning_claim.md`
- **Handoffs to David:** `docs/DKANA_CONTRIBUTOR_HANDOFF.md` (15 KB), `docs/DKANA_INTEGRATION_GUIDE.md`, `docs/DAVID_DKANA_THESIS_FAITHFUL_TUTORIAL.md`, `scripts/export_trajectories_for_david.py`

---

## Three things worth flagging before you plan the submission

1. **Two incompatible manuscripts exist.** The compiled Elsevier `main.tex` is the *"When Is RL Warranted"* eligibility-framework paper (Program Q/O/G lane). The C&IE target declared in `docs/GARRIDO_WRAP_CURRENT_STATE_2026-08-01.md` is the *Garrido–WRAP/v0* paper, for which **only the outline exists** (`CIE_MANUSCRIPT_REWRITE_MAP_...`) and whose prose lives in the v0 DOCX with §4.2/§4.3 empty. Deciding which one goes to C&IE is the first fork.
2. **The claim ledger is stale by one day** relative to the science. `GARRIDO_WRAP_CLAIM_LEDGER_2026-08-01.md` still lists H3′ as `ARTIFACTS_PRESENT_MERGE_PENDING`; the adjudication on 2026-08-02 promoted it to `H3_PRIME_SUSTAINED_AT_N120`, and WRAP-288 was reconciled the same day. Nothing dated 2026-08-03..05 has been folded back into the ledger either.
3. **G3-obs's positive number is formally non-promotable** (`SOURCE_ARTIFACT_PRESERVED_SCOPE_MISMATCH_NOT_PROMOTABLE`) even though the underlying run passed 8/8 falsifiers with published power. If you want the `H_obs +0.0207 [LCB95 +0.0147]` number in a C&IE paper as a confirmed result, a prospective re-execution under a correctly-sealed v2 contract is required.
