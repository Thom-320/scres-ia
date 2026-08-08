# Explore — Repo: /Users/thom/Projects/research/scres-ia

- **agent_id**: `aad5fc6e24d7410ee`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T00:49:12.499Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-aad5fc6e24d7410ee.jsonl`
- **turns**: 122 · **assistant messages**: 2

## Task given

```text
Repo: /Users/thom/Projects/research/scres-ia

GOAL: inventory everything that already exists toward a Computers & Industrial Engineering (C&IE) submission, so I can judge how close we are to a submittable manuscript.

Look for and report, with exact paths:

1. `docs/ESTRATEGIA_CIE_2026-08-06.md` and `docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md` — read both fully and summarise: what paper was planned, what claims it was to carry, what was already drafted.
2. Any C&IE guide-compliance work: search branches for `codex/submission-a-cie-guide-compliance` and `codex/submission-a-program-q` and `codex/submission-a-finalization`. Use `git show <branch>:<path>` to read key docs. Report what a submission package contained (cover letter? highlights? declarations? anonymised version? figure formats?).
3. Manuscript directories: `docs/manuscript_current/`, `docs/manuscript_draft/`, `docs/manuscript_notes/`, `results/manuscript/`. List what is in each, how many words/sections any draft has, and its date.
4. Figure pipeline: find `scripts/build_cie_outer_loop_figures.py` or similar. Report which figures it generates, whether values are loaded from result.json (no hardcoded numbers), and output formats.
5. `scres_ia_manuscript_overleaf.zip` at repo root — list its contents (unzip -l only, do not extract).
6. Any LaTeX/Word manuscript files anywhere (`*.tex`, `*.docx` under docs/ or deliverables/).

OUTPUT: a compact readiness assessment — what exists, what is reusable, and what is genuinely missing to submit. Read-only, exact paths, no narrative padding.
```

## Final answer

## Readiness assessment — C&IE submission

**Bottom line: there is no submittable manuscript for the current science.** There is a *complete submission package* for a paper whose scientific base was abandoned, and *complete, confirmed evidence* for the paper you actually want to write, with zero prose written. Those two things are on opposite sides of a 425-commit branch gap.

---

## 1. The two planning docs

### `/Users/thom/Projects/research/scres-ia/docs/ESTRATEGIA_CIE_2026-08-06.md` (5.8k, Aug 6)
Planned paper: the **outer loop as Garrido's AI bridge**. Reads Garrido 2024 page-by-page and anchors the paper to his own figure nodes — Fig. 2 node ③ (`Decision variables, ρ`) → node ⑧ (`Metric of SCRES`), with `L` accumulating *across runs and configurations*. Claim: our outer loop **is** that bridge — simulation-optimization over configuration space, not intra-episode control. Justifies C&IE as venue because Garrido 2024 cites **seven** C&IE papers (refs 8, 9, 26, 28, 34, 38, 41), incl. Bruckler et al. 2024 C&IE 192 (SCRES *metrics* review) — which the `ret_excel`-rewards-abandonment finding speaks to directly.

Stated gap at that date: **R0–R7 were all development; nothing confirmatory.** Two runs launched to fix that (R7 prospective confirmation on virgin block 8.200.001–060; VPS KAN/MLP/DMLPA bake-off at matched 200k params).

### `/Users/thom/Projects/research/scres-ia/docs/RESULTADOS_CIE_CONSOLIDADOS_2026-08-05.md` (15k, Aug 5)
The results package, explicitly "sin prosa de manuscrito" — **it is a results inventory, not a draft**. R0–R7 with artifact paths and contracts:
- R1 normaliser leak repaired (prefix normaliser), R1b twin-surface falsifier
- R2 `NON_SEPARABLE_BUT_CONTEXT_INVARIANT`, `H_regime` +0.0038
- R3/R4 comparator ladder → **the ingredient is retention, not the approximator** (4 families gain)
- R5 KAN≡MLP≡neuron, neuron wins `Δ_efficiency` (5 params, 74× fewer)
- R6 extended 4,608 grid → `H_regime` ×7.4, still fails gate
- R7 grid transfer → **only the factorised bandit beats its marginal replay**

Also declares the confirmation attempt **aborted and quarantined** (`CONFIRMATION_BLOCK_QUARANTINED_NO_SCIENTIFIC_RESULT`).

**That last statement is now out of date** — see §7.

---

## 2. C&IE guide-compliance work (branch-only)

`codex/submission-a-cie-guide-compliance` exists **only as `origin/`** (no local ref). `origin/codex/submission-a-program-q` (`1fad6bb`) is the superset — it merged guide-compliance. All three are **12 ahead / ~425 behind `main`**, and `main` itself is stale (`89acc81`, 28-Jul; the science branch is ~823 ahead).

Package at `papers/submission_a_program_q/` (38 files, **absent from `main` and from your working tree**):

| Item | Path (via `git show origin/codex/submission-a-program-q:<path>`) |
|---|---|
| Cover letter | `papers/submission_a_program_q/COVER_LETTER_DRAFT.md` — full prose, PENDING author/DOI placeholders |
| Highlights | `HIGHLIGHTS.txt` — 5 bullets, all <85 chars |
| Title page (non-anon) | `TITLE_PAGE.tex` — elsarticle, all human fields `PENDING` |
| Guide audit | `CIE_GUIDE_AUDIT_2026-07-29.md` — 17-row requirement table vs live guide |
| Checklist | `RELEASE_AND_SUBMISSION_CHECKLIST.md` — 8 done, 11 open |
| Readiness JSON | `submission_readiness.json` — 10-item `pending` list |
| AI declaration | `GENERATIVE_AI_DISCLOSURE_DRAFT.md` + in-manuscript section |
| Anonymised build | `scripts/build_submission_a_cie_review_bundle.py` — identity-token-guarded zip, excludes `TITLE_PAGE.tex` |
| Manuscript | `main.tex` (~3,620 words), `references.bib`, 6 generated `.tex`/`.csv` tables, 4 PNG figures, `output/pdf/submission_a_program_q_draft.pdf` |
| Other | `CLAIMS_TABLE_2026-07-28.md`, `SUBMISSION_HANDOFF.md`, `INDEPENDENT_REVIEW_PACKET.md`, `SECURITY_AND_DISCLOSURE_CHECKLIST.md`, `GARRIDO_FACE_VALIDATION_REQUEST.md`, `AUTHORSHIP_AND_PORTFOLIO.md`, `EXECUTION_BOARD_8_WEEKS.md`, `source_of_truth.json`, `submission_metadata.json` |

Paper title: *"When Feedback Beats an Exhaustive Open-Loop Frontier but Not Structured Control: Exact Benchmarking of Recurrent RL in a Supply-Chain DES."*

**The problem:** this is a **RecurrentPPO / 65,536-calendar** paper. `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §7 now forbids the surrounding claim family, and `RESULTADOS_CIE_CONSOLIDADOS` states "RL, PPO, MLP y KAN siguen fuera del carril." Its `C1_VIRGIN_BLOCK` is `NO-GO`. The *mechanics* are reusable; the *science* is not.

---

## 3. Manuscript directories

| Path | Contents | Verdict |
|---|---|---|
| `docs/manuscript_current/submission/elsevier/` | `main.tex` (167 lines, 1,192 w), 10 section `.tex` (**two competing numberings**: `03_eligibility_framework`+`04_mfsc_case`+`05_results`+`06_discussion` vs `03_methodology`+`04_results`+`05_discussion`+`06_conclusion`), `references.bib` (15k), `main.pdf` (661k, 10 Jul), 22 figures ×(pdf+png) | **~17k words total but a *third*, different paper**: *"When Is Reinforcement Learning Warranted in a Supply Chain DES? Decision-Rights Eligibility…"*. Mtimes 18 Jul except `04_results.tex` (31 Jul). Marked `STALE_SUPERSEDED.md` on the submission branch: "not a source for Submission A." |
| `docs/manuscript_draft/` | `section_3_3_rl_formulation.md` (8.9k), `section_4_2_hybrid_results.md` (5.7k), both 24 Jun | Dead — RL formulation, off-carril |
| `docs/manuscript_notes/` | 16 `.md`, 24 Jun–29 Jul (reward audits, ReT tail branch audit, paper strategy memo, backlogs) | Process notes, not prose |
| `results/manuscript/` | `h1_h3_v1`, `h1_h3_v2`, `h1_h3_v2_1`, `h1_h3_originales_v3` (7 Aug), `h2_learning_curve` (7 Aug) — `result.json` only | **Live evidence**, not text. H1/H2 SUPPORTED, H3 NOT SUPPORTED |

---

## 4. Figure pipeline — `scripts/build_cie_outer_loop_figures.py` (11k, Aug 5 14:38)

Generates 5 figures → `docs/manuscript_current/submission/elsevier/figures/`:
`fig_a_normaliser_leak`, `fig_b_surface_gates`, `fig_c_comparator_ladder`, `fig_d_memory_price`, `fig_e_delta_efficiency`.

**Formats: correct.** Vector `.pdf` for LaTeX + 300-dpi `.png` for the Word port, Okabe-Ito colourblind-safe, STIX serif. **No hardcoded numbers** — confirmed: `load()` at line 52 reads every value from a `result.json`.

**But the artifacts it reads are superseded.** The script hardcodes:
- `search_ladder_v2/result.json` (figs c, d) — `results/search_ladder_v2/SUPERSEDED.md` exists; `INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` marks it `SUPERSEDED_FOR_CURRENT_CLAIMS_BY_V5`. Authoritative is `results/search_ladder_v5/`
- `twin_surface/result.json` — the consolidated doc cites `twin_surface_v2/`
- `surface_gates/result.json` — doc cites `surface_gates_v2/`
- `search_surrogates/result.json` — this one is current

Figures on disk are dated Aug 5 **14:38**; the ordered/corrected artifacts landed Aug 5 **18:18–19:11**, and `search_ladder_v5` on Aug 6 18:04. `grep` confirms **no script references `search_ladder_ordered`, `search_ladder_v2_ordered`, `search_ladder_v5`, or `grid_transfer_ordered_v1` for figures**. The "no drift possible" property holds for values but not for artifact selection — the figures are stale and would need the paths repointed and a re-run.

---

## 5. `scres_ia_manuscript_overleaf.zip` (1.5 MB, root, 18 Jul)

27 files, all dated **2026-07-02** — snapshot of the older Elsevier draft:
`main.tex` (9,347 B), `main.pdf` (391 kB), `references.bib` (15,415 B), `sections/01_introduction` … `06_conclusion` (6 files; `04_results.tex` is 29,968 B), `figures/fig1`–`fig8` as pdf+png pairs.
Same lineage as `docs/manuscript_current/` but an **earlier, smaller** cut (8 figures vs 22, 6 sections vs 10). Superseded by the on-disk copy; no unique content.

---

## 6. All LaTeX / Word manuscripts

**LaTeX**
- `docs/manuscript_current/submission/elsevier/main.tex` + `sections/*.tex` (10) — stale paper
- `papers/submission_a_program_q/main.tex` + `TITLE_PAGE.tex` — branch only, stale science
- Meeting/note documents (not manuscripts): `docs/garrido_meeting_decision_variables_2026-07-03/main.tex`, `main_es.tex`; `docs/garrido_track_b_alternatives_2026-07-03/track_b_alternatives_garrido.tex`; `docs/action_space_design_note/main.tex`; `docs/garrido_meeting_2026-06-24/reporte_garrido.tex`, `reporte_cuello_botella.tex`
- `papers/paper2/` exists on disk (results, not manuscript)

**Word**
- `deliverables/Program_Q_DES_Model_Description_CIE.docx` (152 kB, 23 Jul)
- `deliverables/v0_neuralNet-scres_DES_and_oracle_metric.docx` (2.5 MB, 26 Jul) — the "borrador v.0" whose H1–H4 the Aug 7 amendment adjudicates
- `deliverables/v0_neuralNet-scres_DES_section_updated.docx` (2.2 MB, 23 Jul)
- Generators: `deliverables/build_des_model_doc.py`, `update_v0_model_section.py`, `insert_oracle_metric_section.py`
- `deliverables/REVIEW_figures_and_formulas_2026-07-23.pdf` (982 kB)

---

## 7. What changed since the two planning docs — this is the important part

The Aug-5 claim that "no confirmatory result exists" is **superseded**. Per `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` and `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §3:

- `results/grid_transfer_confirmation_v2/result.json` — `GRID_TRANSFER_CONFIRMED__UCB1`, **virgin block 8.200.001–060**. UCB1 +0.03073 [+0.01990] vs state-blind marginal replay; **the neuron loses**: −0.01178 [−0.01849, −0.00484]. R7 is now prospective confirmation.
- `results/garrido_h2_h3_confirmation_v1/result.json` — `CONFIRM_H2_H3_ALL_SIX_PANELS`, 12 virgin roots, Holm p 4.8e−17…1.9e−15 (rescued from `codex/paper-b-retained-v5`)
- `results/gsa_confirmation/result.json` — `GSA_CONFIRMED_ON_VIRGIN_BLOCK` 7.700.001–120
- `results/manuscript/h2_learning_curve/` — `H2_SUPPORTED_LEARNING_CURVE`, slope +0.042201 [+0.034664, +0.049922]; H1 original SUPPORTED (+126.0 h [+98.4, +154.5]); H3 original NOT SUPPORTED

`TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §8 records **`MANUSCRITO_Q1_Q2 = GO`**.
Authoritative claim text is frozen in `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` (16.5k, 8 numbered sections incl. citable artifacts, frozen Q1/Q2 claims, permitted/forbidden vocabulary, retired figures, claim ladder) + `_ENMIENDA_1.md` (13.5k).

---

## Verdict

**Reusable as-is**
- The entire C&IE compliance apparatus in `papers/submission_a_program_q/` — guide audit table, checklist, title page template, highlights format, AI-disclosure section, and especially `scripts/build_submission_a_cie_review_bundle.py` (identity-guarded anonymous bundle). Content must be rewritten; structure is done and audited against the live guide.
- The elsarticle scaffolding and `references.bib` (15k) in `docs/manuscript_current/submission/elsevier/`.
- `scripts/build_cie_outer_loop_figures.py` — architecture is right (result.json-sourced, dual pdf/png, 300 dpi); needs artifact paths repointed to `search_ladder_v5`, `twin_surface_v2`, `surface_gates_v2` and a re-run.
- The frozen claim text in `GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md` — this is effectively the results section in note form.

**Genuinely missing**
1. **Every word of the new paper.** No abstract, introduction, related work, methodology, discussion, or conclusion exists for the outer-loop/Q1–Q2 paper. `RESULTADOS_CIE_CONSOLIDADOS` explicitly declines to be prose. This is the whole job.
2. **Figures regenerated** against authoritative artifacts, plus a new figure for the confirmed R7 (`grid_transfer_confirmation_v2`) which the current 5-figure set predates.
3. **The three submission branches merged or cherry-picked forward** — the package is 425 commits behind and invisible from your working tree.
4. **All human-blocked items** (unchanged since 29 Jul): author order/affiliations/CRediT, corresponding-author contact, funding, Elsevier declarations-tool `.docx`, archival DOI + double-anonymous routing, Garrido's written face validation, independent language/format QA.
5. A decision on the two competing section numberings in `docs/manuscript_current/submission/elsevier/sections/` — that directory currently holds two half-papers.

**Honest distance:** you have a fully-audited submission *chassis* and fully-confirmed *evidence*, and no manuscript joining them. Call it a solid 2–3 weeks of writing plus a figure re-run, before the human-gate items start.
