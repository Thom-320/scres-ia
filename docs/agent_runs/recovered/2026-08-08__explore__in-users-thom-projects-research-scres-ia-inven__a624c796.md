# Explore — In <HOME>/Projects/research/scres-ia, inventory everything a manuscript titled roughl

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a624c796fc1e3bea2`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T01:26:42.815Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-a624c796fc1e3bea2.jsonl`
- **turns**: 98 · **assistant messages**: 2

## Task given

```text
In <HOME>/Projects/research/scres-ia, inventory everything a manuscript titled roughly "Retained search state before neural architecture" would need. Read-only, be concrete with numbers and paths.

1. `results/search_ladder_v5/` — the comparator ladder. How many methods, what is the ranking, is it true that the top six are exactly the six that retain state? Give the ordered list with scores, the endpoint, the seed block, evidence grade and scope string.
2. `results/garrido_fig5_surrogate/` and `results/garrido_wrap_q1/` — the "Fig. 5 is an algebraic identity" diagnostic. Exact residual, how many columns are identically zero, claim_status.
3. `results/surrogate_architecture_bakeoff/` (or architecture_bakeoff) — KAN vs MLP. Exact contrasts and CIs, and whether `use_kan` was on.
4. THE CONFLICT I need resolved: what is the CURRENT authoritative value for the neuron_memory vs ofat_transfer contrast? One audit says the canonical table gives +0.01071 [+3.56e-5, +0.02171], i.e. it DOES exclude zero. But `results/.../ofat_lcb_reconciliation*` (find it) concluded OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE with the lower bound positive in only ~65% of resampling seeds. Find BOTH artifacts, report both numbers, their dates/SHAs, and say which one supersedes which and on what evidence.
5. The normaliser audit: `results/twin_surface_v2/`, `results/garrido_normaliser_audit_v3/` (find exact names) — what is the prefix vs oracle normaliser result and which artifact is authoritative.
6. The H1–H4 adjudications: `results/manuscript/h1_h3_originales_v3/`, `h2_learning_curve/`, `h3_merge_adjudication/`. For each: claim_status, scope string, and whether it is development or confirmation.

Give me a table of artifact -> claim_status -> evidence grade -> scope -> key numbers.
```

## Final answer

I read every artifact. Here is the full inventory.

## Master table

| # | artifact | claim_status | evidence grade | scope | key numbers |
|---|---|---|---|---|---|
| 1 | `results/search_ladder_v5/result.json` | `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH` | **REPLAY** | `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`; `run_role=CACHE_ANALYSIS` | 16 arms (15 + oracle), 6 contexts × 12 seeds, budget 24; top-6 = the 6 memory arms |
| 2a | `results/garrido_fig5_surrogate/result.json` | `DEVELOPMENT_FIG5_SURROGATE` | **UNCONTRACTED** (no contract hash) | none declared; seed 20260731 | identity error **3.219646771412954e-15**, r²=1.0, **3 of 5** columns identically zero |
| 2b | `results/garrido_wrap_q1/result.json` | `DEVELOPMENT_FIG5_SURROGATE` | **UNCONTRACTED** | none declared | same task-A block; adds `decision: NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL` |
| 3a | `results/surrogate_architecture_bakeoff/result.json` | `KAN_SEARCHES_WORSE_THAN_A_MATCHED_MLP` | **REPLAY** | `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION` | `kan − mlp_matched = +0.01036905 [+0.00301822, +0.01892608]`, p=0.0012; params 532 vs 529 |
| 3b | `results/architecture_bakeoff/result.json` | `DEVELOPMENT_ARCHITECTURE_BAKEOFF_NO_ADJUDICATION` | **UNCONTRACTED** | `DEVELOPMENT_ONLY_NO_VIRGIN_SEEDS_NO_LEARNER_AUTHORISATION` | `KAN−MLP = −0.4751 [−1.5484, +0.5982]`; all 3 contrasts cross zero |
| 3c | `results/architecture_bakeoff/sealed_record.json` | `CONTENT_SEALED_PROVENANCE_NOT_CERTIFIABLE` | — (custody record) | `CUSTODY_ONLY_NO_NUMBER_CHANGES_NO_ADJUDICATION` | file sha `d641ab0a54ecf2c8…`, 6242 bytes |
| 4a | `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md` §2 | canonical claims table (v0) | doc | — | says `+0,01071 [+3,56e−05, +0,02171] → excluye cero` |
| 4b | `results/ofat_lcb_reconciliation/result.json` | **`OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE`** | **DEVELOPMENT** | `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS` | B=50 000, 40 RNG seeds, LCB>0 in **0.65** of seeds |
| 5a | `results/twin_surface_v2/result.json` | `PREFIX_NORMALISER_IS_BLIND_TO_THE_UNRUN_SURFACE` | **REPLAY** | `DEVELOPMENT_ON_BURNED_TAPES_STRUCTURAL_SPY_TEST` | prefix: all paths unchanged; oracle: 12/12 memory+reset paths change |
| 5b | `results/garrido_normaliser_audit_v3/result.json` | `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER` | **REPLAY** | `DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER` | prefix `memory−reset = +0.06070331 [+0.04556081, +0.08019782]` |
| 6a | `results/manuscript/h1_h3_originales_v3/result.json` | `H1_SUPPORTED__H3_NOT_SUPPORTED` | **DEVELOPMENT** | `DEVELOPMENT_ALREADY_OPEN_BLOCK_NO_VIRGIN_SEEDS_NO_ADJUDICATION` | H1 vs static `+125.985 [+98.347, +154.545]`; H3 p_holm = 1.0 |
| 6b | `results/manuscript/h2_learning_curve/result.json` | `H2_SUPPORTED_LEARNING_CURVE` | **DEVELOPMENT** | `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION` | slope `+0.04220148 [+0.03466394, +0.04992206]`, n=120 |
| 6c | `results/garrido_h3_merge_adjudication/result.json` | `H3_PRIME_SUSTAINED_AT_N120` | **DEVELOPMENT** | **no `scope` key at all** | `memory−reset = +9.31444 [+2.34913, +16.34741]`, n=120 |

Grades are from `research/evidence_registry.jsonl` (216 artifacts, 211 distinct experiments), which derives grade from contract/custody facts, not from `claim_status`. Corpus-wide: CONFIRMATORY 3, DEVELOPMENT 87, DIAGNOSTIC 5, NEGATIVE_OR_HALTED 20, REPLAY 39, UNCONTRACTED 57.

---

## 1. The comparator ladder — `results/search_ladder_v5/result.json`

- **16 arms**, of which `oracle` is the reference, so **15 ranked methods**.
- `self_sha256` `f7dfb1e8bc1c036bcac28c83535e7a5dd8ebd723dd01d3ec604e8228c20cc511`; created `2026-08-06T23:01:39Z`; commit `46d56bc` (2026-08-06).
- **Endpoint**: `primary_metric = "auc_regret_norm"`, contract `contracts/garrido_expanded_des_e_star_v3_metric.json`, `contract_sha256` `25f33961a6f43713…`; budget 24; contexts `R1r, R2r, R1r+R2r, R1r|esc, R2r|esc, R1r+R2r|esc`.
- **Seed block**: `5300001–5300012` (12 seeds), `replay_of = "garrido_q2_des288"`, `run_role = CACHE_ANALYSIS`.
- **Scope**: `DEVELOPMENT_ON_BURNED_TAPES_NO_ADJUDICATION`. Evidence grade **REPLAY** ("declared re-execution of an already-open block"). Not prospective confirmation.
- 7 falsifiers, `all_passed: true` (f7 marked `not_applicable`, declared replay).

Ordered list (mean `auc_regret_norm`, lower is better; `oracle` = 0.0 by construction and is excluded from `ranking_best_first`):

| # | method | AUC regret | retains state? |
|---:|---|---:|---|
| 1 | `ucb1_transfer` | 0.04502300698358852 | yes |
| 2 | `neuron_memory` | 0.05203274760040456 | yes |
| 3 | `ofat_transfer` | 0.06274306709376977 | yes |
| 4 | `lookahead_kg_transfer` | 0.08018189890560519 | yes |
| 5 | `gp_ei_transfer` | 0.08390483487553962 | yes |
| 6 | `thompson_transfer` | 0.08907820238728746 | yes |
| 7 | `ucb1` | 0.09655146536034616 | no |
| 8 | `ofat` | 0.10024197702481123 | no |
| 9 | `gp_ei` | 0.10661346477746912 | no |
| 10 | `thompson` | 0.10893184647329195 | no |
| 11 | `lhs_local` | 0.10948882277743688 | no |
| 12 | `neuron_reset` | 0.11273605766343564 | no |
| 13 | `lookahead_kg` | 0.11479477721224257 | no |
| 14 | `random` | 0.1397945613431999 | no |
| 15 | `annealing` | 0.1742042148688457 | no |

**Yes, the claim is exactly true.** The artifact's own `memory_arms` list is `["gp_ei_transfer", "ucb1_transfer", "ofat_transfer", "neuron_memory", "lookahead_kg_transfer", "thompson_transfer"]` — set-identical to the top six of `ranking_best_first`. The gap is clean: worst memory arm 0.08908, best stateless arm 0.09655. Falsifier `f5_memory_arms_actually_carry_state` passes, so the label is checked, not asserted.

Caveat for the manuscript: the ladder's own headline arm is **not** the neuron. `ucb1_transfer` ranks first, and `neuron_memory − ucb1_transfer = −0.00700974 [−0.02444132, +0.01407644]` — a tie. The one prospective confirmation in the project (`results/grid_transfer_confirmation_v2`, virgin block 8200001–060) has the neuron **losing** to a state-blind marginal replay at `−0.01178 [−0.01849, −0.00484]` while UCB1 wins. The retention story survives; the "neuron" framing does not.

## 2. Fig. 5 is an algebraic identity

Both `results/garrido_fig5_surrogate/result.json` (`self_sha256` `40bf64852c6e6f44…`, 2026-07-31) and `results/garrido_wrap_q1/result.json` (`self_sha256` `dc2e431338f605b3…`, 2026-08-01) carry the identical `task_A_figure_as_drawn` block:

- `status: "IDENTITY_NOT_A_LEARNING_TASK"`, `r2: 1.0`
- **Exact residual: `max_abs_identity_error = 3.219646771412954e-15`**
- **3 of 5 driver columns identically zero**: `Re_APj`, `Re_DPj_RPj`, `not_in_his_ReT`
- Identified coefficients: `Re_RPj = 0.9999999999999677`, `Re_FRt = 0.9999999999999998`
- `claim_status` on both files: `DEVELOPMENT_FIG5_SURROGATE`; both graded **UNCONTRACTED** (no `contract_sha256` — nothing fixes what the run was allowed to claim). 5 falsifiers all pass; n=90 configurations, 30 groups, 87 pairs.

`garrido_wrap_q1` adds the decision layer: `NO_GO_NEURAL_PREMIUM_IN_WRAP_PANEL`, `promotion_eligible: false`, SESOI 0.05 on held-out R². Neither neural arm clears it — kan−linear `+0.02158 [+0.00416, +0.03900]`, backprop−linear `+0.01657 [+0.00484, +0.02830]`, both `passes_sesoi_and_ci: false`.

## 3. KAN vs MLP — two different bakeoffs, opposite signs

**`results/surrogate_architecture_bakeoff/result.json`** (search-efficiency bakeoff, 2026-08-07, `self_sha256` `965c477d91ebc322…`), contract `docs/PREREGISTRO_SURROGATE_ARCHITECTURE_BAKEOFF_2026-08-07.md` sha `212743cefdf526e3…`, seeds 5300001–012, budget 24:

- `kan_minus_matched_mlp = +0.01036905263158129 [+0.00301822, +0.01892608]`, `p_two_sided = 0.0012` — **excludes zero, KAN searches worse.**
- Parameters actually matched: **kan 532 vs mlp 529** (falsifier f1 exists precisely because an earlier claim compared 532 against 31).
- `best_vs_random` (harness sensitivity): `−0.08776181 [−0.10430897, −0.07131315]`, p=0.0.
- Arm means: `neuron_5p` 0.05203, `mlp_matched` 0.08853, `spline_poly` 0.09754, `kan` 0.09889, `gbt` 0.10832, `gp_matern` 0.11379, `random` 0.13979.

**`results/architecture_bakeoff/result.json`** (RL control bakeoff on `track_b_v1`, 60 000 steps, seeds 9491–9495, 200k-param target):

- `KAN_minus_MLP = −0.47510094 [−1.54836, +0.59816]`
- `DMLPA_minus_KAN = +0.61080308 [−0.34998, +1.57158]`
- `DMLPA_minus_MLP = +0.13570213 [−0.56925, +0.84066]`
- All three cross zero. Params 204 816 / 199 215 / 187 404.

**`results/architecture_bakeoff_200k/result.json`** (200 000 steps, macOS): `KAN_minus_MLP = −0.22681 [−0.68044, +0.22682]`, also crossing zero.

**Was `use_kan` on? No — and this is a live defect, not a footnote.** `scripts/run_architecture_bakeoff_v1.py:113` defines the DMLPA constructor with `use_kan=False` as the default, and the file's own comment at line 118 says what this repo has measured as "DMLPA" is a de-KAN'd network. `docs/PREREGISTRO_DMLPA_KAN_LATENT_2026-08-07.md:14` and `docs/ANALISIS_EXPORT_DAVID_2026-08-07.md:134-137` state the notebook shipped to David declares `use_kan=False` while the cell he executed declares `use_kan=True`. So the "DMLPA" arm in both `architecture_bakeoff` and `architecture_bakeoff_200k` is not David's architecture. `results/dmlpa_variants/` and `scripts/run_dmlpa_kan_latent_v1.py` (arms `dmlpa_mlp` hidden_dim 152 / `dmlpa_kan` hidden_dim 10) are the repair.

The `surrogate_architecture_bakeoff` KAN arm is a genuine KAN surrogate — the `use_kan` flag belongs only to the DMLPA/track_b family, not to it.

Custody note: `results/architecture_bakeoff/sealed_record.json` (2026-08-07, contract `a31fc25e4ed8c029…`) is an **external, retroactive** seal. Its own text: it certifies content as of the record's timestamp and **explicitly does not certify provenance**, because the runner did not seal at execution time. `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md:80` bans citing `results/architecture_bakeoff*` as "sealed".

## 4. THE CONFLICT — resolved, and the reconciliation wins

Both artifacts exist. Here is each, then the supersession.

**Artifact A — the canonical table's "excludes zero" line.** `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, line 56, commit `eb132a1` (2026-08-07), file mtime 11:36:

> `ofat_transfer`: **+0,01071 [+3,56e−05, +0,02171]** → excluye cero **por 3,6e−05**

Its source is `results/search_ladder_v5/result.json` → `vs_neuron_memory.ofat_transfer`: `mean 0.01071031949336522`, `lcb95 3.564844184833131e-05`, `ucb95 0.02171366105332719`, n=12. Artifact date 2026-08-06T23:01:39Z, `self_sha256` `f7dfb1e8bc1c036b…`, commit `46d56bc`.

**Artifact B — the reconciliation.** `results/ofat_lcb_reconciliation/result.json`, created `2026-08-07T18:44:35Z`, `self_sha256` `a35bb6ec721d683892000e74aae4a1c08b901aa13d36ee1f7aa5c956536e93ba`, contract `docs/ENMIENDA_RECONCILIACION_LCB_OFAT_2026-08-07.md` sha `c0b17233460fd3e4…`, commit `3b36875` (2026-08-07), `rng_seed 20260807`, B=50 000, 40 stability seeds. `claim_status: OFAT_LCB_SIGN_IS_RESAMPLING_UNSTABLE`.

The two sealed ladders score **byte-identical replicate arrays** (falsifier f1: `arrays_identical` true for both contrasts) and produce opposite signs on the same mean:

| sealed artifact | mean | stored `lcb95` | sign |
|---|---:|---:|---|
| `results/search_ladder_v2_ordered/result.json` (`a79f67c7d58c8f13…`, 2026-08-05, commit `5d941af`) | 0.01071031949336522 | **−2.761381942678142e-05** | includes zero |
| `results/search_ladder_v5/result.json` (`f7dfb1e8bc1c036b…`, 2026-08-06, commit `46d56bc`) | 0.01071031949336522 | **+3.564844184833131e-05** | excludes zero |

The reconciliation's own recomputation at the declared RNG seed gives a **third** value: `lcb95 = +1.551174847040456e-04`, `ucb95 = 0.021477852311749167`. And repeating the whole bootstrap under 40 independent RNG seeds: **`lcb_positive_fraction_over_rng_seeds = 0.65`**, inside the preregistered instability band [0.05, 0.95].

**Which supersedes which: B supersedes A.** Grounds, in order of force:

1. **Dates and lineage.** B (2026-08-07T18:44Z) postdates A's source (2026-08-06T23:01Z) and A itself (commit `eb132a1`, 2026-08-07 morning). B was committed in `3b36875`, the same commit that wrote `docs/TABLA_CANONICA_DE_CLAIMS_2026-08-07_ENMIENDA_1.md`.
2. **The amendment is the successor document and says so.** ENMIENDA_1 opens by declaring it succeeds `TABLA_CANONICA_DE_CLAIMS_2026-08-07.md`, "que **no se edita**". Its §3 is titled "prohibido escribir «excluye el cero»". The base table is preserved unedited by repository rule (retired results are labelled, not deleted) — so A still existing is not evidence that A is current.
3. **B subsumes A's evidence rather than contradicting it.** The mean is identical to the last digit in both ladders. B is not a competing measurement; it is a stability analysis *of* A's estimator, and it demonstrates that A's bound sign is a property of the bootstrap draw, not of the data.
4. **The runner met its own preregistered reading rule.** ENMIENDA_RECONCILIACION §4 fixed the thresholds *before* the run: fraction in (0.05, 0.95) → `UNSTABLE`; ≥0.95 → name the positive sign; ≤0.05 → name the non-positive sign. Observed 0.65 falls in the middle band. Falsifier `f3_a_stable_bound_would_be_reported_as_stable` exists to prevent hedging when the fraction is 0 or 1 — and it passes.

**Mandated citation form** (from the artifact's `how_to_report_it` and ENMIENDA_1 §3):

> `neuron_memory` and `ofat_transfer` are **indistinguishable** in regret AUC (mean **+0.01071**; the bootstrap lower bound falls on both sides of zero depending on the resample: **−2.76e−05** in `search_ladder_v2_ordered`, **+3.56e−05** in `search_ladder_v5`).

Both sealed bounds must be quoted. Writing "excludes zero" for this contrast is explicitly prohibited.

Two things worth carrying into the manuscript. First, ENMIENDA_1 §3 and the amendment §2 both record that an **external audit quoted −0.0000276 and was told it was wrong** — the audit was right, it was quoting the other sealed artifact, and the correction is owed in writing. Second, the neighbouring contrast is **stable**: `neuron_memory − ucb1_transfer` has LCB positive in **0%** of the 40 seeds (`mean −0.00700974`, stored bounds `−0.0243436` in v2_ordered and `−0.0244413` in v5). "The neuron does not beat UCB1-with-transfer" is robust and can be stated without hedging.

## 5. The normaliser audit

**`results/twin_surface_v2/result.json`** — the structural spy test. `PREFIX_NORMALISER_IS_BLIND_TO_THE_UNRUN_SURFACE`, grade **REPLAY**, scope `DEVELOPMENT_ON_BURNED_TAPES_STRUCTURAL_SPY_TEST`, `self_sha256` `04b8137157e83a612a81e2d10995e5adf5553cb58370bceb18046559034dd559`, seed 5300001, created 2026-08-05T23:18:27Z.

The test builds a twin surface with perturbed *unrun* tail cells and checks whether each arm's visit path changes. Under **prefix**, `all_paths_unchanged: true` — all four strategies × six contexts unchanged. Under **oracle**, `all_paths_unchanged: false` — `neuron_memory` and `neuron_reset` change path in **all 6/6 contexts**, while `ofat` and `random` (which do not consult the normaliser) stay put. Both required conditions hold: `prefix_passed: true` and `oracle_reacted: true`. The oracle normaliser reads cells that were never executed. That is the leak.

**`results/garrido_normaliser_audit_v3/result.json`** — the panel. `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER`, grade **REPLAY**, scope `DEVELOPMENT_REPLAY_ON_BURNED_TAPES_NO_ADJUDICATION_NO_LEARNER`, `run_role = BURNED_REPLAY_AUDIT`, `self_sha256` `b1cf3705a070a191…`, contract `docs/PREREGISTRO_AUDITORIA_NORMALIZADOR_2026-08-05.md` sha `bc9d8cf78d370e82…`, 288 configurations, 12 repeats, seeds 5300001–012.

`memory_vs_reset`, prefix vs oracle:

| endpoint | prefix | oracle |
|---|---|---|
| `auc_regret_norm` | **+0.06070331 [+0.04556081, +0.08019782]** | +0.09014693 [+0.07419196, +0.10802658] |
| `runs_to_within_1pct` | **+5.83333 [+4.44444, +7.30556]** | +7.90278 [+6.81944, +8.93056] |

Arm means under prefix: `neuron_memory` 0.05203, `neuron_reset` 0.11274, `ofat` 0.10024, `random` 0.13851. Under oracle `neuron_reset` inflates to 0.14131 — the leak's whole footprint sits on the reset arm.

**Authoritative artifact: `garrido_normaliser_audit_v3/`.** `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md:39` names it, superseding `_smoke` (which is `HALTED_FALSIFIER_FAILED`), the base, and `_v2` (registry marks `_v2` as `duplicate_of` v3). Falsifier f1 confirms it reproduces the sealed `garrido_meta_learner_v2` means with `abs_delta = 0.0` on all four arms.

**Consequence for the manuscript, and it is binding.** The draft's H4 figure `+7.90 runs [+6.88, +8.93]` is the *oracle* number and is on the prohibited list (ENMIENDA_1 §2, INDICE §4). Mandatory substitution: primary **AUC +0.06070 [+0.04556, +0.08020]**; censored secondary **5.83 runs [+4.44, +7.31]**, always labelled as censored. Also retired: `7.24 / 13.54 / 12.42`. This is easy to miss because the draft quotes the number without naming its normaliser.

## 6. H1–H4 adjudications

**`results/manuscript/h1_h3_originales_v3/result.json`** — `H1_SUPPORTED__H3_NOT_SUPPORTED`, grade **DEVELOPMENT**, scope `DEVELOPMENT_ALREADY_OPEN_BLOCK_NO_VIRGIN_SEEDS_NO_ADJUDICATION`, `self_sha256` `dc46ce6069755a28…`, `file_sha256` `39061791dd37eef4…`, contract `24a1fe62fcaab9ad…`, created 2026-08-07T16:09:05Z, seeds 6000001–6000120.

**Development, not confirmation** — its own scope string says so, and the registry rationale reads "contracted, but not a confirmation over a virgin block."

- `H1_hybrid_vs_static = +125.98541666666667 h [+98.346796875, +154.54471354166665]`, n_cells 960, p_holm 0.0
- `H1_hybrid_vs_reset = +74.04583333 h [+51.52382813, +97.46838542]`, p_holm 0.0
- Mechanism is **absorption**: hybrid absorbs 875/960 shocks vs static 755/960, reset 823/960
- Levels: hybrid 75.7 h, reset 149.75 h, static 201.69 h; τ = 1344 h, H1 horizon 6048 h
- `H3_hybrid_vs_reset` p_raw 0.8208 / p_holm 1.0; `H3_hybrid_vs_static` p_raw 0.5248 / p_holm 1.0 — **sign is wrong and the CI crosses zero, with the estimand present.** Not a power failure, a refutation.
- Endpoint redefinition is declared in the artifact: `restricted_ttr = min(TTR, τ)` with a paired placebo, not `system_ttr`. Written 2026-08-06, before this preregistration. The artifact states it is a different estimand, not a loosened one — the earlier `system_ttr` version returned 1.000 and halted (`h1_h3_v1` = `HALTED_FALSIFIER_FAILED`, graded NEGATIVE_OR_HALTED, and per ENMIENDA_2 must be cited *with* its failure). Handle this paragraph carefully in the manuscript; it is the one place a reviewer will read endpoint-shopping unless the timeline is on the page.

**`results/manuscript/h2_learning_curve/result.json`** — `H2_SUPPORTED_LEARNING_CURVE`, grade **DEVELOPMENT**, scope `DEVELOPMENT_REANALYSIS_OF_SEALED_ARTIFACTS_NO_SEEDS_NO_ADJUDICATION`, `self_sha256` `74b75141241ba7639f629e2ba8db0ef28bff480fe706489f8977dcd4d4bf48fa`, `file_sha256` `2894e525dc360c8f…`, contract `7741309cf93fa4e0…`, created 2026-08-07T18:39:39Z.

**Development — a reanalysis of sealed artifacts, zero new seeds.**

- Primary slope `+0.04220148 [+0.03466394, +0.04992206]`, n=120
- Null control (`random − ofat`) `−0.00508806 [−0.01557022, +0.00565800]` — **crosses zero**, so the trend is not escalating context difficulty
- Advantage by context: R1r **+0.00000** (structural — with nothing to carry, memory and reset are the same arm), R2r +0.28275, R1r+R2r +0.19052, R1r|esc +0.22111, R2r|esc +0.31709, R1r+R2r|esc +0.26869
- Estimand is explicit: slope, not level. A large but flat advantage would support H4, not H2.
- Falsifier f2 verifies the normaliser is **prefix, not oracle** — reproducing the oracle panel would reintroduce the leak that retired the original H2 figure.
- Sources: the contracted n=120 pair, 90 replicates local (`e768e0f69f187272…`) + 30 vps (`ad240fbb9dda507a…`).

**`results/garrido_h3_merge_adjudication/result.json`** — `H3_PRIME_SUSTAINED_AT_N120`, grade **DEVELOPMENT**, `self_sha256` `1ac02efa1618e5a9…`, contract `docs/PREREGISTRO_H3_POTENCIA_2026-08-01.md` sha `576d02b5de7609eb…`, created 2026-08-02T14:18:26Z.

**It has no `scope` key and no `run_role` key.** Every other artifact in this inventory carries one. If the manuscript needs a scope string for H3′, it must be derived from the registry grade (DEVELOPMENT) and the custody note, not quoted — there is nothing to quote. Flagging it because a scope string is exactly what an auditor will ask for.

- `memory_vs_reset = +9.31444444 [+2.34912500, +16.34740972]`, n=120
- `memory_vs_ofat = +16.22027778 [+9.61070833, +22.74053472]`
- Reading rule preregistered: LCB95 > 0 → sustained; ≤ 0 at n=120 → refuted with power
- Per-slice diagnostic: local_90 `+10.26593 [+2.36085, +18.19272]`, vps_30 `+6.46000 [−7.30878, +20.32814]` — the 30-replicate slice alone does not clear zero; the merge does
- Estimand is narrow and the artifact says so: **variance of search cost across contexts**, memory minus reset. Explicitly *not* the Alzheimer mean. Merging seeds 6000001–090 and 6000091–120, disjoint, identical module manifests.

**H4** has no dedicated directory. Its canonical value is the `memory_vs_reset` prefix contrast from `garrido_normaliser_audit_v3`: **AUC +0.06070 [+0.04556, +0.08020]**, secondary censored 5.83 runs [+4.44, +7.31]. See §5 for the prohibited draft figure. The related `results/manuscript/h1_h3_v2_1/result.json` (`self_sha256` `76f303d8e09b1e52…`, 2026-08-01) is the declared **H1′** reformulation on `service_loss_auc_ration_hours`, and `h1_h3_originales_v3` explicitly does not reopen it.

**Net status of the four hypotheses** (ENMIENDA_1 §1): H1 sustained, H2 sustained, H4 measured, **H3 not sustained** with H3′ sustained as a declared reformulation. Three of four. The amendment's own instruction: do not round to four.

---

## What is missing for a publishable manuscript

- **Zero confirmatory evidence for the title claim.** All three CONFIRMATORY artifacts are `grid_transfer_confirmation_v2`, `gsa_confirmation`, `garrido_h2_h3_confirmation_v1`. The ladder, both bakeoffs, both normaliser artifacts and all H1–H4 adjudications are REPLAY or DEVELOPMENT on burned tapes. The one confirmed learning result runs *against* the neuron.
- **No virgin seed blocks remain.** `docs/TABLA_CANONICA_GENERADA.md`: `BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `new_seed_opening: false`, **0 virgin blocks left**. No further confirmation is possible without authorising new seeds. (INDICE §5 still describes `g3a_v2_development` / 7700001–7700120 as `RESERVED_NOT_OPENED` — but the generated table, which is newer and derives from the registry, reports zero. These two disagree and it is worth reconciling before anyone plans a confirmation run.)
- **Two of the Fig. 5 artifacts are UNCONTRACTED**, as is `architecture_bakeoff`. Nothing fixes what those runs were allowed to claim, and the residual/identity result is one of the manuscript's cleaner findings.
- **`architecture_bakeoff`'s DMLPA arm is a de-KAN'd network** and its provenance is not certifiable. If the manuscript cites a DMLPA number, it is citing the wrong architecture.
- **`garrido_h3_merge_adjudication` has no scope string.**
- 191 of 216 registry keys are incomplete (88%), mostly missing `execution` (140), `endpoint` (139), `seed_block` (110).
- Branch is `codex/expanded-contract-comparators-v2` at `e027fd5`; `main` (`89acc81`, 28-Jul) is ~823 commits behind and is not a current source.
