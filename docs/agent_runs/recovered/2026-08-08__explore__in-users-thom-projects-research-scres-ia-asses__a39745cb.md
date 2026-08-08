# Explore — In <HOME>/Projects/research/scres-ia, assess release readiness for a journal submissi

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a39745cb91016db33`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T01:38:43.686Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-a39745cb91016db33.jsonl`
- **turns**: 256 · **assistant messages**: 3

## Task given

```text
In <HOME>/Projects/research/scres-ia, assess release readiness for a journal submission. Read-only. Be concrete.

1. Run `.venv/bin/python -m pytest tests/ -q --timeout=600 2>&1 | tail -40` (read-only test run is fine) and report the exact collected/passed/failed counts and the names of failing tests grouped by cause. Four audits claim "21 known failures". Verify the real current number.
2. Find any triage doc for those failures (grep docs/ for triage / TRIAGE / fallos / test). Report what it says.
3. Provenance: find `docs/` files recording gaps in Paper-2 provenance — one audit claims `supply_chain.py` changed AFTER the manifests that anchor some results, breaking reproducibility. Verify: grep for manifest/provenance gap registries, and check whether `module_manifest` hashes stored in `results/search_ladder_v5/` and `results/grid_transfer_confirmation_v2/` still match the current `supply_chain/` files. Report matches/mismatches explicitly.
4. Figure builders: read `scripts/build_cie_outer_loop_figures.py` — list which artifact paths it reads, and check each against the authoritative index to see whether it points at superseded artifacts (search_ladder_v2 vs v5, twin_surface vs twin_surface_v2, surface_gates vs surface_gates_v2).
5. Find `scripts/build_submission_a_cie_review_bundle.py` or any anonymised-bundle builder / identity guard anywhere in the repo INCLUDING other branches (`git log --all --diff-filter=A --name-only -- '*bundle*'`). Report the real path and branch.
6. Report the current branch and its exact HEAD SHA, plus `git rev-parse origin/codex/submission-a-program-q` if it exists, and the ahead/behind counts between them.

Report findings as a concrete list with file paths and numbers.
```

## Final answer

Suite finished. Here are the verified findings.

---

## 1. Test suite — real current number: **21 failed, 2277 passed**, 2 skipped, 2 xfailed, 705.27s (11:45)

`--timeout=600` is **not a valid flag** — `pytest-timeout` is not installed in `.venv`; the command in the request errors out immediately (`unrecognized arguments: --timeout=600`, `inifile: <HOME>/Projects/research/scres-ia/pytest.ini`). Re-run without it. Working tree was clean (`git status --porcelain` → 0 lines), so this reflects committed state at `e027fd5`.

The **count** 21 is right; the **set** is not what the audits describe. Grouped by cause:

**A. Markov-completeness guard / exact-transducer family — 16**
- `tests/test_paper2_bottleneck_exact_transducer.py` — 8 (`test_immutable_sim_contract_state_is_bound_into_every_key`, `test_sequential_run_prefix_calls_do_not_create_natural_binding_drift`, `test_full_completed_order_and_risk_histories_are_conservatively_serialized`, `test_small_horizon_primary_transducer_matches_every_brute_calendar`, `test_real_collision_certificate_binds_finite_markov_theorem`, `test_two_fresh_isolated_w2_processes_emit_distinct_bound_receipts`, `test_signed_harness_receipt_normalizes_to_exact_chain_w1`, `test_every_live_simulator_attribute_is_classified_fast`)
- `tests/test_paper2_bound_execution_harness.py` — 6 (all checksum/session tests)
- `tests/test_paper2_bottleneck_full_frontier.py` — 2 (`test_w6_real_canonical_frontier_matches_unaccelerated_brute_and_replay`, `test_checkpoint_resume_invalidates_on_dependency_hash_drift`)

**B. Post-freeze hash drift on `supply_chain/supply_chain.py` — 3**
- `tests/test_paper2_exhaustive_search_registry.py::test_reproducibility_manifest_hashes_every_listed_artifact_and_source`
- `tests/test_paper2_exhaustive_search_registry.py::test_canonical_v2_metric_governance_quarantines_every_visible_v1_claim`
- `tests/test_program_j_request_snapshot_v2_structure_audit.py::test_program_j_structure_audit_is_content_and_source_addressed`

Both assertion messages are the same pair: expected `d8fd93475b3904bc…`, actual `74ed0909950c377f…`. **This contradicts the triage doc**, which attributes this failure to `scripts/bound_program_o_affected_orders.py`. That is stale — the current binding that fails is `supply_chain/supply_chain.py`.

**C. Stale artifact certificate — 1**
- `tests/test_k3_frontloading_dominance.py::test_direct_cli_needs_no_pythonpath_and_checked_json_is_current` → `stale or missing K3 certificate: research/paper2_exhaustive_search/k3_frontloading_dominance_certificate.json`

**D. New failure, in none of the audits — 1**
- `tests/test_program_q_contract.py::test_live_program_q_seed_custody_declarations_do_not_fake_a_collision` → `STOP_PROGRAM_Q_SEED_COLLISION`

I ran the scanner directly. The offending file is **`docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md:95`**, added by `a9b878d` (2026-08-07 12:35): `Quemados por confirmación: 8200001–060, 7490001–256.` The en-dash range is not parsed as a bounds declaration, so `seeds_declared_as_bounds` is empty and 7490001/7490256 read as *live use* of reserved seeds. **The authoritative-artifact index broke the seed-custody guard.** Same failure mode the triage doc warns about (the July 31 custody repair regressed the same way).

---

## 2. Triage docs — three, and they disagree with each other and with reality

- **`docs/TRIAJE_SUITE_21_FALLOS_2026-08-06.md`** — the primary one. Says suite went 0 → 21 between 07-31 and 08-06; verified none came from that day's work. Claims one *fixed* (`test_seed_custody_registry::test_seed_registry_is_fail_closed_before_submission_receipt`, whitelisting `BURNED_CONFIRMATION_COMPLETE`). Blames the registry failures on `ca84f39` editing `scripts/bound_program_o_affected_orders.py` after `9eb2433` froze it. States the PI decision explicitly: *re-freeze with written justification, or declare the provenance broken*. ~10 failures listed as "sin triar".
- **`docs/SUITE_FAILURE_TRIAGE_2026-07-31.md`** — the earlier 27-failure triage, six root causes, all marked ARREGLADO. Cause A (Markov guard, 15 tests) documents 29 unclassified simulator fields and concludes *"el bound exacto de Paper 2 no debe citarse como vigente sobre el simulador actual."* Those 16 A-group failures above are that guard, back.
- **`docs/REGISTRO_DE_HUECOS_2026-08-07.md` §B3** — claims **"20 tests siguen en rojo. De 21, uno arreglado."** **This is wrong.** The count is still 21: the seed-custody fix landed, but the new `test_program_q_contract` failure replaced it one-for-one.

---

## 3. Provenance — the audit's claim is **CONFIRMED**, with the precise scope

Gap registry: **`docs/REGISTRO_DE_HUECOS_2026-08-07.md` §A2 "Procedencia rota de Paper 2"** — *"`supply_chain/supply_chain.py` lleva 12+ commits desde el manifiesto del 14-jul… Re-congelar sería afirmar algo falso. Cierre: decisión del PI — re-correr los artefactos bajo la física actual, o retirar por escrito los claims que anclan."* Also `docs/PROVENANCE_REPAIR_AUDIT_2026-07-31.md` (both original blockers withdrawn; one real gap remains: the dense-frontier `+0.000542` has no located artifact) and `docs/ENMIENDA_SELLADO_RETROACTIVO_BAKEOFF_2026-08-07.md` (`run_architecture_bakeoff_v1.py` never sealed; retroactive record ≠ provenance).

Machine-checkable registry: **`scripts/classify_paper2_hash_drift.py`**. Current output:
```
program_j_structure_audit     post_freeze_source_edit  supply_chain/supply_chain.py
reproducibility_manifest      post_freeze_source_edit  supply_chain/supply_chain.py
                              missing_external_source  x8 (Raw_data1+Re.xlsx, Rsult_1.xlsx, WRAP thesis PDF, …)
v2_implementation_audit       post_freeze_source_edit  supply_chain/supply_chain.py
resumen: {"missing_external_source": 8, "post_freeze_source_edit": 3}
```

**Manifest hash verification (recomputed SHA-256 against current working tree):**

`results/search_ladder_v5/result.json` → `module_manifest` (entry `scripts/run_search_comparator_ladder_v5.py`):
| file | status |
|---|---|
| `scripts/run_search_comparator_ladder_v5.py` | MATCH `bfc62c5c341631fa` |
| `supply_chain/seed_custody.py` | MATCH `3d46668e6cb62ee5` |
| **`supply_chain/arm_runner.py`** | **MISMATCH** — manifest `35135c537d926083`, current `1afea4fa248db5be` |

`results/grid_transfer_confirmation_v2/result.json` → `module_manifest` (entry `scripts/run_grid_transfer_v1.py`):
| file | status |
|---|---|
| `scripts/run_grid_transfer_v1.py` | MATCH `727a8d7d2d5ed6fd` |
| `supply_chain/seed_custody.py` | MATCH |
| **`supply_chain/arm_runner.py`** | **MISMATCH** (same as above) |

…and `cache_module_manifests` (**both** `base` and `extended`, entry `scripts/build_transfer_confirmation_cache_v1.py`):
| file | status |
|---|---|
| `scripts/build_transfer_confirmation_cache_v1.py` | MATCH `ccb85cbfa0f12636` |
| `supply_chain/config.py` | MATCH `ae702a4991966504` |
| `supply_chain/episode_metrics.py` | MATCH `e09199d48a747f67` |
| `supply_chain/seed_custody.py` | MATCH |
| **`supply_chain/supply_chain.py`** | **MISMATCH** — manifest `2f348e5922cbbecf`, current `74ed0909950c377f` |

**The "changed AFTER the manifests" ordering is confirmed by commit timestamps:**
| | commit | timestamp |
|---|---|---|
| `search_ladder_v5/result.json` sealed | `46d56bc` | 2026-08-06 18:02:47 |
| `grid_transfer_confirmation_v2/result.json` sealed | `ed16b9e` | 2026-08-06 04:40:49 |
| `supply_chain/arm_runner.py` last edit | `8ddf6f7` | **2026-08-07 17:56:05** |
| `supply_chain/supply_chain.py` last edit | `cc3af32` | **2026-08-07 20:03:49** |

At both manifest commits the two files still hashed to the recorded values (`git show 46d56bc:…` and `ed16b9e:…` reproduce `35135c53…` / `2f348e59…` exactly). The drift is strictly post-seal, introduced yesterday.

**Two mitigating facts:**
- The artifacts themselves are intact. I recomputed `self_sha256` using the real seal algorithm (`supply_chain/arm_runner.py:194`, `json.dumps(payload, indent=1, sort_keys=True, default=str)`): `SEAL_OK` for `search_ladder_v5`, `grid_transfer_confirmation_v2`, `twin_surface`, `surface_gates`, `search_ladder_v2`. On-disk file hashes also match the frozen claim table (`search_ladder_v5` → `f648a1da5aefaf2f` = A1; `grid_transfer_confirmation_v2` → `7bc33823ccd90b5e` = A2; `surface_gates` → `954ac48301ff1234` = A7). Nothing was tampered with — only the source side moved.
- **Nothing in `tests/` verifies these `module_manifest` hashes.** The four files matching `module_manifest` (`test_grid_transfer_confirmation.py:15`, `test_garrido_surface_cache_custody.py:41`, `test_g3c_temporal_coupling_physics.py`, `test_seed_custody_module.py`) only assert *shape* (`{"modules": {"physics": "frozen"}}`, `{"schema": "module_manifest_v2"}`). **A2 — the project's only `CONFIRMATION`-grade result — lost source identity yesterday and the suite was silent about it.** `supply_chain/seed_custody.py:162` docstring anticipates exactly this: *"it cannot retroactively rescue artifacts that were sealed without one"* — but here the manifest exists and simply isn't checked.

---

## 4. `scripts/build_cie_outer_loop_figures.py` — 4 of 5 figures read non-citable or superseded artifacts

Loads via `RESULTS = Path("results")` (line 28):

| line | figure | reads | status |
|---|---|---|---|
| 66 | `fig_a_normaliser_leak` | `twin_surface/result.json` | **STALE** — citable table lists `twin_surface_v2` (§7, `04b8137157e83a61…`). `twin_surface` is `NOT_ADMITTED` |
| 92 | `fig_b_surface_gates` | `surface_gates/result.json` | **OK but contested** — is A7 in the claim freeze; but amendment 2 admits `surface_gates_v2` for §5 and the index lists `_v2` as superseded. Two governance docs disagree |
| 135 | `fig_c_comparator_ladder` | `search_ladder_v2/result.json` | **SUPERSEDED** |
| 160 | `fig_d_memory_price` | `search_ladder_v2/result.json` | **SUPERSEDED** |
| 189 | `fig_e_delta_efficiency` | `search_surrogates/result.json` | **NOT CITABLE** — not in A1–A9 nor amendment 2. A5 is `surrogate_architecture_bakeoff` |

Cross-checked against `docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md:37`: *escalera de búsqueda → **`search_ladder_v5/`***, superseded: *`search_ladder`, `_v2`, `_ordered`, `_v3`, `_v4`, `_v2_ordered`*. And `docs/GARRIDO_Q1_Q2_CLAIM_FREEZE_2026-08-07.md:16`: A1 = `search_ladder_v5`.

**This isn't cosmetic — the claim changes.** `search_ladder_v2` `claim_status` = `NEURON_LEADS_BUT_NOT_ABOVE_EVERY_MEMORY_ARM`; `search_ladder_v5` = `THE_NEURON_HOLDS_AGAINST_LOOKAHEAD_SEARCH`. v5 adds 4 arms (`lookahead_kg`, `lookahead_kg_transfer`, `thompson`, `thompson_transfer`) and reorders: `neuron_reset` falls from rank 8 of 12 to rank 13 of 16. The figures on disk (`fig_*.pdf`, all mtime **5 Aug 14:38**) predate `search_ladder_v5/result.json` (**6 Aug 18:04**) and `twin_surface_v2` (5 Aug 18:18).

Good news: **v5 is drop-in compatible** — it has `mean_auc_regret` and `per_arm[*].auc`, exactly what lines 135/160 consume. Repointing needs `LABEL` entries for the 4 new arms and `MEMORY_ARMS` (line 49) extended.

**Root cause worth fixing:** the sibling builder `scripts/build_paper1_evidence_v1.py:84-87` has an admission guard — `load_admitted()` reads `results/claim_freeze_amendment_2/result.json` and `raise SystemExit` if a source isn't admitted. `build_cie_outer_loop_figures.py` has **no such guard**, and no test in `tests/` (217 files) references it or `search_ladder_v5`. Its own docstring claims *"a figure cannot drift away from the evidence it claims to show"* — true for values, false for which artifact.

Also: none of `fig_a`…`fig_e` are referenced anywhere in `docs/manuscript_current/submission/elsevier/` (that `main.tex` is the July Paper-2 Elsevier build). And `docs/REGISTRO_DE_HUECOS_2026-08-07.md` §D4 says the C&IE manuscript is *"No escrito. Todos los números existen."*

---

## 5. Anonymised-bundle builder — **not on any local branch**

Real path: **`scripts/build_submission_a_cie_review_bundle.py`**, 134 lines, added by **`ba968909cd0f4a5fe952c7ff8732ef8e6ca56345`** ("Prepare Program Q double-anonymous CIE package", Thomas Chisica, 2026-07-28 22:50:37).

Present on exactly two refs, **both remote-only**:
- `origin/codex/submission-a-cie-guide-compliance` (`ba968909`)
- `origin/codex/submission-a-program-q` (`1fad6bb6`)

It does **not** exist in the current working tree, and it does **not** exist on the local `codex/submission-a-program-q` branch (`031d0af9`, which is **0 ahead / 10 behind** origin). I scanned every branch tree — no other repo has an anonymiser. `papers/submission_a_program_q/` also does not exist locally (`papers/` has only `cie_chassis`, `paper1_unobserved_frontier`, `paper2`).

The identity guard is `assert_anonymous()`, raising `STOP_ANONYMITY_GUARD`, scanning `.tex/.bib/.txt/.md/.json` for `IDENTITY_TOKENS = ("thom", "chisica", "urosario", "github.com/thom-320", "<USER>", "alexander.garrido@")`. Deterministic ZIP via `FIXED_ZIP_TIME = (1980,1,1,0,0,0)` + `ZIP_STORED`.

---

## 6. Branch and SHAs

| | |
|---|---|
| Current branch | **`codex/expanded-contract-comparators-v2`** |
| HEAD | **`e027fd5ca46a3fe2809103591a4e4350db31ff1f`** (2026-08-07 20:18:43, *"Paper 1: one extraction, three figures, and a macro layer the prose cannot bypass"*) |
| vs its own upstream `origin/codex/expanded-contract-comparators-v2` | **7 ahead, 0 behind** (7 unpushed commits) |
| `origin/codex/submission-a-program-q` | **`1fad6bb6218edca63a5bb9bc295e823ac2b6cd9d`** (2026-07-28 22:53:24, merge of PR #20) |
| HEAD vs `origin/codex/submission-a-program-q` | **442 ahead, 25 behind**; merge base `c2fa5cb` (2026-07-17) |
| HEAD vs `origin/main` (`89acc813…`) | **839 ahead, 8 behind** |

---

## Release-readiness bottom line

Five hard blockers, in severity order:

1. **A2 (`grid_transfer_confirmation_v2`), the only `CONFIRMATION`-grade result in the project, lost verifiable source identity on 2026-08-07** — `supply_chain.py` and `arm_runner.py` both drifted post-seal, and no test catches it. The gap registry already frames this as a PI decision (re-run, or withdraw the claims in writing); it now applies to the confirmed claim, not just Paper 2.
2. **The C&IE figures ship superseded science.** `fig_c`/`fig_d` draw the v2 ladder whose `claim_status` was replaced; `fig_a` and `fig_e` draw artifacts absent from the citable table. The builder lacks the admission guard its sibling has.
3. **The 21 red tests are not the 21 that were triaged.** The registry failure has a new cause, and one failure (Program Q seed collision) is entirely new and self-inflicted by the index doc committed yesterday. `REGISTRO_DE_HUECOS` §B3's "20" is wrong.
4. **`docs/INDICE_DE_ARTEFACTOS_AUTORITATIVOS.md` and `..._CLAIM_FREEZE_..._ENMIENDA_2.md` disagree** on whether `surface_gates_v2` is authoritative or superseded. Both dated 2026-08-07.
5. **The anonymised submission bundle and its identity guard are not in the working tree** — they live 25 commits back on a remote branch, together with the entire `papers/submission_a_program_q/` package. Whatever gets submitted from this branch has no anonymity check.

Not blockers but worth flagging: `pytest-timeout` isn't installed (any CI invoking `--timeout` fails at collection); the K3 certificate is stale; and the C&IE manuscript itself is unwritten per §D4.
