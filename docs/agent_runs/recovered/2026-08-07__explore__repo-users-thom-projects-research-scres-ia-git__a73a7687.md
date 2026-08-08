# Explore — Repo: <HOME>/Projects/research/scres-ia (git, ~70 branches, ~112 dirs under results/)

> **`UNADJUDICATED_DO_NOT_CITE`** — transcripción cruda de un agente, guardada por el hook
> `SubagentStop`. Puede contener afirmaciones **retiradas**; el canon vive en los artefactos
> sellados y en las enmiendas, nunca aquí.

- **agent_id**: `a73a768783cc8d34f`
- **session**: `c4c70a0d-d0da-4c53-824e-5d7a181768e6`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-07T16:44:29.518Z`
- **effort**: `high` · **version**: `2.1.222`
- **transcript**: `<HOME>/.claude/projects/-Users-thom-Projects-research-scres-ia/c4c70a0d-d0da-4c53-824e-5d7a181768e6/subagents/agent-a73a768783cc8d34f.jsonl`
- **turns**: 55 · **assistant messages**: 4

## Task given

```text
Repo: <HOME>/Projects/research/scres-ia (git, ~70 branches, ~112 dirs under results/).

GOAL: build a census of which scientific results are CONFIRMATION-GRADE versus DEVELOPMENT-GRADE, across the WHOLE repository AND across all git branches (not just the current one).

Method:
1. Under `results/`, find every JSON with a `claim_status` field. For each, extract: path, `claim_status`, `scope`, `run_role` (if present), `seeds` (first/last + count), `created_at`, `contract_path`, whether `self_sha256` exists.
2. Classify each into: CONFIRMATION (run_role == "CONFIRMATION", or scope mentioning CONFIRMATION / VIRGIN / PROSPECTIVE), DEVELOPMENT, DIAGNOSTIC, HALTED/STOP, or REPLAY.
3. Separately: use `git log --all --diff-filter=A --name-only -- 'results/**/result.json'` or `git ls-tree` on other branches to find result artifacts that exist ONLY on other branches and never landed on the current branch `codex/expanded-contract-comparators-v2`. For those, report branch, path, and claim_status (read them with `git show <branch>:<path>`).
4. Also read `research/seed_custody_registry.json` and report which seed blocks are marked BURNED / OPEN / RESERVED / never-opened.

OUTPUT (be compact and factual, no narrative):
- Table A: every CONFIRMATION-grade artifact (path, claim_status, seed block, date). This is the most important output — there should be only a handful.
- Table B: the 15 highest-value DEVELOPMENT artifacts by apparent claim strength (positive effects with intervals excluding zero), with the actual numbers.
- Table C: artifacts that exist only on other branches, with branch + claim_status.
- Table D: seed blocks by state, and which blocks remain unopened/virgin.

Do NOT modify anything. Read-only. Report exact paths and numbers.
```

## Final answer

Census complete. Findings below.

## Scale

- 212 JSON files under `results/` on HEAD (`codex/expanded-contract-comparators-v2`) carry a top-level `claim_status`; 113 top-level dirs; 95 refs (heads+remotes).
- 226 result JSONs exist on other branches and never on HEAD; 147 of those carry a `claim_status`.
- Only **1** artifact on HEAD has `run_role == "CONFIRMATION"`.

---

## Table A — CONFIRMATION-grade artifacts

| Path | claim_status | scope / run_role | Seed block | Date |
|---|---|---|---|---|
| `<HOME>/Projects/research/scres-ia/results/grid_transfer_confirmation_v2/result.json` | `GRID_TRANSFER_CONFIRMED__UCB1` | `CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER` / `run_role=CONFIRMATION` | 8200001–8200060 (n=60), block `garrido_grid_transfer_v2_confirmation`, now `BURNED_CONFIRMATION_COMPLETE` | 2026-08-06T09:39:38Z |
| `<HOME>/Projects/research/scres-ia/results/metric_audit/ret_metric_repair_confirmation_v1/result.json` + `adjudication.json` | `PROSPECTIVE_CORRECTIVE_CONFIRMATION` | no `scope`, no `run_role`, no `created_at`; contract `contracts/ret_metric_repair_confirmation_v1.json` | not declared in artifact | (absent) |
| `<HOME>/Projects/research/scres-ia/results/sensitivity/backlog_confirmation_v1/result.json` | `REFUTED_BACKLOG_SENSOR_WAS_SELECTION_NOISE` | preregistered confirmation, refuted | 5100001–5100012 (n=12), block `reconciled_5100001` = `USED_DEVELOPMENT_NOT_VIRGIN` | 2026-08-01T01:49:09Z |
| `<HOME>/Projects/research/scres-ia/results/k3/confirmation.json` | (no `claim_status`) `verdict=CONFIRM_K3_OBSERVABLE_RET_HEADROOM`, **`effective_verdict=RETRACT_K3_ADAPTIVE_AND_NEURAL_CLAIMS_STATIC_PERIOD8_CONFOUND`**, superseded by `results/k3/open_loop_confound_audit.json` | `stage=locked_confirmation` | 6800001–6800120 | — |

Confirmation-adjacent custody records (no science): `results/custody/garrido_grid_transfer_confirmation_preflight.json` and `..._v2_preflight.json` (`PREOPEN_CONFIRMATION_PREFLIGHT_PASS`), `..._v2_open.json` (`CONFIRMATION_BLOCK_OPEN_NO_RESULT_YET`), `garrido_grid_transfer_confirmation_abort.json` (`CONFIRMATION_BLOCK_QUARANTINED_NO_SCIENTIFIC_RESULT`), `results/headroom/g3_obs_conversion_v2_replay_20260803/result.json` (`REPLAY_OF_BURNED_BLOCK_NO_NEW_CONFIRMATION`, seeds 7800001–7800140).

The one real confirmation's own numbers (`results/grid_transfer_confirmation_v2/result.json`, n=60 each, all 8 falsifiers passed):

| arm | vs_cold mean [lcb95, ucb95] | vs_marginal_replay mean [lcb95, ucb95] |
|---|---|---|
| ucb1 | 0.057438 [0.049889, 0.064806] | **0.030733 [0.019897, 0.042562]** (only arm > 0) |
| neuron | 0.054394 [0.042896, 0.066871] | −0.011783 [−0.018489, −0.004836] |
| gp | 0.014326 [0.008788, 0.020561] | −0.021595 [−0.030506, −0.012267] |
| ofat | 0.014219 [0.008003, 0.019319] | −0.024672 [−0.032580, −0.016662] |

Caveat recorded inside `f4_seed_custody`: `registry_is_complete: false` → verdict is `NO_KNOWN_COLLISION`, not proof of virginity.

---

## Table B — 15 highest-value DEVELOPMENT artifacts (positive effect, 95% interval excludes zero)

439 positive-CI entries total across development artifacts. Ranked by strength/independence:

| # | Path | claim_status | Quantity | mean [lcb95, ucb95] | Seeds |
|---|---|---|---|---|---|
| 1 | `results/manuscript/h1_h3_originales_v3/result.json` | `H1_SUPPORTED__H3_NOT_SUPPORTED` | `contrasts.H1_hybrid_vs_static` | 125.985 [98.347, 154.545] | 6000001–6000120 (n=120) |
| 2 | `results/manuscript/h1_h3_originales_v3/result.json` | same | `contrasts.H1_hybrid_vs_reset` | 74.046 [51.524, 97.468] | 6000001–6000120 |
| 3 | `results/manuscript/h1_h3_originales_v3/result.json` | same | `secondary_descriptive.H1_hybrid_vs_static_on_differing_cells` | 159.981 [123.993, 195.614] | 6000001–6000120 |
| 4 | `results/garrido_h3_merge_adjudication/result.json` | `H3_PRIME_SUSTAINED_AT_N120` | `h3_prime.memory_vs_ofat` | 16.220 [9.611, 22.741], n=120 | 6000001–6000120 |
| 5 | `results/dmlpa_variants/result.json` | `A_DMLPA_VARIANT_SEPARATES` | `trained_minus_untrained` = `f3_training_beats_not_training` | 20.4377 [18.9606, 22.0691] | none declared |
| 6 | `results/garrido_meta_learner_h3power_local/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | `memory_vs_random` | 12.1870 [11.3574, 13.0333] | 6000001–6000090 (n=90) |
| 7 | `results/garrido_meta_learner_h3power_vps/result.json` | same | `memory_vs_random` | 11.6889 [10.1000, 13.2333] | 6000091–6000120 (n=30) |
| 8 | `results/garrido_meta_learner_v2/result.json` | same | `memory_vs_random` | 12.5556 [10.6524, 14.5556] | 5300001–5300012 (n=12) |
| 9 | `results/garrido_meta_learner_v2/result.json` | same | `alzheimer_effect_runs_saved_by_memory` | 7.9028 [6.8750, 8.9306] | 5300001–5300012 |
| 10 | `results/garrido_normaliser_audit_v3/result.json` | `ALZHEIMER_EFFECT_SURVIVES_AN_HONEST_NORMALISER` | `summary.oracle…runs_to_within_1pct.memory_vs_random` | 12.5556 [10.7778, 14.6250], n=12 | 5300001–5300012 (declared burned replay) |
| 11 | `results/garrido_normaliser_audit_v3/result.json` | same | `summary.prefix…memory_vs_random` | 12.4583 [10.3469, 14.7083], n=12 | 5300001–5300012 |
| 12 | `results/garrido_meta_learner_h3power_local/result.json` | `ALZHEIMER_EFFECT_HAS_A_MEASURED_PRICE` | `memory_vs_ofat` | 5.0389 [4.4926, 5.5945] | 6000001–6000090 |
| 13 | `results/manuscript/h1_h3_v2/result.json` | `H1_SUPPORTED__H3_NOT_SUPPORTED` | `H1.contrasts.primary_all_cells` | 6.167e7 [3.893e7, 8.725e7] | 5800001–5800005 (n=5) |
| 14 | `results/manuscript/h1_h3_v2/result.json` | same | `H1.contrasts.hybrid_vs_reset` | 3.439e7 [1.807e7, 5.386e7] | 5800001–5800005 |
| 15 | `results/manuscript/h1_h3_v2_1/result.json` | same | `H1.contrasts.primary_all_cells` | 6.135e7 [1.443e7, 1.218e8] | 6200001–6200005 (n=5) |

Notes: rows 6–12 are largely the **same 5300001–5300012 / 6000001–6000120 tapes re-analysed** (`garrido_meta_learner`, `_v2`, `_v2_vps_crosscheck`, `garrido_q2_des288_reconciled_v2`, `garrido_normaliser_audit{,_v2,_v3}`, `_h3power_*_v2` all report byte-identical intervals) — not independent evidence. `results/sensitivity/expedite_headroom_v2*` has large positive intervals (e.g. `.comparisons.672.tape_oracle_vs_placebo` 459028.94 [334238.63, 632425.56]) but its own `claim_status` is `NO_TIMING_HEADROOM_UNDER_SERVICE_FIRST` / `HALTED_FALSIFIER_FAILED`, so it is excluded from "high value".

---

## Table C — Artifacts existing only on other branches (never on HEAD)

147 off-HEAD JSONs carry `claim_status`. Confirmation-grade ones:

| Path | Branch(es) | claim_status / status |
|---|---|---|
| `results/q_r1/successor_confirmation_v1/merged/result.json` (+ `source_shards/s01..s04/result.json`) | `codex/paper-b-cf1-cf20-replication` (present on 9 refs) | **`PROSPECTIVE_CONFIRMATION`**, 32 histories, roots 7572001–7572032, 2026-07-22 |
| `results/garrido_h2_h3_confirmation_v1/result.json` (+ `completion_receipt.json`, `tape_level_deltas.json`) | `codex/paper-b-retained-v5` + `origin/` (2 refs only) | no `claim_status`; `status=CONFIRM_H2_H3_ALL_SIX_PANELS`, `global_confirmation_pass=true`, `confirmation_roots_opened=true`, 12 roots 96111336–97836128, Holm all 6 panels pass (p 4.8e-17…1.9e-15), 2026-07-29 |
| `results/q_r1/cold_start_replication_v1/adjudication.json` | `codex/paper-b-cf1-cf20-replication` (12 refs) | `BURNED_PROSPECTIVE_REPLICATION_NO_CONFIRMATORY_CLAIM` |
| `results/garrido_h2_h3_corrective_v1/result.json` | `codex/paper-b-retained-v5` (2 refs) | `COMPLETE_DEVELOPMENT_TAPE_LEVEL_ANALYSIS`, "no confirmatory claim" |

The other ~143 off-HEAD artifacts are development/exploratory, clustered by branch:
- `codex/paper-b-cf1-cf20-replication` — `results/q_r1/**`, `results/program_q2/**`, `results/program_u/**`: overwhelmingly `BURNED_DEVELOPMENT_NO_CLAIM`, `EXPLORATORY_NO_CLAIM`, `BURNED_INSTRUMENT_*_NO_CLAIM`.
- `codex/causal-residual-belief-diagnostic` — `results/program_t/**`, `results/program_u1/**`: `EXPLORATORY_NO_CLAIM`, `BURNED_T0_ROUTING_EVIDENCE`, `BURNED_SMOKE_NOT_SCIENTIFIC_EVIDENCE`.
- `codex/paper-b-retained-v5` — `results/q_r1/matched_retention_factorial_v4_development/*`: `DEVELOPMENT_SELECTION_NO_CONFIRMATORY_CLAIM`, `DEVELOPMENT_ONLY_NO_CONFIRMATORY_CLAIM`.
- `codex/submission-a-cie-guide-compliance`, `codex/authority-ladder-v1-prereg` — `results/program_q/**` receipts: `*_NO_NEW_SCIENTIFIC_RESULT`.
- `qr1-c1-natural-continuation` — `results/q_r1/gate1_*`, `gate2_*`: `EXPLORATORY_NO_CLAIM`.
- `codex/q-r1-comparator-reconciliation` — `results/oracle_curve_v2/VOID_scheduler_mapping_defect.json` (`VOID_NO_CLAIM`), `results/q_r1/factorial_v3/STOP_incomplete_implementation.json` (`NO_RESULT_NOTHING_TO_INTERPRET`).

---

## Table D — Seed blocks by state (`<HOME>/Projects/research/scres-ia/research/seed_custody_registry.json`)

Registry header: `status = BASELINE_INVENTORY_INCOMPLETE_NO_NEW_SEEDS_AUTHORIZED`, `scientific_execution_authorized = false`, `new_seed_opening = false`, reconciled 2026-08-05T21:26:49Z. 32 blocks.

**RESERVED / NEVER OPENED (virgin) — 1 block only:**

| Block | Range | Status |
|---|---|---|
| `g3a_v2_development` | 7700001–7700120 | `RESERVED_NOT_OPENED` — "never touched and stays virgin"; gated by `submission_a_receipt_required_before_g3a_open` |

**BURNED / consumed by confirmation:**

| Block | Range | Status |
|---|---|---|
| `garrido_grid_transfer_v2_confirmation` | 8200001–8200060 | `BURNED_CONFIRMATION_COMPLETE` (opened 2026-08-06T00:46, closed 2026-08-06) |
| `program_q_terminal_confirmation` | 7490001–7490256 | `BURNED` (immutable, outside Garrido-WRAP) |

**QUARANTINED (`ATTEMPTED_NO_SEALED_ARTIFACT`, not reusable as virgin):**

| Block | Range |
|---|---|
| `garrido_grid_transfer_v1_confirmation` | 8100001–8100060 (SIGKILL, no execution receipt; authorization `UNRECORDED`) |
| `g3_obs_v2_confirmation_20260802` | 7900001–7900140 (0-byte log, no result.json) |

**PENDING custody / merge (artifact present, not confirmatory):**

| Block | Range | Status |
|---|---|---|
| `garrido_q2_des288` | 5300001–5300012 | `ARTIFACT_PRESENT_PENDING_CANONICAL_CUSTODY`; consumer `garrido_v0_recovery_holdout` seeds 5300007–5300012 opened **against preregistration** |
| `garrido_h3_local` | 6000001–6000090 | `ARTIFACT_PRESENT_PENDING_MERGE` |
| `garrido_h3_vps` | 6000091–6000120 | `ARTIFACT_PRESENT_PENDING_MERGE`; original VPS snapshot source identity NOT DEMONSTRABLE |

**USED_DEVELOPMENT_NOT_VIRGIN — 24 blocks:** `contention_headroom` 5200001–5200016; `reconciled_5100001` 5100001–5100012; `reconciled_5500001` 5500001–5500008; `reconciled_5600001` 5600001–5600008; `reconciled_5700001` 5700001–5700010; `reconciled_5800001` 5800001–5800008; `reconciled_5800101` 5800101–5800102; `reconciled_5910001` 5910001; `reconciled_6200001` 6200001–6200005; `reconciled_6300001` 6300001–6300006; `reconciled_6400001` 6400001–6400010; `reconciled_6500001` 6500001–6500003; `reconciled_6600001` 6600001–6600006; `reconciled_6700001` 6700001–6700006; `reconciled_6800001` 6800001–6800010; `reconciled_6800120` 6800120; `reconciled_6900001` 6900001–6900008; `g2_autotomy_threshold` 7000001–7000008; `reconciled_7100001` 7100001; `reconciled_7400025` 7400025–7400048; `reconciled_7410001` 7410001–7410048; `reconciled_7420001` 7420001–7420096; `reconciled_7430001` 7430001–7430048; `reconciled_7450001` 7450001–7450006; `g3_old_bottleneck_asymmetry` 7600001–7600010; `g3_obs_v2_powered` 7800001–7800140 (outcome `STRUCTURED_CONTROL_SUFFICES_G3_OBS`, MDE 0.0092/0.0085 vs SESOI 0.010).

Registry's own limitation clauses: absence of a `result.json` is explicitly **not** evidence of virginity; untracked/remote artifacts must be reconciled before any confirmation block opens.

---

## Bottom line

- HEAD holds exactly one clean, virgin-block confirmation: `results/grid_transfer_confirmation_v2/result.json` (UCB1 grid transfer, n=60, block 8200001–8200060).
- Two further confirmations live **only** on `codex/paper-b-retained-v5` (`garrido_h2_h3_confirmation_v1`, all six Holm panels pass) and `codex/paper-b-cf1-cf20-replication` (`q_r1/successor_confirmation_v1`, `PROSPECTIVE_CONFIRMATION`) — neither has ever landed on the current branch.
- The `results/k3/confirmation.json` confirmation is self-retracted (`effective_verdict = RETRACT_...`).
- Everything else — including all the large `ALZHEIMER_EFFECT_*` / `H1_SUPPORTED` / `H3_PRIME_SUSTAINED` intervals — is development on burned or already-open blocks, and much of it is the same two tape sets re-analysed under different names.
- Only one seed block in the whole registry remains virgin: `g3a_v2_development`, 7700001–7700120.
