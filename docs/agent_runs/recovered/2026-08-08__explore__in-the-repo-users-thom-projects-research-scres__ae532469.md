# Explore — In the repo /Users/thom/Projects/research/scres-ia, investigate the current state precisel

- **agent_id**: `ae532469e68dbc21b`
- **session**: `1fab0490-70ae-4d2e-a373-23cec6b3ba5f`
- **branch**: `codex/expanded-contract-comparators-v2`
- **finished**: `2026-08-08T02:09:18.021Z`
- **effort**: `medium` · **version**: `2.1.222`
- **transcript**: `/Users/thom/.claude/projects/-Users-thom-Projects-research-scres-ia/1fab0490-70ae-4d2e-a373-23cec6b3ba5f/subagents/agent-ae532469e68dbc21b.jsonl`
- **turns**: 33 · **assistant messages**: 2

## Task given

```text
In the repo /Users/thom/Projects/research/scres-ia, investigate the current state precisely. Read-only. Report facts with exact values, no interpretation.

1. REMOTE vs LOCAL: run `git rev-parse HEAD`, `git log --oneline -12`, `git rev-parse origin/codex/expanded-contract-comparators-v2`, and `git status --short | head -20`. Report whether local commits are pushed, and the exact SHA the remote branch points to. Also `git log --oneline origin/codex/expanded-contract-comparators-v2 -3`.

2. CERTIFICATE COLLISION: the file `scripts/verify_frozen_path_equivalence_v1.py` may have been rewritten by a concurrent session. Report: (a) its current content — specifically which cache directories it reads, whether it covers ONLY the base 288 surface or also the extended 4608 surface, its falsifier names, and its sample/limit behaviour; (b) `git log --oneline -5 -- scripts/verify_frozen_path_equivalence_v1.py`; (c) whether a DIFFERENT verification script exists, e.g. anything matching `scripts/*frozen*`, `scripts/*equivalence*`, `scripts/*cache_live*`, `scripts/check_garrido_surface_cache_live*`.

3. CERTIFICATE RESULTS: list everything under `results/frozen_path_equivalence/` and any similar dir. For each result.json found, report `claim_status`, `scope`, `run_role`, `cells_replayed`/`slices_replayed` or equivalent sample-size fields, and the falsifier pass/fail summary. I launched a full 103,680-cell run earlier (PID 73823) — report whether a completed full-run artifact exists or whether only a smaller-sample artifact is present.

4. WHICH CACHES DOES THE CONFIRMATION USE: read `results/grid_transfer_confirmation_v2/result.json` and report which surface_cache directories it references (search for 'surface_cache', 'base', 'ext', 'cache' keys), plus its `module_manifest`. Then list `results/surface_cache/` subdirectories with their sizes.

5. TEST SUITE: report the result of `git log --oneline -8 -- tests/` and list any test file matching `*frozen*`, `*equivalence*`, `*manifest*`, `*custody*`. If a test-run log or marker config exists (pytest.ini, pyproject markers, `release_paper2`), report it.

Report each numbered item separately with exact command output where short. Do not modify anything.
```

## Final answer

Findings, item by item. All read-only.

## 1. REMOTE vs LOCAL — 12 commits unpushed

```
HEAD                = f99ba5fda8ea080642958c9ac274f05a034b8ce8
origin/codex/expanded-contract-comparators-v2 = 7187d14dffc5efcf39078c11c6bdde1454a61b30
```
`git branch -vv` reports `codex/expanded-contract-comparators-v2 f99ba5f [origin/...: ahead 12]`. **The 12 local commits are NOT pushed.**

Local `git log --oneline -12`:
```
f99ba5f Re-run Confirmation 2's own anchor under today's code, cell by cell
bc2624f I read the contract instead of the registry, and said a burned block was reserved
18a4174 Prove the frozen path survived yesterday's edits, and give two custody guards their teeth back
6cc3c61 Paper 2 Results, written from the two usable confirmations rather than the closed programme
78b13d0 Three confirmations exist, and no naive census can find all three
e027fd5 Paper 1: one extraction, three figures, and a macro layer the prose cannot bypass
3e9100e Admit the sixteen Paper-1 artifacts to the citable table, with both hashes labelled
cc3af32 The seasonal engine, and the forecast that structurally cannot see the turn
1443e21 Bring the C&IE chassis machinery forward, and nothing of its science
3c6c8d9 Preregister the seasonal demand engine, with the three facts we decide ourselves
92ba0cf Exactly six orders a week, so the aliasing story was mine and it was wrong
1b4b50f Measure the demand process instead of deriving its CV off config.py
```
Remote `-3`: `7187d14 Five questions for Garrido…`, `e50242d Run David's architecture…`, `f34db8d Close Program L on reading B…`.

`git status --short`: one line only — `?? results/dmlpa_kan_latent/`. Working tree otherwise clean.

## 2. CERTIFICATE COLLISION — the script WAS rewritten by commit f99ba5f

(a) Current `/Users/thom/Projects/research/scres-ia/scripts/verify_frozen_path_equivalence_v1.py` (209 lines, mtime Aug 7 21:03):
- **Cache read: exactly one** — `CACHE = Path("results/surface_cache/garrido_transfer_confirmation_v2_base")` (line 44). It does *not* read `..._v2_ext` and does *not* read `wrap288_v1`.
- **Surface coverage: BASE 288 only.** It replays cells via `BASE_CONFIGS` imported from `build_transfer_confirmation_cache_v1`; indexing is `BASE_CONFIGS[idx]` per cached cell. No extended/4608 path exists in the file.
- **Falsifier names (4):** `f1_every_cell_reproduces_the_sealed_cache`, `f2_mutation_control_is_detected`, `f3_declared_manifest_is_narrower_than_the_physics`, `f4_a_declared_module_actually_drifted`.
- **Sample/limit behaviour:** `--limit-slices` (int, **default 0 = every slice**); `slices = sorted(CACHE.rglob("*.json"))`, truncated only if the flag is nonzero. There are 360 slice files under the base cache (6 context dirs: `R1r`, `R1r_esc`, `R1r_R2r`, `R1r_R2r_esc`, `R2r`, `R2r_esc`). 360 × 288 = 103,680 cells. Verdicts: `FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_PHYSICS` / `RERUN_REQUIRED_PROVENANCE_NOT_RECOVERABLE`; declared `scope = PROVENANCE_CERTIFICATE_NO_SCIENCE_RERUN_NO_SEEDS_OPENED`, `run_role = PROVENANCE_CERTIFICATE`.

(b) `git log --oneline -5 -- scripts/verify_frozen_path_equivalence_v1.py`:
```
f99ba5f Re-run Confirmation 2's own anchor under today's code, cell by cell
18a4174 Prove the frozen path survived yesterday's edits, and give two custody guards their teeth back
```
`git show --stat f99ba5f`: single file changed, **448 lines touched, 179 insertions / 269 deletions** — a rewrite, not an extension. The prior (18a4174) version read TWO cache roots — `results/surface_cache/garrido_transfer_confirmation_v2_ext` and `results/surface_cache/wrap288_v1` — with `--cells-per-cache` default 120 and required `--contract`.

(c) Other verification scripts present in `/Users/thom/Projects/research/scres-ia/scripts/`:
- `check_garrido_surface_cache_live_v1.py` (4.4k, Aug 5, last commit `ad8163c`) — "Recompute eight cached cells against the live DES"; uses `run_meta_learner_normaliser_audit_v1` CONFIGS, the 5.3M seed block.
- `audit_v2_metric_freeze_equivalence.py` (11k, Jul 31)
- `audit_track_bp_frozen_posture.py` (6.0k, Jul 18)
- `seal_garrido_surface_cache_v1.py`
No other `*frozen*`/`*equivalence*` script exists.

## 3. CERTIFICATE RESULTS — only the OLD small-sample artifact exists; the full run is still running

`results/frozen_path_equivalence/` contains exactly one file: `result.json`, 10,980 bytes, mtime **Aug 7 20:49**, i.e. **older than the rewritten script** (21:03). `git status` shows it unmodified since commit `18a4174` — so **the on-disk artifact was produced by the OLD script and does not match the current script's schema.**

Its contents:
- `schema_version` = `frozen_path_equivalence_v1`
- `claim_status` = **`FROZEN_PATH_EQUIVALENT_UNDER_CURRENT_SOURCE`** (not the current script's `..._UNDER_CURRENT_PHYSICS`)
- `scope` = `PROVENANCE_ONLY_NO_SCIENTIFIC_CLAIM_NO_NEW_SEEDS`
- `run_role` — **absent** (key not present; top-level keys are `by_cache, calibration_provenance, claim_status, contract_path, contract_sha256, created_at, declared_drift, elapsed_seconds, endpoint, falsifiers, manifest_drift, module_manifest, mutation_controls, preregistration, reference_path, reference_sha256, schema_version, scope, self_sha256, what_this_does_not_certify`)
- `cells_replayed` / `slices_replayed` — **absent.** Sample size lives in `by_cache`: `grid_transfer_confirmation_v2` → `n_cells = 300`, `n_distinct_configs = 290`, `n_distinct_seeds = 60`, cache_root `results/surface_cache/garrido_transfer_confirmation_v2_ext`; `search_ladder_v5` → `n_cells = 300`, `n_distinct_configs = 198`, `n_distinct_seeds = 12`, cache_root `results/surface_cache/wrap288_v1`. **Total 600 cells sampled, over the EXT and wrap288 caches**, not the base 103,680.
- `elapsed_seconds` = 56.69; `created_at` = 2026-08-08T01:49:25Z
- Falsifiers, `all_passed = True`, five of them (old names): `f1_every_sampled_cell_reproduces_exactly` PASS (0 differing of 300+300), `f2_the_sample_spans_the_grid` PASS, `f3_the_comparator_detects_a_planted_difference` PASS (1e-9 perturbation, both caches, value + panel), `f4_the_measured_drift_is_the_declared_drift` PASS (declared 2 = drifted 2: `arm_runner.py`, `supply_chain.py`; undeclared 0), `f5_no_seed_outside_the_burned_blocks` PASS (72 seeds touched).

**PID 73823 is STILL RUNNING** — `.venv/bin/python scripts/verify_frozen_path_equivalence_v1.py`, started 9:05PM, elapsed 02:23 at check time, 80.7% CPU, state `RN`. **No completed full-run (103,680-cell) artifact exists.** Only the 600-cell artifact is on disk, and when PID 73823 finishes it will overwrite `results/frozen_path_equivalence/result.json` (that is the script's default `--output`).

No other similar results dir: `results/` contains `custody`, `surface_cache_custody`, `grid_transfer_confirmation_v2`, `gsa_confirmation`, `gsa_confirmation_corrective`, `garrido_h2_h3_confirmation_v1` — none is a second frozen-path certificate.

## 4. WHICH CACHES THE CONFIRMATION USES

`results/grid_transfer_confirmation_v2/result.json` (29k, Aug 6 09:49, `claim_status = GRID_TRANSFER_CONFIRMED__UCB1`, `run_role = CONFIRMATION`, `scope = CONFIRMATION_ON_RESERVED_VIRGIN_BLOCK_NO_RL_NO_NEURAL_LEARNER`, `created_at` 2026-08-06T09:39:38Z):

- **It contains no `results/surface_cache/...` path strings at all** (regex over the whole JSON returned zero matches). Cache references are structural, under `cache_module_manifests`, with keys `base`, `base_variants`, `extended`, `extended_variants` (plus `base_entry_script`, `extended_entry_script`, `base_manifest_variants`, `extended_manifest_variants`). Both `base` and `extended` declare the same entry script `scripts/build_transfer_confirmation_cache_v1.py` (sha `ccb85cbf…`) and the same four modules: `config.py ae702a49…`, `episode_metrics.py e09199d4…`, `seed_custody.py 3d46668e…`, `supply_chain.py 2f348e59…`.
- `n_base_configs = 288`, `n_ext_configs = 4608`, `seed_block = garrido_grid_transfer_v2_confirmation`, seeds `8200001…`, `budget = 24`, `replay_of = None`.
- Top-level `module_manifest`: `entry_script = scripts/run_grid_transfer_v1.py` (sha `727a8d7d…`), `schema = module_manifest_v2`, modules only **two**: `supply_chain/arm_runner.py = 35135c537d9260832677ab6b4e344229d6b9f4810df8bdef9efd51b58fcac68a`, `supply_chain/seed_custody.py = 3d46668e…`; `scope` = "declared modules and entry script only; NOT the full execution dependency set…".

Note the mismatch relevant to item 2: the confirmation's own `n_ext_configs = 4608` and its `extended` cache manifest have no counterpart in the rewritten script, which reads only `..._v2_base`.

`results/surface_cache/` sizes:
```
2.0M  garrido_transfer_confirmation_base_v1
 31M  garrido_transfer_confirmation_ext_v1
 42M  garrido_transfer_confirmation_v2_base      <- the only cache the current script reads (360 slices)
649M  garrido_transfer_confirmation_v2_ext
 70M  garrido_v0_recovery216_v1
284K  manifests
130M  wrap288_compat_extended_v1
8.3M  wrap288_v1
```

## 5. TEST SUITE

`git log --oneline -8 -- tests/`:
```
18a4174 Prove the frozen path survived yesterday's edits, and give two custody guards their teeth back
77193ca Open the last virgin block, and keep the guardrail fail-closed for anyone else
c1a819b Holdout opened against the preregistration, and it replicates the STOP exactly
95fcb27 Commit the v0 recovery-surface instrument, with its cache at 1 seed of 12
c35f5a4 Repair the custody guardrail that was red instead of guarding
45aef89 Repair the v0 recovery gate at the operative clock
ab9d5f5 Gate v0 recovery without rewarding censored TTR
b1ec6ac David asked for retained weights, and the price is the replication we lose
```
**`f99ba5f` (the script rewrite) touched no tests** — its `--stat` shows one file changed. The last tests change is `18a4174`, which added `tests/test_paper2_manifest_identity.py` (101 lines), `pytest.ini` (+4), and `results/frozen_path_equivalence/result.json` (345 lines).

Matching test files in `/Users/thom/Projects/research/scres-ia/tests/`:
```
test_garrido_surface_cache_custody.py
test_garrido_wrap_custody.py
test_paper2_manifest_identity.py
test_paper2_switch4_producer_custody.py
test_program_m_shared_lift_hpi_custody.py
test_program_m_shared_lift_hpi_validation_custody.py
test_program_o_full_des_custody.py
test_seed_custody_module.py
test_seed_custody_registry.py
```
**No test file matches `*frozen*` or `*equivalence*`.**

Config: `pytest.ini` exists (no pyproject.toml, setup.cfg, tox.ini, or root conftest.py). It sets `testpaths = tests`, `norecursedirs = outputs tmp kaggle archive_untracked deliverables .git *.egg-info`, and defines one marker:
```
release_paper2: submission gate for the Paper 2 manuscript. Must be 100% green before any
    number is cited. Failures outside this marker belong to closed programmes and are
    quarantined with an issue; failures inside it block submission.
```
`release_paper2` appears in exactly two files repo-wide: `pytest.ini` and `tests/test_paper2_manifest_identity.py` (2 occurrences). No test-run log or marker artifact was found elsewhere.
