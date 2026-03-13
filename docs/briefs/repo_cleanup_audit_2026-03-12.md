# Repo Cleanup Audit (2026-03-12)

## Status

The repository is currently clean on `main`.

- `git status --short`: no pending changes
- `.gitignore` already excludes:
  - `tmp/`
  - `docs/artifacts/control_reward/*_smoke/`

This means the cleanup work requested for the benchmark/publication lane has already been applied.

## Commit-By-Lane Audit

Recent history shows the work has been split into sensible lanes instead of one mixed commit.

### Cleanup / generated artifacts

- `740be88` `Exclude tmp/ and smoke-test artifacts from version control`

Scope:
- housekeeping only
- no benchmark semantics changed

### Core environment / observation contract

- `6f44f1b` `Add observation v2 contract, parametric risk profiles, and centralized interface helpers`
- `bc68185` `Refactor trajectory export to use centralized interface helpers`

Scope:
- `supply_chain/env_experimental_shifts.py`
- `supply_chain/external_env_interface.py`
- export/interface plumbing

Role in paper:
- formalizes the observed-state contract
- supports `v1` historical compatibility and `v2` augmented observation

### Benchmark / algorithms / heuristics

- `0aaceb5` `Add heuristic baselines, multi-algorithm benchmark, and publication experiment suite`

Scope:
- `scripts/benchmark_control_reward.py`
- `scripts/run_publication_experiments.sh`
- heuristic baselines
- cross-scenario evaluation
- PPO/SAC/RecurrentPPO benchmark support

Role in paper:
- primary experimental lane for phase 2

### DKANA lane

- `3f09771` `Add DKANA policy starter, dataset pipeline, and integration guide`

Scope:
- `supply_chain/dkana.py`
- `scripts/build_dkana_dataset.py`
- DKANA integration docs/tests

Role in paper:
- separate exploratory lane
- should not be mixed into the main benchmark narrative

### Docs / strategy / briefs

- `58dce63` `Add project guidance, research briefs, and manuscript strategy notes`

Scope:
- `docs/briefs/*`
- `docs/manuscript_notes/*`

Role in paper:
- freezes the current publication strategy
- keeps decision documents separate from code changes

## Audit Verdict

The requested repo cleanup has effectively already happened.

What is true now:

- the worktree is clean
- the main benchmark lane is separated from DKANA
- generated smoke artifacts are ignored
- docs/strategy notes are separated from implementation commits

What is not true:

- the publication experiments have not yet been executed at full scale
- post-experiment statistical reporting is not yet complete

## Recommended Next Step

Do not spend more time on repo hygiene unless a new mixed worktree appears.

Proceed with:

1. `bash scripts/run_publication_experiments.sh --smoke`
2. inspect the generated benchmark bundles
3. launch the full publication run
4. compute statistical comparison tables for the paper
