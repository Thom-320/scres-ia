# Track B Kaggle Runner Bug Audit — 2026-07-01

## Scope
Audited `scripts/run_track_b_kaggle.py` and the Kaggle wrapper after the optimized R2/R24 campaign produced inconsistent verdicts.

## Bugs Found And Fixed

1. **Incorrect CI for ReT delta**
   - Old logic subtracted marginal confidence intervals (`ppo_lo - static_hi`) instead of computing a CI on paired CRN deltas.
   - Fix: `paired_deltas(..., key=(eval_seed, cell))` + bootstrap CI on paired differences.
   - Added guard: if no pairs exist, CI defaults to `(ret_delta, ret_delta)` instead of `(nan, nan)`.

2. **H3 volatility CV used all static policies**
   - Old logic computed static cell means from every static policy, inflating static volatility.
   - Fix: H3 static CV now uses only the selected `best_static` policy.

3. **Campaign-cell/BC mismatch risk**
   - The runner could select mixed `psi` cells and silently fallback to family-level BC teacher actions.
   - Fix: default `--target-psi 1.0`; `--no-psi-filter` required for exploratory mixed-psi runs.
   - Fix: no silent BC fallback. Exact BC teacher rows are required unless `--allow-bc-fallback` is intentionally passed.

4. **Kaggle payload confusion**
   - A stale embedded payload could override the freshly built dataset payload.
   - Fix attempted: remove stale embedded payload and rely on dataset.
   - Operational finding: current Kaggle kernels did not mount `thomaschisica/scres-ia-payload` despite `dataset_sources`; local/tmux confirmatory was launched as the fastest reliable path.

## Validation

- `python3 -m py_compile scripts/run_track_b_kaggle.py` passes.
- Smoke test with `--target-psi 1.0` selects exactly six psi=1 campaign cells:
  - current/increased/severe × R2/R24.
- Smoke test confirms paired CI is coherent and H3 CV is computed against `best_static`.

## Current Correct Run

Local confirmatory is running in tmux:

```bash
outputs/experiments/track_b_campaign_psi1_confirm_local_5seed_2026-07-01/run.sh
```

Config:
- `--target-psi 1.0`
- 6 campaign cells: current/increased/severe × R2/R24
- 5 seeds
- 60k PPO steps
- 200 BC epochs
- dense 48-static frontier
- paired CRN delta CI

Watcher:

```bash
outputs/experiments/track_b_campaign_psi1_confirm_local_5seed_2026-07-01/watch.log
```
