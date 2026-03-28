# CD_GARRIDO2024 Night Report — 2026-03-27

**Session:** Overnight implementation + analysis  
**Repo:** `proyecto_grarrido_scres+ia/`  
**Actual task:** ReT_thesis → Cobb-Douglas continuous bridge (ReT_cd_v1)

---

## What Was Implemented

### 1. `ReT_cd_v1` — New primary reward mode
**File:** `supply_chain/env_experimental_shifts.py`

Raw Cobb-Douglas continuous bridge for the piecewise ReT_thesis:
```
FR_t = max(ε, 1 − backorder_qty / demand)   # fill rate
AT_t = max(ε, 1 − disruption_frac)          # availability
R_t  = FR_t^0.70 × AT_t^0.30               # ∈ (0, 1]
```
- Continuous, smooth gradients → suitable for PPO
- Non-compensable (if either factor → 0, product → 0)
- Weights sum to 1.0 → proper weighted geometric mean

### 2. `ReT_cd_sigmoid` — Experimental (NOT recommended)
Same computation but with sigmoid wrapper applied. Included only to document
the systematic bias when log-inputs are already in (0, 1].

### 3. Wired everywhere
- `REWARD_MODE_OPTIONS` in `env_experimental_shifts.py`
- `SHIFT_ENV_REWARD_MODES` in `train_agent.py`
- `--reward-mode` argparse choices in `train_agent.py`
- Notes in `external_env_interface.py`

---

## Test Results

```
121 passed in 28.87s
```
All 121 existing tests pass. No regressions.

---

## Calibration Results (20 episodes × 500 steps per policy)

**Environment:** increased risk + stochastic_pt

| Policy | Fill Rate | ReT_thesis mean | ReT_cd_v1 mean | ReT_cd_sigmoid mean |
|--------|-----------|-----------------|----------------|---------------------|
| S1     | 0.616     | 0.933           | 0.333          | 0.222               |
| S2     | 0.809     | 0.872           | 0.507          | 0.312               |
| S3     | 0.806     | 0.812           | 0.502          | 0.309               |

**Sigmoid bias confirmed:**
- S1 ratio: 0.667 | S2: 0.614 | S3: 0.617
- Maximum sigmoid reward is σ(0)=0.500 — ~38% compression vs raw C-D

**Step-level correlations:**
- ReT_thesis vs ReT_cd_v1: r≈0.20 (different signals, as expected)
- ReT_cd_v1 vs ReT_cd_sigmoid: r≈0.98 (same signal, just scaled)

**Key finding:** ReT_thesis inflates rewards because most steps have zero
disruption and the formula collapses to fill_rate; the piecewise structure
doesn't penalize availability in calm periods. ReT_cd_v1 is more discriminating.

---

## Training Runs Status

**Launched:** 2026-03-27 ~23:15 GMT-5 (background PID 19126)  
**Log:** `/tmp/ret_cd_v1_overnight.log`

Configuration:
- `reward_mode=ReT_cd_v1`
- `risk_level=increased` + `severe` (sequential)
- seeds: 11, 22, 33, 44, 55 (5 each)
- `timesteps=500_000`, `n_envs=4`, `stochastic_pt=True`
- `observation_version=v1`
- Output: `outputs/ret_cd_v1_500k/{increased,severe}/seed_N/`

Status at report time: actively training, seed 11 / increased, 10+ policy updates.

---

## Key Finding: Raw C-D vs Sigmoid

**Sigmoid is WRONG for RL when inputs are already in (0,1]:**

Since FR_t ∈ (0,1] and AT_t ∈ (0,1], their logs are always ≤ 0.
Therefore the inner log-linear sum ≤ 0, and σ(≤0) ≤ 0.5.
The maximum achievable reward is σ(0) = 0.500, not 1.0.

Sigmoid is only appropriate when variables are unbounded (e.g., the IJPR
Garrido 2024 5-variable formulation with ζ, ε, φ, τ, κ̇).

**Raw C-D** (ReT_cd_v1) is the correct form: output ∈ (0,1], max=1.0.

---

## Artifacts

| File | Location |
|------|----------|
| Analysis doc | `docs/RET_CD_ANALYSIS.md` + `~/Downloads/RET_CD_ANALYSIS.md` |
| Calibration JSON | `outputs/ret_cd_comparison/comparison_results.json` + `~/Downloads/RET_CD_COMPARISON.json` |
| Calibration report | `outputs/ret_cd_comparison/comparison_report.txt` |
| Training script | `scripts/run_ret_cd_v1_benchmark.sh` |
| Env implementation | `supply_chain/env_experimental_shifts.py` (ReT_cd_v1, ReT_cd_sigmoid) |
| Training log | `/tmp/ret_cd_v1_overnight.log` |

---

## For Garrido Meeting

The key document is `docs/RET_CD_ANALYSIS.md`. It covers:
1. Why ReT_thesis piecewise fails as RL training objective
2. The C-D continuous bridge (ReT_cd_v1) formulation and weight justification
3. Why sigmoid produces a systematic ~50% downward bias here
4. Empirical comparison across 3 static policies
5. Relationship to Garrido et al. (2024) IJPR 5-variable formulation
