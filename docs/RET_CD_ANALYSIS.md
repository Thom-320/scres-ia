# RET_CD_ANALYSIS.md

**Date:** 2026-03-27  
**Author:** Thommy / Langosta  
**Status:** Draft — for Garrido meeting

---

## 1. The Problem with ReT_thesis (Piecewise)

The thesis (Garrido-Rios 2017, Eq. 5.5) defines resilience as a piecewise
function that selects one sub-indicator per order based on the disruption state:

| Case | Condition | Formula |
|------|-----------|---------|
| 1 — No disruption | d_frac = 0 | R = fill_rate ∈ [0, 1] |
| 2 — Autotomy | disruption + fill ≥ 0.95 | R = 1 − d_frac ∈ [0, 1] |
| 3 — Recovery | disruption + fill < 0.95 | R = 1 / (1 + d_frac) ∈ (0.5, 1] |
| 4 — Non-recovery | high disruption + low fill | R = 0 |

**Problems for RL training:**
1. **Discontinuities at case boundaries:** The reward surface has sharp jumps
   where the condition changes. Policy-gradient algorithms (PPO) rely on smooth
   gradients; discontinuities destabilize learning.
2. **Non-monotone at autotomy/recovery boundary:** Case 2 uses `1 − d_frac`
   (linear in disruption) while Case 3 uses `1/(1+d_frac)` (hyperbolic). At
   the boundary d_frac → threshold, the two formulas produce different values,
   creating a local non-monotonicity.
3. **Hard case classification:** The binary condition (fill ≥ 0.95) means a
   policy that achieves fill = 0.949 vs 0.951 gets classified into entirely
   different reward branches.

These properties are acceptable for *post-hoc analysis* (the thesis use case)
but are known to fail as RL training objectives (cf. REWARD_DESIGN.md).

---

## 2. The Solution: Cobb-Douglas Continuous Bridge

The Cobb-Douglas (C-D) form is the standard continuous alternative for
multi-factor resilience indices (Garrido et al. 2024, IJPR; Fan et al. 2022;
Human Development Index UNDP 2010). It preserves:

- **Non-compensability:** if any factor → 0, the product → 0
- **Smooth gradients:** differentiable everywhere
- **Proper scaling:** with w₁ + w₂ = 1, output ∈ (0, 1]

### ReT_cd_v1 Formulation

```
FR_t  = max(ε, 1 − backorder_qty / demand)   # fill rate  ∈ (0, 1]
AT_t  = max(ε, 1 − disruption_frac)          # availability ∈ (0, 1]

R_t   = FR_t^0.70 × AT_t^0.30               # raw C-D, ∈ (0, 1]
```

Equivalent log-linear form:
```
ln(R_t) = 0.70 · ln(FR_t) + 0.30 · ln(AT_t)
```

**Weight justification (0.70 / 0.30):**
- FR dominates (0.70): the thesis assigns Re^max = 1.0 to no-disruption and
  autotomy cases — both service-dominant. Fill rate is the primary signal.
- Availability secondary (0.30): thesis assigns Re_bar ≈ 0.5 to recovery —
  partial weight to the disruption dimension.
- Sum = 1.0 → proper weighted geometric mean, no rescaling needed.

**Thesis continuity:**
- When disruption = 0: R_t = FR_t^0.70 ≈ FR_t for high fill rates (≥0.9)
- When fill = 1.0: R_t = AT_t^0.30 → reward still positive under disruption
- When both fail: R_t → 0 (non-compensability, same as Case 4)

---

## 3. Why Sigmoid is Wrong Here

A sigmoid wrapper was proposed as a variant (`ReT_cd_sigmoid`):

```
log_score = 0.70·ln(FR_t) + 0.30·ln(AT_t)   ← always ≤ 0
R_sigmoid = σ(log_score) = 1/(1 + exp(−log_score))
```

**The fundamental problem:**

Since FR_t and AT_t are in (0, 1], their logs are **always ≤ 0**.
Therefore `log_score ≤ 0` always, and `σ(log_score) ≤ σ(0) = 0.500`.

The maximum achievable reward — with perfect fill rate and zero disruption —
is **0.500, not 1.0**. This means:

1. The effective reward range is compressed by ~50%
2. The learning signal is weakened proportionally
3. VecNormalize cannot fully compensate because the upper bound itself is wrong

**When is sigmoid appropriate?**
Sigmoid is appropriate for the Garrido (2024) IJPR formulation because its
variables (ζ, ε, φ, τ, κ̇) are macroeconomic/operational quantities that can
take large positive values, making the log-linear sum **unbounded above**.
The sigmoid then maps an unbounded score to (0, 1). That is not our case.

### Empirical Confirmation (20 episodes × 3 policies)

| Policy | ReT_cd_v1 mean | ReT_cd_sigmoid mean | Ratio |
|--------|----------------|---------------------|-------|
| S1     | 0.3327         | 0.2216              | 0.667 |
| S2     | 0.5073         | 0.3115              | 0.614 |
| S3     | 0.5018         | 0.3094              | 0.617 |

The sigmoid rewards are consistently ~38% lower than raw C-D.
Step-level correlation between ReT_cd_v1 and ReT_cd_sigmoid: r ≈ 0.98
(same ordering/ranking, just uniformly scaled down).

---

## 4. Comparison: ReT_thesis vs ReT_cd_v1

| Metric | ReT_thesis | ReT_cd_v1 |
|--------|------------|-----------|
| S1 mean step | 0.933 | 0.333 |
| S2 mean step | 0.872 | 0.507 |
| S3 mean step | 0.812 | 0.502 |
| Step-level correlation (S2) | — | r=0.20 |
| Fill rate (all modes equal) | 0.809 (S2) | 0.809 (S2) |

**Key observation:** ReT_thesis has high mean rewards (~0.93 for S1) because
most steps have low disruption and the piecewise formula collapses to fill_rate.
ReT_cd_v1 is more discriminating — the availability factor (AT_t^0.30) applies
even in low-disruption steps, producing a lower but more informative signal.

The low step-level correlation (r≈0.20) confirms these reward signals are
**not equivalent** and capture different aspects of performance. This is
expected: ReT_thesis is piecewise and case-based; ReT_cd_v1 is continuous.

---

## 5. Summary Table

| Property | ReT_thesis | ReT_cd_v1 | ReT_cd_sigmoid |
|----------|-----------|-----------|----------------|
| Continuous | ✗ (piecewise) | ✓ | ✓ |
| Smooth gradients | ✗ | ✓ | ✓ |
| Max reward | 1.0 | 1.0 | 0.5 |
| Non-compensable | partial | ✓ | ✓ |
| Thesis-aligned | ✓ (exact) | ✓ (continuous bridge) | ✗ (biased) |
| Recommended for training | ✗ | ✓ | ✗ |
| Recommended for reporting | ✓ | ✓ | ✗ |

---

## 6. Relationship to Garrido et al. (2024)

Garrido et al. (2024, IJPR) proposes a **5-variable** C-D resilience function
with a sigmoid wrapper for factory/supply chain resilience:

```
R = σ(a·lnζ − b·lnε + c·lnφ − d·lnτ − n·lnκ̇)
```

Their variables (ζ: inventory accumulation, ε: backorders, φ: capacity,
τ: time-to-fill, κ̇: cost deviation) are macroeconomic quantities that are
**not bounded to (0, 1]** — hence sigmoid is appropriate there.

Our `ReT_cd_v1` is a **2-variable** C-D that bridges the thesis Eq. 5.5 into
a continuous form. Both the 2-var (ours) and 5-var (IJPR) are valid C-D
formulations for different scopes:

- **ReT_cd_v1:** step-level service + availability, thesis-continuity, RL training
- **cd_garrido2024 (future):** system-level rolling average, 5 variables, IJPR reporting

They are **complementary, not competing** formulations.

---

## 7. Overnight Training Status

Training launched: 2026-03-27 ~23:15 GMT-5  
Configuration:
- Reward: `ReT_cd_v1`
- Risk levels: `increased` + `severe`
- Seeds: 11, 22, 33, 44, 55 (5 seeds per level)
- Timesteps: 500,000 per seed
- n_envs: 4, stochastic_pt: True
- Output: `outputs/ret_cd_v1_500k/`

Log: `/tmp/ret_cd_v1_overnight.log`

---

## References

- Garrido-Rios (2017). *Thesis: Supply Chain Resilience Index.*
- Garrido et al. (2024). *IJPR: Cobb-Douglas Resilience Function.*
- Fan et al. (2022). *Multi-factor resilience with C-D aggregation.*
- UNDP (2010). *Human Development Index: C-D aggregation rationale.*
