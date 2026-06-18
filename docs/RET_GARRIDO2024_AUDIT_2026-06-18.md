# Audit: `ReT_garrido2024_sigmoid` vs Garrido, Pongutá & García-Reyes (2024)

Paper: *Zero-inventory plans, constant workforce, or hybrid approach? Analysing pure production
strategies for enhancing factory resilience with demand variability*, IJPR 2024
(DOI 10.1080/00207543.2024.2425771).

Audited file: `supply_chain/env_experimental_shifts.py::_compute_ret_garrido2024` (line ~1700);
calibration `supply_chain/data/ret_garrido2024_calibration.json`; generator
`scripts/calibrate_cd_exponents.py`.

## Verdict (one line)

The **formula is faithfully transcribed** (Eq 3–6), but the **calibration is STALE** — the offset
exponents and `kappa_ref` were Monte-Carlo–derived on the *legacy/broken* env, so on the current
faithful env the five resilience dimensions are mis-weighted. Re-calibrate before trusting the
cost-aware verdict (it currently gates the running `scresia-track-a-continuous-costaware` experiment).

## What the paper actually does

- Resilience index `R` = a Cobb–Douglas over FIVE output variables measured at the end of a 36-week
  horizon (Algorithm 1/2, lines 50–55):
  - `ζ = ΣI_t/T` avg finished inventory, `ε = ΣB_t/T` avg backorders, `φ = ΣU_t/T` avg spare
    capacity, `τ = Σ(NR_t/min{GR_{t+v},Θ_t})/T` avg time-to-fulfill, `κ̇ = 7κ(S_ij)/Σκ(S_ij)` relative
    cost deviation.
  - Eq 3: `R = ζ^a · (1/ε^b) · φ^c · (1/τ^d) · (1/κ̇^n)`; Eq 4 log-linear; Eq 6 sigmoid of the log score.
  - Exponents are OFFSET FACTORS to make the five terms comparable: each `exponent = 0.20 / ln(var_max)`
    so every term contributes ≈0.20 at its MCS maximum. Paper values (their MCS): a=0.024, b=0.026,
    c=0.04, d=0.06, n=0.1771 (from ζ_max≈3612, etc.).
- Uncertainty source = **demand variability ONLY** (Holt exponential-smoothing forecast Eq 1 with random
  α,γ∈[0,1]; cv=σ_GR/GR); 10,000 Monte-Carlo runs. Processing times are deterministic.
- Cost `κ(S_ij) = Σ[c_p P + c_h H + c_ℓ L + c_u U + c_i I + c_b B + c_o O]`, all c=1 by assumption;
  Section 5 relaxes this and varies them in [1,2].
- Key result (Eq 9): `R(S12) ≻ R(S11) ≻ R(S32) ≻ R(S22) ≻ R(S31) ≻ R(S13) ≻ R(S21)`. **The most
  resilient strategy is `S12` (match-chase, a ZERO-INVENTORY plan); the winning feature is SPARE
  CAPACITY (more workers than strictly needed), NOT inventory.** Most influential cost params: `c_i`,
  `c_p`, `c_b`.

## Findings

**F1 — Formula & sigmoid: FAITHFUL.** The log-linear score and sigmoid match Eq 4/6 exactly, and
`calibrate_cd_exponents.py` implements Garrido's `exponent = 0.20/ln(max)` rule correctly. The runtime
loads the calibrated exponents (a=0.0134, b=0.0168, c=0.0258, d=0.0203, n=0.2495) from the JSON, not the
paper constants. ✓

**F2 — STALE CALIBRATION (main finding).** The JSON `run_config` shows the 1000-episode calibration MCS
ran on the **legacy env**: `observation_version=v1`, and NO `risk_occurrence_mode` / `raw_material_flow_mode`
set → env defaults `legacy_renewal` (the ~2× risk-frequency bug) + `legacy_validated` (the inert
open-loop inventory bug). Its maxima (ζ_max=2.96M, κ_ref=1.41M, ε_max=149k, φ_max=2321, τ_max=18816)
are from the broken world. On the current faithful env (`thesis_periodic` + `kit_equivalent_order_up_to`
+ m2.0, obs v5) the variable scales differ (e.g. observed ζ_avg≈140k–230k), so the offsets no longer
satisfy "each term ≈0.20 at max" → the five dimensions are mis-balanced and the index is mis-scaled.
Worse, `calibrate_cd_exponents.py` does not even EXPOSE `--risk-occurrence-mode` /
`--raw-material-flow-mode` / `--raw-material-order-up-to-multiplier`, so it cannot currently re-calibrate
on the faithful env without a small CLI change.

**F3 — κ̇ normalization is a declared deviation.** Paper `κ̇ = 7κ/Σκ` (this substrategy's cost relative
to the SUM over all 7 substrategies; bounded, mean≈1). Ours `κ̇ = avg_cost / kappa_ref` (fixed MC
reference). A single RL policy cannot compute the cross-substrategy sum, so a per-env reference is a
reasonable adaptation — but it is a DIFFERENT quantity and depends entirely on `kappa_ref` (itself stale,
F2). Since n=0.25 is the LARGEST exponent, κ̇ mis-scaling distorts the index most.

**F4 — Cost uses 4 of 7 components.** Ours `κ = c_p P + c_u U + c_i I + c_b B`; the paper's `c_h H`
(hiring), `c_ℓ L` (firing), `c_o O` (overtime) are omitted — justified because the DES models shifts `S`,
not hire/fire/overtime. The paper's three MOST influential cost params (`c_i`, `c_p`, `c_b`) ARE all
present, so the dominant cost levers are covered. Acceptable, but document.

**F5 — Different uncertainty source (conceptual).** Garrido's `R` was designed and validated under
DEMAND variability (the sole uncertainty). Our env's uncertainty is DISRUPTIONS (risk events) +
`stochastic_pt`. We are applying a demand-variability resilience index to a disruption-driven world. The
five variables are still well-defined, but the "resilience" the index was validated to capture is not
the same construct. Garrido's own future-work sanctions extending to disruptions — so it is defensible,
but must be stated, not silently assumed.

## Recommended fixes (in priority order)

1. **Re-calibrate on the faithful env (blocks trusting the cost-aware verdict).** Add
   `--risk-occurrence-mode`, `--raw-material-flow-mode`, `--raw-material-order-up-to-multiplier` (and
   `--reward-mode`/obs-version already there) to `scripts/calibrate_cd_exponents.py`; run the
   1000-episode MCS with `thesis_periodic` + `kit_equivalent_order_up_to` + m2.0 + obs v5 over the same
   policy mix (static_s1/s2/s3 + random) AND the continuous buffer grid; regenerate
   `ret_garrido2024_calibration.json`. This re-derives a,b,c,d,n and `kappa_ref` so each term ≈0.20 at
   the faithful-env max.
2. **Document the κ̇ and cost-component deviations** (F3, F4) in `docs/RET_GARRIDO2024_IMPLEMENTATION.md`
   as declared adaptations.
3. **State the uncertainty-source gap** (F5): we use the index as a cost-aware *evaluation* over
   disruption scenarios, extending Garrido's demand-variability construct (which his future-work invites).

## Ideas the paper hands us (beyond the audit)

- **Garrido's headline supports our cost-aware interior optimum:** zero-inventory + spare capacity wins
  (Eq 9). Our resilience story should lean on the SHIFT/spare-capacity lever (φ) and LOW inventory, not
  buffer maximisation — exactly what the cost-aware index rewards.
- **Codex's "realistic extension" decoded against the paper:** `demanda variable` = Garrido's CORE
  uncertainty (Holt forecast, the most faithful extension we are MISSING); `costo ci-cp-cb` = his cost
  params (Section 5 sensitivity, ci/cp/cb the influential ones); `stochastic PT spread` = NOT Garrido
  (our own addition — his PT is deterministic).
- **Garrido's future-work IS our project:** he explicitly calls for (a) other risky events — power
  failure, machine breakdowns, absenteeism, material shortages, reprocessing — and (b) the LEARNING
  capability via combined DES + robust learning-based algorithms, plus using `R` in multi-objective
  optimisation of the APP problem. Our RL + DES work is the direct realisation of his stated agenda —
  strong framing for the paper.
