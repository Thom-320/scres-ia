# Probe spec — nonstationary observable-conversion densification (EXPLORATORY_NO_CLAIM)

**CORRECTED 2026-07-21 (v2).** The original v1 of this spec anchored on a MISREAD: it claimed
CROSS_ECHELON_SURGE had observable classical conversion H_obs ≈ +0.119. **That is wrong.** The
+0.119 was CROSS_ECHELON's clairvoyant `h_pi_sampled_mean` (actually 0.091) conflated with the
verdict-summary `best_isolated_classical_h_obs_mean`; the auditor caught it and the artifact
confirms it. Corrected facts below. Same error class as the raw/safe-LCB fuzzy-match — recorded,
not hidden.

## Verified reality (results/program_u1/direct_classical_conversion_v1/result.json)

The `classical_h_obs_loo_mean` (leave-one-tape-out observable conversion vs best static) per
candidate:

| Mask | Cell | H_PI (clairvoyant) | **H_obs (observable classical)** |
|---|---|---:|---:|
| CROSS_ECHELON_SURGE | rho90_share90 | 0.091 | **−0.043** |
| CROSS_ECHELON_SURGE | rho90_share75 | 0.077 | **−0.048** |
| LOC_SURGE | rho75_share90 | 0.039 | **+0.018 (only promoted point)** |

So CROSS_ECHELON is a Program-O-shaped dead end: large clairvoyant H_PI, but observable control
**cannot convert it** (negative H_obs). The ONLY observably-convertible point is **LOC_SURGE /
rho75_share90 / H_obs +0.018** — and `direct_connected_region_v1` already
`STOP_U1_NO_CONNECTED_CLASSICAL_CONVERSION_REGION` (1 promoted point, no connected region, and
its static worst-product fill is very low ≈ 0.084 → a worst-product problem, not just headroom).

## Honest status of the nonstationary lead

Much weaker than v1 implied. There is **one isolated point of +0.018 observable conversion**
(LOC_SURGE), not a +0.119 region. This is not a strong lead; it may well be an isolated
artifact. Densification is still the correct *test* — but its prior is now "probably
ISOLATED_ARTIFACT," and it must not be oversold.

## Design (if pursued — a small parallel probe, not the main compute)

- **Anchor: LOC_SURGE, rho75_share90**, group 1 / trajectory 8 / point 1 (the promoted row).
  Reconstruct its factor vector (φ/ψ on the LOC risks, concurrency, onset/recovery timing) and a
  dense local grid around it.
- Direct-SimPy only; 24→48 burned tapes, CRN; matched stationary control (equal risk mass).
- Selection axis, frozen: `H_PI_safe`, observable `H_obs_classical`, ranking reversals,
  worst-product, resources, connectivity — **NEVER learner return**.

## Connectivity gate + verdicts

Region = ≥3 adjacent points all with `LCB95(H_obs_classical) ≥ 0.015` AND ranking reversals AND
worst-product/resource guardrails AND headroom over the matched stationary control.
- `REGION_CONFIRMED` → a nonstationary envelope exists → authorizes the dynamic hybrid there.
- `ISOLATED_ARTIFACT` (the likely outcome) → no promotable nonstationary env in this DES family;
  the isolated point is an exploratory boundary. Then the learner story is stationary-only.

## Compute + invariants

Direct-SimPy (the risk masks fail transducer exactness); exact resources; burned tapes, no
sealed seeds; EXPLORATORY_NO_CLAIM; compute preflight (the discovery ran in ~48 s, so a dense
LOC_SURGE grid is minutes). This is a SMALL parallel probe; the main compute goes to the
dynamic residual-warm-start ablation.

## Relationship to the corrected picture

The auditor's deeper correction stands: the flat STATIC top does NOT explain the MPC tie — Q
already beats best static by H_OL ≈ 0.06–0.10, so **feedback value exists in the stationary
env**; what is missing is an observable residual belief-MPC does not capture. The primary
win-hunt is therefore the **dynamic residual warm-start** (does starting from a strong policy
let a learner acquire a state-dependent residual over belief-MPC?), not this nonstationary probe.
