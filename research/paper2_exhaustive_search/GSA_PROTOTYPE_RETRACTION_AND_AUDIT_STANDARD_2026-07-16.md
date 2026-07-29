# GSA prototype retraction and canonical audit standard

**Status:** binding audit standard for `war_stress_gsa_overlay_v1`

The prototype at commit `7d5e34425922f0bd817180e4e24fe277b6bb274d` is
retained on `war-risk-interaction` only as an audit trail. It is not an ancestor
of this branch and must not be merged or imported. Commits `087daa6` and
`e5dc135` correctly retracted its false OAT premise and documented measured
prototype defects; this file integrates the surviving audit requirements
without importing the obsolete implementation or preregistration.

## Corrected factual boundary

- Cf1–Cf20 vary risks concomitantly; Cf20 raises all four R2 risks and did not
  open the prior profile-tailoring door. The prior screen was not purely OAT.
- The untested space is continuous frequency/impact variation, the frozen
  inter-family masks, and temporal coupling under the new within-cell timing
  estimand.
- The Ishigami table with injected Gaussian noise demonstrates a possible
  failure mode of naive stochastic Sobol. It does **not** measure the direction
  or magnitude of bias in this DES. Any DES-specific noise statement requires a
  populated DES calibration artifact.
- `ST-S1` is an overlapping total-involvement gap, never unique interaction
  mass. This contract authorizes no interaction or additivity conclusion.
- The old custom PRIM did not manufacture a dense box in its reported pure-noise
  test, but it named irrelevant factors. Current PRIM therefore requires an
  explicit permutation-null control in addition to holdouts and stability.

## Binding A1–A11 standard

1. Morris uses SALib; every trajectory changes exactly one factor per step.
2. The estimand is `H_timing_safe` against the strongest comparator in the same
   configuration, selected without tape-level future information.
3. CRN replications are modeled as a configuration-by-tape panel. Tape main
   effects, configuration signal, configuration-by-tape residual variation,
   mean-emulator error and internal-variance-emulator error are reported
   separately.
4. No interaction/additivity threshold may be calibrated on Ishigami and then
   applied to the DES. Until a DES calibration passes, those verdicts are
   prohibited rather than guessed.
5. Sensitivity indices remain raw and include uncertainty intervals.
6. An interaction claim requires a separately frozen `S_ij` or Shapley design.
   `ST-S1` alone cannot support it.
7. Classical Sobol is restricted to independently sampled inputs within a fixed
   stratum. Dependent inputs require another contract.
8. PRIM requires peeling, pasting, configuration-level holdout, factor
   stability, a separate tape block, permutation false-box control, and
   uncertainty for density/coverage/support.
9. The discrete 144-cell atlas is primary. The GSA overlay cannot run first,
   open validation, alter R3, or replace the connected-region gate.
10. Confirmation opens a frozen region once, with canonical ReT and every
    anti-shed/resource guardrail.
11. Every conclusion must cite a populated machine-readable field. Labels,
    defaults, expectations and projected values are not observations.

## Current status

Implementation and non-scientific preflight exist, but scientific execution is
blocked by the exact-family compute gate. A populated short-horizon fixed-policy
DES fixture validates the CRN decomposition mechanics, but it does not calibrate
an interaction/additivity threshold for `H_timing_safe`; those claims remain
prohibited. No scientific Morris, Sobol, PRIM, validation or learner result
exists.
