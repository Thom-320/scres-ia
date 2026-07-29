# War-stress GSA overlay v1 — preregistration

**Status:** `FROZEN_BEFORE_SCIENTIFIC_SEED_ACCESS`

**Parent:** `war_stress_timing_atlas_v1`

**Implementation base:** `93e3bb9823816ed577acfc7a7820e0193c4bd464`

This overlay adds interaction diagnostics and scenario discovery without
replacing the parent 144-cell experiment. It corrects the unmerged `7d5e344`
prototype in four load-bearing ways:

1. Morris sampling and analysis use SALib 1.5.2; no hand-built clipping.
2. The response is `H_timing_safe` within the same risk cell, not tailoring
   among constants or a posture-reversal indicator.
3. Every response is a vector over paired CRN tapes. A balanced two-way
   decomposition separates configuration signal, shared tape effects and
   configuration-by-tape residual noise.
4. Sobol is computed only on a cross-fitted OOS-approved mean surrogate inside
   a fixed independent-factor stratum. PRIM uses EMA Workbench peeling/pasting
   plus repeated configuration-level holdouts.

## Frozen sampling

There are 12 strata: three parent risk masks by four coupling modes. Factors are
risk-specific frequency and impact multipliers sampled independently and
uniformly in log2 space inside each stratum. In coupled strata, only `phi_R24`
is a frequency factor: the parent contract replaces every other native
occurrence process with the R24-driven cluster schedule. Keeping the replaced
frequency factors would create inert dimensions by construction.

The exact committed manifest contains:

- 570 Morris configurations: 20 candidate trajectories, 10 optimized retained
  trajectories, eight levels;
- a 1,536-point QMC pool: 128 scrambled Sobol space-filling points per stratum;
- exact IDs, log2 values, physical multipliers and content hash for every point.

No new point may be generated after tape access.

CRN has two levels: configurations share the same exogenous base random streams,
while policies inside one fixed configuration must reproduce an identical
realized event tape. Event hashes may legitimately differ across configurations
because φ/ψ transform the common base stream.

## Sequential cost control

Morris runs first on tapes `7470001–7470003`. A stratum advances only when its
lower influence proxy on `H_timing_safe` is at least 0.001, its estimated
Monte-Carlo fraction is at most 0.50, and all CRN/safety/saturation checks pass.
At most three strata advance, under a frozen ordering.

Only their existing QMC pools run on `7470004–7470012`. This prevents direct
Sobol from requiring millions of ten-year DES episodes.

## Stochastic surrogate gate

The mean emulator is cross-fitted by configuration, never by tape. The internal
variance emulator models configuration-by-tape residual variance after removing
the common tape effect; it does not relabel metamodel residuals as DES noise.
Sobol and PRIM remain blocked unless:

- OOS R² ≥ 0.80;
- normalized RMSE ≤ 0.15;
- Spearman rank correlation ≥ 0.90.

Classical Sobol is restricted to independently sampled factors within one fixed
mask/coupling stratum. Raw negative or >1 Monte-Carlo estimates are retained;
they are not clipped. `ST−S1` is reported only as an overlapping higher-order
gap, never as a unique interaction mass. No interaction/additivity claim is
authorized because this contract computes neither `S_ij` nor Shapley and no DES
noise-floor calibration has run.

## Scenario discovery

EMA Workbench PRIM supplies peeling and pasting. It uses the independent
`7470009–7470012` tape block, not the surrogate tapes `7470004–7470008`.
Twenty stratified configuration-level splits test each discovered box, and 99
label permutations calibrate the false-box rate. A stable development box
requires the frozen train/holdout gates, factor stability, permutation p≤0.05
and reported split-distribution intervals. It remains a hypothesis.

The parent atlas must run first and remains binding: neither Sobol nor PRIM can open validation, replace
the connected-region gate, authorize a learner or alter the risk envelope. A
continuous box outside a parent passing component requires a new contract and
new seeds.

## Claim boundary

This overlay can describe which wartime risk factors and interactions influence
safe timing headroom. It cannot establish H_PI, H_obs, Paper 2 or Paper 3.
