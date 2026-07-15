# Program O — corrected conditional development certificate

**Date:** 2026-07-15
**Status:** `ACTIVE_FIXED_CLOCK_PHYSICAL_PREFLIGHT — NOT TERMINAL`

The former statement that the internal search was exhausted and that Garrido Q14 was the only
remaining move is retracted. The dual-resource diagnostic identifies a real development signal under
fixed-clock accounting, but the parent DES charges empty capacity without physically executing empty
missions. A new internal falsification is therefore both possible and required: implement the
dedicated schedule physically and test parity before prospective validation.

## Verified evidence

| Quantity | Verified value | Claim boundary |
|---|---:|---|
| Safe full-DES H_PI | 0.15151; simultaneous LCB95 0.11562 | perfect-information ceiling only |
| Fully fungible null | exactly 0 | causal mechanism control |
| Primary belief-MPC ΔReT on burned tapes | 0.0648–0.1013 across the passing component | development only |
| Operational state given current belief, LCB95 | **0.0182–0.1446** across passing belief-MPC contrasts | development only |
| Corresponding contrast means | **0.0244–0.1659** | development only |
| Total-information placebo LCB95 | 0.0490–0.1862 | development only |
| Belief given current state LCB95 | 0.00535–0.01709 | development only |
| Pay-per-use resource gate | fails | no pay-per-use positive claim |

Custody identities: diagnostic result
`e48606e79c4fcbcc7a1f2955476057d2cd9432b95a0bac7337f38f5b44cba535`, source fit
`d67ac97a359a307ca632b6a13493e3ff5940a97e9440a6bc4b7d77c08a147875`, scientific
commit `a9733d033b0ac0969ab9ccc994219bda2b66215e`. All 360 placebo contrasts and all
resource, guardrail, trajectory, and counterfactual records were independently recomputed. Validation
seeds `7420049–7420096` remain unopened.

## What the diagnostic changes

The earlier statement that operational state added no value was false and remains retracted. With
placebos actually executed, `belief_mpc__3` and `belief_mpc__4` show a connected three-cell component
(`rho75_share90`, `rho90_share75`, `rho90_share90`) in which current operational state adds material
value beyond belief alone. The fourth cell (`rho75_share75`) is not eligible because CVaR10 worsens by
about 0.0061.

This is not confirmed H_obs: the controllers and resource interpretation were diagnosed on burned
development tapes. It is nevertheless sufficient to justify one prospective route if the fixed-clock
resource envelope is made physically real.

## Load-bearing physical issue

The thesis specifies daily downstream freight and 24 h legs and takes vehicle availability for
granted. It does not specify empty running or a flat freight contract. In the current code, empty daily
slots are charged in the ledger but not simulated. Thus the favorable fixed-clock result cannot yet be
described as literal physical resource equality.

Program O now proceeds as a disclosed researcher extension:

- one daily downstream mission executes loaded or empty;
- every mission occupies Op10 and Op12 for 24 h each;
- every policy receives exactly the same missions, vehicle-hours, crew-hours, and payload capacity;
- the production action changes only the nonfungible C/H mix;
- pay-per-use remains a reported boundary interpretation.

The frozen contract is `contracts/program_o_fixed_clock_physical_hobs_validation_v1.json`; the
preflight is `research/paper2_exhaustive_search/PROGRAM_O_FIXED_CLOCK_PHYSICAL_PREFLIGHT_FREEZE_2026-07-15.md`.

## Prospective route

1. Pass physical identity, convoy-occupancy, conservation, resource-equality, null, and parity tests
   using only burned fixtures.
2. Commit an immutable execution freeze and custody manifests.
3. Open `7420049–7420096` once for the frozen `belief_mpc__3` controller, three connected cells, nine
   placebos, complete 65,536-calendar denominator, guardrails, and action audit.
4. A fail closes Program O without policy, cell, physics, or metric rescue.
5. A pass establishes classical H_obs under the dedicated-shuttle extension. Only then may one learner
   be frozen and tested against the maximum of the open-loop and classical comparator set on new tapes.

## Current claim boundary

`full_des_h_pi_established: True` · `h_obs_confirmed: False` ·
`fixed_clock_physical_preflight: ACTIVE` · `learner_authorized: False` ·
`paper2_confirmed: False` · `paper3_authorized: False`.
