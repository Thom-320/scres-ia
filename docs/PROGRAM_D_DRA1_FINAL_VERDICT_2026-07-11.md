# Program D DRA-1 — final authoritative verdict

Status: **CLOSED — `STOP_NO_DYNAMIC_ORACLE_HEADROOM`**.

Authoritative preregistration:
`docs/PROGRAM_D_DRA1_V3_PREREGISTRATION_2026-07-11.md`.

Authoritative corrective result:
`results/program_d/dra1_prefix_balanced_branching/verdict.json`.

## Chronology

1. `07b3904`: conserved two-CSSU allocation primitive.
2. `a6068d5`: opt-in split topology, actuator and localized risks.
3. `35ea776`: nine-policy same-contract static frontier.
4. `5f518d6`: first exact branching result; later superseded on diversity
   interpretation because its 0.25 prefix induced A/B state imbalance.
5. `5115ef0`: neutral/opposite-prefix verification resolves the confound.
6. `9b1a1f2`: authoritative 180-state prefix-balanced corrective branching.
7. `bcab1fb`: independent V5 verification confirms the terminal STOP.

## Final evidence

- 60 calibration tapes; no virgin tape opened.
- 180 states: 66 A-stressed, 65 B-stressed, 49 balanced.
- 1,620 exact one-epoch branches across nine actions.
- exact replay-prefix identity and mass conservation: PASS.
- only 11/180 states had strictly positive oracle headroom.
- mean oracle delta ReT: 0.000087895;
  clustered CI95 [0.000027576, 0.000165914].
- normalized action diversity: FAIL; no action level approached the frozen 15%
  support threshold.
- guardrails: PASS.
- PPO trained: false.
- retained/reset/frozen tested after DRA-1: false.

## Allowed claim

After balancing policy-prefix histories, stress direction and live-state
categories, daily CSSU allocation under fixed aggregate capacity, automatic
unused-capacity reallocation and the tested service rules exposes sparse and
practically negligible dynamic oracle value. It is not promoted to an
observable policy or RL action dimension.

## Forbidden claims

- DRA-1 does not reject all spatial logistics control.
- Allocation A=0.25 is not claimed intrinsically optimal.
- No adaptive-control, PPO, retained-learning, path-dependency or real-world
  organizational-learning claim is supported.
- The first 58/60 raw-allocation diversity result is superseded, not deleted.

No DRA-1 parameter, action, tape or threshold may be altered to rescue this
closed lane. A spatial commitment model with persistent vehicle location would
be a new decision family and require a new preregistration.
