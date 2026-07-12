# Program E final verdict — 2026-07-12

## Terminal decision

`STOP_PROGRAM_E_VALIDATION`

Program E executed the frozen observable-policy conversion study through its
validation gate. The gate did not authorize the confirmatory test. No tape from
the preregistered virgin universe `920001–920060` was opened, and retained
learning was not authorized.

This verdict does not reopen or alter the terminal DRA-2b result. DRA-2b
identified restricted clairvoyant headroom. Program E tested whether observable
closed-loop policies could convert that headroom.

## Frozen genealogy

- DRA-2b terminal freeze: `3bcf6e9`, tag `dra2b-stop-2026-07-12`.
- Program E preregistration: `86f1cc0`, tag
  `program-e-preregistered-2026-07-12`.
- Observable environment and data freeze: `55fe595`, tag
  `program-e-data-freeze-2026-07-12`.
- Baseline and trainer freeze: `4e7c4c6`, tag
  `program-e-training-freeze-2026-07-12`.
- Ten-model pre-validation freeze: `3cae431`, tag
  `program-e-model-freeze-2026-07-12`.

Preregistration SHA-256:
`a16344cf0b4bf92bb31ab21dacb70ffd989c5b0f225556a96f7b251f3a8a8969`.

Contract SHA-256:
`e63e9425cf5c68e728dc75348b37e7c3d393e53db9f18d34dca147d2f2e269df`.

## What was executed

- 80 new training tapes and 20 new validation tapes, balanced across four
  disruption families.
- A depth-3 observable policy tree trained on restricted-oracle first actions.
- The frozen observable heuristic.
- Ten preregistered MaskablePPO learner seeds (`9301–9310`), 200,000 steps each.
- Nine same-contract static policies and a resource-envelope convex static
  comparator.
- The restricted 14-day open-loop oracle at validation episode start, used only
  as a diagnostic denominator.
- Zero confirmatory/virgin tapes.

## Primary validation result

Against the frozen resource-envelope static comparator:

- PPO mean delta ReT: `-0.000050621`;
- two-way bootstrap CI95: `[-0.000320016, 0.000200923]`;
- restricted-oracle headroom: `0.011557538`;
- conversion efficiency: `-0.004380`;
- positive learner seeds: `0/10`;
- favorable validation tapes: `70%`;
- mean service-loss reduction: `-0.0484%`, CI95
  `[-0.1567%, 0.0215%]`;
- lost-order delta: exactly `0`;
- resource-envelope and tail-risk gates: passed.

PPO failed the preregistered ReT, conversion, service-noninferiority, and
positive-seed gates. The favorable-tape fraction alone cannot promote a model
whose mean effect is negative for every learner seed.

## Observable baseline result

The policy tree achieved `82.6%` grouped cross-fitted first-action accuracy on
training oracle labels, but did not convert that classification accuracy into
episodic resilience. It used fewer resources than every static policy, so no
resource-dominated static comparator exists inside its envelope. Even against
the least-resource static doctrine, it lost `0.131277` ReT descriptively.

The heuristic had a valid resource-envelope comparator
(`threshold_2500__wait_72h`) and also failed:

- mean delta ReT: `-0.000052230`;
- favorable validation tapes: `0/20`;
- mean service-loss reduction: `-0.0440%`;
- lost-order delta: `0`.

## Allowed claim

> Under the frozen finite-convoy physics, observation contract, reward, policy
> classes, training budget, and resource-envelope comparison, restricted
> clairvoyant headroom did not translate into out-of-sample episodic advantage
> for the tested observable tree, heuristic, or MaskablePPO policies.

This supports the methodological distinction:

`physical authority -> clairvoyant headroom -/-> observable conversion -/-> learned advantage`.

## Forbidden claims

- RL is impossible in every supply-chain DES.
- PPO can never beat a constant under any alternative observation, reward,
  physical contract, or training budget.
- DRA-2b had no headroom.
- The 14-day oracle was deployable.
- The tree's classification accuracy constituted policy value.
- Retained learning or path dependency was tested in Program E.

## Consequence

Program E is terminal. Do not open the virgin tapes, retune the reward, add a
new architecture, or begin persistent/reset/frozen arms. The manuscript claim
is now the preregistered decision-right eligibility and oracle-to-policy
conversion framework, with this null conversion as its final experimental
stage.
