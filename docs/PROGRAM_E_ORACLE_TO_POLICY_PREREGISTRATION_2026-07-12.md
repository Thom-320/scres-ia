# Program E — oracle headroom to observable policy

Status: **FROZEN BEFORE SEEDS 900001+ OR ANY PROGRAM E POLICY FIT**.

## Scope and non-rescue boundary

DRA-2 and DRA-2b remain terminal under their own preregistered gates. Program E
does not relabel either STOP. It asks a different question: whether a daily
closed-loop policy using only contemporaneous information can convert any of the
finite-convoy headroom into out-of-sample performance.

Program E is the last experimental campaign before the manuscript freeze. It is
reported whether PPO wins, ties or loses.

## Physical and information contract

The finite-convoy physics, risks, ReT implementation, resource ledger and action
mask are unchanged. The lane remains a stylized extension, not an exact recovery
of Garrido's transport resources.

The daily policy observes staging, inventory, backlog, recent realized flows,
convoy availability/ETA, current route status, prior action, phase and cumulative
resources. It never observes future demand, shocks, repairs, regimes, oracle
labels or future outcomes.

## Policies

1. Nine threshold/wait static policies.
2. Their validation-selected convex mixture under each candidate's departures and
   unavailable-hours envelope.
3. One frozen observable heuristic.
4. A depth-3 tape-cross-fitted policy tree trained on Program E training labels.
5. MaskablePPO, MLP 2x64 tanh, exactly the hyperparameters and ten learner seeds in
   `contracts/program_e_policy_realizability_v1.json`.

No architecture or reward sweep is allowed.

## Reward and evaluation

PPO receives one dense reward: negative daily service-loss increment plus 0.1
times daily backlog-age increment, divided by p95 normalizers estimated only from
training static rollouts. ReT is never the reward. Normalizers are frozen before
validation.

The primary outcome is paired ReT against the validation-frozen convex static
mixture. Resource use is constrained externally, not hidden in adjustable reward
weights.

## New data universes

- Training: 80 tapes, seeds 900001–900080.
- Validation: 20 tapes, seeds 910001–910020.
- Virgin confirmation: 60 tapes, seeds 920001–920060.

Every split is balanced across the four frozen disruption families.

## Validation gate

PPO opens the virgin test only if all hold:

- mean ReT delta at least +0.01 and CI95 lower bound above zero;
- conversion efficiency at least 0.50;
- service loss non-inferior;
- no increase in lost orders;
- resource-envelope validity;
- at least 8/10 learner seeds favorable;
- at least 70% validation tapes favorable;
- no tail-risk deterioration.

The 5% service-loss threshold is retained only for the stronger managerial-value
claim. It is not retroactively declared satisfied by DRA-2b.

If validation fails, emit `STOP_PROGRAM_E_VALIDATION`; do not open seeds 920001+.
If it passes, freeze weights, tree, heuristic, convex-mixture weights and analysis
before opening the 60 virgin tapes.

## Claim ladder

- Tree and PPO lose: clairvoyant headroom is not observably convertible.
- Tree wins and PPO does not: interpretable adaptive control suffices.
- Tree and PPO tie above static: no additional neural value.
- PPO beats static but not tree: adaptive value, no neural incremental value.
- PPO beats both under resources: adaptive neural-policy value.
- Only a confirmed out-of-sample PPO win authorizes persistent/reset/frozen.
