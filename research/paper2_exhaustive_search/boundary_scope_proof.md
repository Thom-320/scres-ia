# Boundary of what can and cannot be certified

Date: 2026-07-13
Current machine status: `OPEN_ACTIVE_BOUND_REQUIRED`.

## Result

The current repository does not contain a valid Paper 2 environment. It also does not yet support terminal outcome B over the complete user-permitted extension class.

This is not because a positive result is assumed to exist. It is because the extension class is open-ended and one implemented family still lacks a quantitative ceiling:

1. The finite, versioned Track A-C and Program D-K3 contracts are null, retracted, guardrail-failing, physically invalid, or comparator-incomplete as recorded in `phase0_failure_taxonomy.json`.
2. The new product, transport, information, storage, inspection, transshipment, queue and mission-loadout families require operational facts absent from the thesis. They are `blocked_domain_fact`, not falsified. Mission loadout is thesis-motivated (one pack covers 24 hours, up to 1.4 kg, typically about five packs depending on mission/resupply) but absent from the executable theatre-delivery state.
3. The integrated M/T/R response-team contract has a policy-level null but not a family ceiling. It has 11,184,811 effective full-horizon calendars; the experiment tested three constants. Direct enumeration is about 741.79 serial CPU-days at the latest timing. An exact event-effect quotient reduces the required DES runs 22.70× to about 32.68 serial CPU-days, but those executions have not been completed and the simulator still has no branchable weekly state or additive canonical objective.

Accordingly, the exact scientific state is:

- Paper 2 confirmed: **false**.
- Paper 3 authorized: **false**.
- Finite-envelope boundary terminal: **false**.
- Further PPO/retention training authorized: **false**.

## Why a universal boundary over arbitrary extensions is impossible

The allowed extension language admits new persistent latent states, leading signals, incompatible actions, commitment delays, product types, and scarce resources, provided they are disclosed and later validated. Such a class can embed a headroom-bearing two-state partially observed control problem.

Construct an idealized two-product screen:

- a fraction `alpha` of orders require non-substitutable type A or B;
- the type must be produced one period before demand;
- the two demand regimes are balanced;
- a deployable signal identifies the next regime with accuracy `q > 0.5`;
- a correct type is on time and a wrong type cannot serve the order;
- total production is identical across policies.

The best symmetric open-loop decision is correct with probability 0.5. Signal-following is correct with probability `q`, so the idealized observable increment is

\[
\Delta ReT = \alpha(q-0.5)
\]

under the deliberately simplified one-for-one endpoint. For `alpha=0.05` and `q=0.75`, the constructed increment is `0.0125`; the signal-null `q=0.5` gives exactly zero.

This example is not MFSC evidence and is not a proposed result. It is a counterexample to a universal impossibility claim: without bounds on product nonfungibility, affected-order share, signal quality, and commitment physics, past nulls cannot prove that every permitted future extension has `H_obs <= 0.01`.

## Finite-envelope certification rule

A terminal boundary is permitted only after the searched envelope is frozen as a finite set of versioned contracts. For every row, one of the following must exist:

1. an exact or certified resource-restricted `H_PI` UCB below 0.01;
2. a formal dominance/information-inclusion proof;
3. an exact zero-liveness result because the action is absent from the transition kernel;
4. an exact strong classical optimum leaving less than 0.01 learner headroom;
5. a guardrail impossibility under every feasible action;
6. a domain-fact blocker explicitly outside the frozen numerical envelope.

A failed heuristic, learner, or signal policy is not a family ceiling.

## Quantitative boundary currently established

| Mechanism | Strongest quantitative closure currently justified |
|---|---|
| Track B eight-dimensional control | PPO minus same-contract constant `-0.00001805`, CI95 `[-0.00002862,-0.00000809]`. Historical positive claim was comparator-restricted. |
| D1 priority | Restricted oracle `+0.0010945`, CI95 `[+0.0004225,+0.0019773]`; observable tree is negative and loses orders. |
| DRA-1 allocation | Restricted oracle `+0.0000879`, UCB95 `0.0001659`. |
| Program I resource-equal transport branching | Oracle `+0.00001045`, UCB95 `0.00001420` for the versioned branch portfolio; not a global reservation ceiling. |
| Program J alarm/maintenance | Maximum screened restricted oracle `0.0001338`; observable rules negative. This closes alarm scaling within J, not every domain-valid maintenance extension. |
| K3 budgeted replenishment | Learned policy exactly equals one fixed period-8 schedule; adaptive and neural claims retracted. |
| Multi-echelon lag alone | Formal information inclusion: `O_lag` is a subset of `O_current`, so lag cannot improve the frozen G/H optimum. |
| Native reservation, storage, inspection, lateral and multimodal actions | Exact zero action liveness because those actions are not in the thesis transition kernel. Their introduced versions remain domain-blocked. |
| Alternate-route recourse | Exact thesis-native `H_PI=H_obs=0` because there is no alternate-route action. The Program-L development heuristic reaches at most mean `+0.004017` with LCB95 below zero and extra departures; this is not a ceiling for the unvalidated extension. |
| Mission-loadout carried autonomy | Exact current-kernel `H_PI=H_obs=0`: the DES has no cohort carried-inventory state, loadout action, sealed carried-consumption ledger or return rule. A Garrido-approved downstream extension remains domain-blocked. |
| Native R14 rework sequencing | Rework consumes the same Op5-Op7 hourly capacity before new raw material in the one-product simulator. Discard violates the thesis/conservation guardrails; a new sequencing benefit would require product-specific consequence or deadline physics absent from the kernel. |
| Integrated M/T/R team | Signal policy `-0.001309`, CI95 `[-0.006384,+0.003093]`; feasible clairvoyant lower bound `+0.002336`, CI95 `[-0.000993,+0.005841]`; no ceiling. |

A deterministic per-tape canonical upper bound was also computed on the 120 burned locked tapes. Because `ret_excel_visible_v1` has an un-clipped `0.5/RP` recovery branch and the action-sensitive causal cone begins before almost every order, its mean upper gap versus constant M is `138.185` (bootstrap interval `[87.003,200.842]`); zero tapes have a bound at or below `0.01`. This is rigorous but vacuous. It proves that an affected-order/unit-score shortcut cannot close M/T/R, not that the family has large headroom.

## Smallest next closure step

The only current numerical `active_for_bound` family is the integrated M/T/R team. It can change state only after:

- the full 11,184,811-calendar frontier is certified or replaced by a valid upper-bound method;
- a resource-restricted PI ceiling is computed through the canonical aggregator;
- the one-team-week resource is confirmed to be truly fungible across M/T/R, while reserve issue/replenishment is frozen as an outcome or separately matched resource;
- the retrospective selected-metric null equality is replaced by a preregistered full-ledger null test and the action-trajectory replacements are persisted;
- the 80% efficacy and 0.85 signal assumptions receive a domain-valid envelope or the contract is explicitly excluded from the Garrido-defensible finite envelope.

Until then, reporting a terminal exhaustion certificate would overstate the evidence.

The implemented quotient is a valid route to the first two requirements, not their completion. For each tape and week it records only whether the active action mitigates an R11, R22/R23 or R24 event. Calendars with the same effect word are outcome-equivalent in the frozen code, so one DES run per effect word suffices; the final calendar optimization must still scan all calendars by table lookup under the frozen reserve rule.
