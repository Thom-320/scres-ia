# Authority Ladder V1 preregistration

Date: 2026-07-28

Status: `DRAFT_PROSPECTIVE_UNOPENED_NOT_AUTHORITY`

Machine-readable mirror: `contracts/authority_ladder_v1.json`

## Binding temporal boundary

Program Q is submitted first.

Before an editorial-system submission receipt exists, this study permits only:

- preregistration;
- fail-closed action routing;
- tests on synthetic or already burned data;
- specification of physical sourcing, resources, and native cadences;
- the flags-off equivalence harness.

No fresh root, tape, optimizer seed, DDMRP search, MPC screen, PPO, MLP,
DMLPA, or KAN run is authorized.

Program Q remains scientifically immutable and cannot be revised using this
study.

## Research question

When a thesis-grounded supply-chain DES exposes additional physically valid
decision authority, which decision families create deployable contingent
value, how much is captured by structured control, and does any observable
residual remain for a learned controller?

The study separates:

1. value of perfect information;
2. value convertible from deployable observations;
3. structured-feedback value;
4. specifically neural value;
5. architectural efficiency conditional on comparable outcomes.

## Domain identity

### Historical reference

`E_Garrido` denotes the monoproduct, thesis-grounded reference environment. It
is a reconstruction with documented attribution limits, not a validated
digital twin.

Program Q is not A0. Its researcher-defined products, risk-off demand process,
eight-week horizon, and count-four product-mix action remain exclusive to
Submission A.

### Prospective superset

`E_star` is one maximal physical kernel constructed after Submission A is
submitted. From its first scientific execution it contains:

- all approved upstream and downstream nodes;
- all approved buffers, finite capacities, and locations;
- procurement sources and contract state;
- routes, transport capacity, and lead times;
- inventory in transit;
- the complete resource ledger;
- the fixed risk and demand contract;
- one superset causal observation;
- native review cadences;
- action families controlled only through masks.

No level may introduce a new node, transition, source, cost, observation field,
or risk process.

### Flags-off bridge

With all new rights fixed to Garrido-approved baseline values, `E_star` must
reproduce `E_Garrido` within a tolerance frozen before the bridge is run.

The bridge covers:

- event-sequence hashes;
- trajectories;
- Excel ReT;
- inventory levels;
- orders and backlog;
- risks;
- flow and resource ledgers.

A failed bridge yields `STOP_ESTAR_FLAGS_OFF_NON_EQUIVALENT`. No authority
screen may proceed.

## A0

Recommended A0:

> Monoproduct `E_star`, with the original buffer/shift rights approved by
> Garrido and all new procurement and positioning rights fixed at approved
> baseline values.

The final A0 manifest must enumerate commit, horizon, step cadence, risks,
products, observation fields, reward, primary endpoint, parameter values,
resource budgets, action keys, neutral values, and review epochs.

Garrido's written approval of that manifest is required before freeze.

## Factorial decision masks

Decision families:

- `P`: procurement authority for Op1--Op2;
- `U`: upstream-buffer authority;
- `D`: downstream-buffer authority.

The pre-screen contains all eight masks:

| Mask ID | Rights |
|---|---|
| `M000` | none beyond A0 |
| `M100` | P |
| `M010` | U |
| `M001` | D |
| `M110` | P + U |
| `M101` | P + D |
| `M011` | U + D |
| `M111` | P + U + D |

This factorial identifies composition and interaction; a presentation ladder
does not replace it.

All masks share the exact same kernel, observation vector, risk/demand tapes,
horizon, endpoint, resource budgets, SESOI, and search budget.

Every policy admissible in a lower mask is embedded in each superset mask by
holding added rights at baseline. Search output that fails to recover the
embedded incumbent is reported as search failure, not evidence that authority
reduces performance.

## Multirate action contract

Every action dimension declares:

- `review_epoch`;
- `next_review_in`;
- current value;
- feasible range or discrete set;
- neutral/baseline value;
- carry-forward rule;
- resource consequences.

The environment exposes both the action mask and time to the next eligible
review in the common observation.

Outside its native review epoch, a dimension cannot change:

- Op1 changes only at contract-renewal epochs;
- Op2 changes only at procurement-review epochs;
- buffers change only at their signed review cadence;
- shifts and transport follow their own signed cadences.

Off-epoch values carry forward. Unknown action keys fail closed.

## Physical sourcing and resource contract

Strategic top-up injection is prohibited:

```text
total_strategic_raw_injected = 0
total_strategic_rations_injected = 0
```

Legitimate external procurement remains a source, with orders, receipts,
capacity, price/budget, and lead-time commitments recorded.

A conservative transfer is bounded by:

```text
q_move = min(shortfall, upstream_available, transport_capacity)
```

Every transfer records origin, destination, departure, quantity, transit
inventory, lead time, arrival, and resource use.

The primary objective is Excel ReT subject to frozen resource budgets for:

- raw material ordered and received;
- supplier-order count;
- contract activations;
- lead-time commitments;
- inventory-time;
- terminal stock;
- transport or vehicle hours;
- expediting;
- shift-hours.

A budget-violating policy is ineligible. A preregistered Pareto frontier may
be reported as a companion analysis.

The optional scalarization

```text
J_lambda = ReT - lambda_I*I - lambda_P*P - lambda_T*T - lambda_S*S
```

is secondary sensitivity over a frozen preference grid. It never replaces the
primary endpoint.

## Common SESOI

One absolute Excel-ReT SESOI applies to all masks because they share one
physical contract, population, horizon, and endpoint.

Garrido must sign the SESOI and its operational interpretation before any
fresh data are opened. Physical companion endpoints receive separately frozen
guardrails.

Normalized headroom capture is secondary and is interpreted only when the
absolute denominator is material.

## Estimands

Perfect-information diagnostic:

```text
H_PI = V(pi_PI) - V(pi_static)
```

This does not authorize neural training.

Observable convertibility:

```text
H_obs = V(pi_flex_obs) - V(pi_static)
```

`pi_flex_obs` is cross-fitted, strictly non-anticipative, and restricted to
the common deployable observation.

Deployable residual:

```text
Delta_obs = V(pi_flex_obs) - V(pi_BestStructured)
```

Neural premium:

```text
Delta_N = V(pi_learner) - V(pi_BestStructured)
```

Architecture contrast:

```text
Delta_KAN = V(pi_KAN) - V(pi_matched_MLP)
```

Where exhaustive enumeration is impossible, results are called
`best-found incumbent` or `lower bound within the frozen library`. They are
never called exact ceilings.

## Gates

### G0 — physical validity

- flags-off bridge passes;
- mass and provenance close by product;
- strategic injections are zero;
- replay and campaign identity pass;
- every exposed dimension passes liveness at its native epoch;
- resource ledgers close.

### G1 — diagnostic decision relevance

- action ranking changes materially under at least one admissible state;
- absolute perfect-information or frozen-library improvement is reported;
- no perfect-information result authorizes a learner.

### G2 — observable conversion

- cross-fitted non-neural policy uses only deployable information;
- its absolute ReT contrast exceeds the common SESOI with positive LCB95;
- required physical guardrails pass;
- placebos and open-loop-collapse checks pass.

### G3 — deployable residual

`Delta_obs` exceeds the common SESOI with positive LCB95 under matched
information, actions, and resources.

Only G3 authorizes an architecture bakeoff in that mask.

Failure in one mask does not cancel the other masks.

### G4 — architecture bakeoff

- PPO+MLP;
- RecurrentPPO+MLP;
- PPO+DMLPA;
- PPO+Real-KAN.

David's results may enter only as an external replication and never affect
selection.

## Comparator envelope

`BestStructured` is the best eligible member among comparator families
applicable to a mask:

- constant full-contract;
- open-loop best-found;
- order-up-to;
- `(s,S)`;
- DDMRP with net-flow and dynamic buffer adjustment;
- direct-DES MPC;
- robust/scenario MPC;
- an interpretable observable policy.

DDMRP applies to inventory/buffer decisions, not universally to every right.

All comparators receive the same causal information, action rights, physical
resources, horizon, and development budget appropriate to the comparison.

MPC has two reported bars:

1. `quality_bar`: reasonable compute for the strongest scientific comparator;
2. `equal_time_bar`: the learner's online decision-time budget.

Beating only the second supports an amortization or latency claim, not
decision-quality superiority.

## Architecture matching

Any authorized bakeoff reports separate panels for:

- identical DES interactions and development configurations;
- matched parameter count;
- matched FLOPs or action latency;
- matched wall-clock and hardware.

KAN is a candidate for a non-dominated accuracy/compute/interpretability
frontier. It is not a novelty claim and does not appear in the final title
unless it reaches G4.

## Custody and splits

This draft contains no roots, tapes, or optimizer seeds.

A later freeze must add disjoint development, selection, test, and
confirmation blocks plus immutable receipts. Confirmation remains sealed until
all predeclared gates allow it.

No historical STOP is retroactively changed by this study.
