# Program T T0 theory and residual-headroom gate

## Binding status

`FROZEN_DEVELOPMENT_ONLY_NO_NEW_SCIENTIFIC_SEEDS`

T0 implements the defensible part of the master plan without prespecifying a
winning architecture. It does not reopen Program O, O-R, or Q and cannot itself
support a Paper 2 claim.

## Critical correction to the proposed master plan

The historical Program O/Q `belief_mpc` is not a full-DES optimum. Its frozen
implementation plans for three or four weeks, propagates expected product demand,
and lexicographically minimizes backlog area, worst backlog, terminal shortage,
and switching. It does not optimize the canonical ReT aggregator. The bounded
`belief_dp` is likewise documented as an approximate seven-outcome planning
comparator, not a certified POMDP bound.

Consequently, a learned method may legitimately beat those controllers. But the
comparison would be unpublishable if the gain disappeared after aligning MPC to
ReT, lengthening its horizon, or enforcing the same service constraints. T0 must
strengthen the comparator before training the hybrid.

## What is frozen now

1. An exact reduced POMDP benchmark with a complete open-loop frontier and exact
   belief-state dynamic program.
2. Nested attribution: strong MPC, terminal value only, belief correction only,
   both components, then the confidence gate.
3. Equal observable history, physical resources, and online accounting.
4. Residual-headroom and placebo gates before any large neural training.
5. Component-level interfaces that can later support retained/reset experiments
   for Paper 3.

The GRU size, number of quantiles, loss weights, conformal method, and online
latency limit are deliberately not frozen. Choosing them now would create false
precision before we know whether residual headroom exists.

The confidence gate is also not called a safety certificate. Split-conformal
coverage for a local prediction does not by itself guarantee joint coverage for
an adaptive eight-week trajectory. Repeated switching can accumulate persistent
consequences. The gate is a calibrated abstention mechanism; episode-level
worst-product and lost-demand safety remains a confirmatory estimand.

## Evidence ladder

### T0-A: exact benchmark

The reduced benchmark must establish the ordering between exact belief-DP,
full-horizon open-loop, and receding-horizon MPC. Full-horizon MPC must recover
the exact policy. Short-horizon residual gaps are diagnostic evidence of a
continuation-value mechanism, not evidence for the full DES.

### T0-B: strengthened full-DES comparator

On burned data only, fit and evaluate:

- ReT-aligned MPC at horizons 1, 3, 4, 6, and 8;
- analytic and particle beliefs under identical history;
- nominal, scenario, and robust variants;
- explicit worst-product and lost-demand constraints;
- a quality-versus-online-time frontier.

The full canonical aggregator remains the scorer. A planning surrogate is
allowed only when its approximation error is measured and frozen.

### T0-C: learning attribution

Cross-fit the M1-M4 arms on disjoint burned partitions. The full simulator state
may be a training-only auxiliary label, but it is forbidden at evaluation. Real
history must beat shuffled, delayed, and wrong-channel histories.

### T0-D: go/no-go

Large training is authorized only when the residual advantage over the strongest
deployable MPC has LCB95 at least 0.015, remains positive against the complete
open-loop frontier, preserves worst-product service, and survives the history
placebos and resource ledgers.

If horizon-8 MPC absorbs the gap, Program T stops in this environment. That is a
mechanism result: the historical gap was planning truncation, not learned value.

## Paper 3 continuity

The component boundaries are intentional. A later Paper 3 can separately reset
or retain the recurrent belief, terminal value, policy weights, and structured
posterior while resetting all physical state. No retained-learning claim or seed
is authorized by this document.
