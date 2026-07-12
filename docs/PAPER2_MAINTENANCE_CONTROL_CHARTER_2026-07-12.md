# Paper 2 — Condition-Based Maintenance and Shared Repair-Crew Allocation

## Binding question

Can a non-anticipative policy allocate one fixed 24-hour weekly maintenance
slot across Op5, Op6 and Op7 and outperform every periodic same-contract
calendar on unseen tapes?

This is a new decision family. Programs E, G, H and I remain terminal and are
used only as boundary evidence.

## Thesis anchor and declared extensions

The thesis models Op5-Op7 as a serial assembly line, exposes Op5/Op6 to R11,
Op7 to R14, and deducts 24 hours of maintenance per week from effective
capacity. It also describes DES entities as passive and identifies stationarity
as a limitation. The weekly allocation decision, persistent degradation,
condition sensor and single shared crew are declared researcher-imposed
extensions. They are not represented as recovered historical facts.

## Scientific contract

Each week exactly one of `PM5`, `PM6`, or `PM7` receives the 24-hour slot.
Maintenance causes immediate downtime and persistent condition improvement.
Utilization and exogenous wear worsen condition. R11 candidates and R14
innovations are materialized before policy evaluation and shared under exact
CRN; policy changes vulnerability, not the threat tape. Finite WIP creates
blocking and starvation. Corrective work occupies the same crew and queues
against preventive work; the primary contract does not interrupt maintenance
already in progress because the thesis does not specify a safe preemption rule.

Paper 2 succeeds if any deployable policy beats the best periodic calendar
under equal maintenance resources. A neural claim requires PPO to beat the
best structured policy as well. Paper 3 remains locked until Paper 2 passes a
virgin confirmation.

## Non-rescue boundary

The primary endpoint remains the canonical full-ledger Excel ReT, with service
loss, quantity-weighted ReT, lost orders, tail risk and resources as binding
guardrails. Cobb-Douglas is secondary. No risk amplitude, metric, action,
horizon, sensor or efficacy parameter may be changed using learner results.
