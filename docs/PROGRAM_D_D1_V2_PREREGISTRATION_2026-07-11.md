# Program D D1-v2 — preregistration (frozen before results)

Status: **FROZEN 2026-07-11**. Primary metric: Garrido Excel `ret_excel`.
No PPO, retained learning, KAN, RNN, SAC, TD3, or MARL is authorized.

## Evidence synthesis

**Thesis facts.** The chain contains a standing downstream order queue, daily
freight, contingent R24 demand, and a fixed SPT-like service rule. Op1/Op2 are
outside the PI-authorized decision surface.

**Repository facts.** The proxy implements six pure Op9 reorderings. D1-v1 ran
without active risks. Track B's general adaptive-win framing is superseded by
its same-contract challenge; Program L stopped for lack of observable headroom.

**Experimental evidence.** D1-v1 proves nominal static authority only. A
non-promotional active-risk diagnostic found different ReT and service-loss
winners across families; this justifies but does not promote D1-v2.

**Unproven inference.** A state-dependent rationing rule may outperform the best
admissible constant. Only branching, sequential tree, and virgin gates can show it.

## Frozen contract and universes

Rules: `spt_contingent`, `fifo_contingent`, `lpt_contingent`, `spt_flat`,
`fifo_flat`, `age_threshold`; threshold fixed at 336 h. Rule changes only reorder
the queue. Selection seeds 810001–810030; validation 811001–811030; virgin
820001–820060. Each calibration split has five tapes per R1/R2/mixed ×
current/increased cell, 104 weeks, with Figure 6.2/text downstream quantity
source (2400–2600). Tapes are materialized post-warm-up and replayed identically.

## Phase 1–2 — admissible constant frontier

Select the highest-mean-ReT rule whose mean degradation versus thesis SPT is at
most 1% in service-loss AUC and 2% in lost orders. Freeze it before validation.
Validation authorizes branching only if liveness and conservation pass and either
(a) delta ReT is at least 0.001 with paired CI95 lower bound above zero, or
(b) service loss falls at least 2% with CI95 lower bound above zero.
Contradictory endpoints permit branching but no managerial claim.

## Phase 3 — exact-prefix branching

Sample ten states per calibration tape: two each nominal queue, occupancy at
least 80%, maximum age at least 336 h, R24/contingent present, and downstream
disruption/recovery. Missing strata use a logged nearest-quantile fallback.
Replay from scratch under the frozen constant, verify identical predecision and
exogenous hashes, switch for 24 h before dispatch, then return to the common
continuation. Report 72 h and 28 d for orders open at branching plus new orders.

Oracle promotion requires all: two rules optimal in at least 15% of states; none
above 85%; service loss at least 5% lower with CI95 lower bound above zero; ReT
co-directional with CI95 lower bound at least zero; lost-order relative increase
CI95 upper bound at most 2%; no 72 h benefit becoming 28 d damage; no eviction or
disappearing-demand mechanism. Otherwise emit
`STOP_NO_STATE_DEPENDENT_RATIONING_HEADROOM`.

## Phase 4 — observable convertibility

Only after Phase 3: depth-3 tree, five GroupKFold splits by tape, using current
SB inventory, queue quantity/count/contingent share/size and age quantiles,
oldest age, transit, current Op9–Op12 state, recent demand/delivery/fill, prior
rule, and operational day. Exclude future risk, repair, regime, demand, outcomes.
Each fold is evaluated sequentially on held-out tapes.

Calibration requires at least 50% oracle headroom, ReT CI95 lower bound above
zero, service loss at least 5% lower with CI95 lower bound above zero, lost no
worse than 2%, and Pareto non-dominance. Freeze code, tree, features, comparator,
analysis, manifest, and git SHA before one virgin opening. Virgin confirmation:
ReT CI95 lower bound above zero; service loss at least 3% lower with CI95 lower
bound above zero; favorable in at least 4/6 cells and 70% tapes; lost and CVaR10
no worse than 2%. Only then emit `PROMOTE_TO_SEQUENTIAL_CONTROL`; this still does
not establish retained learning.

