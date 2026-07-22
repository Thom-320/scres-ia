# Q-R1 comparator challenge v2: burned preflight (2026-07-22)

## Scope

This is `BURNED_DEVELOPMENT_NO_CLAIM`.  It uses roots 7570801--7570804,
eight selected campaign states, and no fresh confirmation roots.  Learner
returns and retained-minus-reset performance were not used to select a
configuration.  The result does not alter the frozen Q-R1 successor STOP.

## Instrument corrections

Comparator v2 differs from the earlier exploratory prototype in four material
ways:

1. It enumerates all six latent `(theta, Z)` states with exact posterior
   weights.
2. Its nested conditional-demand bank is keyed only by history root, campaign,
   week, latent state, and path id.  It is independent of arm, belief,
   observation hash, and action.
3. It optimizes `early_ret_complete_cohort` and reports visible and whole-
   campaign metrics in parallel.
4. It abstains when no candidate satisfies the declared constraints; it never
   labels an infeasible action safe.

The conditional-demand integral remains an approximation.  The controller is
therefore the strongest tested member of this bounded family, not an optimal
MPC or POMDP policy.

## Convergence

The cheap c4-to-c16 screen did not converge.  The c16-to-c64 screen reduced
value errors below tolerance but agreed on only 15/16 first actions (93.75%).
The frozen 95% action gate was not rounded down.

The c64-to-c256 comparison passed for both H4 scenario and H4
constraint-aware (expected worst-product floor 0.70):

| Diagnostic | Scenario | Constraint-aware |
|---|---:|---:|
| First-action agreement | 16/16 | 16/16 |
| Mean absolute planning-value error | 0.000355 | 0.000355 |
| q95 absolute planning-value error | 0.000780 | 0.000780 |
| Abstentions | 0 | 0 |

A monolithic c64/c256 invocation first reached Pareto evaluation and then hit
its 3,600-second hard cap without writing the already-computed convergence
rows.  The runner was corrected to separate convergence from Pareto through a
validated receipt.  The repeated convergence used the same burned states,
family, thresholds, and budgets.

## Preliminary ReT--service point

The c64 policy was then evaluated over the same eight retained/reset pairs:

| Controller | Delta early full-cohort ReT | Delta worst-product fill | Delta unresolved orders | Delta lost orders |
|---|---:|---:|---:|---:|
| H4 scenario | +0.04340 | -0.00690 | +0.375 | 0 |
| H4 constraint-aware, floor 0.70 | +0.04340 | -0.00696 | +0.375 | 0 |

Mass residual was exactly zero and neither controller abstained.

These eight burned pairs suggest that a converged structured controller can
retain materially more than the +0.01 ReT SESOI while showing a smaller
worst-product point penalty than the frozen confirmation.  They do **not**
establish safety or a causal improvement over the frozen H4/p16 controller:
the populations differ, no clustered interval is available, and the 0.70
constraint appears inactive because its result is effectively identical to
scenario mode.  The positive unresolved delta also remains unresolved.

## Routing

1. Do not authorize a learner from this preflight.
2. Obtain Garrido's neutral definition of any service floor or unresolved-
   demand requirement.
3. Expand the burned frontier with raw paired rows and an actually active,
   domain-grounded constraint; select a universal structured controller only
   by convergence, feasibility, and calibration performance.
4. Run a prospective power audit with SESOI +0.01 before assigning fresh roots.
5. Only a safe residual against that frozen controller may authorize M1/M2/M4.

