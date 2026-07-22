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
2. The PI reports that Garrido's acceptance criterion is canonical mean ReT.
   Therefore no unrequested service floor will be invented as a successor
   claim gate.  Worst-product, unresolved, lost and resource use remain
   mandatory secondary disclosures.
3. The 24-root c64/c256 expansion failed the frozen action-agreement gate:
   87/96 arm-states agreed (90.625%), despite mean and q95 planning-value
   errors of 0.000340 and 0.000892.  A targeted c256/c1024 audit of the nine
   disagreements agreed in 8/9 states, with mean error 0.000095.  This is
   evidence of numerically near-indifferent actions, not permission to round
   the 95% gate down.
4. Run one burned, ReT-first tie-aware convergence audit.  The primary remains
   `early_ret_complete_cohort`; among sequences within 0.002 planned ReT of
   the maximum, prefer higher worst-product fill and then fewer unresolved
   orders.  The 0.002 band is fixed before this audit and is one fifth of the
   final +0.01 SESOI.  It is a numerical indifference rule, not a safety claim.
   This audit failed: c64/c256 agreed on 80/96 first actions (83.33%), while
   mean and q95 value errors remained only 0.000383 and 0.001014.  The
   service tie-breaker therefore increases numerical ranking instability and
   is not eligible for freeze.
5. Return to the pure-ReT selector and run the already specified global
   c256/c1024 convergence check.  This is a budget escalation, not a changed
   outcome gate.  If it fails, comparator convergence remains unresolved.
6. If and only if the original convergence thresholds pass, freeze that
   universal comparator, expand its raw retained/reset rows, and run the
   clustered prospective power audit with SESOI +0.01.
7. M1/M2/M4 require a deployable residual against that frozen comparator;
   oracle knowledge remains a clairvoyant upper bound, never learned value.
