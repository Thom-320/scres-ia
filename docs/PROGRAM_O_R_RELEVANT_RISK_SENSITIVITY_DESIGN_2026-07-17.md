# Program O-R — Garrido-style relevant-risk sensitivity design

## Purpose and boundary

This is a secondary sensitivity for Program O-R, not a search over all risk
combinations and not a rescue of the risk-off learner verdict.  It follows the
thesis strategy of enabling risks that act on the operations controlled by the
decision under study.

The Program O-R action allocates three weekly production rights between two
products on the shared Op5--Op7 line.  The directly relevant implemented risks
are therefore:

- `R11`: failures at Op5/Op6;
- `R14`: defects at Op7;
- `R21`: joint disruption of Op5/Op6/Op7 and the downstream Op9 stock point.

`R22` is retained as a negative-control mask: it affects product-blind LOCs
Op4/8/10/12 and should not create product-mix information by itself.  `R12`,
`R13`, `R23`, and black-swan `R3` are excluded.

`R24` is scientifically relevant because it changes demand, but the current
one-product risk process has no frozen rule assigning a contingent order to
`P_C` or `P_H`.  Activating it now would silently invent the most important
part of the signal.  Every R24 mask is therefore blocked until its product
label, observation time, and request-snapshot ledger treatment are frozen.

## Frozen cheap panel

The executable masks are:

1. `R11+R14` — production/quality risk;
2. `R21` — joint line and Op9 capacity risk;
3. `R11+R14+R21` — combined relevant production risk;
4. `R22` — product-blind downstream negative control.

Each mask is evaluated at the thesis `current` and thesis `severe` tables.  No
continuous frequency/impact grid, PRIM, Sobol, commonality search, or R3
escalation is permitted.  This is eight profiles rather than the abandoned
global atlas.

## Sequence

1. Complete the risk-off Program O-R learner contract first.
2. Add a product-tag-preserving risk adapter and prove risk liveness for every
   enabled risk; a factor that is enabled by label but does not change the
   affected transition invalidates the profile.
3. Evaluate the **frozen risk-off learner** and the same open-loop/classical
   comparators on all eight profiles using burned sensitivity tapes.
4. Report ReT as primary and CVaR/AUC/backlog as secondary.
5. Fine-tuning is not automatic.  It requires a later preregistration, new
   training tapes, one frozen initialization/budget, and a fresh confirmation
   block.  It cannot change the risk-off Paper-2 verdict.

The sensitivity answers whether relevant Garrido risks preserve, amplify, or
erase already-learned product-mix adaptation.  It cannot select the environment
in which the learner happens to win.
