# Program H — Information Sufficiency and Belief-State Audit

Prepared and frozen on 2026-07-12 for the 2026-07-13 study boundary. No Program H tape has
been opened and no Program H policy or learner has been fitted.

> Program G remains terminal.  
> Program H estimates whether observable policy value is information-limited or
> policy-class-limited.  
> No risk, signal, metric, physics, horizon, or action parameter is selected using learner
> performance.

## Question and estimands

Under the corrected stylized spatial contract, does a strong non-anticipative policy using the
available operational information materially beat frozen ABAB, or does the perfect-information
headroom depend on inaccessible future information?

Let $J_{static}$ be ABAB, $J^*_{obs}$ the optimal non-anticipative value under the frozen
information contract, and $J_{PI}$ the perfect-information ceiling. Program H estimates
$H_{PI}=J_{PI}-J_{static}$, $H_{belief}=J_{belief}-J_{static}$, and
$\eta_{information}=H_{belief}/H_{PI}$. Its central diagnostic is an optimistic QMDP upper
bound. If $J_{QMDP}-J_{static}<0.01$, even the bound leaves no practically material observable
advantage. QMDP is not a deployable policy or an exact POMDP solution.

## Frozen contract

Program G v1.2 is inherited without modification: two CSSUs; `{A,B,HOLD}` weekly priority;
four-week episode; S1 production; frozen convoy capacity and cycle; the same 12 surge-1.50
cells sampled uniformly; the same tempo, signal, route, and order processes; canonical
`ret_order`; and ABAB as comparator. `ret_quantity`, attended orders, worst-CSSU fill, and
unfulfilled rations are guardrails. Program H cannot alter cell, signal quality, persistence,
horizon, risks, metric, action rights, reward, or practical thresholds.

## Belief state

The reported tempo marginal has nine joint labels
$Z=(Z_A,Z_B)\in\{low,routine,surge\}^2$. Because tempo is semi-Markov, the actual filtering
state is augmented to $\widetilde Z_t=(Z_{A,t},d_{A,t},Z_{B,t},d_{B,t})$, where $d_i$ is
dwell age or an exactly equivalent residual-duration phase. Filtering only nine labels is
prohibited because it discards transition-relevant history. True tempo and dwell are never
passed to a policy.

Before policy evaluation, the filter must pass normalization, one-step predictive calibration,
label-swap symmetry, and synthetic recovery tests. Its log loss must improve on the frozen
marginal-prior predictor on development tapes; otherwise it is reported uninformative and no
learner is authorized.

## Information contracts

O0 is primary and exactly matches Program G: current local signals, inventory fractions, cover,
and episode phase. Its filter may condition only on that observation history and the resulting
observable physical trajectory. It may not use realized demand merely because demand exists in
the tape.

O1 is an optional sensitivity adding lagged realized demand per CSSU, backlog quantity and age,
recent deliveries, convoy ETA, and current route state. O1 is
`BLOCKED_AND_EXCLUDED_UNLESS_SIGNED_OFF`. A dated Garrido record must confirm that every field
is operationally available before any O1 fit or tape. Without that record, Program H completes
on O0 alone. Neither contract includes future demand, latent tempo, next risk, future closure
duration, oracle labels, or future outcomes.

## Policy ladder and bound

The frozen ladder is ABAB; prior cover/MPC as references; a regret-weighted observable policy
fitted to counterfactual action values rather than action accuracy; belief-MPC; POMCP or a
point-based solver with prespecified search budget; and the perfect-information oracle as a
diagnostic ceiling. The primary upper bound is QMDP. An information-relaxation bound is allowed
only through a hashed amendment freezing its algorithm and penalty before seed 1060001 opens.
Failure of a particular policy is never described as a bound on $J^*_{obs}$.

## Universes and order

- Development/calibration: 1060001--1060200 (200 tapes).
- Locked belief-policy test: 1070001--1070400 (400 tapes).
- Virgin learner test: 1080001--1080400 (400 tapes), unopened unless the complete gate passes.

No Program D--G tape is reused. Execution order is filter preflight; O0 bound and policy
development; freeze policies and search budgets; locked test; gate; and only then, if authorized,
learner training followed by frozen virgin evaluation.

## Learner authorization

A belief-aware non-neural policy must achieve on 1070001+: order ReT improvement at least 0.01
with paired CI95 lower bound above zero; quantity ReT and attended orders non-inferior;
worst-CSSU fill deterioration no greater than 0.02; unfulfilled rations non-inferior; equal
resource rights; at least 70% favorable tapes; and at least 30% conversion of PI headroom.

Failure keeps 1080001+ closed and terminates Program H. Passing authorizes only a 2x64 PPO using
$(X_t,b_t)$ and one RecurrentPPO raw-history ablation, ten seeds, with frozen learning rate,
normalization, reward, compute, and checkpoints. A learner must beat ABAB and the strongest
non-neural belief policy before persistent/reset is allowed.

## Terminal interpretations

If the optimistic observable bound lies less than 0.01 above ABAB, the allowed conclusion is
that the frozen information contract cannot support a practically material observable advantage
despite PI headroom. If a belief policy passes, the allowed conclusion is that Program G's tested
policy classes were insufficient; Program G remains terminal. Other outcomes are inconclusive
or tested-policy losses, never proof that every POMDP policy fails.

Program H is the final computational extension of the stylized spatial contract. No Programs I,
J, or K; no full-DES spatial port before manuscript submission; and no rescue through a new
metric, horizon, signal, risk, reward, architecture, or tape universe.
