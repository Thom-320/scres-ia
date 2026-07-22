# How to beat the MPC — the honest analysis and the winning plan (2026-07-22)

North star: a learner (learning-augmented MPC, not pure PPO) that beats the strongest structured
retained MPC on canonical ReT, with the SAME information, safety, and compute budget. This document
maps every path, the decision criteria, and the one path that is both winnable and defensible.

## The barrier, stated exactly

For a learner to beat the MPC, the MPC must be **sub-optimal** in a **decision-relevant** way. Three
independent facts close the easy paths in the stationary two-product environment:

1. **Belief is inference-optimal.** The retained regime belief is `cross_campaign_transition` = the
   exact 2-state Markov posterior under known κ; for a 2-state regime the scalar posterior is a
   sufficient statistic. No learner can form a better belief from the same history.
2. **The decision IS belief-sensitive on the regime carrier** (unlike the parametric carrier): the
   optimal first-2 mix action changes with the retained belief in 119/264 pairs → that is what
   produces the +0.0226 retained value. So ranking reversals already exist; retained knowledge has
   authority. This is necessary but not sufficient for a learner to win.
3. **The MPC's model equals the true model** (by construction, stationary env). Given an optimal
   belief and the true model, an MPC that plans well is optimal. The only remaining gap is CONTROL
   (planning depth).

So in the stationary env, the ONLY door is control sub-optimality (horizon). If the MPC is also
control-optimal, the stationary env is fully exhausted and **no learner can win there** — that is a
certificate, not a failure.

## The three doors, in order of test

### Door 1 — Control sub-optimality (horizon). TESTING NOW.
If the true value depends on lookahead beyond H4 (H6/H8 materially > H4 on the retained arm), the H4
MPC is control-suboptimal and a **learned terminal value (M1)** that amortizes deep lookahead can beat
it. Test: the horizon-convergence probe (H3/H4/H6, burned roots, stratified planner). Preliminary
(n=1): H4 == H6. Decision:
- **H6 − H4 LCB > 0 materially** → Door 1 OPEN → build M1 (learned terminal value); a learner beats
  the H4 MPC by cheap deep lookahead. (Lower probability given the n=1 tie.)
- **H6 ≈ H4** → stationary env control-saturated → Door 1 CLOSED → go to Door 2/3.

### Door 2 — Model intractability under complex risk (THE defensible winning door).
Make the environment's true dynamics **richer than any tractable structured planner can integrate**,
and decision-relevant. The MPC must then APPROXIMATE (truncated horizon, particle belief, no exact
risk-state DP) — not because we designed a bad model, but because **exact planning is intractable**.
A learner trained on the true environment amortizes the intractable lookahead and beats the
approximation. This is the Bertsekas ADP value proposition and it is **defensible against Reviewer
#2**: the claim is "when risk dynamics exceed tractable structured planning, a learned controller
beats the best tractable MPC," NOT "we handicapped the MPC."

Concrete mechanism: a **product-specific risk with a large/continuous, slowly-recovering state** —
e.g. a capacity loss on one product's Op5–Op7 share with stochastic multi-week recovery, or
product-specific rework (R14) cascades. Properties required:
- product-asymmetric (breaks the mix symmetry → strong ranking reversals);
- a risk state too large for exact belief-DP (so the MPC must truncate/approximate);
- persistent across campaigns (so retained risk-type knowledge has cross-campaign value);
- Garrido-native family (R14/R22/R24) extended with product coupling, physically justified.

The learner: the **confidence-gated learning-augmented MPC (M4)** — already ~70% built and unit-tested
(`program_t_confidence_gated_mpc.py`). It keeps the MPC's feasibility/safety/fallback and adds a
learned belief-correction + terminal value where the tractable MPC is weakest. Pure RecurrentPPO is
the ablation, not the contender.

### Door 3 — Belief insufficiency (misspecification). Fallback only.
Continuous-θ / drift / semi-Markov regime where the exact 6-state filter is no longer sufficient.
Defensible ONLY with Garrido-validated physical ranges and the SAME fit data for learner and MPC;
otherwise it reads as designed-to-fail. Lower priority than Door 2.

## Winning plan (gates, criteria, execution)

**G-H (tonight, running): horizon door.** Horizon-convergence probe. Routes Door 1 open/closed.

**G-R (next): ranking-reversal + intractability screen (learner-blind).** Build a minimal
product-specific complex-recovery risk in direct-SimPy. Measure, WITHOUT any learner:
1. H_PI (oracle-of-risk vs best static) ≥ 0.02 — physical opportunity exists;
2. ranking reversals: the optimal mix action varies with the risk-type state in a connected region;
3. tractable-MPC gap: `V(oracle/clairvoyant) − V(best tractable MPC*)` ≥ 0.015 — the approximation
   leaves convertible headroom (this is the intractability signature the learner targets);
4. observable classical conversion: a deployable classical controller captures part of H_PI.
Selection NEVER by learner return. If (1)+(2)+(3)+(4) hold in a connected region → GO to G-L. Else →
exhaustion certificate for this mechanism.

**G-L (if G-R passes): train the learning-augmented MPC vs comparator v2.** Comparator v2 =
"strongest preregistered deployable comparator suite under the specified information, action, safety,
and computational contract" (per Codex): stratified/exact integration, H1–H6 (+H8 only after
DP/caching preflight), convergence-gated against a high-precision reference (p16/p64/p256 sensitivity;
p64 is NOT ground truth), fail-closed-or-abstain. Ladder M0→M4 + PPO ablation, each with independent
holdout, ranking-change + placebo + ledger/service gates. **WIN condition:**
`LCB95[RetainedLearningAugmentedMPC − BestTractableRetainedMPC*] ≥ 0.01`, no adverse cell, exact
resources, no lost-demand increase, worst-product within margin, multi-seed stable, same online
compute budget. Fresh sealed seeds; confirmation one-shot.

**Automatic pre-learner STOP (binding):** if G-R shows no deployable residual over the best tractable
MPC*, NO learner is trained — that is the exhaustion certificate (quantitative ceilings: oracle
headroom, horizon plateau, tractable-MPC gap) + exact Garrido questions (which risks are
product-coupled in reality; is the demand model richer than the 3-θ grid; is downstream fleet
fixed-clock or pay-per-use).

## Why this can actually win (and is not another null)

The stationary-env nulls all share one cause: the MPC's model = truth + tractable planning = optimal.
Door 2 breaks exactly that: it makes exact planning **intractable** while keeping the mechanism
physically real. That is the regime where learned control has a genuine, defensible edge — and it is
the regime Garrido's military supply chain actually lives in (concurrent risks, recovery dynamics,
route/product coupling). If Door 2 also closes, the exhaustion certificate is itself a strong result:
"structured Bayesian control is optimal up to the tractable frontier; learning helps only when risk
dynamics exceed it."
