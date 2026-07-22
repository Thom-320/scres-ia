# How to beat the MPC — the honest analysis and the winning plan (2026-07-22)

> **SUPERSEDED IN PART — CORRECTED ROUTE BINDING (2026-07-22, Codex review accepted).**
> A second Codex review found real overstatements in the strategy below. All accepted:
> 1. **"MPC model==truth + tractable planning==optimal" is FALSE.** The full-DES MPC is a
>    receding-horizon, ret_proxy-based, demand-approximate, scenario-family-restricted controller,
>    NOT a certified full-DES POMDP optimum. Belief sufficiency for inference ≠ control optimality.
> 2. **The horizon sweep does not certify exhaustion** (p16 only, single θ, binary carrier, burned,
>    no p64/p256; partially redundant with the existing calibration_v3). EXPLORATORY_NO_CLAIM;
>    authorizes nothing. (My H3=H4=H6 paired result stands only as a descriptive prior.)
> 3. **Door 2 was framed backwards.** "Make the env intractable so the learner wins" is attackable
>    environment-engineering. The defensible question is: *do physically-plausible Garrido-recognized
>    risks create persistent, product-specific, decision-relevant uncertainty that tractable
>    structured controllers do not fully convert?* Physics first → headroom → learner. Never design
>    complexity to induce a neural win.
> 4. **The thresholds H_PI≥0.02 and residual≥0.015 are arbitrary** (same class as the retired 0.02).
>    Replace with the scientific SESOI +0.01 + a prospective power audit (uncertainty + experiment
>    cost → derived diagnostic gate), never a heuristic multiplier.
> 5. **"M4 ~70% built" is overstated.** `program_t_confidence_gated_mpc.py` is an interface + tested
>    gate; `program_t_learning_components.py` is a GRU + a loss. Still MISSING: causal dataset,
>    targets, trainer, split/calibration, real MPC adapters, terminal-value integration,
>    belief-correction integration, checkpointing, end-to-end eval, compute matching. It is a useful
>    scaffold, not a 70%-complete system.
>
> **Corrected north star (binding):** demonstrate that retained decision knowledge *causally* improves
> cold-start resilience after a full physical reset; and, AS A STRETCH, that a learning-augmented MPC
> converts a residual the best retained *tractable* controller does not capture. NOT merely "beat the
> MPC."
>
> **Binding sequence (Codex's corrected route; supersedes the door ordering below):**
> 1. Finish + adjudicate the canonical Q-R1 (VPS `ovh-agent-lab`, commit 983c197) — SOLE Q-R1 source.
> 2. Build comparator challenge v2 AFTER adjudication, without mixing branches.
> 3. Freeze the best *tested deployable* comparator — never call it "optimal MPC".
> 4. Measure the residual with full-cohort ReT, policy-independent CRN, full constraints, power audit
>    at SESOI +0.01.
> 5. Residual is horizon-type → test M1 (terminal value). 6. Residual is inference-type → test M2
>    (belief correction). 7. No stationary residual → learner-blind sensitivity of R11/R14/R21/R22/R24.
> 8. Only if those risks are decision-inert → product-coupled extension **validated by Garrido**,
>    physics + grid frozen BEFORE any RL. 9. M3/M4 only if M1/M2 change rankings and pass
>    holdout/placebos. 10. Final confirmation vs the best retained structured controller, same rights.
>
> Nothing past step 1 executes until the canonical Q-R1 is adjudicated. The sections below are the
> earlier (partly-flawed) analysis, kept for transparency; where they conflict with this block, THIS
> block wins.

---


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

### Door 2 — Model intractability under complex risk (candidate door; REFINED per Codex 2026-07-22).
Make the environment's true dynamics **richer than any tractable structured planner can integrate**,
and decision-relevant. The MPC must then APPROXIMATE (truncated horizon, particle belief, no exact
risk-state DP) — not by design, but because **exact planning is intractable**.

**CRITICAL REFINEMENT (accepted from Codex): intractability alone is NOT a neural win.** Under
intractability the fair comparator is NOT the exact MPC — it is the **best CLASSICAL structured
approximation under the same compute budget** (rollout / approximate DP / a hand-designed or
classical structured terminal value). A learned (neural) value function must beat THAT, not just the
exact MPC. So the win hierarchy has two distinct levels:
- **Level 1 (amortization, EXPECTED):** learning-augmented MPC beats the tractable exact-ish MPC.
  This is not a neural premium — any amortization of intractable lookahead achieves it.
- **Level 2 (the REAL Δ_N):** the neural value/belief beats the **best classical structured ADP**
  under the same budget. Only this is a defensible neural premium. My earlier win condition
  (`learner − MPC*`) tested Level 1; the real bar is `neural − best_classical_structured_ADP`.

**Gating (binding, per Codex):** Door 2 is defensible ONLY if, FIRST, the EXISTING Garrido risks
R11/R14/R21/R22/R24 are shown **decision-inert** (a learner-blind screen: the optimal mix action does
not vary with them — the prior product-BLIND screen found invariant posture, but product-coupling is
untested), AND Garrido provides **written physical validation** of any product-coupled risk (amplitudes,
frequencies, timing, persistence). Complexity added without those two is not defensible.

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

**G-H (tonight): horizon door.** Horizon-convergence probe (burned, `/tmp/qr1_horizon`). **Honest
scope (per Codex): this is a burned diagnostic, NOT a saturation certificate** — it uses p16 only
(not p16/p64/p256 convergence; p16 is not ground truth), a single θ=(0.90,0.90), and the binary
regime carrier. It INDICATES horizon effect at those settings, it does not certify control-optimality.
Result so far: the **paired `retained_H4 − retained_H3` contrast is exactly 0.000000 across all 176
campaigns** (identical retained decisions at H3 and H4) — a strong saturation signal at p16/this-cell;
H6−H4 pending. A full certificate would add p64/p256 convergence and multiple cells. Routes Door 1
open/closed as a diagnostic prior, not a proof.

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
holdout, ranking-change + placebo + ledger/service gates. **WIN condition (two levels, per Codex):**
must include a **classical structured ADP** baseline (non-neural learned/rollout terminal value) in
the comparator suite. Level 1 `LCB95[learning-augmented MPC − best tractable MPC*] ≥ 0.01` shows
amortization (necessary, not sufficient). **The publishable neural premium is Level 2:**
`LCB95[neural-value learner − best classical structured ADP] ≥ 0.01` under the SAME compute budget.
Both with no adverse cell, exact resources, no lost-demand increase, worst-product within margin,
multi-seed stable. Fresh sealed seeds; confirmation one-shot. If Level 1 holds but Level 2 fails →
"amortization helps, neural value does not" (still a result, not a neural premium).

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
