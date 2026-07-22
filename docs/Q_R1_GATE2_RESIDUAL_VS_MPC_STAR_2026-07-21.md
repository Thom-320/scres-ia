# Q-R1 Gate 2 — deployable residual vs the frozen MPC*

**EXPLORATORY_NO_CLAIM. Burned roots only. Canonical ReT unchanged.**

Gate 2 asks whether any deployable headroom survives beyond the strongest structured controller
(MPC*, frozen at Gate 1) — the precondition for authorizing a learner. The answer, on both memory
carriers, is **no deployable premium in the stationary environment**. The route defined by the plan
for this outcome is the product-coupled persistent-risk extension.

## Two carriers, tested separately

### Parametric (theta) carrier — NULL against MPC*
`results/q_r1/gate2_mpc_star_retained_value_v1/result.json`. MPC* runs on both arms; the retained arm
carries the between-campaign joint θ posterior, the reset arm is uniform; the regime is physically
reset to 0.5 on both, so the delta isolates retained parametric knowledge.

| Mode | n | first-2 actions changed | Δ full-cohort ReT (mean, CI95) |
|---|---:|---:|---:|
| persistent κ=0.90 | 24 | 1/24 | **+0.000000** [0, 0] |
| iid (null) | 24 | 0/24 | +0.000000 [0, 0] |

Retained θ knowledge produces **zero** decision change and zero ReT under the strengthened MPC — the
product-mix decision is insensitive to the demand-share parameter. This is the same
belief-insensitivity mechanism Program Q found (RL≈MPC 5 independent ways), now confirmed for the
exact-integration comparator. Resources exact; 0/24 favorable.

### Regime carrier — retained value real, but the headroom above it is CLAIRVOYANT
Oracle-headroom decomposition from the frozen D0 artifact
(`results/q_r1/cold_start_replication_v1/d0_retained_context.json`), κ=0.90, full-cohort endpoint,
264 pairs, history-clustered bootstrap:

| Contrast | mean | LCB95 | favorable/adverse | reading |
|---|---:|---:|---:|---|
| retained − reset | +0.02261 | +0.01911 | 110/9 | retained structured value (= Gate 0) |
| **oracle − retained** | **+0.03155** | **+0.02727** | 147/5 | the learner ceiling over the retained MPC |
| oracle − reset | +0.05416 | +0.05194 | 257/7 | total value of knowing the initial regime |

The retained Bayesian belief is formed by `cross_campaign_transition(belief_c, kappa)` — the **exact**
two-state Markov transition under the known persistence κ — applied to the exact within-campaign
posterior. For a two-state regime with known κ, the scalar posterior is a **sufficient statistic**, so
the retained belief is the optimal deployable estimate of the current initial regime given past
campaigns. The +0.0316 gap to the oracle is therefore the **irreducible clairvoyant uncertainty** (the
oracle knows the true current regime; no deployable policy with the same information can), not
convertible headroom. Consistently, among the tested deployable arms (retained, reset, shuffled,
wrong) **retained is already the best** — shuffled and wrong are negative — so the deployable residual
over the retained controller is ≤ 0.

## Combined verdict

| Source | deployable residual over the best structured controller |
|---|---|
| Parametric carrier vs MPC* (this gate) | 0.000 (24 pairs, both modes) |
| Regime carrier, best tested deployable arm (D0) | ≤ 0 (retained is the best deployable) |
| Joint-carrier D3 clean deployable bound (prior) | +0.0094, LCB +0.0057 (< the 0.015 gate) |
| Regime oracle headroom | +0.0316 — but **clairvoyant**, not deployable |

**Gate 2 outcome: `STOP_NO_DEPLOYABLE_NEURAL_PREMIUM_STATIONARY_ENV`.** The structured Bayesian
controller sits at the deployable frontier on both carriers; the large regime oracle headroom is
clairvoyant. No learner is authorized in this environment — training one would spend PPO to
approximate (at best equal) an already-optimal Bayesian controller, the predicted null of Program Q's
5-way RL≈MPC and the belief-insensitivity mechanism.

## What IS established (Paper 3 spine, structured form)
Retained structured (Bayesian) decision knowledge improves cold-start canonical ReT by **+0.0226
(LCB95 +0.019, full-cohort)** over a physical reset — Garrido (2024)'s cumulative-learning ask,
delivered in structured form, service-clean under natural continuation (Gate 0). No neural net needed
or warranted for it.

## Route (per the plan)
The only authorized new mechanism is the **product-coupled persistent-risk extension**: a
researcher-defined risk (Garrido-approved physical params, selected learner-blind by H_PI / ranking
reversals / observable classical conversion — never learner return) that makes the product-mix
decision belief-SENSITIVE, so retained knowledge can convert to a decision change the stationary env
forbids. This is a NEW environment; the stationary-env neural-premium question is closed.
