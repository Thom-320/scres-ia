# Appendix A. Reconciliation with the v0 hypotheses

Every row below is generated from `results/v0_adjudication_matrix/result.json`, whose falsifier `f1`
requires each figure to resolve inside the artifact its own row names. Nothing here is transcribed.

## A.1 The two research questions

| v0 question | Verdict | Answered on | Not answered |
|---|---|---|---|
| Can the model be operationalised dynamically? | **Partially** | an outer loop over restarted DES runs: search state persists, the physical system does not | dynamic operationalization *inside* an episode. The DES is reset between runs and no arm observes or acts during one |
| Does the neural component improve predictive accuracy? | **Not answered** | — | no held-out predictive validation of a resilience index exists. Fit quality was measured only inside search contracts, where it is an input to a search comparison, not a predictive claim |

## A.2 The formal expression

The v0 draft writes `R_t = f(S_t, D_t, L_{t−1})`. `L` survives as an endogenous state variable and
the contribution is real, but it is a **factorised bandit statistic on the run index `k`**, not on
physical time:

```
Y_k = DES(x_k, c_k, ω_k)          one execution of the DES
L_k = U(L_{k−1}, x_k, Y_k)        search state updates with what was observed
x_{k+1} = π(L_k, c_{k+1})         the next configuration is chosen with what was retained
```

The physical state resets between runs; only `L_k` persists. The subscript `t` in the original
promises within-episode adaptive control, which nothing in this work tests.

## A.3 The four hypotheses

| | v0 wording | Verdict | Estimand actually measured |
|---|---|---|---|
| **H1** | a hybrid model recovers faster than a static simulation | **Supported on a redefined endpoint, development only** | `restricted_ttr = min(TTR, τ)`, τ = 1344 h, paired placebo, isolated shocks: **+125.99 h [+98.35, +154.54]**, 960 cells |
| **H2** | performance improves over successive disruptions | **Supported on its own estimand, development only** | OLS slope of (reset − memory) AUC against the context ordinal 1..6: **+0.04220 [+0.03466, +0.04992]**, n = 120, with a null control at −0.00509 [−0.01557, +0.00566] that crosses zero |
| **H3** | learning reduces variance across disruption intensities | **Not supported — live estimand, wrong sign** | **−1.109e15 [−3.649e15, +1.293e15]**, *p* = 0.821, 360 cells |
| **H4** | resilience at *t* depends on accumulated learning | **Supported for search only, development only** | retained state lowers AUC regret in **6 of 6** matched families, all six surviving max-*T* simultaneous inference (critical value 2.591) and Holm |

**Three of four are supported, and the qualifiers are not optional.** H1 holds on an endpoint
redefined for this lane, and a companion surface gate on the same recovery question returned
`STOP_NO_RECOVERY_LEARNING_HEADROOM`, so a *learner* exploiting that endpoint is not established;
the H1 contrast is between postures, and identifies no neural causality. H2's ordinal is a context
sequence in an outer loop, not successive disruptions within a run — and that artifact's own
estimand note records that a large but *flat* advantage would support H4 rather than H2. H4 measures
the cost of **finding** a good configuration, not the resilience **delivered** at time *t*; on the
simple regret of the recommendation actually deployed, only one of six families keeps a simultaneous
lower bound above zero.

**None of the four is confirmatory.** All four artifacts sit on already-open blocks and three say so
in their own `scope` field. H3 is no longer "no estimand available" — it is "no effect", which is
the stronger negative, and H3′ (variance of *search* cost) is a different construct that does not
rescue it.

The formal expression is therefore a **theoretical proposal for an outer-loop state variable**, not
an empirically established law of the supply chain. The manuscript does not use H1–H4 as a passing
hypothesis family.

**Withdrawn figure.** The v0 draft cites `+7.90 runs` for H4. That number is real but comes from the
**oracle-normaliser** panel and was quoted without naming its normaliser; it is forbidden in the
claim lock and does not appear in this manuscript.
