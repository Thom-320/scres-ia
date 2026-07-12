# Program G — metric triangulation (exploratory) — 2026-07-12

Status: **`METRIC_INDUCED_POLICY_REVERSAL_CONFIRMED`.** Different resilience functionals rank the
Program G policies differently under IDENTICAL physical trajectories. Executed per the external
dictamen (2026-07-12): one trajectory / many lenses, NEW tapes (calibration 1020001+, locked test
1030001+; the G5/bridge universe 1010001+ retired to development), the repo's Cobb-Douglas index
(`ReT_garrido2024` frozen exponents a=.024,b=.026,c=.040,d=.060,n=.1771). **Cobb-Douglas is a
SECONDARY construct index here — it does NOT replace ret_excel and is NOT used to rescue the G5 win.**

## Result (CORRECTED — 168h calendar + canonical cumulative ledger; locked test 1030001+, 200 tapes)

> Two code bugs (found by external review, fixed): (1) the order adapter used 144h "weeks"
> (`(w·6+dow)·24`) instead of `w·168 + dow·24` — with LTj=48 this mis-classified on-time vs late and
> inflated the ABAB-vs-cover gap; (2) a hand-rolled ledger incremented Bt/Ut AFTER scoring instead of
> before. Now ret_order uses the repo's canonical `compute_order_level_ret_excel_formula`, and
> `ret_quantity` (ration-weighted) is added. G0 tests 10/10.

| Metric | ABAB static | cover | retexcel-tree | mpc | **winner** | cover−ABAB CI95 |
|---|---:|---:|---:|---:|---|---|
| ret_order (each order equal — thesis) | 0.500 | 0.480 | 0.485 | 0.482 | **ABAB** | −0.021 [−0.027,−0.015] |
| ret_quantity (each ration equal — mass) | 0.492 | 0.485 | 0.487 | 0.484 | **ABAB** | −0.007 [−0.009,−0.004] |
| worst-CSSU fill (spatial fairness) | 0.831 | 0.657 | 0.782 | 0.705 | **ABAB** | −0.173 [−0.213,−0.137] |
| attended orders (/48) | 42.7 | 39.3 | 42.0 | 40.2 | **ABAB** | — |
| Cobb-Douglas-inspired sigmoid | 0.5061 | 0.5109 | 0.5097 | 0.5089 | **cover** | +0.005 [+0.004,+0.006] |
| Cobb-Douglas-inspired spatial (geo-mean) | 1.118 | 1.166 | 1.119 | 1.150 | **cover** | +0.048 |

## Reading (honest, corrected — the reversal is NOT order-vs-mass)
- **Every thesis-grounded resilience metric favours blind alternation ABAB**: ret_order AND
  **ret_quantity** (so the earlier "cover optimizes mass, ABAB optimizes order-equity" hypothesis is
  **REFUTED** — ABAB wins BOTH the order-weighted and ration-weighted ReT), plus attended orders and
  worst-CSSU fill.
- **Only the Cobb-Douglas-inspired aggregate index favours the concentrating cover policy** (+0.005),
  and it does so while cover destroys spatial fairness (worst-CSSU fill −0.173 — cover starves one
  theatre). The multiplicative aggregate rewards concentration precisely because it contains NO
  worst-node protection — exactly the dictamen's warning about an aggregate CD.
- Therefore the CD preference for the adaptive/concentrating policy is a **documented artifact of a
  non-fairness-preserving aggregate index**, NOT adaptive value, NOT a ret_excel replacement, NOT a
  rescue of G5.

## The paper-facing claim (the actual contribution)
> **Adaptive-control eligibility in a supply-chain simulation is not invariant to the resilience
> functional.** Under identical physical trajectories and resource constraints, every thesis-grounded
> resilience metric — order-weighted ReT, ration-weighted ReT, order completion, and worst-node fill —
> ranks a fixed alternating schedule above the adaptive concentrating policy. Only a multiplicative
> Cobb-Douglas aggregate index (which contains no worst-node protection) reverses the ranking, and it
> does so while the favoured policy sacrifices spatial fairness. Metric choice — not the algorithm —
> decides whether "adaptive control" appears valuable.

This is stronger and more honest than "PPO won 560 rations in a proxy": the metric-dependence IS the
finding, and it warns that a multiplicative aggregate index can flatter a policy that starves a node.
Corrected caveat: the earlier order-vs-mass explanation is refuted — ration-weighted ReT also favours
alternation; the divergence is continuity/fairness metrics vs a non-fairness-preserving aggregate.

## Discipline / disclosed limits
- **Exploratory metric-sensitivity, NOT virgin-confirmatory** (policies were known from G5).
- **Stylized daily order adapter, NOT the full Op1–Op13 DES**; no-risk fill-rate branch (R22 off).
- **φ (spare capacity) and κ̇ (cost) held CONSTANT** in Program G v1.2 (S1 fixed, resources matched) →
  the CD index is driven by ζ (inventory), ε (backorders), τ (fulfil time); the 5-D framing is
  honest only for those three here.
- **CD-spatial used pooled SB inventory** (ζ_i aggregate) so it under-captures spatial unfairness; the
  `worst_cssu_fill` column is the trustworthy spatial-fairness signal and it favours ABAB.
- Per the dictamen: a proper paper-faithful vs spatial Cobb-Douglas study needs its own preregistration,
  Garrido-approved exponents/normalizers, and full-DES variables before any managerial CD claim or a
  CD-reward PPO. Not done here; this is the sensitivity probe only.

`results/program_g/triangulation/verdict.json`. Supersedes nothing; complements the ret_excel
reanalysis. Program G remains a boundary result under ret_excel; CD does not change that.
