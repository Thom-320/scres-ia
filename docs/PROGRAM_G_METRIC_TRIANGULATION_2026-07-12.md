# Program G — metric triangulation (exploratory) — 2026-07-12

Status: **`METRIC_INDUCED_POLICY_REVERSAL_CONFIRMED`.** Different resilience functionals rank the
Program G policies differently under IDENTICAL physical trajectories. Executed per the external
dictamen (2026-07-12): one trajectory / many lenses, NEW tapes (calibration 1020001+, locked test
1030001+; the G5/bridge universe 1010001+ retired to development), the repo's Cobb-Douglas index
(`ReT_garrido2024` frozen exponents a=.024,b=.026,c=.040,d=.060,n=.1771). **Cobb-Douglas is a
SECONDARY construct index here — it does NOT replace ret_excel and is NOT used to rescue the G5 win.**

## Result (locked test 1030001+, 200 tapes, one trajectory per policy per tape)

| Metric (winner ↓) | ABAB static | cover | mpc | service-tree | retexcel-tree | **winner** |
|---|---:|---:|---:|---:|---:|---|
| ret_excel_full_ledger_guardrailed | 0.541 | 0.509 | 0.516 | 0.509 | 0.526 | **ABAB** |
| attended orders (/48) | 42.7 | 39.3 | 40.2 | 39.3 | 42.0 | **ABAB** |
| worst-CSSU fill (spatial fairness) | 0.831 | 0.657 | 0.705 | 0.657 | 0.782 | **ABAB** |
| Cobb-Douglas sigmoid (aggregate) | 0.5082 | 0.5125 | 0.5107 | 0.5125 | 0.5118 | **cover** |
| Cobb-Douglas spatial (geo-mean) | 1.126 | 1.172 | 1.156 | 1.172 | 1.126 | **cover** |
| service-loss (daily order adapter) | 8536 | 12161 | 10913 | 12161 | 7982 | retexcel-tree |

## Reading (honest, non-compensatory)
- **Order-level continuity (ret_excel), order completion, and spatial fairness (worst-CSSU fill) all
  favour blind alternation ABAB.** The concentrating cover policy has the WORST worst-CSSU fill (0.657
  vs 0.831) — it wins by starving one theatre.
- **The Cobb-Douglas resilience index — both aggregate and the spatial geometric-mean variant — favours
  the concentrating cover policy**, exactly as the dictamen predicted: an aggregate multiplicative index
  of inventory/backlog/time can reward concentrating transport on the starving node while temporarily
  abandoning the other, and it does NOT contain spatial fairness between CSSUs.
- Therefore CD "favouring cover" is a **construct-dependent advantage that comes with a documented
  worst-CSSU-fill penalty**, NOT a universal or SCRES win, and NOT a rescue of G5.

## The paper-facing claim (the actual contribution)
> **Adaptive-control eligibility in a supply-chain simulation is not invariant to the resilience
> functional.** A ration-mass loss metric, an order-level continuity metric (`ret_excel`), and a
> multiplicative Cobb-Douglas resilience index induce DIFFERENT policy rankings under identical physical
> trajectories and resource constraints. Concentration (adaptive) wins under mass/Cobb-Douglas
> aggregates; alternation (a fixed schedule) wins under order-level continuity and spatial fairness.

This is a stronger, more honest contribution than "PPO won 560 rations in a proxy": it makes the
metric-dependence itself the finding, and it warns that a multiplicative aggregate index can flatter a
policy that sacrifices a node.

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
