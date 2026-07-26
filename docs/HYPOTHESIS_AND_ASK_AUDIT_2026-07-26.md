# What we can deliver against H1–H4 and Garrido's asks (audit, 2026-07-26)

Every "delivered" row below points at a committed artifact. Nothing is claimed from memory.

## The four hypotheses of v.0 §3.1

| | Hypothesis as written | Status | Evidence |
|---|---|---|---|
| **H1** | *Learning Effect* — hybrid simulation–neural models achieve significantly **shorter recovery times** than static models | **Split: supported for learning-enabled control, not supported for the neural arm; recovery time never measured** | belief-MPC beats the entire static frontier by +0.07..+0.12 ReT and captures 74% of the exact clairvoyant headroom (LCB95 +0.554); neural learners: 0/10 seeds above the static bar. No TTR readout exists — the outcome measured is ReT, not recovery time |
| **H2** | *Adaptation* — improved performance under **successive disruptions** (learning-curve effect) | **Delivered, with a scope correction** | exact dose-response κ0.90 > κ0.75 > iid = 0 on sealed seeds; +0.066 prospective (LCB +0.052); training-time learning curve now exists (`learning_curve_*.json`). The successions are successive **campaigns**, not successive disruption events — the primary two-product experiment is risk-off |
| **H3** | *Volatility Reduction* — reduce performance **variance across disruption intensities** | **Refuted under everything tested** | service-aware screen 0/9 configs (`STOP_SERVICE_AWARE_NO_SAFE_CONVERSION`); risk-sensitivity screen: the optimal posture is **invariant across all 45 risk profiles**, max H_profile_safe 6.9e-05 vs a 0.01 bar. Planner-side mechanisms cannot convert the mean gain into a safe one |
| **H4** | *Path Dependency* — resilience at *t* is positively influenced by **accumulated learning** from prior events | **Delivered — the strongest result in the programme** | the retained/reset pair isolates it exactly: identical machinery, identical horizon, differing only in whether knowledge crosses campaigns → capture +0.743 vs −0.100, exact optima 42/48 vs **0/48**. L_{t−1} is operationalized as the retained posterior |

**Consequence for the paper.** H4 is the spine, H2 supports it, H1 needs rewording (from *neural* to *learning-enabled*, and from *recovery time* to the measured resilience outcome — or a TTR readout must be added), and H3 should be reported as a measured boundary rather than quietly dropped. A refuted hypothesis with an exact ceiling behind it is publishable; a silently deleted one is not.

## v.0 §2's four promised resilience metrics

| Metric | Status |
|---|---|
| Service level stability | measured — it is precisely where the boundary sits (worst-product fill LCB95 breaches in both strata) |
| Adaptation speed | measured as of today — the capture-versus-experience curve |
| Time to recovery | **not measured**; cheap to add from existing tapes (backlog-age and service-loss AUC readouts) |
| Cost volatility | **not measurable in the current model** — the two-product extension carries no validated cost layer, and costing is a documented limitation of the source thesis. Either drop the claim or scope it to resource use (vehicle hours, dispatch slots), which the ledger does record |

## Garrido's meeting asks (2026-07-22)

| Ask | Status |
|---|---|
| DOCX describing the DES model in detail, sentence per line, Baseline 0, NN learning context | **delivered** — `v0_neuralNet-scres_DES_section_updated.docx` §3.3, 8 subsections, 4 figures, native ReT equation |
| Explicit learning metric measured over time (the oracle) | **delivered today** — §3.4 + Table 4 + Figure M5 |
| Three-model comparison (Baseline 0 / RecurrentPPO+MLP / KAN+PPO) | **two of three**: Baseline-0 anchors and both neural arms are graded on the oracle metric; the KAN arm is David's and is pending |
| KAN as the *caballito de batalla* (superior architecture, interpretable) | **contested by our own evidence** — KAN is the best fitter (R² 0.985) but the worst locator (argmax regret 0.037 vs MLP 0.000) and the worst transferer (0.238 vs 0.030). The interpretability argument stands; the predictive-operational superiority argument does not, as a surrogate |
| Reward function vs number of parameters curve | **not done**; feasible with the existing trainers, moderate cost |
| Mean shifts as a resource-efficiency Pareto argument | **not done**; the shift count is not a decision variable in Program Q, but the resource ledger (vehicle hours, dispatch slots, payload) supports the readout on the thesis-native track |
| End-of-horizon hoarding check ("die sick, not healthy") | **designed against and documented, never measured** — fixed production entitlement, clearance tail, terminal ledger, no reward on ending inventory. Cheap to measure from existing tapes |
| Literature-review proof of novelty (first DES + AI for resilience) | **not started** — and Garrido made it a condition for publishing |
| Digital twin on real data (DHL) | future work by his own framing; not now |

## Garrido (2024) Figure 5 — the surrogate ask

**Delivered.** Backprop MLP, KAN and simulation-optimization were all graded against exact answer
keys. Guided search reaches zero regret with 64 of 65,536 evaluations (≈0.1% of exhaustive cost),
which is the "replace costly manual configuration search" claim executed literally, with the
disclosure that the landscape carries large tie plateaus.

## Recommended order of the remaining work

1. **Literature-review novelty proof** — it is a publication precondition, and it is the only
   remaining item with no technical risk.
2. **TTR and hoarding readouts** — both are cheap, both come from tapes we already have, and
   together they close H1's wording problem and Garrido's end-of-horizon concern.
3. **Params-versus-capture curve** — directly serves the KAN discussion and reuses the oracle
   metric as its y-axis.
4. **David's KAN arm** on the same oracle metric, so the three-model comparison is graded on one
   scale rather than three.
