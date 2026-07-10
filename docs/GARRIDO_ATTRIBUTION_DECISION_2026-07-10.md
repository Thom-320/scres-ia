# DECISION: attribution unidentifiability accepted — two-tier fidelity framework (2026-07-10)

**PI decision (Thomas, 2026-07-10): the original Simulink assignment logic is
not recoverable (Garrido no longer has the code). The event→order attribution
is therefore formally an UNIDENTIFIED inverse problem** — established by the
mechanism sprint: multiple mechanisms are compatible with the workbook
outputs (superset-violation discriminator shows non-window marks for
R12/R13/R21 without singling out genealogy; the complete R13 FIFO gate failed
jointly on odd CFs, `docs/GARRIDO_HYBRID_ATTRIBUTION_GATE_2026-07-10.md`).
Continuing to fit attribution mechanisms would be optimizing an unidentified
model — a stopping problem, not a science problem. This document removes the
blocker under honest labeling.

## The two-tier fidelity framework

**Tier 1 — identifiable physical fidelity (pass/fail, ALL PASS):**
- Mass conservation (raw + rations residual 0 on all 20 CFs);
- Placed-order counts match the workbook placement ledger (max j) to ±0.3%;
- Attended/lost/pending ledger structure reproduced (workbook view);
- Zero on-time artifact (no CTj ≤ 48 h, matching every workbook);
- Causal liveness for all 9 risks (R12/R13 physically delay material);
- Op1→Op2 contract coupling; route-aware replenishment; daily-freight Op9;
- Throughput within ~4% of workbook attended quantity per year.

**Tier 2 — distributional match under a DISCLOSED attribution proxy
(measured dispersion, reported not gated):**
- Attribution proxy: overlap attribution + fixed 168 h R24 stress window —
  the best-scoring simple mechanism on odd CFs (ret_gap 0.026 calibration /
  0.0355 validation; risk-share gap 0.066 validation), mechanism-plausible
  for R24/R14 (near-zero superset violations) and explicitly a PROXY for
  R12/R13/R21 (whose true assignment rule is unidentified).
- Measured validation gaps (even CFs, one run, frozen before measurement):
  ReT MAE 0.0355; risk-share 0.0656; CT/RP quantile ratios per
  `outputs/audits/garrido_reference_v2_gate_r3/`.
- Precedent: Garrido's OWN model validation accepted annual production gaps
  of −21.6%..+14.1% (Table 6.10, RMSE 87,918). Our self-imposed 0.02/0.05
  bars demanded distributional identity that the lost internals make
  unattainable; Tier 2 reports the achieved dispersion instead.

## What is decided

1. **`garrido_proxy_v1` is frozen** (`supply_chain/data/garrido_proxy_v1_freeze_2026-07-10.json`)
   as the sanctioned endogenous research environment: Tier-1 physics + the
   disclosed Tier-2 attribution proxy. **RL training is permitted on this
   environment** under its label.
2. **The label `garrido_reference_v2` (and any "thesis-faithful" claim for
   the endogenous environment) remains permanently blocked** — it would
   assert an attribution fidelity that is unidentifiable. The preflight
   enforces this.
3. **`excel_risk_tape` / forensic replay remains the exact-fidelity lane**
   for any claim requiring Garrido's actual per-order attribution (it copies
   his tapes; no mechanism assumption).
4. Paper language (binding): "thesis-grounded reconstruction with audited
   physical causality (Tier 1) and a disclosed attribution proxy; order-level
   distributional comparison reported with measured gaps (Tier 2)". Never
   "faithful replication of the Simulink internals"; never silent use of the
   proxy as ground truth for attribution-sensitive claims.
5. Attribution-sensitive analyses (per-risk RP decompositions, risk-branch
   causal claims) must either use the forensic tape lane or carry the proxy
   disclosure inline.
6. The mechanism-sprint corpus (discriminator, counterfactual event-deletion
   auditor, FIFO gate, falsification tests) is retained as a methodology
   contribution: how to audit a legacy-model reconstruction and DETECT
   unidentifiability.
7. **Non-blocking question to Garrido** (email, Spanish, to send):
   > Profesor Garrido: en las hojas Raw_data, ¿qué representa exactamente que
   > una fila de pedido tenga valor positivo en una columna de riesgo
   > (p. ej. R13 o R21_2)? ¿Se marca el pedido cuando su intervalo temporal
   > coincide con el evento, cuando consumió material afectado por el evento,
   > o mediante otra regla del modelo Simulink? Cualquier recuerdo parcial
   > nos sirve para etiquetar correctamente nuestra reconstrucción.

## Stop rules that REMAIN in force

- No more attribution-mechanism fitting against odd CFs (unidentified).
- Even CFs are exhausted for this question (one r3 measurement stands).
- No new windows, propagation heuristics, or per-risk special cases.
- Track C / adaptive lanes keep their own frozen physics (`adaptive_research_v1`).
