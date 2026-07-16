# Preregistration — op11 fair-allocation conversion probe (v1)

> **ADDENDUM 2026-07-16 (post G0–G4, auditor domain review): status →
> `HOLD_PENDING_DOMAIN_FACT` + development negative.** Two independent findings:
> (1) *Domain:* the thesis describes NO allocation rule between its two CSSUs (Op11 is PT=0
> receipt; §6.5.5 takes vehicles for granted) and the raw Excel ledgers aggregate orders without
> destination — so this probe is a stylized multi-CSSU EXTENSION, not thesis-native. It cannot be
> promoted (or sold as a thesis door) unless Garrido confirms multi-CSSU competition with
> allocation authority and per-CSSU observability (email question 5).
> (2) *Result (reduced model, burned blocks only, commit 39517a2):* status
> **`OP11_FAIR_CONVERSION_DEVELOPMENT_NO_GO` / `G1_FAILED` / `G5_NOT_OPENED`.** G0 PASS (exact
> reproduction of the fairness-violating anchor, H ≈ 0.01307); ALL state-contingent fair
> candidates NEGATIVE (≈ −0.0053..−0.0057); the best eligible candidate equals the static
> comparator on all 600 burned development tapes (H = 0.0) — strong diagnostic evidence, NOT a
> universal identity, so no "mathematically cannot pass" claim is made. G1 failed (anchor
> tolerance mis-frozen, disclosed); G2/G3 passed trivially because the selected policy is
> equivalent to the static on that set. The development no-go suffices to keep G5 closed —
> **virgin block 4800001–4800200 preserved, never opened.** This is a development no-go, not a
> refutation; no certificate entry claims a thesis-native closure.

**Date frozen:** 2026-07-16 · **Status:** FROZEN BEFORE IMPLEMENTATION AND ANY EXECUTION
**Contract:** `contracts/op11_fair_allocation_conversion_probe_v1.json`
**Contract SHA-256:** `c6863e7c5e7d29d8f1a9bd8cc7c50620e7e2b1e192eed25e0b3516360bb2316e`

## Why this probe exists

Across the entire Paper 2 search, exactly one region ever showed **OOS-stable observable headroom**:
the Program I GP-located region (`results/headroom_gsa/oos_guardrail_check.json`), H_obs
+0.0100..+0.0131 with CI95 > 0 on three independent tape blocks — refused as a lane
(`qualifies_new_lane: false`) because the win is bought by starving one CSSU
(`worst_cssu_fill_delta` ≈ −0.13 vs the −0.02 guardrail).

The frozen decision-right catalog contains the lever that governs precisely that failure:
`op11_allocation_rule` (fifo / mission_priority / **max_min_fill**), status `requires_adapter`.
The probe asks the sharpest cheap question available: **when the controller is given a
fairness-aware allocation rule, does any of that headroom convert into value that survives the
fairness guardrail?** Both answers are useful: a pass authorizes (only) preregistering the full-DES
adapter; a fail closes the last positive-prior door and strengthens the boundary certificate.

## Design in one paragraph

Reduced two-CSSU shared-convoy model (`supply_chain/headroom_sensitivity.py`), environment
untouched, θ\* copied verbatim from the custodied OOS artifact (signal_q 0.532, lead 2, surge_mult
1.946, persistence short, commonality 0.887, r22 0.107). A frozen finite family of fairness-aware
non-anticipative policies (maxmin realized fill; ratio rule with fairness override, δ ∈ {0.02,
0.05, 0.10}; minimum-service alternation) is compared against the **same, never-weakened** static
comparator used by Program I. Estimand `H_obs_fair` with hard eligibility (worst-CSSU-fill,
attended, ret_quantity all ≥ −0.02 vs comparator). Gates in order: G0 reproduction of the known
violating anchor (env identity), G1 Program-G anchor null, G2 information null at signal_q = 0.5
(**must execute and populate a field** — the 022abd0 lesson), G3 permuted-signal placebo, G4
development selection on burned blocks (3000001/4200001/4500001) with an execution-freeze
artifact, G5 one-shot confirmation on **virgin block 4800001–4800200**: LCB95 > 0, point ≥ 0.01,
guardrails non-inferior, ≥ 120/200 favorable tapes (bimodality guard — the Program O OOS lesson).
No rescues. Series 747 untouched.

## Claim boundary

This is a **development probe on the reduced model** (`ret_order`), not the canonical endpoint.
A PASS authorizes exactly one action: preregistering the full-DES op11 adapter contract (liveness
and ledger tests per the catalog invariant, canonical `ret_excel_request_snapshot_v2` endpoint).
Learner, Paper 2 and Paper 3 claims remain blocked behind that separate contract. A FAIL is
recorded as `FAIR_CONVERSION_REFUTED_IN_REDUCED_MODEL` in the boundary certificate.

## Roles

Implementer: PI agent (new policy functions + runner + fixture tests only). Independent verifier:
the concurrent auditor process, who must verify G0–G4 outputs, the execution-freeze artifact, and
the contract SHA **before** the virgin block opens.
