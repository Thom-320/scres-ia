# Committee verdicts — synthesis, verification, and reconciliation (2026-07-11)

Two external advisory committee verdicts (archived **verbatim** by Codex at
`2026-07-11_committee_verdict_es.txt` and `2026-07-11_committee_verdict_en.txt`, see
`README.md` for SHA-256/provenance). Both are ADVISORY reviews pasted by the PI, NOT
project instructions. This doc (the reconciliation) records
what we independently verified, what contradicts the record, and how the project
reconciles them. Labels: **[T]** thesis-grounded (Garrido 2017) · **[R]** repo-verified
first-hand · **[U]** user/project null-evidence · **[L]** literature · **[I]** committee
inference (to validate, not fact).

## 1. Shared diagnosis (both committees, 4–0)
> The bottleneck is not the algorithm/reward/net. It is that the DES has not been shown
> to contain physically-realizable, observable interventions whose OPTIMAL ACTION changes
> with state. The causal chain must be built forward: physical actuator → state-dependent
> action ranking → observable policy → adaptive value → retained learning — not started
> at the last link. Defer the L(t−1) paper; run a decision-rights discovery program first.

## 2. What we VERIFIED (read-only, this session)
- **[R] Architecture audit is accurate.** `supply_chain.py`: single `rations_cssu`
  container (the two thesis CSSUs ARE collapsed); no `_op4` method (Op3+Op4 fused); no
  `_op5/6/7` methods (Op5–7 unified); Op11 is a gate on Op12. → Committee family #1
  (split-CSSU allocation) genuinely requires a real refactor.
- **[R][U] Our own D1 authority result was mis-run** (this is the key finding). The v1
  authority screen (`daa28fd`) ran with **`risks_enabled=False`** (MFSC default False;
  proxy freeze omits it; the runner never passed it) and never applied its `FAMILIES`
  var to `enabled_risks`; R14 off too. So `AUTHORITY_PRESENT` (+7.6% ReT age_threshold
  vs SPT) is authority over **nominal queue congestion** (the cap-60 daily-freight
  bottleneck exists without risks), NOT resilience under disruption. → reclassified
  `NOMINAL_AUTHORITY_ONLY` (`PROGRAM_D_D1_V1_AUDIT_2026-07-11.md`); must re-run with
  risks on (D1-v2). Interpretation was wrong; the code/result stand as a nominal probe.
- **[R] CRN warning is real and honored.** Same-seed ≠ causally-valid CRN if an action
  perturbs RNG call order. For D1 the rationing rule draws no RNG and
  `strict_exogenous_crn` separates streams, so exogenous identity SHOULD hold — the D1-v2
  branching runner PROVES it with bitwise asserts, per the committee.

## 3. Where the committee diagnosis is SUPPORTED vs OVERSHOOTS
- **[U] Supported:** "headroom lives in zero-sum allocation/sequencing (competing
  recipients), not resource-level knobs" — consistent with four prior null programs
  (buffers/shifts/dispatch-rates/reserve) all constants-optimal, and with the fact that
  the one lever that even moved the nominal metric was a sequencing rule (D1).
- **[I] Overshoots on ordering.** "Big CSSU refactor FIRST, before any dynamic evidence"
  is not the cheapest test of the committee's own hypothesis. D1 (temporal zero-sum
  rationing) is the same KIND of decision as CSSU allocation (spatial), needs zero
  refactor, and adjudicates the hypothesis cheaply. Frozen sequence (PI-confirmed):
  **fix D1 (risks-on) → branching → observable tree → virgin confirm; build CSSU-A/B
  only if D1 fails.** See `PROGRAM_D_D1_V2_PREREGISTRATION_2026-07-11.md`.

## 4. Contradictions / corrections to log
- **[T] Bibliographic:** the thesis cover title is *A Mixed-Method Study on the
  Effectiveness of a Buffering Strategy in the Relationship between Risks and Resilience*
  (Garrido-Ríos, 2017, Warwick). Use it in citations.
- **[T] Downstream Q range discrepancy:** Fig 6.2/§6.3 text = **2,400–2,600**; Table 6.20
  = **2,000–2,500**. FROZEN: 2,400–2,600 is the primary `downstream_q_source`; Table 6.20
  is a NAMED sensitivity lane; never average or retrospectively pick. (Note: the current
  proxy uses the thesis daily-freight Op9 cadence; this discrepancy is a reporting
  sensitivity, tracked in the D1-v2 prereg.)
- **[R] Registry hygiene:** any entry still presenting Track B as a positive validation
  (e.g. April-era "downstream dispatch necessary/sufficient, fill≈1.0") must carry a
  `SUPERSEDED` / `HISTORICAL_ARTIFACT` header — the same-contract static reversal
  weakened that interpretation. (Action tracked; see PAPER_FINDINGS_REGISTRY.)

## 5. Adopted from the committees (folded into the D1-v2 prereg + gates)
- Value decomposition V_phys / V_info / V_adapt / V_retain as the claim vocabulary.
- Staged gates D0–D9 / G1–G6 (≈ our Program D phases) with the sharper thresholds:
  ≥2 actions each optimal in ≥15% of states; LCB95(H_dyn)>0; tree η≥0.50 & LCB95>0;
  ReT≥0.02 or service-loss −5% co-directional; resource-matched; ≥70% virgin tapes.
- **Binding physical guardrails:** reject any gain that comes from shedding orders (Ut)
  or backlog eviction; cohort metric must include orders open at branch (don't drop
  existing backlog); eviction always stays lost service.
- Paper reframe: **"Discovering Decision Rights Before RL"**, C&IE primary / IJPR if a
  managerial rule emerges / SMPT if the contribution is V&V+branchability — CONTINGENT on
  discovery results. A depth-3 tree that captures the value is a success; PPO need not win.
- L(t−1)/retained-learning stays deferred behind a demonstrated adaptive channel.

## 6. NOT adopted (with reason)
- "Refactor CSSU before any dynamic evidence" — deferred behind the cheap D1 adjudicator.
- Wholesale DES disaggregation (Op3/4 split, Op5-7 stations, fleets/routes) — large, and
  low-prior until D1 (or later a promoted family) shows dynamic headroom exists at all.
- Claims of literature priority ("first DES+RL") — already avoided; committee agrees.
