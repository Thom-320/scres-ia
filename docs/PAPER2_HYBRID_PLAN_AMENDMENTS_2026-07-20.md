# Paper 2 hybrid plan — binding amendments (2026-07-20)

**Status:** BINDING_PROSPECTIVE, adopted by the PI in writing ("Me encantan las enmiendas.
ejecutalas y guardalas"). Machine-readable companion:
`contracts/paper2_hybrid_plan_amendments_v1.json`. These amendments modify the
learning-augmented event-triggered MPC plan (C&IE routing) **before** any of its remaining
contracts (U1/S1b, U2, U3/U4, U5) are frozen or any scientific seed opens. No terminal verdict
is modified; in particular `STOP_T0_NO_SAFE_RESIDUAL_HEADROOM` (2026-07-20, burned routing
evidence: best reinforced MPC `ret_proxy_scenario_h3_p4`, residual LCB95 −0.059) stands and
routes the campaign to the U1/S1b nonstationary-envelope screen exactly as the plan prescribes.

## Amendment 1 — U5 graded outcome set (`PASS_HYBRID_SAFE_EQUIVALENT`)

The plan's U5 primary gates were superiority-only. Given the program's entire history
(Program Q: TOST equivalence to belief-MPC + worst-product-fill guardrail failure), the most
probable *good* outcome of the hybrid is **safety recovery at classical equivalence** — the
constraint-aware MPC layer restores the worst-product floor the pure learner traded away,
without surpassing classical ReT. Under the unamended plan that outcome maps to STOP,
reconstructing the compound-gate failure mode this program has now paid for twice (O-R
calibration; Q adjudication).

U5 therefore adjudicates to one of **four** terminals, frozen before any U5 seed opens:

1. `PASS_HYBRID_SUPERIOR` — the plan's original superiority gates.
2. `PASS_HYBRID_SAFE_EQUIVALENT` — **all** of:
   - simultaneous CI95[V(Hybrid) − V(BestClassical)] ⊂ [−0.01, +0.01] in every promoted cell
     (TOST, two one-sided bounds);
   - simultaneous LCB95 of the worst-product-fill contrast vs BestClassical ≥ −0.02 in every
     promoted cell (the exact guardrail Q failed);
   - simultaneous LCB95[V(Hybrid) − V(PureRL)] ≥ 0;
   - every Class-B integrity gate of the plan unchanged.
   Licensed claim: *deployable worst-product safety recovered at classical-equivalent
   canonical resilience and amortized online cost.* Superiority claims are forbidden under
   this outcome.
3. `BOUND_HYBRID_ADAPTATION_ONLY` — value over the static frontier and PureRL floor but
   neither superiority nor certified safe-equivalence.
4. `STOP_HYBRID_NO_VALUE`.

## Amendment 2 — compute preflight gate + publication tripwire

**Preflight (binding for every upcoming contract freeze).** Before U1/S1b, U2, U3/U4 or U5 is
frozen, a `compute_preflight.json` must be committed and hash-referenced by the contract,
containing *measured* seconds/episode on the actual target hardware, the worst-case total
episode count implied by the frozen design (no early-exit credit), projected wall-clock and
peak memory, and the capacity cap. PASS requires projected wall-clock ≤ 0.5 × time remaining
to the tripwire and memory within capacity. A contract frozen without a passing preflight is
invalid. (Lessons: S1 died at 48% after ~3 days; the war-stress atlas died
`STOP_COMPUTE_INFEASIBLE` at 86.5M episodes.)

**Tripwire.** If U2 has no timing PASS (LCB95(H_timing) ≥ 0.015) by **2026-08-03**, the
executed Program Q manuscript (branch `paper2-program-q-integration`) is submitted as Paper 2
with its truthful claim, and the hybrid campaign continues *uninterrupted* as the next paper.
Only the publication routing changes; nothing is cancelled. The date is adjustable by the PI
in writing before the U1 freeze; absent adjustment it is binding.

## Amendment 3 — Garrido source attribution

The manuscript's Garrido-alignment matrix (required supplement) must attribute every inherited
element to exactly one source:

| Source | What it grounds |
|---|---|
| Garrido-Ríos **2017 thesis** | Physics/fidelity: Op1–Op13, risk taxonomy, cap-60 list, warm-up, Annex B snapshot semantics, Table 6.12, the 0/47,546 workbook verification. |
| **Garrido et al. 2024** | Closed-loop motivation (DES output → updated decisions); Cobb-Douglas `ReT_garrido2024` — secondary outcome bar only, never a reward. |
| Garrido **proposal/draft** | The superseded v0 scope (Deep Learning framing, predictive-accuracy claim, Track-A contract, DKA/KAN). Cited only as replaced scope. |

No claim of continuity may cite "Garrido" without naming the specific source.
