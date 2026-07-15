# Program O — terminal-outcome certificate (PI synthesis)

**Date:** 2026-07-15
**Outcome:** `BOUNDARY_CERTIFICATE` (Branch B). The full-DES perfect-information ceiling
is real and material; **no finite observable controller — minimal or state-rich —
establishes state-dependent adaptive value beyond a regime-belief heuristic**, and the
one residual signal is resource-contested. Sealed validation tapes `7420049–7420096`
were never opened (`validation_seed_accessed: False`).

This is the PI-level synthesis. It references the committed operational artifacts.

---

## 1. Established quantitative ceiling (H_PI) — custody-verified

| Quantity | Value |
|---|---|
| safe H_PI (holdout, least-favorable cell) | **0.15151** |
| simultaneous safe LCB95 | **0.11562** |
| exact fungible-null H_PI (causal control) | **0.0** |
| direct full-DES parity | 25,177 episodes |
| resource match | Δgross_production = 0; charged/reserved resources equal |

Commit `6ad6f10`; verdict `98ce2ce`; `result_sha256 f5f2da8d…`; evidence
`results/program_o/full_des_hpi_translation_v1/validation_custody_verdict_v1.json`
(`PASS…CUSTODY_VERIFIED`). First of 18 screened families whose clairvoyant ceiling is
material, survives Op9–Op12 buffering, and has an exact causal null. Certified **at
conserved throughput** (the oracle reallocates the C/H split, not the total).

## 2. Observable-headroom gates — both families STOP

### 2a. Label-only HMM (refuted earlier)
Strong development signal (Δ=0.0625 primary) but excluded: oracle-true (ρ,share) changed
0/192 trajectories (share-magnitude-invariant sign rule), and the gain leaned on transport.

### 2b. State-rich classical family — `STOP_RESOURCE_OR_GUARDRAIL_CONFOUND`
Run `program-o-state-rich-fit-v1-20260715`, commit `041dcef`, result
`sha256 d67ac97a…` (transfer-verified), producer exit 0, burned fit tapes `7420001–48`
only, sealed validation untouched. `stability.passing_cells: []`.

Ten enumerated controllers (base-stock, max-pressure, min-cost-flow, belief-MPC, belief-DP).
Per cell the best controller (belief-DP) captures **material** ReT vs the full open-loop
frontier: 0.038 / 0.068 / 0.074 / 0.102 across (ρ75s75, ρ75s90, ρ90s75, ρ90s90) — **η ≈ 0.25–0.7**.
All configurations: `metric_guardrails_pass: True`, `reserved_capacity_equal: True`.

**Every configuration fails on two independent axes:**

1. **State adds nothing (resource-independent, load-bearing).** `information_placebos_pass:
   False` for **all** controllers — a stale-state, masked (no operational state), swapped, or
   cross-tape policy matches the real-state controller. The rich operational state
   (inventory / backlog / WIP) is **not load-bearing**; the value traces to the
   **regime belief** (label-history-derived) already available to the refuted label-only lane.
   belief-MPC is genuinely state-responsive (state-perturbation counterfactuals 287/287 pass)
   yet still fails the placebos; belief-DP / min-cost-flow additionally miss the week-0
   channel-swap counterfactual (a no-initial-state artifact).

2. **Actual-transport frontier (resource-contested, NOT load-bearing).** `strict_actual_use_pass:
   False` and the matched-resource frontier is empty (`eligible_calendar_count: 0`) for all —
   every controller out-transports the entire 65,536-calendar frontier. Mechanism: at **equal
   production** (Δ=0) and **equal reserved fleet** (5,376 charged-hours, Δ=0, but only ~2,280
   used → **42 % utilization**), the belief policy delivers **+~22,000 otherwise-stranded
   rations** (worst-product-fill +0.26, omitted −22,370) by producing the right non-fungible
   mix and filling **idle, already-charged** freight. Whether that counts against the policy
   depends on the freight model (see §4); it is **not** the load-bearing failure — axis 1 stands
   regardless.

## 3. Terminal finding

Perfect information is worth ≈0.15 ReT. A **regime-belief heuristic** captures ≈0.04–0.10 of
it via non-fungible mix-matching, but **rich operational-state observation adds nothing beyond
that belief** (every state-rich controller fails the information placebos). Therefore **no
learned or classical state-rich adaptive controller is warranted**: a learner with full
operational-state observation has no headroom to exploit beyond a simple belief rule that a
minimal policy already realizes. This is the sharpest "when not to train" instance in the
search — the value of observability **saturates at the belief** — and it extends the Program
D–K null pattern and the Program K cost-efficiency-not-resilience finding. The `STOP` is
**overdetermined**: it holds under either resource-fairness convention.

## 4. Exact open questions (the only reopeners)

1. **Freight economics (resource convention).** Is the downstream fleet **fixed-clock reserved**
   (charged whether loaded or empty — the contract's stated model, under which the +425 actual
   hours within 5,376 reserved are free) or **pay-per-use**? Reserved → the belief heuristic's
   service gain (worst-product-fill +0.26) is resource-honest and a narrower **interpretable**
   positive could stand; pay-per-use → the resource `STOP` is also correct. **Either way the
   state-rich boundary (§3, axis 1) is unchanged.** This refines Q2/Q13.
2. **Construct (Garrido Q13).** Whether ≥2 mutually non-substitutable ration classes share the
   Op5–Op7 bottleneck (validates MFSC-representativeness) or the mix is fungible/deterministic
   (collapses to the exact null). Non-blocking; the internal boundary stands regardless.

## 5. Disposition

- **Paper 2 (positive learned adaptive control): not warranted** on Program O — state-rich
  observation adds no value over a belief heuristic. No learner authorized. Paper 3 not reached.
- **Boundary/exhaustion certificate: delivered**, with quantitative ceilings (H_PI 0.152;
  belief-captured ≈0.04–0.10; state-rich increment ≈0) and the two exact questions above.
- **Residual (governance decision):** if freight is reserved-fixed, a *narrower* interpretable
  paper — "a regime-belief mix heuristic materially improves worst-product fill in a
  non-fungible military supply chain at equal reserved resources" — is defensible, but it is a
  simple heuristic result, not learned adaptive control, and needs the §4.1 fact.

Custody: fit result `d67ac97a` transfer-verified; H_PI verdict `f5f2da8d`; sealed tapes intact.
