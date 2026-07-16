# GSA layer — handoff spec + audit standard

**Date:** 2026-07-15 · **Implementer:** concurrent audit process (SALib) · **Auditor:** PI agent
**Prototype status:** `7d5e344` (war_risk_gsa.py) is **NOT integrable** — see defects below. Retained only
as (a) an Ishigami validation bench and (b) the source of the **measured constraints** in §1, which are the
salvageable output of the failed build.

**Primary contract stays `war_stress_timing_atlas_v1` (`93e3bb9`)** — 144 concomitant cells, connected-region
rule, seeds **747**. The GSA layer is a **diagnostic on top**, never a replacement.

---

## 1. Measured constraints (from auditing the prototype — these bind the design)

### C1 — CRITICAL: DES stochasticity MANUFACTURES FALSE INTERACTION
Injecting noise into a deterministic benchmark (Ishigami) shows internal noise **deflates first-order and
inflates total-effect** indices:

| noise sd | S1[x2] (true 0.442) | ST[x3] (true 0.244) |
|---|---|---|
| 0.0 | 0.468 | 0.235 |
| 1.5 | 0.402 | **0.347** |
| 3.0 | **0.281** | **0.546** |

Since apparent interaction ≈ `ST − S1`, **simulation noise inflates it** — i.e. a naive Sobol on the
stochastic DES is **biased toward *finding* the wartime-interaction hypothesis**. This is a false positive in
the direction we *want* to be true, which is exactly the direction that must be guarded hardest.

⇒ **A stochastic emulator (or explicit parametric/internal-noise separation) is MANDATORY before any Sobol on
the DES.** Averaging a few seeds and treating the result as a deterministic `f(X)` is invalid.
⇒ **All interaction/additivity thresholds must be calibrated ON THE DES with replications — never on Ishigami.**

### C2 — `ST − S1` is not an "interaction mass"
Measured on Ishigami: `Σᵢ(ST_i − S_i) = 0.455` vs the true single interaction `V₁₃/V = 0.244` → **1.87×
double-counting** (the x1–x3 interaction appears in *both* factors' totals).
⇒ Report `ST_i` as **total involvement of factor i**. Any *interaction* claim must rest on **second-order
indices (S_ij)** or **Shapley effects**, not on `ST − S1`.

### C3 — The additivity verdict needs a calibrated, DES-measured noise floor
On a known-additive function (true interaction = 0) the estimated floor was: **~0.038 @ N=1024**, **~0.031 @
N=4096**, **~0.001–0.012 @ N=16384**. Per C1 the floor will be **larger** on the stochastic DES.
⇒ N must be chosen so the **DES-measured** floor sits well below the decision tolerance, and the API must
**refuse** an additivity verdict below that N.
⇒ Even then, **"additive ⇒ the OAT null generalises" is logically insufficient**: additive ≠ linear, and the
prior design was **discrete**, not a continuous traverse.

### C4 — Greedy PRIM produces spurious structure
On **pure noise** the prototype returned `restricted = ['a','c']` at **density 0.109 vs base rate 0.10** — i.e.
it did *not* manufacture density (my earlier claim that it did was **wrong**), but it **named irrelevant
factors as restricting**, and `min_support` was **inert** (identical 0.20→0.03; the peel loop exits on
no-improvement first).
⇒ PRIM requires **pasting, cross-validation, bagging/stability, multiple-box handling, false-box control, and
coverage/density uncertainty**. A named box without stability analysis is not a finding.

### C5 — Cost forbids naive Sobol
At the N that C3 requires, Sobol ≈ `16384 × (8+2) × 18 postures × 3 seeds ≈ **8.85M** ten-year DES episodes`
(**17.7M** at 6 seeds). ⇒ **Infeasible without an emulator** or a hard dimensional/stratum restriction.

### C6 — Morris must be SALib's
The prototype's hand-rolled trajectory generator used clipping: **208/800 steps NULL (26.0%) + 186/800
off-Δ (23.2%) = 49.2% invalid**, silently skipped. Its tests passed only because Ishigami is tolerant.
⇒ **Use SALib 1.5.2's validated `morris.sample` / `morris.analyze`.** Do not hand-roll.

## 2. Required design

1. **Atlas first:** run the discrete 144-cell atlas (`93e3bb9`) as the primary screen.
2. **Morris (SALib)** *within* each mask/coupling stratum, over **independent continuous** parameters only:
   φ per risk, duration/recovery per risk, R24 magnitude, offsets, warning lead, repair capacity.
3. **Target = `H_timing_safe`** (best event-triggered control at X − best comparator **of that same cell**).
   **Not** constant-posture tailoring — timing value can exist even when the optimal constant never changes.
   Analyse separately: physical ReT, timing headroom, losses/CVaR, clipping/saturation.
4. **Stochastic emulator**: CRN replications; mean model **and** variance/heteroscedasticity model; **report
   metamodel error**; validate OOS.
5. **Sobol only on strata with independent inputs**, with confidence intervals. For dependent/coupled factors
   use **Shapley effects** or stratified analysis.
6. **PRIM** on a **second development block**, with CV/bagging. The box remains a **hypothesis**.
7. **Confirm only the frozen region** on **7470101–7470148**. Once. No rescues.
8. **R3 frozen** throughout. Canonical `ret_excel_request_snapshot_v2` is the sole promotion endpoint;
   temporal indices are secondary. Anti-shed guardrails mandatory.

## 3. Audit standard (the bar I will apply)

| # | Check |
|---|---|
| A1 | Morris via **SALib**; verify zero null/off-Δ steps; report r, levels, trajectory count |
| A2 | Estimand is **`H_timing_safe`** against the **cell's own** comparator — not tailoring |
| A3 | **Stochastic emulator present**; CRN replications; **metamodel error reported**; OOS-validated |
| A4 | Interaction/additivity thresholds **calibrated on the DES** (not Ishigami), with the floor shown |
| A5 | **Raw, unclipped** indices reported (negatives left visible as noise diagnostics) + CIs |
| A6 | Interaction claims backed by **S_ij or Shapley** — never `ST − S1` alone |
| A7 | Inputs **independent within stratum**; otherwise Shapley/stratified, not classical Sobol |
| A8 | PRIM: pasting, CV/bagging, box **stability**, false-box control, coverage/density **uncertainty** |
| A9 | Governance: atlas primary (not replaced); seeds **747**; **R3 frozen**; discovery = hypothesis only |
| A10 | Confirmation opens the frozen region **once**; anti-shed guardrails; canonical ReT primary |
| A11 | **Every stated conclusion traceable to a populated field** — no verdict inferred from a label, a default, or an expectation |

A11 is not boilerplate. Four times in this program a conclusion was asserted from a label or an expectation
rather than read from the data (`information_placebos_pass` default read as a result; `one_at_a_time` label
read as a design; a hand-rolled Morris passing a tolerant benchmark; and a PRIM claim contradicted by its own
printed numbers). Every one was caught by adversarial review, not by the author. **I will apply A11 to this
implementation, and it should be applied to my audit of it.**
