# ⚠️ SUPERSEDED — DO NOT EXECUTE

**Retracted 2026-07-15.** This pre-registration justified itself on the claim that the completed risk screen was
**one-factor-at-a-time and therefore interaction-blind**. **That premise is FALSE:** `thesis_design.py`
`RISK_PATTERNS` shows Cf1–Cf20 vary risks **concomitantly** (Cf20 elevates **all four R2 risks at once** and
produced **no door**). The intra-family concomitant region was therefore already covered.

It is further superseded in **governance** by the frozen `war_stress_timing_atlas_v1` (144 concomitant cells,
connected-region rule, seed series **747**, not the 746 proposed here), and its **estimand is wrong**: it targets
constant-posture *tailoring* rather than **`H_timing_safe`** — timing value can exist even when the optimal
constant never changes across regimes.

Retained only as an audit trail of the reasoning error. Any GSA layer must sit **on top of** the atlas, target
`H_timing_safe`, use SALib's validated Morris, and explicitly handle DES stochasticity + input dependence.

---

# Pre-registration — Wartime multi-risk INTERACTION sensitivity (discovery → frozen virgin confirmation)

**Date:** 2026-07-15 · **Science base:** `adbfb8f` · **Status:** **SUPERSEDED / NOT FOR EXECUTION** — its §1b rationale rested on a false premise (see banner).
**Primary endpoint:** `ret_excel_request_snapshot_v2` via `supply_chain.episode_metrics.compute_episode_metrics`
**R3 (black swan): FROZEN — never activated, never scaled, excluded from every phase.**

---

## 1. Why this is required, not a fishing expedition

Two independent grounds, both stated **before** any result:

### 1a. Domain
The MFSC is a **military** supply chain: its purpose is to function in crisis. Garrido himself activated only
the risks relevant to the operations he modelled, and proposed war-level risks as a scenario. Evaluating a
wartime posture is therefore a *defensible, realistic* construct — not a contrivance.

### 1b. Methodological — the decisive reason (the OAT gap)

The completed risk screen (`DEVELOPMENT_NO_DOOR_UNDER_TESTED_FRONTIER`, max `H_profile_safe` 6.93e-05 vs
bar 0.01) used **Garrido's Cf1–Cf20 + one-at-a-time R11–R24**. That is a **one-factor-at-a-time (OAT)**
design, and the sensitivity-analysis literature is unambiguous about its scope:

> **"By design, the OAT approach cannot account for interactions of parameter combinations, since multiple
> parameters are never varied concomitantly."**
> OAT explores only a **"hypercross"** around the nominal point, with **negligible coverage** of the parameter
> volume; and where a response deviates from additivity, OAT conclusions are **strongly dependent on the point
> at which the other inputs are fixed** (Saltelli et al.).

**Therefore the screen's finding — "the optimal posture is invariant across 45 risk profiles" — is valid
ONLY ALONG THE AXES from the baseline. It says nothing about the interior of the risk space.** If a posture
reversal requires (say) **R22 AND R24 elevated simultaneously**, the completed design was **structurally
incapable of detecting it**.

A wartime regime *is* the simultaneous multi-risk region. Probing it is the **methodologically required
complement** to an OAT screen, not a second bite at the apple.

**Honest prior (stated up front):** evidence against is substantial — (i) Track C already tested a *combined*
campaign regime (R22/R23/R24 + R21, non-stationary) and returned 6.5e-05 vs a 5% bar; (ii) at extreme
intensity the system enters **binary collapse** (everything fails equally); (iii) the OAT screen showed
posture ordering changes with **budget**, never with **regime**. We expect a null and pre-commit to reporting it.

## 2. What legitimises searching (and what does not)

Searching the risk space **is legitimate — as discovery**. Scenario discovery (PRIM) inside Robust Decision
Making is published methodology for exactly this: *find the region of the uncertainty space where a policy's
performance changes*. The discipline is not "don't search"; it is **where the search lives**:

- **Discovery runs on DEVELOPMENT tapes and produces a HYPOTHESIS — never a result.**
- **Only a frozen, one-shot confirmation on VIRGIN tapes may support a claim.**
- The discovered scenario must **additionally be domain-defensible** (a wartime regime we would defend to
  Garrido), not an arbitrary corner. **Dual test: statistical AND doctrinal.**

**Explicitly forbidden:** reporting a discovered box as the result; re-discovering after a failed confirmation;
selecting amplitudes after seeing outcomes; scaling R3; any rescue of a failed confirmation.

## 3. Method (NOT a grid search)

### Phase A — DISCOVERY (development tapes; hypothesis-generating only)

Target is the **decision**, not output variance: the indicator is **posture reversal / `H_profile_safe` ≥ 0.01**
at equal resources, i.e. *does the argmax posture change?*

1. **Morris screening (elementary effects)** over the full recurrent-risk space **with concomitant variation**
   (R11–R14, R21–R24 frequency × impact). Cheap; identifies which factors and which **interactions** move the
   decision. Directly repairs the OAT blindness.
2. **Sobol / Saltelli indices** on the decision indicator: report first-order `S_i` **and total-effect `S_Ti`**;
   **`S_Ti − S_i` quantifies exactly the interaction mass the OAT screen could not see.** If `S_Ti ≈ S_i` for
   all factors, the response is additive and the OAT null generalises — *that alone would be a publishable,
   decisive strengthening of the negative.*
3. **PRIM / scenario discovery**: if any reversal exists, extract the **box** (bounds on risk parameters)
   where it occurs, with coverage/density reported.
4. **Doctrinal screen:** the box must be expressible as a wartime regime with a written doctrinal rationale.
   If the box is not domain-defensible → **discovery ends, no confirmation.**

**Phase A output:** either (i) "additive, no reversal region" → the negative is strengthened and *closed on
interactions*; or (ii) a frozen candidate wartime scenario + its doctrinal justification.

### Phase B — CONFIRMATION (VIRGIN tapes, one shot)

Only if Phase A yields a domain-defensible candidate:

- Freeze: the scenario (risk activation/frequency/impact), the constant-posture family, the contract, the
  metric, the guardrails, the seeds — **committed before opening any tape**.
- Open a **fresh virgin seed block (746-series; everything ≤745 is burned/dev)** **exactly once**.
- **Promotion gate (all required):**
  - **ΔReT canonical (`ret_excel_request_snapshot_v2`) ≥ 0.01**, paired **LCB95 > 0**;
  - ≥ pre-registered minimum favorable seeds;
  - **resources equal** (production, charged/reserved capacity);
  - **anti-shed guardrails non-inferior**: worst node/product, quantity-weighted ReT, full-order ReT, lost/
    omitted orders, backlog age, `ret_excel_cvar10`, service-loss AUC;
  - action/trajectory certificate (not a disguised constant).
- **Temporal indices are secondary sensitivities.** A temporal-only gain authorises **neither learner nor
  Paper 2**.

## 4. The wartime scenario (defined a priori, doctrinally)

Frozen **before** Phase A results:
- **Active risks:** R21, R22, R23, R24 (+R11–R14 as the plant-side set) — the recurrent, doctrine-relevant set.
- **R3: NOT activated, NOT scaled** — the black swan stays exactly as Garrido specified.
- **Intensity:** simultaneous elevation of the *recurrent* risks (frequency and impact together) to a level
  justified in writing by wartime doctrine — **not** chosen by outcome. Amplitudes are pre-registered in the
  implementing contract and may not be re-selected afterwards.
- **Rationale requirement:** each activation/amplitude carries a one-line doctrinal justification defensible
  to Garrido (this is the artifact that separates "wartime MFSC" from "the corner where RL wins").

## 5. Pre-committed outcomes

1. **Additive response (`S_Ti ≈ S_i`), no reversal region** → the OAT null **generalises to the interior**;
   the exhaustion certificate is upgraded from "negative along the axes" to "negative including interactions."
   **Strongest outcome for the negative paper.**
2. **Reversal region exists but is not domain-defensible** → reported as a modelling artifact; no confirmation;
   negative stands.
3. **Domain-defensible wartime candidate → confirmation FAILS on virgin tapes** → closed, no rescues; the
   negative is further strengthened (it now survives a purpose-built wartime interaction regime).
4. **Confirmation PASSES all gates** → a legitimate **new same-physics pre-learner contract** ("under a
   doctrinally-justified wartime regime, regime-tailored posture converts material ReT at equal resources").
   This establishes at most **classical H_obs > 0** — it does **not** confirm learned Paper 2, which still
   requires a learner to beat the max over {open-loop frontier, all constants, MPC/DP/interpretable triggers}
   on further fresh tapes. **Paper 3 remains blocked.**
5. **Gain accompanied by worse worst-node / more losses / more resources** → shed-to-win; **rejected** (the
   retracted stylized-atlas precedent).

## 6. Claim boundary

`H_PI_established: Program O only` · `H_obs_established: false` · `learner_authorized: false` ·
`paper2_confirmed: false` · `paper3_authorized: false`. Nothing in this pre-registration alters those until
a Phase-B confirmation passes every gate.

---

**Method sources:** Saltelli et al. on OAT's interaction-blindness and hypercross under-coverage; Morris
elementary effects (screening); Sobol/Saltelli variance-based indices (`S_Ti − S_i` = interaction mass);
PRIM scenario discovery within RAND's Robust Decision Making.
