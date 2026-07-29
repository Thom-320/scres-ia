# Face-validation request for Prof. Garrido-Ríos (2026-07-11)

Purpose: obtain Garrido's domain sign-off BEFORE we write the CSSU-A/B refactor
(DRA-1) and before we freeze two fidelity points surfaced by re-reading his Excel
workbooks. Nothing below is implemented yet — these questions gate the code.
Keep the ask tight and respectful; the four blocks are ordered by how much they
block us. Blocks 1–2 gate DRA-1; blocks 3–4 are fidelity confirmations for the paper.

Context for him (one paragraph): we reconstructed the MFSC DES in Python and ran a
pre-registered "decision-rights discovery" program to find whether any dynamically
adjustable decision beats the best fixed policy under disruptions. Five decision
families (strategic buffers, work-shifts, downstream dispatch rates, an emergency
reserve, and the Op9 order-rationing rule) all showed that a single fixed policy is
already near-optimal — no deployable adaptive advantage, measured before training any
RL. The last untested family is **spatial allocation between forward support units**,
which is why we ask about the CSSUs.

---

## Block 1 — Are the two CSSUs a real allocation decision? (gates DRA-1 split)

Figure 6.2 labels operation 11 "Combat Service Support Units **(2)**", but Figure 6.1
and the modelled flow treat the CSSU level as a single aggregated node, which our
code mirrors. We would like to split it into CSSU-A / CSSU-B to test whether
*allocating the daily supply between two forward units* is a decision whose best
choice changes with the situation. Before we build this, please confirm:

1. In the real MFSC, are there genuinely **two (or more) CSSUs** operating in
   parallel, each serving a **distinct zone / set of troops** with its own demand?
2. Is the **total daily downstream dispatch capacity shared** across them (i.e., more
   to one unit means less to the other on a given day)? This "shared scarce capacity"
   is what makes the allocation a real tradeoff rather than two independent pipes.
3. During scarcity, is there any **doctrine** for how stock is prioritised between
   units (e.g., mission criticality, oldest-demand-first, equal service floors)?
4. Is **lateral transfer** between two CSSUs ever done in practice? (We plan to
   exclude it from the first experiment to keep identification clean, but want to
   know if it exists.)

If the honest answer is "the model always treated them as one and there is no real
per-unit allocation decision", we will NOT present CSSU-A/B as a thesis feature — we
would drop or heavily caveat it. We need your call here.

## Block 2 — Is a localized elevated-threat regime plausible? (gates the DRA-1 treatment)

You authorized us to enable/disable and change the frequency/impact of each risk. For
DRA-1 we would use that ONLY to create an identifiable spatial decision: direct
**R22 (attacks on lines-of-communication)** at one lane and **R23 (attacks on forward
logistics-support units)** at one CSSU at a time, at an elevated but bounded rate,
described as a "localized high-intensity threat" scenario. Please confirm:

5. Is a scenario where **R22/R23 concentrate on one route / one forward unit at a
   time** (rather than hitting everything uniformly) **operationally realistic**?
6. Is there an **upper bound** on frequency/impact you would consider still
   defensible for such a scenario (so we pre-register it, not tune it to results)?

We will pair this with a "shuffled-location" placebo (same total attack intensity but
random target) so any adaptive gain must come from *knowing where* the attack is, not
from the attacks themselves.

## Block 3 — Confirm the ReT population and the >1 values (fidelity for the paper)

Re-reading Raw_data1/Raw_data2 (Cf1–Cf20) we found two things we want to state
correctly in the paper:

7. Your workbooks appear to compute mean ReT over the **attended orders only** — the
   visible rows — while lost/pending orders survive only through the ΣBt/ΣUt columns
   (e.g., Cf1 shows 4,241 visible rows but max order index 5,714). Is that right — you
   averaged ReT over the **delivered/visible orders**, not over every order placed?
8. The recovery branch `0.5·(1/RPj)` is not clipped, so a few orders score **above 1**
   (some Cf reach well beyond 1 when RPj is very small). Did your reported ReT keep
   these un-clipped, or did you cap them at 1? We currently reproduce them un-clipped
   and disclose it; we just want to match what you reported.

## Block 4 — Which downstream quantity range is canonical?

9. The text and Figure 6.2 give the downstream batch as **2,400–2,600 rations/day**,
   while Table 6.20 gives **2,000–2,500**. We use 2,400–2,600 as primary and
   Table 6.20 as a sensitivity. Which did your final runs actually use?

---

## What we do with each answer
- **Block 1 "yes, two real units, shared capacity"** → we build the CSSU-A/B refactor
  and run DRA-1 as a disclosed structural extension with your validation.
- **Block 1 "no real per-unit decision"** → we drop CSSU-A/B; the decision-rights
  paper closes with five boundary families and the discovery protocol as the
  contribution (still a strong result).
- **Blocks 2–4** → freeze the regime bound and the two fidelity statements in the
  paper's methods, with your confirmation cited.

(Standing separate question already drafted — the per-order risk-attribution rule in
`docs/GARRIDO_ATTRIBUTION_DECISION_2026-07-10.md` §7 — can be sent in the same message.)
