# Cover letter — *Computers & Industrial Engineering*

**Status:** draft v2, 2026-08-25. Not for dispatch until the checks in
`papers/cie_submission/references_checklist.md` are closed and the one remaining open item in
`papers/paper2/05_conclusion.md` (the pending CODEX CRN companion analysis, §5.2) is computed.
Narrative conforms to the two dated PI decisions of 2026-08-25
(`docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md`,
`docs/ENDPOINT_PRIMARY_DECISION_2026-08-25.md`).

---

**To:** The Editor-in-Chief, *Computers & Industrial Engineering*
`[EDITOR NAME — verify against the current masthead before dispatch; do not guess]`

**Re:** Submission of an original research article —
**"Measuring what there is to learn: a falsification-grade evaluation protocol for learning-based
supply-chain-resilience control, and what it finds in a validated military-food-supply DES"**

Dear Editor,

We submit the manuscript above for consideration as an original research article in *Computers &
Industrial Engineering*. It has not been published elsewhere and is not under consideration by
another journal. All authors have approved the submission.

## Why this manuscript belongs in CIE

*Computers & Industrial Engineering* recently published Guzmán, Andrés & Torres-Polo (2026,
218:112044, `10.1016/j.cie.2026.112044`), whose declared primary contribution is **methodological**:
*"a controlled evaluation protocol with matched seeds, fixed horizons, and 95% confidence intervals
… to enable reproducible comparison across baselines, disruption scenarios, and sector
archetypes."* We read that paper as a statement, made inside this journal, that the evaluation
standard of learning-based supply-chain-resilience research is the binding constraint on what the
subfield can claim.

**We agree, and our manuscript is the next step on exactly that axis.** We extend the matched-seed
+ CI95 protocol with four elements it does not contain, and we show — on a validated
discrete-event simulation of a real military food supply chain — that adding them changes the
conclusion one would otherwise draw:

1. **A measured ceiling, computed before any training.** A perfect-information headroom screen
   establishes how much value is available to capture at all, paired with a mechanism placebo that
   must return *exactly zero*. In our environment the ceiling is 0.15151 (LCB95 0.11562) and the
   placebo returns exactly 0. Without this number, a null result cannot be distinguished from an
   under-powered one — and a positive result cannot be sized.

2. **An exhaustively enumerated comparator frontier.** Rather than a handful of fixed heuristics or
   a No-Op baseline, our comparator is the maximum over all 4⁸ = 65,536 admissible open-loop
   production calendars, *reselected inside every bootstrap resample*. Comparator selection is where
   apparent wins are manufactured; enumerating the frontier removes the degree of freedom entirely.

3. **Equivalence as an estimand, with declared power.** A pre-registered indifference zone
   (δ = 0.01, frozen before any evaluation seed existed) lets us test "no difference" as
   equivalence by two one-sided bounds, with power stated ex ante — instead of inferring sameness
   from a non-significant superiority test, which is the field's standard and incorrect practice.

4. **A per-product equity guardrail that is allowed to fail.** And it does fail. Under an aggregate
   resilience endpoint, the learned policy buys mean fill by unbalancing the weakest product. This
   is a mechanistic failure mode that **no scalarised resilience index in the current literature can
   detect** — not the source thesis's `ReT`, not Garrido's `R`, not Ding's `R̂` (2026), not Kong's
   composite score — because none carries a per-product term that could fail.

## What we find, stated plainly

Applied to three pre-registered operating cells of a two-product, shared-capacity extension of the
Garrido-Ríos (2017) military food supply chain, the protocol yields a decomposition rather than a
headline:

- **Closed-loop control beats the complete enumerated open-loop frontier** in 3/3 cells
  (simultaneous LCB95 +0.043 / +0.037 / +0.066), with adversarial certification that the advantage
  is genuine state feedback: information-replacement placebos, action-trajectory audits, exact
  scheduled-resource equality, and 990 bit-exact physical replays.
- **The recurrent learner shows no premium over the strongest state-rich classical controller**
  (Δ_N = −0.0017 / −0.0027 / −0.0015). The value of closing the "Alzheimer loop" is the value of
  *feedback*, not of function approximation.
- **The aggregate objective buys the mean at the cost of the minimum** — the per-product guardrail
  fails, mechanistically, and we show the same substitution recurs at the level of the endpoint
  itself: two admissible members of the same resilience construct return opposite signs on
  identical tapes with an identical policy pair.

We also report, as a sensitivity table rather than as a caveat, that the sign of a measured
advantage in this model family depends jointly on endpoint, incumbent and tape block — and we
identify which of the three readings survives without assuming block comparability. We publish this
because it is the strongest objection a referee could raise against work of this kind, and because
it generalises: it is a property of scalarised resilience constructs, not of our implementation.

## Fit with the journal's scope and readership

The paper is an applied-engineering contribution with an explicit managerial implication, which we
state as a decision rule rather than a slogan: **do not commission a learning system before
measuring the ceiling.** In our programme, seven prior mechanism families failed the Level 1–2
screen; no learner could have manufactured value in any of them, and the screen costs a small
fraction of a training budget. For practitioners weighing an investment in learned control, the
protocol answers "how much is available, and would a well-specified OR controller already get it?"
before the investment is made. That is a *support-for-decision* result of the kind CIE publishes,
and it is the reason a negative finding here is informative rather than merely disappointing.

The manuscript sits directly among recent CIE work — Guzmán et al. (2026, `10.1016/j.cie.2026.112044`),
Habibi, Chakrabortty & Abbasi (2023, `10.1016/j.cie.2023.109531`), Tian et al. (2024,
`10.1016/j.cie.2023.109829`), Park & Lee (2025, `10.1016/j.cie.2025.111312`) and Sriprateep et al.
(2026, `10.1016/j.cie.2025.111583`) — all of which we cite and engage. It includes a substantive
computational case (the exhaustive frontier, the exact transducer, the replay certification) and
system-level figures, as the journal's readership expects.

## Relation to prior work, including where we are not first

We are explicit about this. Ding et al. (2026, IJPE 297:109995) published multi-agent RL for network
reconfiguration under disruption before us; we cite it, we do not compete with it, and we note
technically that its episodes are single-step, which makes the reported comparison a contextual
bandit result rather than evidence about sequential credit assignment. Kong (2026) reports a
composite resilience score without confidence intervals or a declared seed count. Gijsbrechts et al.
(2022, MSOM) and Kaynov et al. (2024, IJPE) already establish the prior expectation that deep RL
*matches* rather than beats strong heuristics — which is precisely why a well-measured equivalence,
with a declared margin and declared power, is a more defensible contribution than a marginal
premium would have been.

Our novelty is not an architecture. It is the measurement apparatus and what it can falsify.

## Openness, and what we disclose against ourselves

Every experiment is pre-registered as a hash-addressed contract frozen before seeds open; sealed
evaluation blocks open once, with no post-hoc changes to thresholds, comparators or physics; and
terminal verdicts — including the failures — are committed with their raw artifacts. The repository
is public and every reported number is machine-generated from a custodied artifact with its
SHA-256, never transcribed by hand.

In that spirit the manuscript states its own limits without prompting: a preregistered joint tail
gate was **not met**, and we show by instrument audit that a zero-margin "non-inferiority" bound is
arithmetically a superiority test, so we make no deployment-safety claim; the strongest comparator
plans with the correct model and is therefore not deployable, so the honest scope is equivalence to
a model-aware controller; the scope questions posed to the source author are unanswered, so the
risk-scenario envelope is labelled a researcher stress extension rather than a validated operating
range; and the preregistered frozen-policy replication was **executed** on virgin tapes (N = 256
per cell) and is reported exactly as adjudicated: superiority over the open-loop frontier and
equivalence within ±0.01 held in every cell, while the per-product equity guardrail failed in all
three, so the preregistered composite returned a STOP verdict — a compound-gate outcome that we
unpack rather than compress into either "replication succeeded" or "replication failed"
(`docs/PROGRAM_Q_CANONICAL_EVIDENCE_STATUS_2026-08-25.md`). We would rather a referee
read these from us than find them.

## Suggested reviewer expertise

Reviewers competent in (i) discrete-event simulation and simulation-optimisation methodology,
(ii) reinforcement learning for inventory and production control, and (iii) statistical
ranking-and-selection or equivalence testing would be best placed to assess the contribution. We
declare no conflicts of interest and request no exclusions.

We hope the manuscript is of interest to the journal, and we thank you for your consideration.

Sincerely,

`[AUTHOR BLOCK — names, affiliations, corresponding author and ORCIDs to be inserted]`

---

### Assembly notes (delete before dispatch)

- `[EDITOR NAME]` and `[AUTHOR BLOCK]` are the two unresolved fields.
- All numbers above trace to `papers/paper2/results_table.md` (machine-generated) and
  `results/paper_prep/endpoint_block_inversion_v1/`. Do not edit them here; regenerate.
- Title is provisional and differs from the working title in `01_introduction_draft.md`; pick one
  before dispatch and make both files agree.
- The Guzmán 2026 quotation is verbatim from the paper's abstract as recorded in
  `/home/ubuntu/scres-sources/reports/REVISION_CLAUDE_ESTADO_ARTE.md:75`; re-check it against the
  published PDF before dispatch, since a misquoted positioning anchor is worse than none.
- Every DOI in this letter was resolved against api.crossref.org on 2026-08-24 (HTTP 200, title,
  container, volume and year matching as cited).
