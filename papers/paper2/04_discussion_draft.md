# Paper 2 — Discussion (draft v1, 2026-07-18)

## 4.1 What the ladder shows

The four-level decomposition localizes adaptive value precisely. The mechanism (non-fungible
products sharing capacity) creates a large clairvoyant opportunity that vanishes exactly under
fungibility; observable belief-state control converts most of what is convertible; a generic
recurrent learner re-acquires that conversion from experience alone; and the learner collects no
premium over the structured controller. Each level is falsifiable separately, and in this study
the levels disagree in exactly the way that compound success criteria hide: Level 3 is a clear
positive while Level 4 is a clear null. A single PASS/STOP boolean over their conjunction would
have reported — and briefly did report — "failure" for a system that had just produced the
program's first genuine learned adaptation. The prospective replication repeated the pattern a
second time: both scientific endpoints replicated cleanly, yet the contract's compound verdict
is STOP because a single distributional guardrail (worst-product fill versus the classical
controller) crossed its frozen margin. Component-wise reporting is what makes both compound
verdicts interpretable rather than misleading.

## 4.2 Why "no neural premium" is the informative outcome

The environment is small and highly structured: four actions, eight decisions, a 21-feature
observation that already includes an HMM belief summary, and a belief-MPC comparator that plans
explicitly over short action sequences with the correct decision model. Under those conditions
the network has no hidden variable to discover; the best it can do is approximate the structured
solution — which is what the data show, to within a few thousandths of ReT. This aligns with the
operations-management literature: generic deep RL rarely dominates strong structure-exploiting
policies in well-modeled problems, while learned policies that *imitate or amortize* model-based
controllers are valued for cost, speed, and deployability rather than outcome superiority. The
correct reading is not "RL failed" but "structured decision theory already sits at the observable
frontier here" — a statement the field needs measured examples of, given how routinely RL-in-DES
papers benchmark against weak open-loop baselines only.

## 4.3 Amortization as the practical contribution

Matching belief-MPC with a single forward pass is operationally meaningful even at zero outcome
premium: the MPC replans online at every decision, requires an explicit decision model and belief
machinery at run time, and its cost scales with the enumeration horizon; the learned policy is a
constant-time function of observations. Where decision latency, compute at the edge, or model
maintenance matter, an amortized policy certified non-inferior is a deployable artifact. The
prospective replication therefore reports outcome equivalence and computational cost as separate,
jointly interpretable endpoints.

## 4.4 Methodological lessons (paid for, in order)

1. **Report preregistered estimands separately.** Compound gates answer deployment questions;
   science needs the components.
2. **Zero-margin "non-inferiority" is superiority in disguise.** Our instrument audit showed the
   tail gate's 80%-power threshold exceeded the mean effect itself; margins must be prespecified
   with power analysis, and equivalence demonstrated by two-sided bounds, never inferred from
   non-significance.
3. **Comparator selection is where wins are manufactured.** Best-by-mean selection (never
   per-tape maxima), complete frontiers instead of curated baselines, and reselection inside
   every bootstrap resample are the difference between an estimand and an artifact.
4. **Fail-closed gates, not conventions.** Every integrity component (placebos, trajectory
   audits, resource ledgers) is code that fails when absent — a lesson learned from a default
   flag once read as a result.
5. **Sealed one-shot validation works.** Twice in this program, development-tape optimism died on
   sealed tapes; both deaths are in the record, and both made the surviving claims stronger.
6. **Machine-generate every reported number.** Two transcription-class errors in our own
   reporting were caught only because tables are built from hashed artifacts; the second was
   caused by fuzzy field matching — exact schema paths are part of the method.

## 4.5 Limitations

The two-product extension is a researcher-defined mechanism study motivated by the thesis's
documented aggregation of 21 ration classes into one product — not a calibrated representation of
the real portfolio; (ρ, s) ranges are an experimental envelope. Training and evaluation share the
three-cell demand family, so out-of-distribution robustness (including the thesis's own
operational risks, which were disabled throughout Program O) remains a preregistered but
unexecuted extension. Joint tail safety was not established and is not claimed. The event-tie
semantics of the source metric awaits domain confirmation; our dual-semantics sensitivity bounds
the exposure. The learned-adaptation and equivalence findings are replication-grade (frozen
policies, fresh sealed tapes, N=256 per cell), but the replication's compound verdict is a
guardrail STOP: worst-product fill against the classical controller could not be certified
non-inferior at the frozen −0.02 margin in any cell, so the learned policies are equivalent in
ReT while conceding a small, not-provably-bounded amount of worst-product balance. No compound
deployment claim is made, and external validity claims await the source author's answers.

## 4.6 Implications and next questions

For practitioners: do not commission a learner before establishing Levels 1–2 — every one of the
seven prior mechanism families in this program failed there, and no learner can manufacture value
that clairvoyance cannot find. For researchers: the interesting open question is no longer
"can RL beat a schedule" but *what a learned policy should retain between related campaigns* —
whether carrying inferred demand-mix beliefs (rather than weights) buys cold-start resilience, a
question with a clean causal design (persistent vs reset, independent-campaign null) that this
benchmark now makes executable. That study is gated, deliberately, behind the replication.
