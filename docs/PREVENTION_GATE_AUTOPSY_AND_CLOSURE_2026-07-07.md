# Prevention gate autopsy and Phase-1 closure — 2026-07-07

## Purpose

Consolidated closing document for the Ruta B preventive-learning investigation:
why the original causal gate failed, what replaced it, what it found, and what
is and is not claimed going forward. This supersedes ad-hoc references to
"prevention" scattered across the session's docs — the authoritative chain is
listed at the bottom.

## Why the splice gate failed (autopsy)

`R_full - R_reset(pre-risk)` replaces a policy's actions in the 4 weeks before
a target-risk event with a "calm action," reruns from the same seed, and
checks whether resilience (Garrido Excel ReT) drops. The estimand is sound in
principle. Two independent implementations of it (the original inlined Ruta B
counterfactual, and a from-scratch "corrected" reimplementation with an
exclusion halo) both produced false positives — a reactive PPO baseline with
no auxiliary loss scored 80-90% positive, and a Ruta B variant with a
contentless permuted label scored 67-78%. Root cause, confirmed by re-running
the ORIGINAL trusted script (`scripts/audit_track_b_risk_event_counterfactual.py`,
which correctly gives null results on the same checkpoints): the reimplementations
defined "calm action" as a simple statistic over the episode's own actions
(full-episode median, or an exclusion-halo mean that still degraded under
R24's high frequency). Under Case C, R24 fires roughly every 1.2 weeks, so
almost every pre-event window overlaps with legitimate reactive response to a
*different*, nearby event. Substituting a contaminated "calm" action removes
real reactive value, not preparation — producing a positive delta for any
sufficiently reactive policy, independent of whether it has any predictive
component at all.

**Lesson for any future causal-prevention gate in a frequent-event DES:** the
calm-action baseline must be validated with negative controls before trusting
its output. Minimum bar, all four required:

1. **Reactive-null control**: a policy with no risk-anticipation mechanism at
   all must score in the null band under the gate.
2. **Permuted/contentless-label control**: if the candidate policy has a
   predictive component, a version trained on a shuffled or constant version
   of the same label must score null.
3. **Placebo anchors**: event-free weeks matched on episode phase must score
   null.
4. **Cross-implementation agreement**: if a new gate reimplementation
   disagrees with a previously-validated one on the same frozen checkpoint,
   trust the previously-validated one and find the discrepancy before
   promoting any result.

## What replaced it: preventive-headroom ceiling tests

Rather than iterating on gate design indefinitely, the question was inverted:
does this environment have ANY preventive headroom at all, independent of any
particular gate's sensitivity? Two direct, gate-independent tests:

- **Forced-prep response surface** (`scripts/audit_prevention_headroom_sweep.py`):
  branch rollouts forcing a fixed posture before isolated real and placebo
  anchors, outcome restricted to orders causally exposed to the anchor
  (`[OPTj, OATj-or-now]` overlapping the event window) — not episode-wide ReT.
- **Clairvoyant PPO**: a policy trained and evaluated with the TRUE future
  risk label visible in its observation (a genuine upper bound via perfect
  information, not reward shaping).
- **Independent corroboration**: Codex's Gate v2
  (`scripts/audit_track_b_prevention_gate_v2.py`, event-tape replay/removal
  design — `event_on_off` and `forced_prep_sweep` modes) reached the same null
  conclusion the same day via a differently-implemented mechanism.

## Result

Both ceiling tests, and Codex's independent Gate v2, agree: **preventive
headroom ≈ 0** for R22 under the `track_b_v1` action contract in this DES. The
clean-physics (R22-only) tier of the forced-prep sweep showed an *exact* zero
(not just small) across 84 real + 96 placebo anchors, traced to a mechanistic
cause: R22's damage (exogenous recovery duration, direct operation knockout)
has no causal path from the dispatch/shift levers available to the policy.
Clairvoyant PPO with perfect foreknowledge did not beat the reactive baseline
and was the most resource-expensive variant tested.

## Standing verdict (do not re-litigate without new evidence)

- **No causal-prevention claim** for Track B under the studied action
  contract and risk configuration. This is a boundary result, not a training
  failure — reported as "this DES rewards fast adaptive recovery, not
  anticipatory preparation."
- **Ruta B's efficiency effect stands independently**: same ReT as reactive
  PPO at ~55% of its resource cost. Control ladder (λ=0 head-present,
  constant-label, permuted-label, true-label) shows the effect is
  architectural (from `RutaBAuxFeaturesExtractor`'s trunk), not from the
  auxiliary loss or its predictive content — confirmed by contrast with
  clairvoyant PPO (extra feature, default extractor, reactive's high-cost
  profile).
- **Reopening conditions**: only with a pre-registered environment change that
  gives preparation a genuine causal channel it currently lacks (e.g., a real
  lead time between a dispatch/inventory decision and its effect, or
  irreversible pre-positioning that reacting after the fact cannot substitute
  for) — not by tuning the gate or the architecture further under the current
  physics.

## Document chain (chronological, for citation)

1. `docs/TRACK_B_RUTA_B_LIVE_AUX_CASE_C_VERDICT_2026-07-06.md` — screen-scale apparent win (erratum added)
2. `docs/TRACK_B_RUTA_B_LIVE_AUX_CASE_C_CONFIRM_VERDICT_2026-07-06.md` — confirmatory-scale apparent win (erratum added)
3. `docs/TRACK_B_RUTA_B_COUNTERFACTUAL_GATE_AUDIT_2026-07-07.md` — adversarial controls, retraction
4. `docs/TRACK_B_RUTA_B_ESTABLISHED_GATE_FINAL_2026-07-07.md` — re-verification on the trusted gate, correct environment
5. `docs/TRACK_B_PREVENTION_GATE_V2_IMPLEMENTATION_2026-07-07.md` (Codex) — independent gate reimplementation, same null conclusion
6. `docs/TRACK_B_PREVENTIVE_HEADROOM_CEILING_VERDICT_2026-07-07.md` — ceiling tests, efficiency-ladder attribution
7. This document — consolidated closure and standing verdict
