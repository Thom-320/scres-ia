# Paper 2 — Results (draft v1, 2026-07-18)

*(Prose around the machine-generated master table — `results_table.md`, built by
`build_results_table.py` from custodied artifacts with per-cell source hashes. Numbers here are
never hand-typed into prose without appearing in that table. Program Q slots into §3.5.)*

## 3.1 Physical opportunity is material and mechanism-specific (Level 1)

Under the safe per-tape oracle, the two-product extension exposes a resource-conserving
clairvoyant resilience gap of **H_PI = 0.152** (simultaneous safe LCB95 = 0.116) against the best
of all 65,536 open-loop calendars. The effect is the mechanism, not the environment: under
complete substitution — the exact fungibility null — the same oracle recovers **precisely zero**
headroom. Ordering-sensitivity replays (blocked-left / blocked-right event conventions) move the
safe bound by less than 0.002, so the opportunity is not an artifact of event-tie semantics.

## 3.2 Observable classical control converts the opportunity in mean (Level 2)

The frozen belief-MPC controller, using only non-privileged observations, converted a material
share of the ceiling on fresh sealed tapes: simultaneous mean-advantage lower bounds of
**+0.066 / +0.043 / +0.059** across the three demand-regime cells, with 44/42/46 of 48 tapes
favorable and **all 27 information placebos beaten** under exactly equal scheduled resources.
The preregistered joint tail gate — CVaR10 non-inferiority at zero margin — was not met in two
cells (simultaneous LCB95 −0.009 and −0.016), although the tail *point estimates* were positive
(+0.035, +0.020). A post-hoc instrument audit (numerics retained) showed that a zero-margin
lower-bound requirement is mathematically a superiority test whose 80%-power threshold at this
sample size is a true tail improvement of ≈ +0.079 — larger than the mean effect itself. We
therefore report Level 2 as: *mean conversion established; deployment-grade joint tail safety
not claimed.*

## 3.3 A recurrent policy learns the adaptation (Level 3)

Ten independently seeded recurrent policies — trained without access to the latent regime, tape
identity, or true cell parameters, and rewarded only by terminal canonical ReT — beat the
**complete** open-loop frontier on the held-out calibration block in all three cells:

- H_OL point estimates **+0.076 / +0.063 / +0.105**, simultaneous LCB95
  **+0.043 / +0.037 / +0.066**;
- **41 / 42 / 44 of 48** tapes favorable against the best-by-mean of 65,536 calendars;
- feedback certified: per-seed trajectory audits (≥8 distinct calendars, modal fraction ≤ 0.50),
  all three replacement policies (modal, phase-only, frequency-matched) executed and beaten;
- integrity exact: scheduled-resource deviation 0.0 across learners, classical controllers and
  the full frontier; demand-ledger identities exact; 144/144 raw matrices SHA-manifested; and
  **990 direct full-DES replays with zero failures** (maximum absolute error 5.6 × 10⁻¹⁶).

This is, to our knowledge, the first learned policy in this validated military DES family to show
material out-of-sample adaptive value under a complete open-loop comparator frontier and an
adversarial feedback certification.

## 3.4 No neural premium over structured belief control (Level 4)

Against the strongest frozen classical controller selected per cell, the learned policies show
**no incremental outcome value**: Δ_N point estimates **−0.0017 / −0.0027 / −0.0015** with lower
bounds −0.009 / −0.014 / −0.008, and only 1 / 0 / 2 of 10 optimizer seeds beating both
comparators simultaneously. The compound preregistered gate (which required a neural premium of
+0.01) therefore terminated the learner contract as `STOP_CALIBRATION_NOT_ELIGIBLE`; both
estimands are reported separately here exactly as they were preregistered. The differences from
belief-MPC are an order of magnitude smaller than the learned advantage over open-loop
scheduling: the network arrives at the structured controller's frontier — a single forward pass
where the MPC runs an online planning enumeration — and stops there.

## 3.5 Prospective replication (Program Q) — [SLOT]

A frozen-policy replication contract (`program_q_frozen_policy_replication_v1`) tests, on
entirely new tapes with the ten policies frozen by hash: (E1) replicated superiority over the
complete open-loop frontier, and (E2) the neural relation graded as premium, TOST equivalence
within ±0.01, or a non-inferiority floor. *(Outcome — PASS_PREMIUM / PASS_EQUIVALENT / BOUND /
STOP — and its table row are inserted here by the build script when the terminal artifact
exists; the framing of §3.1–3.4 does not depend on which outcome obtains.)* A mandatory
computational benchmark (per-decision latency, memory, planner cost at equal hardware)
accompanies the replication to quantify amortization value separately from outcome value.

## 3.6 Where the ladder localizes the value

Reading the four levels together: the mechanism (non-fungibility under shared capacity) creates
the value; observable information converts most of what is convertible; a generic recurrent
learner *re-learns* that conversion from experience without privileged access; and structured
decision theory already sits at the observable frontier, leaving no premium for the network to
collect. Each of the historical negative programs in this search (buffer/shift posture, queue
rationing, spatial allocation, risk-magnitude tailoring) failed at Level 1 or 2 — which is
precisely why end-to-end learners could never manufacture value there, and why the compound
"learner must also beat the OR controller" reading of Level 4 should not be mistaken for a
failure of Level 3.
