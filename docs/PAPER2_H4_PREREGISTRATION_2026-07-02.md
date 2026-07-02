# Paper 2 pre-registration: retained learning (L_{t-1}) in Track B — 2026-07-02

**Status: PRE-REGISTERED, NOT LAUNCHED.** No experiment below has been
run at the time of writing. This document fixes the designs, arms,
seeds, metrics, and claim thresholds *before* any Paper 2 compute, so
that Paper 2 never accumulates the forking-paths debt Paper 1 had to
pay down (see `docs/REVIEWER2_DEEP_AUDIT_2026-07-01.md` Part 4 and
`docs/REV2_PRE_SUBMISSION_FIXES_2026-07-02.md`). Any deviation from
this document must be recorded here with a dated amendment note before
results are interpreted.

## Motivating pilot (already run; NOT part of Paper 2's confirmatory evidence)

`docs/H4_RETAINED_VS_RESET_VERDICT_2026-07-02.md`: 10 seeds
(8101-8110), 8 online-adaptation cycles x 3,000 timesteps,
`control_v1`, `adaptive_benchmark_v2`. Retained-minus-reset Excel ReT:
`obs_full` CI95 [+0.0000095, +0.0000525]; `obs_hidden` CI95
[+0.0000167, +0.0000819], 9/10 seeds positive. The pilot motivates the
study; it is treated as exploratory and will not be pooled with any
confirmatory run below.

## Research question

Does a Track B policy that retains parameters across sequential
disruption campaigns accumulate a performance advantage that (a) grows
with exposure, (b) survives regime shifts, and (c) exceeds retraining
from scratch at matched total compute? This is the operational
correlate of Garrido's L_{t-1} construct; it is explicitly NOT a claim
of organizational learning in the Levitt-March institutional sense.

## Fixed experimental backbone (all experiments)

- Env: `track_b_v1` 8D contract, observation v7, `control_v1` reward,
  h104, 168h step, stochastic PT, thesis year basis — identical to the
  Paper 1 canonical contract.
- Primary observation arm: **`obs_hidden`** (fields 30-36 masked, as in
  E2/H4). `obs_full` is reported as secondary only.
- Primary metric: **Excel ReT (`order_ret_excel`)**, evaluated on
  held-out CRN episodes with a frozen evaluation harness; training
  reward is never a reported outcome.
- Inference: the training seed is the inferential unit. 10 seeds per
  arm, drawn from a NEW block (9101-9110) — no reuse of pilot seeds
  8101-8110 or canonical seeds 1-10. Seed-clustered t-based CI95.
- Claim threshold (pre-registered, same as registry C12): a claim is
  made if and only if the seed-clustered CI95 lower bound of the
  relevant contrast is strictly > 0 under `obs_hidden`. Sign
  consistency (>= 8/10 seeds) is reported but is not by itself a claim
  basis.
- Every run writes to a dated `outputs/experiments/paper2_h4_*`
  directory; failed or aborted lanes are recorded in this doc, not
  deleted.

## E-P2-1: Dose-response (first experiment; go/no-go for Paper 2)

**Design.** Cycles C in {2, 4, 8, 16}, online budget fixed at 3,000
timesteps/cycle. Arms per C: retained, reset, frozen. 10 seeds
(9101-9110). Contrast: retained-minus-reset Excel ReT at final cycle,
per C.

**Pre-registered predictions.**
- H-dose: the retained-minus-reset delta is monotonically
  non-decreasing in C (tested by seed-clustered slope over C; claim
  requires slope CI95 lower bound > 0).
- Null outcome interpretation (fixed now): if the delta at C=16 is not
  larger than at C=8 (slope CI95 includes 0) the pilot effect is
  interpreted as warm-start/optimization noise, NOT learning, and
  Paper 2 pivots or stops. This is the go/no-go gate.

**Budget note.** Retained and reset arms consume identical total online
timesteps by construction at each C; frozen consumes zero.

## E-P2-2: Cross-regime retention (transfer vs interference)

Run only if E-P2-1 passes its gate.

**Design.** 4-phase campaign sequence: current -> increased -> severe ->
current (each phase 4 cycles x 3,000 timesteps). Arms: retained across
phases; reset at each phase boundary; frozen. 10 seeds (9201-9210).
Evaluation after each phase on that phase's regime AND re-evaluation on
phase-1 (current) at the end.

**Pre-registered contrasts.**
- Transfer: retained-minus-reset within each post-shift phase.
- Interference/forgetting: end-of-campaign performance on `current`
  (retained) minus phase-1 performance on `current` (same seeds) — a
  negative CI95 upper bound < 0 is claimed as forgetting.
- Either direction is publishable; the doc commits to reporting both.

## E-P2-3: Matched-compute control (the reviewer-killer comparison)

Run only if E-P2-1 passes its gate.

**Design.** At C=8: (a) retained (8 x 3,000 online on top of the
canonical checkpoint), (b) scratch-matched: trained from scratch for
60,000 + 24,000 = 84,000 timesteps (identical total gradient budget,
identical final-cycle evaluation), (c) canonical frozen. 10 seeds
(9301-9310).

**Pre-registered contrast.** Retained minus scratch-matched at equal
total compute. Claim of genuine path-dependence requires CI95 lower
bound > 0 under `obs_hidden`. If scratch-matched wins or ties, the
honest conclusion is "retention is a compute-efficiency convenience,
not path-dependence" — also publishable, pre-committed here.

## Reporting commitments

1. All three experiments reported regardless of sign (no file-drawer).
2. No new reward modes, observation versions, or action contracts may
   be introduced mid-study; any such change starts a new pre-registered
   amendment.
3. Effect sizes always reported relative to (a) the Paper 1 headline
   gain (+0.000438) and (b) the ReT ceiling (0.006944), to prevent
   small-effect overclaiming.
4. The Paper 2 manuscript claim vocabulary is fixed in advance:
   "retained online adaptation," "path-dependence at matched compute"
   (only if E-P2-3 passes), never "organizational learning" without a
   dedicated construct-validity section.

## Sequencing constraint

Nothing in this document launches until Paper 1 is submitted
(currently: E3 per-cell dense frontier running; then cold read; then
submission). First launch after that: E-P2-1 only.
